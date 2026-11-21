"""
fairclip_analyze.py

Offline analysis of the per-sample .npz files written by fairclip_full_eval.py.
CPU only, runs in seconds to a couple of minutes. No model or dataset needed.

    python fairclip_analyze.py --results_dir results_full --out_dir analysis

Writes to out_dir:
  overall.csv        per model/attack/eps: AUC [95% CI], accuracy, flip rate,
                     attack success rate (ASR, on initially-correct samples),
                     ES-AUC per attribute, DPD and EOD per attribute
  subgroup.csv       per model/attack/eps/attribute/group: n, positives, clean and
                     attacked AUC with CIs, paired-bootstrap CI on the AUC drop, ASR
  flip_budget.csv    per model/attack/attribute/group: fraction of initially-correct
                     samples flipped at each epsilon, median flip budget
                     (+ Kruskal-Wallis p-value across groups)
  calibration.csv    per model: spread of clean P(glaucoma), prompt geometry
                     (||t_g - t_n|| and the bound it puts on P), class-wise margins
  mechanism.csv      per model/attack/eps: how far the attack moves the image
                     embedding (1 - cos) vs. how far it moves the margin
  summary.md         the numbers above in readable form
  fig_auc_vs_eps.png, fig_subgroup_pgd.png, fig_calibration.png

Metric definitions follow the FairCLIP repo (src/modules.py) where one exists:
  ES-AUC = AUC / (1 + sum_g |AUC_g - AUC|)
  DPD    = max_g P(pred=1 | g) - min_g P(pred=1 | g), pred = (p >= 0.5)
  EOD    = max(TPR gap, FPR gap) across groups (fairlearn's definition)
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, kruskal
from sklearn.metrics import roc_auc_score

GROUP_NAMES = {
    'race': {0: 'Asian', 1: 'Black', 2: 'White'},
    'gender': {0: 'Female', 1: 'Male'},
    'ethnicity': {0: 'non-Hispanic', 1: 'Hispanic'},
    'language': {0: 'English', 1: 'Spanish', 2: 'Other'},
}
MIN_POS_NEG = 5  # need at least this many positives AND negatives to report a group AUC


# ---------------------------------------------------------------- metrics
def auc_rows(scores, y):
    """AUC for each row of scores [B,n] against labels y [B,n] (Mann-Whitney, ties averaged)."""
    r = rankdata(scores, axis=1)
    n1 = y.sum(1)
    n0 = y.shape[1] - n1
    with np.errstate(invalid='ignore', divide='ignore'):
        return ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_idx(n, B, rng, y=None):
    """Stratified bootstrap indices so every resample keeps the class balance."""
    if y is None:
        return rng.integers(0, n, size=(B, n))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, (B, len(pos))), rng.choice(neg, (B, len(neg)))], axis=1)


def auc_ci(p, y, idx):
    b = auc_rows(p[idx], y[idx])
    return np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5), b


def es_auc(p, y, g):
    overall = roc_auc_score(y, p)
    dev = 0.0
    for v in np.unique(g):
        m = g == v
        if len(np.unique(y[m])) == 2:
            dev += abs(roc_auc_score(y[m], p[m]) - overall)
    return overall / (1 + dev)


def dpd_eod(p, y, g):
    pred = (p >= 0.5).astype(int)
    sel, tpr, fpr = [], [], []
    for v in np.unique(g):
        m = g == v
        sel.append(pred[m].mean())
        if (y[m] == 1).any():
            tpr.append(pred[m & (y == 1)].mean())
        if (y[m] == 0).any():
            fpr.append(pred[m & (y == 0)].mean())
    return max(sel) - min(sel), max(max(tpr) - min(tpr), max(fpr) - min(fpr))


def thresh_stats(pc, pa, y):
    yc = y.astype(bool)
    correct = (pc >= 0.5) == yc
    flip = (pc >= 0.5) != (pa >= 0.5)
    acc = ((pa >= 0.5) == yc).mean()
    asr = flip[correct].mean() if correct.any() else np.nan
    return acc, flip.mean(), asr


# ---------------------------------------------------------------- loading
def load(results_dir):
    runs = []
    for f in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
        d = dict(np.load(f, allow_pickle=False))
        d['model_label'] = str(d['model_label'])
        d['attacks'] = [str(a) for a in d['attacks']]
        d['attr_names'] = [str(a) for a in d['attr_names']]
        runs.append(d)
        print(f'loaded {f}: {d["model_label"]}, n={len(d["labels"])}, '
              f'attacks={d["attacks"]}, eps={np.round(d["epsilons"], 5).tolist()}')
    if not runs:
        raise SystemExit(f'no .npz files in {results_dir}')
    return runs


def conditions(d, use_q):
    """Yields (attack, eps, adv_probs) including a clean 'none' row."""
    yield 'none', 0.0, d['p_clean']
    key = 'p_adv_q' if use_q else 'p_adv'
    for ai, a in enumerate(d['attacks']):
        for ei, e in enumerate(d['epsilons']):
            yield a, float(e), d[key][ai, ei]


# ---------------------------------------------------------------- analyses
def overall_table(runs, B, seed):
    rows = []
    for d in runs:
        y, pc, attrs = d['labels'], d['p_clean'], d['attrs']
        idx = boot_idx(len(y), B, np.random.default_rng(seed), y)
        variants = [('float', False)] + ([('8bit', True)] if 'p_adv_q' in d else [])
        for vname, use_q in variants:
            for att, eps, pa in conditions(d, use_q):
                if vname == '8bit' and att == 'none':
                    continue
                lo, hi, _ = auc_ci(pa, y, idx)
                acc, flip, asr = thresh_stats(pc, pa, y)
                row = dict(model=d['model_label'], pixels=vname, attack=att, epsilon=eps,
                           eps_x255=eps * 255, n=len(y), auc=roc_auc_score(y, pa),
                           auc_lo=lo, auc_hi=hi, acc=acc, flip_rate=flip, asr=asr)
                for j, an in enumerate(d['attr_names']):
                    row[f'es_auc_{an}'] = es_auc(pa, y, attrs[:, j])
                    row[f'dpd_{an}'], row[f'eod_{an}'] = dpd_eod(pa, y, attrs[:, j])
                rows.append(row)
    return pd.DataFrame(rows)


def subgroup_table(runs, B, seed):
    rows = []
    for d in runs:
        y, pc, attrs = d['labels'], d['p_clean'], d['attrs']
        for j, an in enumerate(d['attr_names']):
            for code, gname in GROUP_NAMES[an].items():
                m = attrs[:, j] == code
                yg = y[m]
                npos, nneg = int(yg.sum()), int((1 - yg).sum())
                if min(npos, nneg) < MIN_POS_NEG:
                    print(f'  skip {d["model_label"]} {an}={gname}: {npos} pos / {nneg} neg')
                    continue
                idx = boot_idx(m.sum(), B, np.random.default_rng(seed), yg)
                pcg = pc[m]
                lo_c, hi_c, bc = auc_ci(pcg, yg, idx)
                for att, eps, pa in conditions(d, False):
                    pag = pa[m]
                    lo_a, hi_a, ba = auc_ci(pag, yg, idx)
                    drop_b = bc - ba  # paired: same resamples
                    _, _, asr = thresh_stats(pcg, pag, yg)
                    rows.append(dict(model=d['model_label'], attack=att, epsilon=eps, attribute=an,
                                     group=gname, n=int(m.sum()), n_pos=npos,
                                     auc_clean=roc_auc_score(yg, pcg), auc_clean_lo=lo_c, auc_clean_hi=hi_c,
                                     auc_adv=roc_auc_score(yg, pag), auc_adv_lo=lo_a, auc_adv_hi=hi_a,
                                     auc_drop=roc_auc_score(yg, pcg) - roc_auc_score(yg, pag),
                                     drop_lo=np.nanpercentile(drop_b, 2.5),
                                     drop_hi=np.nanpercentile(drop_b, 97.5), asr=asr))
    return pd.DataFrame(rows)


def flip_budget_table(runs):
    """For each initially-correct sample: smallest epsilon in the grid at which the
    attack makes it wrong. Attacks at different eps are separate runs, so this is
    the first eps (in increasing order) at which the sample is misclassified."""
    rows = []
    for d in runs:
        y, pc, attrs = d['labels'].astype(bool), d['p_clean'], d['attrs']
        eps = np.asarray(d['epsilons'], dtype=float)
        order = np.argsort(eps)
        correct = (pc >= 0.5) == y
        for ai, att in enumerate(d['attacks']):
            wrong = (d['p_adv'][ai][order] >= 0.5) != y[None, :]  # [E,n]
            first = np.where(wrong.any(0), wrong.argmax(0), len(eps))  # index into sorted eps; E = never
            for j, an in enumerate(d['attr_names']):
                samples, names = [], []
                for code, gname in GROUP_NAMES[an].items():
                    m = (attrs[:, j] == code) & correct
                    if m.sum() < 2 * MIN_POS_NEG:
                        continue
                    fg = first[m]
                    samples.append(fg); names.append(gname)
                    med = np.median(fg)
                    row = dict(model=d['model_label'], attack=att, attribute=an, group=gname,
                               n_correct=int(m.sum()),
                               median_flip_eps=(eps[order][int(np.ceil(med))] if med < len(eps) else np.inf))
                    for k, e in enumerate(eps[order]):
                        row[f'flipped@{e:.5f}'] = (fg <= k).mean()
                    rows.append(row)
                if len(samples) >= 2 and len(np.unique(np.concatenate(samples))) > 1:
                    p = kruskal(*samples).pvalue
                    for r in rows[-len(samples):]:
                        r['kruskal_p'] = p
    return pd.DataFrame(rows)


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def calibration_table(runs):
    rows = []
    for d in runs:
        pc, y, s = d['p_clean'], d['labels'], d['sims_clean']
        margin = s[:, 1] - s[:, 0]
        m1, m0 = margin[y == 1], margin[y == 0]
        pooled = np.sqrt((m1.var(ddof=1) + m0.var(ddof=1)) / 2)
        q = np.percentile(pc, [0, 1, 25, 50, 75, 99, 100])
        geo = {}
        if 'delta_norm' in d:
            dn = float(d['delta_norm'])
            geo = dict(text_cos=float(d['text_cos']), delta_norm=dn,
                       p_bound_lo=1 / (1 + np.exp(dn)), p_bound_hi=1 / (1 + np.exp(-dn)))
        rows.append(dict(model=d['model_label'], n=len(y), prevalence=y.mean(), **geo,
                         margin_mean_pos=m1.mean(), margin_mean_neg=m0.mean(),
                         margin_gap=m1.mean() - m0.mean(), margin_dprime=(m1.mean() - m0.mean()) / pooled,
                         clean_auc=roc_auc_score(y, pc), acc_at_0p5=((pc >= 0.5) == y).mean(),
                         frac_pred_pos=(pc >= 0.5).mean(),
                         p_min=q[0], p_q01=q[1], p_q25=q[2], p_median=q[3], p_q75=q[4], p_q99=q[5], p_max=q[6],
                         frac_within_0p001=(np.abs(pc - 0.5) < 0.001).mean(),
                         frac_within_0p01=(np.abs(pc - 0.5) < 0.01).mean(),
                         margin_mean=margin.mean(), margin_std=margin.std(),
                         cos_glaucoma_mean=s[:, 1].mean(), cos_nonglaucoma_mean=s[:, 0].mean()))
    return pd.DataFrame(rows)


def mechanism_table(runs):
    rows = []
    for d in runs:
        if 'emb_shift' not in d:
            continue
        y = d['labels']
        sign = 2 * y - 1  # +1 if moving the margin up is "toward correct"
        m_clean = d['sims_clean'][:, 1] - d['sims_clean'][:, 0]
        ref = np.nan  # median L2 distance between clean embeddings of two random test images
        if 'emb_clean' in d:
            rng = np.random.default_rng(0)
            i, j = rng.integers(0, len(y), (2, 20000))
            keep = i != j
            ref = np.median(np.linalg.norm(d['emb_clean'][i[keep]] - d['emb_clean'][j[keep]], axis=1))
        for ai, att in enumerate(d['attacks']):
            for ei, eps in enumerate(d['epsilons']):
                sh = d['emb_shift'][ai, ei]
                dm = (logit(d['p_adv'][ai, ei]) - m_clean) * sign  # negative = pushed toward wrong class
                rows.append(dict(model=d['model_label'], attack=att, epsilon=float(eps), eps_x255=float(eps) * 255,
                                 emb_shift_median=np.median(sh), emb_l2_median=np.median(np.sqrt(2 * np.clip(sh, 0, None))),
                                 margin_move_median=np.median(dm),
                                 margin_move_over_gap=np.median(dm) / (m_clean[y == 1].mean() - m_clean[y == 0].mean()),
                                 ref_pair_l2_median=ref,
                                 shift_over_ref=np.median(np.sqrt(2 * np.clip(sh, 0, None))) / ref))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def plots(runs, ov, sg, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = ov['model'].unique().tolist()
    attacks = [a for a in ov['attack'].unique() if a != 'none']
    colors = dict(zip(models, plt.cm.tab10.colors))

    fig, axes = plt.subplots(1, len(attacks), figsize=(5 * len(attacks), 4.2), sharey=True, squeeze=False)
    for ax, att in zip(axes[0], attacks):
        for m in models:
            for pix, ls in [('float', '-'), ('8bit', '--')]:
                s = ov[(ov.model == m) & (ov.attack == att) & (ov.pixels == pix)].sort_values('epsilon')
                if not len(s):
                    continue
                ax.plot(s.eps_x255, s.auc, ls, marker='o', ms=3, color=colors[m],
                        label=f'{m}' + (' (8-bit)' if pix == '8bit' else ''))
                if pix == 'float':
                    ax.fill_between(s.eps_x255, s.auc_lo, s.auc_hi, color=colors[m], alpha=0.15)
            clean = ov[(ov.model == m) & (ov.attack == 'none')].auc.iloc[0]
            ax.axhline(clean, color=colors[m], lw=0.8, alpha=0.5)
        ax.axhline(0.5, color='gray', ls=':', lw=0.8)
        ax.axvline(0.5, color='gray', ls='-.', lw=0.8)
        ax.set_xscale('log'); ax.set_title(att); ax.set_xlabel('epsilon (x 1/255)')
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel('AUC'); axes[0][0].legend(fontsize=7)
    fig.suptitle('AUC vs. budget (band: 95% bootstrap CI; dash-dot line: half an 8-bit level)')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig_auc_vs_eps.png'), dpi=160); plt.close(fig)

    main_att = 'pgd' if (sg.attack == 'pgd').any() else next(a for a in sg.attack.unique() if a != 'none')
    s = sg[sg.attack.isin([main_att, 'none'])]
    attrs = s.attribute.unique().tolist()
    fig, axes = plt.subplots(len(models), len(attrs), figsize=(4 * len(attrs), 3.4 * len(models)),
                             sharey=True, squeeze=False)
    for r, m in enumerate(models):
        for c, an in enumerate(attrs):
            ax = axes[r][c]
            for g in s[(s.model == m) & (s.attribute == an)].group.unique():
                gg = s[(s.model == m) & (s.attribute == an) & (s.group == g) & (s.attack == main_att)].sort_values('epsilon')
                n = int(gg.n.iloc[0]) if len(gg) else 0
                ax.errorbar(gg.epsilon * 255, gg.auc_adv, yerr=[gg.auc_adv - gg.auc_adv_lo, gg.auc_adv_hi - gg.auc_adv],
                            marker='o', ms=3, capsize=2, label=f'{g} (n={n})')
            ax.set_xscale('log'); ax.axhline(0.5, color='gray', ls=':', lw=0.8); ax.grid(alpha=0.3)
            ax.set_title(f'{m}: {an}', fontsize=9); ax.legend(fontsize=7)
            if r == len(models) - 1:
                ax.set_xlabel(f'{main_att.upper()} epsilon (x 1/255)')
    axes[0][0].set_ylabel(f'subgroup AUC under {main_att.upper()}')
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig_subgroup_pgd.png'), dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for d in runs:
        m = d['model_label']
        axes[0].hist(d['p_clean'], bins=80, alpha=0.5, color=colors[m], label=m)
        s_ = d['sims_clean']
        axes[1].hist(s_[:, 1] - s_[:, 0], bins=80, alpha=0.5, color=colors[m], label=m)
    axes[0].axvline(0.5, color='k', lw=0.8); axes[0].set_xlabel('clean P(glaucoma)')
    axes[1].axvline(0.0, color='k', lw=0.8); axes[1].set_xlabel('cos(img, "glaucoma") - cos(img, "non-glaucoma")')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig_calibration.png'), dpi=160); plt.close(fig)


# ---------------------------------------------------------------- report
def summary_md(ov, sg, fb, cal, out_dir):
    L = ['# FairCLIP attack analysis: summary\n']
    L.append('## Clean-score calibration\n')
    L.append(cal.round(4).to_markdown(index=False))
    L.append('\nIf P(glaucoma) sits in a very narrow band around 0.5, the 0.5 threshold is fragile and flip '
             'rate mostly reflects clean accuracy, not robustness. Saturated flip rate under a strong attack '
             'should approximately equal clean accuracy at 0.5 (the attack can only flip correct predictions).\n')
    cols = ['model', 'pixels', 'attack', 'eps_x255', 'auc', 'auc_lo', 'auc_hi', 'acc', 'flip_rate', 'asr',
            'es_auc_race', 'es_auc_gender', 'es_auc_ethnicity', 'es_auc_language']
    L.append('## Overall\n')
    L.append(ov[[c for c in cols if c in ov]].round(4).to_markdown(index=False))
    main_att = 'pgd' if (sg.attack == 'pgd').any() else next(a for a in sg.attack.unique() if a != 'none')
    L.append(f'\n## Subgroups: AUC drop with paired 95% CI ({main_att})\n')
    t = sg[sg.attack == main_att][['model', 'epsilon', 'attribute', 'group', 'n', 'n_pos', 'auc_clean',
                                 'auc_adv', 'auc_drop', 'drop_lo', 'drop_hi', 'asr']]
    L.append(t.round(4).to_markdown(index=False))
    L.append('\nRead the drop CIs before claiming a disparity: groups whose CIs overlap are not '
             'distinguishable at this sample size.\n')
    L.append('## Flip budget by group (initially-correct samples)\n')
    L.append(fb.round(4).to_markdown(index=False))
    open(os.path.join(out_dir, 'summary.md'), 'w').write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results_full')
    ap.add_argument('--out_dir', default='analysis')
    ap.add_argument('--n_boot', default=1000, type=int)
    ap.add_argument('--seed', default=0, type=int)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = load(args.results_dir)
    cal = calibration_table(runs)
    print('\ncalibration:\n', cal.round(4).T.to_string())
    ov = overall_table(runs, args.n_boot, args.seed)
    sg = subgroup_table(runs, args.n_boot, args.seed)
    fb = flip_budget_table(runs)
    mech = mechanism_table(runs)
    if len(mech):
        print('\nmechanism (median over samples):\n', mech.round(5).to_string(index=False))
    for name, df in [('overall', ov), ('subgroup', sg), ('flip_budget', fb), ('calibration', cal),
                     ('mechanism', mech)]:
        df.to_csv(os.path.join(args.out_dir, f'{name}.csv'), index=False)
    try:
        summary_md(ov, sg, fb, cal, args.out_dir)
    except ImportError:
        print('pip install tabulate for summary.md')
    plots(runs, ov, sg, args.out_dir)
    print(f'\nwrote CSVs, summary.md and figures to {args.out_dir}/')
    print(ov[ov.pixels == 'float'][['model', 'attack', 'eps_x255', 'auc', 'auc_lo', 'auc_hi', 'acc', 'asr']]
          .round(4).to_string(index=False))


if __name__ == '__main__':
    main()
