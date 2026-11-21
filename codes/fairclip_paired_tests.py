"""
fairclip_paired_tests.py

Formal tests for the comparisons the report makes, using the per-sample .npz
files from fairclip_full_eval.py. CPU only, no GPU rerun.

Why paired: every model is evaluated on the SAME 1,883 images, and clean and
attacked scores come from the SAME images. Resampling images once and computing
both quantities on that resample removes the image-to-image variation that the
two quantities share, so differences get much tighter (and honest) intervals
than comparing two separate confidence intervals by eye.

Three families of tests, at the chosen attack and budgets:

  model   AUC(model) - AUC(reference) under attack, and the same for the clean
          AUC and for the drop. Reference defaults to the CLIP baseline.
  gap     within one model: drop(group A) - drop(group B), e.g. Black - White.
          Groups are resampled independently (different patients), each
          stratified by label; clean and attacked share the resample.
  did     difference-in-differences: gap(model) - gap(reference) on the same
          resamples. Answers "does this FairCLIP variant change the disparity
          under attack compared with CLIP?"

Each row gets a two-sided bootstrap p-value. p_holm applies Holm's correction
within each family across all models, budgets and contrasts, so it is safe to
read the table as a whole.

    python fairclip_paired_tests.py --results_dir results_gray --out_dir analysis_gray
    python fairclip_paired_tests.py --results_dir results_rgb  --out_dir analysis_rgb
    python fairclip_paired_tests.py --results_dir results_uap  --out_dir paired_uap --attack uap --eps 1/255,2/255
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

CONTRASTS = [  # (attribute index, attribute name, group A code, name A, group B code, name B)
    (0, 'race', 1, 'Black', 2, 'White'),
    (0, 'race', 0, 'Asian', 2, 'White'),
    (1, 'gender', 0, 'Female', 1, 'Male'),
    (2, 'ethnicity', 1, 'Hispanic', 0, 'non-Hispanic'),
    (3, 'language', 1, 'Spanish', 0, 'English'),
    (3, 'language', 2, 'Other', 0, 'English'),
]


def auc_rows(scores, y):
    r = rankdata(scores, axis=1)
    n1 = y.sum(1)
    n0 = y.shape[1] - n1
    return ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def strat_idx(idx_pool, y, B, rng):
    """Stratified bootstrap of the indices in idx_pool, keeping its class balance."""
    pos = idx_pool[y[idx_pool] == 1]
    neg = idx_pool[y[idx_pool] == 0]
    return np.concatenate([rng.choice(pos, (B, len(pos))), rng.choice(neg, (B, len(neg)))], axis=1)


def auc_of(p, y, idx):
    return auc_rows(p[idx], y[idx])


def summarize(est, boot):
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return dict(estimate=est, lo=lo, hi=hi, p_boot=min(1.0, max(p, 1 / len(boot))))


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load(results_dir):
    runs = {}
    for f in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
        d = dict(np.load(f, allow_pickle=False))
        runs[str(d['model_label'])] = d
    files = None
    for k, d in runs.items():
        if files is None:
            files = d['files']
        elif not np.array_equal(files, d['files']):
            raise SystemExit(f'{k} was evaluated on different images; paired tests need the same test set')
    return runs


def adv_probs(d, attack, eps):
    ai = [str(a) for a in d['attacks']].index(attack)
    ei = int(np.argmin(np.abs(d['epsilons'] - eps)))
    if not np.isclose(d['epsilons'][ei], eps):
        raise SystemExit(f'eps {eps} not in run (have {d["epsilons"]})')
    return d['p_adv'][ai, ei]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--attack', default='pgd')
    ap.add_argument('--eps', default='0.0002,0.0005')
    ap.add_argument('--reference', default='CLIP baseline')
    ap.add_argument('--n_boot', default=2000, type=int)
    ap.add_argument('--seed', default=0, type=int)
    a = ap.parse_args()
    eps_list = [float(e.split('/')[0]) / float(e.split('/')[1]) if '/' in e else float(e)
                for e in a.eps.split(',') if e.strip()]

    runs = load(a.results_dir)
    if a.reference not in runs:
        raise SystemExit(f'reference "{a.reference}" not found; have {list(runs)}')
    ref = runs[a.reference]
    y = ref['labels']
    attrs = ref['attrs']
    n = len(y)
    rng = np.random.default_rng(a.seed)
    rows = []

    # ---- model vs reference (one shared resample of all images)
    idx_all = strat_idx(np.arange(n), y, a.n_boot, rng)
    for eps in eps_list:
        pr_c, pr_a = ref['p_clean'], adv_probs(ref, a.attack, eps)
        br_c, br_a = auc_of(pr_c, y, idx_all), auc_of(pr_a, y, idx_all)
        for name, d in runs.items():
            if name == a.reference:
                continue
            pm_c, pm_a = d['p_clean'], adv_probs(d, a.attack, eps)
            bm_c, bm_a = auc_of(pm_c, y, idx_all), auc_of(pm_a, y, idx_all)
            from sklearn.metrics import roc_auc_score as AUC
            for what, est, boot in [
                ('auc_clean', AUC(y, pm_c) - AUC(y, pr_c), bm_c - br_c),
                ('auc_adv', AUC(y, pm_a) - AUC(y, pr_a), bm_a - br_a),
                ('drop', (AUC(y, pm_c) - AUC(y, pm_a)) - (AUC(y, pr_c) - AUC(y, pr_a)),
                 (bm_c - bm_a) - (br_c - br_a)),
            ]:
                rows.append(dict(family='model', model=name, reference=a.reference, attack=a.attack,
                                 epsilon=eps, contrast=what, **summarize(est, boot)))

    # ---- group gaps within a model, and difference-in-differences vs reference
    from sklearn.metrics import roc_auc_score as AUC
    for (j, an, ga, na, gb, nb) in CONTRASTS:
        ia = np.where(attrs[:, j] == ga)[0]
        ib = np.where(attrs[:, j] == gb)[0]
        if min(y[ia].sum(), (1 - y[ia]).sum(), y[ib].sum(), (1 - y[ib]).sum()) < 5:
            continue
        sa, sb = strat_idx(ia, y, a.n_boot, rng), strat_idx(ib, y, a.n_boot, rng)
        for eps in eps_list:
            gap_boot, gap_est = {}, {}
            for name, d in runs.items():
                pc, pa = d['p_clean'], adv_probs(d, a.attack, eps)
                drop_a = auc_of(pc, y, sa) - auc_of(pa, y, sa)
                drop_b = auc_of(pc, y, sb) - auc_of(pa, y, sb)
                gap_boot[name] = drop_a - drop_b
                gap_est[name] = ((AUC(y[ia], pc[ia]) - AUC(y[ia], pa[ia]))
                                 - (AUC(y[ib], pc[ib]) - AUC(y[ib], pa[ib])))
                rows.append(dict(family='gap', model=name, reference='', attack=a.attack, epsilon=eps,
                                 attribute=an, contrast=f'{na}-{nb}', **summarize(gap_est[name], gap_boot[name])))
            for name in runs:
                if name == a.reference:
                    continue
                rows.append(dict(family='did', model=name, reference=a.reference, attack=a.attack, epsilon=eps,
                                 attribute=an, contrast=f'{na}-{nb}',
                                 **summarize(gap_est[name] - gap_est[a.reference],
                                             gap_boot[name] - gap_boot[a.reference])))

    df = pd.DataFrame(rows)
    df['p_holm'] = np.nan
    for fam, g in df.groupby('family'):
        df.loc[g.index, 'p_holm'] = holm(g.p_boot.values)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, 'paired_tests.csv')
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    print(df.drop(columns=['reference', 'attack']).round(4).to_string(index=False))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
