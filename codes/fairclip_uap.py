"""
fairclip_uap.py -- universal adversarial perturbations (UAP) for FairCLIP checkpoints.

Weaker attacker than per-image PGD: the attacker learns ONE fixed perturbation per
(checkpoint, budget) on TRAINING images, selects the epoch on VALIDATION images, and
applies it unchanged to every TEST image. No per-image gradient at deployment.

Perturbation types (all grayscale by default, i.e. one [1,1,224,224] delta shared by R,G,B):
  uap      one delta for every image, maximizing CE w.r.t. the true label over all
           training images. Label-agnostic at deployment. Because AUC is a ranking metric,
           a delta that shifts every P(glaucoma) by the same amount does nothing; this can
           only hurt AUC through image-dependent effects.
  uap_cc   class-conditional pair: delta_pos learned on glaucoma images (pushes them to
           non-glaucoma), delta_neg learned on non-glaucoma images (pushes them to
           glaucoma). At test time each image gets the delta of its TRUE class -- the same
           label knowledge as the untargeted per-image PGD in fairclip_full_eval.py, but
           with no per-image optimization.
Controls:
  noise    per-image uniform noise in [-eps, eps] (same as fairclip_full_eval.py)
  randsign one universal random +-eps grayscale pattern (same L_inf norm as the UAP)

8-bit: the learned deltas are rounded to integer gray levels (straight-through estimator
during optimization, so it optimizes the rounded delta). Clean images already sit on the
k/255 grid, so x + delta is on the grid too and p_adv == p_adv_q for uap / uap_cc /
randsign. Use budgets that are whole gray levels (k/255); anything else is rounded.

Transfer: every checkpoint's deltas are also evaluated on every other checkpoint
(p_transfer_q[source, kind, eps, sample]), which gives the CLIP -> FairCLIP transfer
numbers for universal perturbations for free.

Output (out_dir/):
  <label>.npz          one per TARGET checkpoint, same keys as fairclip_full_eval.py
                       (attacks = noise, randsign, uap, uap_cc) + transfer arrays, so
                       fairclip_analyze.py / fairclip_paired_tests.py run unchanged
                       (paired tests: --attack uap_cc --eps 0.00784313725 etc.)
  deltas/<label>.npz   learned deltas in gray levels (int8) + validation history

Run from FairCLIP/FairCLIP/ (needs fairclip_full_eval.py next to it):

  # smoke test (~minutes)
  python fairclip_uap.py --dataset_dir <DATA> --checkpoints <clip.pth> --labels "CLIP baseline" \
      --n_train 256 --n_val 128 --n_test 64 --epochs 1 --epsilons 2/255,8/255 --out_dir results_uap_smoke
  # full run
  python fairclip_uap.py --dataset_dir <DATA> --checkpoints <4 ckpts> \
      --labels "CLIP baseline,FairCLIP (Race),FairCLIP (Gender),FairCLIP (Ethnicity)" \
      --out_dir results_uap
  python fairclip_analyze.py --results_dir results_uap --out_dir analysis_uap
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

sys.path.append('.')

import clip
from src.modules import fair_vl_med_dataset
from fairclip_full_eval import (CLIP_MEAN, CLIP_STD, Unnormalize, parse_eps, load_model_auto_arch,
                                forward, quantize, encode_prompts)

KINDS = ('uap', 'uap_pos', 'uap_neg')


def safe_name(label):
    return ''.join(c if c.isalnum() else '_' for c in label).strip('_')


# ----------------------------------------------------------------------------- data
def cache_split(args, preprocess, subset, n, random_subset):
    """Preprocess a split once and keep it as uint8 on CPU (clean images are on the k/255 grid).
    Test uses the first n images in dataset order (same as fairclip_full_eval.py, so the
    'files' arrays match for paired tests); Training/Validation use a seeded random subset."""
    kw = dict(text_source='label') if subset == 'Training' else {}  # avoids the summarized-note csv
    ds = fair_vl_med_dataset(args.dataset_dir, preprocess, subset=subset, **kw)
    N = len(ds)
    if n <= 0 or n >= N:
        idx = np.arange(N)
    elif random_subset:
        idx = np.sort(np.random.default_rng(args.seed).choice(N, n, replace=False))
    else:
        idx = np.arange(n)
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=64, shuffle=False, num_workers=args.num_workers)
    X, L = [], []
    t0 = time.time()
    for images, _, la in loader:
        x = (images * CLIP_STD + CLIP_MEAN).clamp(0, 1)
        X.append(torch.round(x * 255).to(torch.uint8))
        L.append(la.long())
    X, L = torch.cat(X), torch.cat(L)
    print(f'  cached {subset}: {len(idx)}/{N} images ({X.numel()/1e9:.2f} GB uint8, {time.time()-t0:.0f}s), '
          f'prevalence {L[:, 0].float().mean():.3f}')
    return X, L, [ds.files[i] for i in idx]


def to_float(Xu8, device):
    return Xu8.to(device, non_blocking=True).float() / 255.0


def apply(x, d):
    return (x + d).clamp(0, 1)


def to_grid(d):
    """Straight-through rounding to whole gray levels."""
    return d + (torch.round(d * 255.0) / 255.0 - d).detach()


# ----------------------------------------------------------------------------- probs
@torch.no_grad()
def probs(model, un, tf, X, device, bs, delta=None, delta_by_label=None, y=None, q=False, emb=False):
    """P(glaucoma) for X (+delta). delta_by_label=(d_neg, d_pos) picks the delta of each image's true class."""
    out, embs = [], []
    for i in range(0, len(X), bs):
        x = to_float(X[i:i + bs], device)
        if delta is not None:
            x = apply(x, delta)
        elif delta_by_label is not None:
            yb = y[i:i + bs].to(device).view(-1, 1, 1, 1)
            x = apply(x, torch.where(yb == 1, delta_by_label[1], delta_by_label[0]))
        if q:
            x = quantize(x)
        p, _, f = forward(model, x, un, tf, return_emb=True)
        out.append(p.cpu())
        if emb:
            embs.append(f.cpu())
    p = torch.cat(out).double().numpy()
    return (p, torch.cat(embs)) if emb else p


# ----------------------------------------------------------------------------- learning
def learn_uaps(model, un, tf, Xtr, ytr, Xva, yva, eps, args, device):
    """Sign-SGD on one delta per kind, projected to the L_inf ball, cosine-decayed step,
    straight-through rounding to gray levels. Epoch chosen by validation AUC (8-bit)."""
    C = 1 if args.gray else 3
    H, W = Xtr.shape[2:]
    g = torch.Generator(device=device); g.manual_seed(args.seed)
    deltas = {k: (torch.rand((1, C, H, W), generator=g, device=device) * 2 - 1) * eps for k in KINDS}
    cpu_gen = torch.Generator(); cpu_gen.manual_seed(args.seed)
    n, bs = len(ytr), args.batch_size
    total = args.epochs * math.ceil(n / bs)
    lr0, it = args.lr_ratio * eps, 0
    yva_np = yva.numpy()

    def val_scores(ds):
        dq = {k: torch.round(v * 255) / 255 for k, v in ds.items()}
        a_uap = roc_auc_score(yva_np, probs(model, un, tf, Xva, device, args.eval_batch_size, delta=dq['uap']))
        a_cc = roc_auc_score(yva_np, probs(model, un, tf, Xva, device, args.eval_batch_size,
                                           delta_by_label=(dq['uap_neg'], dq['uap_pos']), y=yva))
        return a_uap, a_cc

    a0 = val_scores(deltas)
    hist = [dict(epoch=0, val_auc_uap=a0[0], val_auc_cc=a0[1])]
    best = {'uap': (a0[0], {k: deltas[k].clone() for k in KINDS}, 0),
            'cc': (a0[1], {k: deltas[k].clone() for k in KINDS}, 0)}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, generator=cpu_gen)
        loss_sum = {k: 0.0 for k in KINDS}
        for b in range(0, n, bs):
            ib = perm[b:b + bs]
            x = to_float(Xtr[ib], device)
            y = ytr[ib].to(device)
            lr = lr0 * (args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * it / total)))
            for k, mask in (('uap', None), ('uap_pos', y == 1), ('uap_neg', y == 0)):
                if mask is not None and not bool(mask.any()):
                    continue
                xb, yb = (x, y) if mask is None else (x[mask], y[mask])
                d = deltas[k].clone().requires_grad_(True)
                _, logits = forward(model, apply(xb, to_grid(d) if args.ste else d), un, tf)
                loss = F.cross_entropy(logits, yb)
                grad, = torch.autograd.grad(loss, d)
                with torch.no_grad():
                    deltas[k] = (deltas[k] + lr * grad.sign()).clamp(-eps, eps)
                loss_sum[k] += float(loss) * len(yb)
            it += 1
        a = val_scores(deltas)
        hist.append(dict(epoch=ep, val_auc_uap=a[0], val_auc_cc=a[1],
                         **{f'train_ce_{k}': loss_sum[k] / max(1, n) for k in KINDS}))
        if a[0] < best['uap'][0]:
            best['uap'] = (a[0], {k: deltas[k].clone() for k in KINDS}, ep)
        if a[1] < best['cc'][0]:
            best['cc'] = (a[1], {k: deltas[k].clone() for k in KINDS}, ep)
        el = time.time() - t0
        print(f'    eps {eps*255:5.2f}/255  epoch {ep}/{args.epochs}  val AUC uap {a[0]:.4f}  cc {a[1]:.4f}  '
              f'({el/60:.1f} min, ~{el/ep*(args.epochs-ep)/60:.1f} min left for this eps)')
    out = {'uap': best['uap'][1]['uap'], 'uap_pos': best['cc'][1]['uap_pos'], 'uap_neg': best['cc'][1]['uap_neg']}
    levels = {k: torch.round(v * 255).clamp(-round(eps * 255), round(eps * 255)).to(torch.int8).cpu().numpy()
              for k, v in out.items()}
    sel = dict(best_epoch_uap=best['uap'][2], best_val_auc_uap=best['uap'][0],
               best_epoch_cc=best['cc'][2], best_val_auc_cc=best['cc'][0])
    return levels, hist, sel


def phase_learn(args, ckpts, labels, epsilons, device, Xtr, Ltr, Xva, Lva):
    os.makedirs(os.path.join(args.out_dir, 'deltas'), exist_ok=True)
    ytr, yva = Ltr[:, 0], Lva[:, 0]
    for ckpt, label in zip(ckpts, labels):
        path = os.path.join(args.out_dir, 'deltas', f'{safe_name(label)}.npz')
        if os.path.exists(path) and not args.overwrite:
            print(f'\n[{label}] deltas exist, skipping learning ({path}); --overwrite to redo')
            continue
        print(f'\n[{label}] learning UAPs on {len(ytr)} training images')
        model, _ = load_model_auto_arch(ckpt, device, args.fallback_arch)
        un = Unnormalize(device)
        tf = encode_prompts_fixed(model, device)
        lv = {k: [] for k in KINDS}
        hists, sels = [], []
        for eps in epsilons:
            if abs(eps * 255 - round(eps * 255)) > 1e-6:
                print(f'    WARNING eps={eps} is not a whole gray level; the rounded delta uses '
                      f'{round(eps*255)} levels')
            levels, hist, sel = learn_uaps(model, un, tf, Xtr, ytr, Xva, yva, eps, args, device)
            for k in KINDS:
                lv[k].append(levels[k])
            hists.append(hist); sels.append(sel)
            print(f'    eps {eps*255:5.2f}/255 selected: {sel}')
        np.savez_compressed(path, label=label, checkpoint=ckpt, epsilons=np.array(epsilons),
                            **{k: np.stack(v) for k, v in lv.items()},
                            history=json.dumps(hists), selection=json.dumps(sels),
                            n_train=len(ytr), n_val=len(yva), epochs=args.epochs, lr_ratio=args.lr_ratio,
                            batch_size=args.batch_size, ste=args.ste, gray=args.gray, seed=args.seed)
        print(f'  saved {path}')
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


@torch.no_grad()
def encode_prompts_fixed(model, device):
    """Same prompts as the Test split of fair_vl_med_dataset ([non-glaucoma, glaucoma])."""
    tokens = torch.cat([clip.tokenize('A photo of non-glaucoma'), clip.tokenize('A photo of glaucoma')]).to(device)
    t = model.encode_text(tokens)
    return t / t.norm(dim=1, keepdim=True)


# ----------------------------------------------------------------------------- evaluation
def phase_eval(args, ckpts, labels, epsilons, device, Xte, Lte, files):
    yte = Lte[:, 0]
    y_np = yte.numpy()
    E, n = len(epsilons), len(yte)
    src = {}
    for label in labels:
        d = np.load(os.path.join(args.out_dir, 'deltas', f'{safe_name(label)}.npz'))
        if not np.allclose(d['epsilons'], epsilons):
            raise SystemExit(f'deltas for {label} have eps {d["epsilons"]}, expected {epsilons}')
        src[label] = {k: torch.from_numpy(d[k].astype(np.float32) / 255.0).to(device) for k in KINDS}  # [E,1,C,H,W]
    C = src[labels[0]]['uap'].shape[2]
    H, W = Xte.shape[2:]
    attacks = ['noise', 'randsign', 'uap', 'uap_cc']
    g_rs = torch.Generator(device=device); g_rs.manual_seed(args.seed + 1)
    randsign = [torch.round((torch.randint(0, 2, (1, C, H, W), generator=g_rs, device=device).float() * 2 - 1)
                            * eps * 255) / 255 for eps in epsilons]

    for ckpt, label in zip(ckpts, labels):
        print(f'\n[{label}] evaluating on {n} test images')
        model, _ = load_model_auto_arch(ckpt, device, args.fallback_arch)
        un = Unnormalize(device)
        tf = encode_prompts_fixed(model, device)
        bs = args.eval_batch_size
        p_clean, f_clean = probs(model, un, tf, Xte, device, bs, emb=True)
        p_clean_q = probs(model, un, tf, Xte, device, bs, q=True)
        sims_clean = np.zeros((n, 2))
        for i in range(0, n, bs):
            fb = f_clean[i:i + bs].to(device)
            sims_clean[i:i + bs] = (fb @ tf.T).double().cpu().numpy()
        A = len(attacks)
        p_adv = np.zeros((A, E, n)); p_adv_q = np.zeros((A, E, n)); emb_shift = np.zeros((A, E, n))
        g_noise = torch.Generator(device=device); g_noise.manual_seed(args.seed)
        for ei, eps in enumerate(epsilons):
            own = src[label]
            dl = {'randsign': randsign[ei], 'uap': own['uap'][ei],
                  'cc': (own['uap_neg'][ei], own['uap_pos'][ei])}
            for i in range(0, n, bs):
                x = to_float(Xte[i:i + bs], device)
                yb = yte[i:i + bs].to(device).view(-1, 1, 1, 1)
                fc = f_clean[i:i + bs].to(device)
                pshape = (x.shape[0], 1 if args.gray else 3, H, W)
                xs = {'noise': apply(x, (torch.rand(pshape, generator=g_noise, device=device) * 2 - 1) * eps),
                      'randsign': apply(x, dl['randsign']),
                      'uap': apply(x, dl['uap']),
                      'uap_cc': apply(x, torch.where(yb == 1, dl['cc'][1], dl['cc'][0]))}
                with torch.no_grad():
                    for ai, att in enumerate(attacks):
                        pa, _, fa = forward(model, xs[att], un, tf, return_emb=True)
                        p_adv[ai, ei, i:i + bs] = pa.cpu().numpy()
                        emb_shift[ai, ei, i:i + bs] = (1 - (fa * fc).sum(1)).cpu().numpy()
                        p_adv_q[ai, ei, i:i + bs] = forward(model, quantize(xs[att]), un, tf)[0].cpu().numpy()
        # transfer: every source's deltas on this target (8-bit; identical to float for grid deltas)
        S = len(labels)
        p_tr = np.zeros((S, 2, E, n))
        for si, sl in enumerate(labels):
            for ei in range(E):
                if sl == label:
                    p_tr[si, 0, ei] = p_adv_q[attacks.index('uap'), ei]
                    p_tr[si, 1, ei] = p_adv_q[attacks.index('uap_cc'), ei]
                    continue
                p_tr[si, 0, ei] = probs(model, un, tf, Xte, device, bs, delta=src[sl]['uap'][ei], q=True)
                p_tr[si, 1, ei] = probs(model, un, tf, Xte, device, bs, y=yte, q=True,
                                        delta_by_label=(src[sl]['uap_neg'][ei], src[sl]['uap_pos'][ei]))

        print(f'  clean AUC {roc_auc_score(y_np, p_clean):.4f}  (8-bit clean {roc_auc_score(y_np, p_clean_q):.4f})')
        print(f'  {"attack":>9s} ' + ' '.join(f'{e*255:7.2f}' for e in epsilons) + '   (x/255; 8-bit AUC)')
        for ai, att in enumerate(attacks):
            print(f'  {att:>9s} ' + ' '.join(f'{roc_auc_score(y_np, p_adv_q[ai, ei]):7.4f}' for ei in range(E)))
        for si, sl in enumerate(labels):
            if sl != label:
                print(f'  from {sl[:22]:>22s}: uap ' +
                      ' '.join(f'{roc_auc_score(y_np, p_tr[si, 0, ei]):.3f}' for ei in range(E)) + ' | cc ' +
                      ' '.join(f'{roc_auc_score(y_np, p_tr[si, 1, ei]):.3f}' for ei in range(E)))

        text_cos = float(tf[0] @ tf[1]); dnorm = float((tf[1] - tf[0]).norm())
        out = os.path.join(args.out_dir, f'{safe_name(label)}.npz')
        np.savez_compressed(out, model_label=label, checkpoint=ckpt, files=np.array(files),
                            labels=y_np.astype(np.int64), attrs=Lte[:, 1:5].numpy().astype(np.int64),
                            attr_names=np.array(['race', 'gender', 'ethnicity', 'language']),
                            p_clean=p_clean, p_clean_q=p_clean_q, sims_clean=sims_clean,
                            attacks=np.array(attacks), epsilons=np.array(epsilons),
                            p_adv=p_adv, p_adv_q=p_adv_q, emb_shift=emb_shift,
                            emb_clean=f_clean.numpy().astype(np.float32),
                            text_cos=text_cos, delta_norm=dnorm,
                            alpha_ratio=args.lr_ratio, steps=args.epochs, seed=args.seed, gray=args.gray,
                            transfer_sources=np.array(labels), transfer_kinds=np.array(['uap', 'uap_cc']),
                            p_transfer_q=p_tr)
        print(f'  saved {out}')
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--checkpoints', required=True, help='comma-separated; "none" = off-the-shelf CLIP')
    ap.add_argument('--labels', default='')
    ap.add_argument('--fallback_arch', default='ViT-B/16', choices=['ViT-B/16', 'ViT-L/14'])
    ap.add_argument('--epsilons', default='1/255,2/255,4/255,8/255,16/255')
    ap.add_argument('--n_train', default=0, type=int, help='0 = whole Training split')
    ap.add_argument('--n_val', default=0, type=int, help='0 = whole Validation split')
    ap.add_argument('--n_test', default=0, type=int, help='0 = whole Test split')
    ap.add_argument('--epochs', default=5, type=int)
    ap.add_argument('--batch_size', default=32, type=int, help='UAP optimization batch')
    ap.add_argument('--eval_batch_size', default=64, type=int)
    ap.add_argument('--lr_ratio', default=0.1, type=float, help='initial sign-SGD step = lr_ratio * eps')
    ap.add_argument('--lr_min_frac', default=0.05, type=float, help='final step as a fraction of the initial')
    ap.add_argument('--no_ste', dest='ste', action='store_false', help='optimize the float delta, round at the end')
    ap.add_argument('--rgb', dest='gray', action='store_false', help='per-channel delta (ablation)')
    ap.add_argument('--num_workers', default=2, type=int)
    ap.add_argument('--seed', default=0, type=int)
    ap.add_argument('--skip_learn', action='store_true', help='only evaluate existing out_dir/deltas/*.npz')
    ap.add_argument('--overwrite', action='store_true', help='re-learn deltas that already exist')
    ap.add_argument('--out_dir', default='results_uap')
    args = ap.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}   gray={args.gray}  ste={args.ste}')
    ckpts = ['' if c.strip().lower() == 'none' else c.strip() for c in args.checkpoints.split(',') if c.strip()]
    labels = ([l.strip() for l in args.labels.split(',')] if args.labels.strip()
              else [os.path.basename(c) if c else 'pretrained CLIP' for c in ckpts])
    assert len(labels) == len(ckpts), '--labels must match --checkpoints'
    epsilons = [parse_eps(e) for e in args.epsilons.split(',') if e.strip()]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args_uap.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=1)

    _, preprocess = clip.load(args.fallback_arch, device='cpu', jit=False)
    if not args.skip_learn:
        Xtr, Ltr, _ = cache_split(args, preprocess, 'Training', args.n_train, random_subset=True)
        Xva, Lva, _ = cache_split(args, preprocess, 'Validation', args.n_val, random_subset=True)
        phase_learn(args, ckpts, labels, epsilons, device, Xtr, Ltr, Xva, Lva)
        del Xtr, Ltr, Xva, Lva
    Xte, Lte, files = cache_split(args, preprocess, 'Test', args.n_test, random_subset=False)
    phase_eval(args, ckpts, labels, epsilons, device, Xte, Lte, files)


if __name__ == '__main__':
    main()
