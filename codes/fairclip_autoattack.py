"""
fairclip_autoattack.py  (v2)

APGD-CE, the main attack of AutoAttack (Croce & Hein 2020), run on EVERY test
image and keeping the highest-loss point, so it is directly comparable to our
PGD and to AUC. Output is the same .npz format as fairclip_full_eval.py (attack
name 'apgd'), so fairclip_analyze.py, fairclip_paired_tests.py and
make_tex_numbers.py work unchanged.

Why not the full AutoAttack wrapper: it only attacks images that are correctly
classified at the 0.5 threshold and stops once each one flips. That suits
accuracy, not AUC, and three of our four checkpoints predict almost every image
"healthy", so it would leave most glaucoma images untouched. Its other members
also do not fit a 2-class rule (APGD-DLR needs >= 3 classes; APGD-T/FAB-T have a
single target; Square stops at misclassification).

Per-channel (RGB), like AutoAttack itself; compare with the RGB PGD table.

    pip install git+https://github.com/fra31/auto-attack
    python fairclip_autoattack.py --dataset_dir <DATA> \
        --checkpoints <clip>,<race>,<gender>,<ethnicity> \
        --labels "CLIP baseline,FairCLIP (Race),FairCLIP (Gender),FairCLIP (Ethnicity)" \
        --from_dir results_rgb --out_dir results_apgd --epsilons 0.0002,0.0005 --n_iter 100
    python fairclip_analyze.py      --results_dir results_apgd --out_dir analysis_apgd
    python fairclip_paired_tests.py --results_dir results_apgd --out_dir analysis_apgd --attack apgd
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

sys.path.append('.')
from src.modules import fair_vl_med_dataset
from fairclip_full_eval import load_model_auto_arch, Unnormalize, forward, encode_prompts, parse_eps, quantize


class LogitWrapper(torch.nn.Module):
    def __init__(self, model, un, text_feats):
        super().__init__()
        self.model, self.un, self.text_feats = model, un, text_feats

    def forward(self, x):
        return forward(self.model, x, self.un, self.text_feats)[1]


def safe(label):
    return ''.join(c if c.isalnum() else '_' for c in label).strip('_')


def run_model(args, ckpt, label, epsilons, device):
    from autoattack.autopgd_base import APGDAttack

    path_in = os.path.join(args.from_dir, f'{safe(label)}.npz')
    if not os.path.exists(path_in):
        raise SystemExit(f'missing {path_in}; run fairclip_full_eval.py first')
    base = dict(np.load(path_in, allow_pickle=False))
    y_np = base['labels']
    n = len(y_np)

    model, preprocess = load_model_auto_arch(ckpt, device, args.fallback_arch)
    un = Unnormalize(device)
    full = fair_vl_med_dataset(args.dataset_dir, preprocess, subset=args.eval_set)
    loader = DataLoader(Subset(full, list(range(n))), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    text_feats = encode_prompts(model, full, device)
    net = LogitWrapper(model, un, text_feats).eval()
    emb_clean = torch.from_numpy(base['emb_clean']) if 'emb_clean' in base else None

    E = len(epsilons)
    p_adv, p_adv_q, emb_shift = np.zeros((1, E, n)), np.zeros((1, E, n)), np.zeros((1, E, n))
    print(f'\n[{label}] n={n}, {E} budgets x {args.n_iter} iterations')

    for ei, eps in enumerate(epsilons):
        apgd = APGDAttack(net, n_iter=args.n_iter, norm='Linf', n_restarts=1, eps=float(eps),
                          loss='ce', seed=args.seed, device=device)
        t0, i0 = time.time(), 0
        for bi, (images, _, la) in enumerate(loader):
            x = un.to_pixels(images.to(device))
            y = la[:, 0].to(device).long()
            apgd.init_hyperparam(x)
            torch.manual_seed(args.seed + bi)
            x_best = apgd.attack_single_run(x, y)[0].detach()  # highest-CE point for every image
            b = x.shape[0]
            with torch.no_grad():
                pa, _, fa = forward(model, x_best, un, text_feats, return_emb=True)
                p_adv[0, ei, i0:i0 + b] = pa.cpu().numpy()
                p_adv_q[0, ei, i0:i0 + b] = forward(model, quantize(x_best), un, text_feats)[0].cpu().numpy()
                if emb_clean is not None:
                    emb_shift[0, ei, i0:i0 + b] = 1 - (fa.cpu() * emb_clean[i0:i0 + b]).sum(1).numpy()
            i0 += b
            if (bi + 1) % 5 == 0 or i0 == n:
                el = time.time() - t0
                print(f'  eps {eps*255:.3f}/255: {i0}/{n}  ({el/60:.1f} min, ~{el/i0*(n-i0)/60:.1f} min left)', flush=True)
        print(f'  eps {eps:.5f}: APGD AUC {roc_auc_score(y_np, p_adv[0, ei]):.4f}   '
              f'8-bit {roc_auc_score(y_np, p_adv_q[0, ei]):.4f}', flush=True)

    out = dict(base)
    out.update(attacks=np.array(['apgd']), epsilons=np.array(epsilons, dtype=float),
               p_adv=p_adv, p_adv_q=p_adv_q, emb_shift=emb_shift, apgd_n_iter=args.n_iter)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'{safe(label)}.npz')
    np.savez_compressed(path, **out)
    print(f'  saved {path}')
    del model, net
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--labels', default='')
    ap.add_argument('--from_dir', required=True)
    ap.add_argument('--out_dir', default='results_apgd')
    ap.add_argument('--fallback_arch', default='ViT-B/16', choices=['ViT-B/16', 'ViT-L/14'])
    ap.add_argument('--eval_set', default='Test', choices=['Validation', 'Test'])
    ap.add_argument('--epsilons', default='0.0002,0.0005')
    ap.add_argument('--n_iter', default=100, type=int)
    ap.add_argument('--batch_size', default=32, type=int)
    ap.add_argument('--num_workers', default=2, type=int)
    ap.add_argument('--seed', default=0, type=int)
    args = ap.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpts = [c.strip() for c in args.checkpoints.split(',') if c.strip()]
    labels = [l.strip() for l in args.labels.split(',')] if args.labels.strip() else [os.path.basename(c) for c in ckpts]
    assert len(labels) == len(ckpts), '--labels must match --checkpoints'
    eps = [parse_eps(e) for e in args.epsilons.split(',') if e.strip()]
    print(f'device: {device}; APGD-CE, {args.n_iter} iterations, all images')
    for c, l in zip(ckpts, labels):
        run_model(args, c, l, eps, device)


if __name__ == '__main__':
    main()
