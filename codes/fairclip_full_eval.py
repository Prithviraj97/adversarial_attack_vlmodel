"""
fairclip_full_eval.py

One GPU pass per checkpoint that replaces running fairclip_model_comparison.py
and fairclip_subgroup_breakdown.py separately. Instead of writing only summary
numbers, it saves PER-SAMPLE results (clean prob, clean cosine similarities,
attacked prob for every attack/epsilon, label, demographics, file name) to one
.npz per model. Every downstream number (overall AUC, subgroup AUC, ES-AUC,
bootstrap CIs, calibration, per-sample flip budget) is then computed offline by
fairclip_analyze.py, with no second GPU run.

Differences from the earlier scripts:
  * evaluates EXACTLY n_samples (Subset), not "first batch past n" (the old 208)
  * text prompts are encoded once (they are identical for every sample)
  * clean pass runs once per model instead of once per attack
  * --quantize also evaluates every adversarial image after rounding it to the
    8-bit grid (k/255). The clean input already sits on that grid: CLIP's
    preprocess converts to RGB uint8 before ToTensor. Any epsilon below
    0.5/255 (~0.00196) is erased entirely by this rounding, so this tells you
    which results survive saving the image as a normal 8-bit file.
  * epsilons can be given as floats or as "k/255"
  * --gray shares the perturbation across R,G,B (the SLO images are grayscale
    replicated to 3 channels, so an independent per-channel perturbation produces
    an image the scanner could never output). Run once without and once with it,
    into different --out_dir folders.

Run from FairCLIP/FairCLIP/ (same place as the other scripts):

    python fairclip_full_eval.py \
        --dataset_dir "C:\\path\\to\\Harvard-FairVLMed" \
        --checkpoints "C:\\...\\clip.pth,C:\\...\\fairclip_race.pth" \
        --labels "CLIP baseline,FairCLIP (Race)" \
        --epsilons 0.0001,0.0002,0.0005,0.001,0.002,1/255,2/255,4/255 \
        --attacks noise,fgsm,pgd --quantize \
        --out_dir results_full
"""

import argparse
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
from clip.model import build_model
from src.modules import fair_vl_med_dataset, compute_vl_prob

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class Unnormalize:
    def __init__(self, device):
        self.mean = CLIP_MEAN.to(device)
        self.std = CLIP_STD.to(device)

    def to_pixels(self, x_norm):
        return (x_norm * self.std + self.mean).clamp(0, 1)

    def to_normalized(self, x_pixels):
        return (x_pixels - self.mean) / self.std


def parse_eps(s):
    s = s.strip()
    if '/' in s:
        a, b = s.split('/')
        return float(a) / float(b)
    return float(s)


def load_model_auto_arch(pretrained_weights, device, fallback_arch='ViT-B/16'):
    _, preprocess = clip.load(fallback_arch, device='cpu', jit=False)
    if pretrained_weights:
        ckpt = torch.load(pretrained_weights, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        width = state_dict['visual.conv1.weight'].shape[0]
        print(f'  detected architecture: {({768: "ViT-B/16", 1024: "ViT-L/14"}).get(width, width)}')
        model = build_model(state_dict).to(device)
    else:
        model, _ = clip.load(fallback_arch, device=device, jit=False)
    model.float()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess


def forward(model, x_pixels, un, text_feats, return_emb=False):
    """Returns (prob_glaucoma [B], logits [B,2]) and optionally the unit image embedding.
    The logits ARE the two raw cosine similarities: evaluate_CLIP.py applies softmax
    to them directly, with no logit scale."""
    f = model.encode_image(un.to_normalized(x_pixels))
    f = f / f.norm(dim=1, keepdim=True)
    tf = text_feats.unsqueeze(0).expand(f.shape[0], -1, -1)  # [B,2,D]
    prob, logits = compute_vl_prob(f, tf)
    if return_emb:
        return prob[:, 1], logits, f
    return prob[:, 1], logits


def attack_batch(model, x, y, un, text_feats, attack, eps, alpha, steps, gen, gray=False):
    """gray=True shares one perturbation across the R,G,B channels. The SLO images are
    single-channel and replicated to RGB, so this keeps the adversarial image grayscale,
    which is the realistic setting; gray=False perturbs channels independently."""
    pshape = (x.shape[0], 1, *x.shape[2:]) if gray else x.shape

    if attack == 'noise':
        noise = (torch.rand(pshape, generator=gen, device=x.device) * 2 - 1) * eps
        return (x + noise).clamp(0, 1)

    def grad_of(x_in):
        x_in = x_in.clone().detach().requires_grad_(True)
        _, logits = forward(model, x_in, un, text_feats)
        loss = F.cross_entropy(logits, y)
        g = torch.autograd.grad(loss, x_in)[0]
        return g.sum(1, keepdim=True) if gray else g

    if attack == 'fgsm':
        return (x + eps * grad_of(x).sign()).clamp(0, 1).detach()

    if attack == 'pgd':
        start = (torch.rand(pshape, generator=gen, device=x.device) * 2 - 1) * eps
        x_adv = (x + start).clamp(0, 1)
        for _ in range(steps):
            x_adv = x_adv.detach() + alpha * grad_of(x_adv).sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0, 1)
        return x_adv.detach()

    raise ValueError(attack)


def quantize(x):
    return torch.round(x * 255.0) / 255.0


@torch.no_grad()
def encode_prompts(model, dataset, device):
    tokens = dataset[0][1].to(device)  # [2,77]: "non-glaucoma", "glaucoma"
    t = model.encode_text(tokens)
    return t / t.norm(dim=1, keepdim=True)  # [2,D]


def summarize(p_clean, p_adv, y):
    pred_c, pred_a = p_clean >= 0.5, p_adv >= 0.5
    correct_c = pred_c == y.astype(bool)
    flipped = pred_c != pred_a
    asr = flipped[correct_c].mean() if correct_c.any() else float('nan')
    return (roc_auc_score(y, p_adv), (pred_a == y.astype(bool)).mean(), flipped.mean(), asr)


def run_model(args, ckpt, label, epsilons, attacks, device):
    print(f'\n[{label}] {ckpt or "(off-the-shelf)"}')
    model, preprocess = load_model_auto_arch(ckpt, device, args.fallback_arch)
    un = Unnormalize(device)

    full = fair_vl_med_dataset(args.dataset_dir, preprocess, subset=args.eval_set)
    n = len(full) if args.n_samples <= 0 else min(args.n_samples, len(full))
    ds = Subset(full, list(range(n)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f'  {args.eval_set}: {len(full)} samples, evaluating {n}')

    text_feats = encode_prompts(model, full, device)
    print(f'  prompt geometry: cos(t_glaucoma, t_non) = {float(text_feats[0] @ text_feats[1]):.4f}, '
          f'||t_g - t_n|| = {float((text_feats[1]-text_feats[0]).norm()):.4f} '
          f'-> P(glaucoma) is confined to [{torch.sigmoid(-(text_feats[1]-text_feats[0]).norm()).item():.4f}, '
          f'{torch.sigmoid((text_feats[1]-text_feats[0]).norm()).item():.4f}]')
    E, A = len(epsilons), len(attacks)
    p_adv = np.zeros((A, E, n), dtype=np.float64)
    p_adv_q = np.zeros((A, E, n), dtype=np.float64) if args.quantize else None
    emb_shift = np.zeros((A, E, n), dtype=np.float64)  # 1 - cos(f_adv, f_clean)
    emb_clean = None  # [n, D] float32, allocated once D is known
    p_clean = np.zeros(n); p_clean_q = np.zeros(n); sims_clean = np.zeros((n, 2))
    labels = np.zeros(n, dtype=np.int64); attrs = np.zeros((n, 4), dtype=np.int64)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    t0 = time.time()
    i0 = 0
    for bi, (images, _, la) in enumerate(loader):
        b = images.shape[0]
        sl = slice(i0, i0 + b)
        x = un.to_pixels(images.to(device))
        y = la[:, 0].to(device)
        labels[sl] = la[:, 0].numpy(); attrs[sl] = la[:, 1:5].numpy()
        with torch.no_grad():
            pc, sc, f_clean = forward(model, x, un, text_feats, return_emb=True)
            p_clean[sl] = pc.cpu().numpy(); sims_clean[sl] = sc.cpu().numpy()
            if emb_clean is None:
                emb_clean = np.zeros((n, f_clean.shape[1]), dtype=np.float32)
            emb_clean[sl] = f_clean.cpu().numpy()
            p_clean_q[sl] = forward(model, quantize(x), un, text_feats)[0].cpu().numpy()
        for ai, att in enumerate(attacks):
            for ei, eps in enumerate(epsilons):
                xa = attack_batch(model, x, y, un, text_feats, att, eps,
                                  args.alpha_ratio * eps, args.steps, gen, gray=args.gray)
                with torch.no_grad():
                    pa, _, fa = forward(model, xa, un, text_feats, return_emb=True)
                    p_adv[ai, ei, sl] = pa.cpu().numpy()
                    emb_shift[ai, ei, sl] = (1 - (fa * f_clean).sum(1)).cpu().numpy()
                    if args.quantize:
                        p_adv_q[ai, ei, sl] = forward(model, quantize(xa), un, text_feats)[0].cpu().numpy()
        i0 += b
        if (bi + 1) % 10 == 0 or i0 == n:
            el = time.time() - t0
            print(f'  {i0}/{n} samples  ({el/60:.1f} min elapsed, ~{el/i0*(n-i0)/60:.1f} min left)')

    print(f'  clean AUC {roc_auc_score(labels, p_clean):.4f}   acc@0.5 {((p_clean>=0.5)==labels).mean():.4f}'
          f'   quantized-clean max |dp| {np.abs(p_clean_q-p_clean).max():.2e}')
    print(f'  {"attack":>6s} {"eps":>9s} {"x255":>6s}  {"AUC":>7s} {"acc":>6s} {"flip":>6s} {"ASR":>6s}'
          + ('   | 8-bit: AUC   ASR' if args.quantize else ''))
    for ai, att in enumerate(attacks):
        for ei, eps in enumerate(epsilons):
            auc, acc, flip, asr = summarize(p_clean, p_adv[ai, ei], labels)
            line = f'  {att:>6s} {eps:9.5f} {eps*255:6.3f}  {auc:7.4f} {acc:6.3f} {flip:6.3f} {asr:6.3f}'
            if args.quantize:
                aq, _, _, sq = summarize(p_clean, p_adv_q[ai, ei], labels)
                line += f'   |  {aq:7.4f} {sq:6.3f}'
            print(line)

    os.makedirs(args.out_dir, exist_ok=True)
    safe = ''.join(c if c.isalnum() else '_' for c in label).strip('_')
    out = os.path.join(args.out_dir, f'{safe}.npz')
    save = dict(model_label=label, checkpoint=ckpt, files=np.array(full.files[:n]),
                labels=labels, attrs=attrs, attr_names=np.array(['race', 'gender', 'ethnicity', 'language']),
                p_clean=p_clean, p_clean_q=p_clean_q, sims_clean=sims_clean,
                attacks=np.array(attacks), epsilons=np.array(epsilons), p_adv=p_adv, emb_shift=emb_shift, emb_clean=emb_clean,
                text_cos=float(text_feats[0] @ text_feats[1]),
                delta_norm=float((text_feats[1] - text_feats[0]).norm()),
                alpha_ratio=args.alpha_ratio, steps=args.steps, seed=args.seed, gray=args.gray)
    if args.quantize:
        save['p_adv_q'] = p_adv_q
    np.savez_compressed(out, **save)
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
    ap.add_argument('--eval_set', default='Test', choices=['Validation', 'Test'])
    ap.add_argument('--n_samples', default=0, type=int, help='0 = whole split')
    ap.add_argument('--batch_size', default=16, type=int)
    ap.add_argument('--num_workers', default=2, type=int)
    ap.add_argument('--epsilons', default='0.0001,0.0002,0.0005,0.001,0.002,1/255,2/255,4/255')
    ap.add_argument('--attacks', default='noise,fgsm,pgd')
    ap.add_argument('--alpha_ratio', default=0.25, type=float)
    ap.add_argument('--steps', default=10, type=int)
    ap.add_argument('--quantize', action='store_true')
    ap.add_argument('--gray', action='store_true',
                    help='share the perturbation across RGB channels (keeps the SLO image grayscale)')
    ap.add_argument('--seed', default=0, type=int)
    ap.add_argument('--out_dir', default='results_full')
    args = ap.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    ckpts = ['' if c.strip().lower() == 'none' else c.strip() for c in args.checkpoints.split(',') if c.strip()]
    labels = ([l.strip() for l in args.labels.split(',')] if args.labels.strip()
              else [os.path.basename(c) if c else 'pretrained CLIP' for c in ckpts])
    assert len(labels) == len(ckpts), '--labels must match --checkpoints'
    epsilons = [parse_eps(e) for e in args.epsilons.split(',') if e.strip()]
    attacks = [a.strip() for a in args.attacks.split(',') if a.strip()]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    for c, l in zip(ckpts, labels):
        run_model(args, c, l, epsilons, attacks, device)


if __name__ == '__main__':
    main()
