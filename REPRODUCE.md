# Reproducing the FairCLIP adversarial-attack study

This guide reproduces every number and figure in the report *One Gray Level Is Enough: Adversarial Attacks on Fairness-Aware Glaucoma Screening with FairCLIP*, from a fresh machine to the compiled PDF. It assumes you have never run FairCLIP before.

The pipeline has one GPU stage per attack family, then CPU-only analysis that reads saved per-image results, so the analysis never needs the GPU again.

```
FairCLIP repo + dataset + released checkpoints
        │
        ├─ fairclip_full_eval.py  (GPU)  noise / FGSM / PGD, grayscale + per-channel  ─┐
        ├─ fairclip_autoattack.py (GPU)  APGD-CE                                       ├─ per-image .npz files
        └─ fairclip_uap.py        (GPU)  universal / class-paired / transfer          ─┘
                                                      │
        fairclip_analyze.py, fairclip_paired_tests.py (CPU)  →  CSVs + figures
                                                      │
        make_all_numbers.sh (CPU)  →  numbers*.tex  →  report PDF
```

---

## 1. System requirements

| | Used for the report | Minimum |
|---|---|---|
| GPU | NVIDIA RTX 5060, 8 GB (Blackwell, `sm_120`) | Any CUDA GPU with ≥ 8 GB; the attacks run in fp32 at batch 16–32 |
| OS | Windows 11 | Windows or Linux (macOS without CUDA is impractically slow) |
| Python | 3.10 | 3.10 |
| CUDA build of PyTorch | cu128 | Whatever build supports your GPU (see §2.2) |
| Disk | Dataset and four checkpoints (several GB; check the sizes on the download pages), < 200 MB of results | same |
| LaTeX (optional) | TeX Live with `newtx`, `tikz`, `wrapfig`, `titlesec` | Overleaf works |

The CPU analysis steps run on any laptop in seconds to minutes.

## 2. Setup

### 2.1 Clone the FairCLIP repository

```
git clone https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP.git
cd FairCLIP/FairCLIP
```

All our scripts run **from `FairCLIP/FairCLIP/`**, next to `evaluate_CLIP.py`, because they import the repository's own `clip` package and `src/modules.py` (`fair_vl_med_dataset`, `compute_vl_prob`). Copy the files from this repository's `attacks/` folder into that directory.

### 2.2 Install PyTorch for your GPU first, then everything else

The FairCLIP repo pins `torch==2.2.2+cu121`. **That build does not run on Blackwell GPUs (RTX 50-series, `sm_120`).** If you have one:

1. Install the rest of FairCLIP's `requirements.txt` with the `torch`, `torchvision` and `torchaudio` lines removed.
2. Install a CUDA 12.8 build:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. Check: `python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"` must list `sm_120`.

On older GPUs the pinned build is fine.

Then install what our scripts add:

```
pip install -r requirements_attacks.txt
```

`requirements_attacks.txt` installs `numpy scipy scikit-learn pandas matplotlib tabulate` and AutoAttack from GitHub (`git+https://github.com/fra31/auto-attack`). The PyPI package named `autoattack` is not the same project; use the GitHub one.

### 2.3 Download the dataset

Download **Harvard-FairVLMed** using the link in the FairCLIP README (hosted on Hugging Face). You need the three folders:

```
<DATA>/Training/data_*.npz     7,000 files
<DATA>/Validation/data_*.npz   1,000 files
<DATA>/Test/data_*.npz         2,000 files
```

The README also mentions `data_summary.csv` and `gpt-4_summarized_notes.csv`. **You do not need them**: the evaluation code reads notes and demographics from inside each `.npz`. If you ever re-run FairCLIP's own fine-tuning scripts, they hard-code `SUMMARIZED_NOTE_FILE=gpt4_summarized_notes.csv`; pass `--summarized_note_file ''` instead.

### 2.4 Download the released checkpoints

From the authors' Google Drive folder (linked in the FairCLIP README), download the ViT-B/16 checkpoints:

| Our label | File |
|---|---|
| CLIP baseline | `clip.pth` |
| FairCLIP (Race) | `fairclip_race.pth` |
| FairCLIP (Gender) | `fairclip_gender.pth` |
| FairCLIP (Ethnicity) | `fairclip_ethnicity.pth` |

File names in the Drive folder may differ slightly; only the paths you pass on the command line matter. Our scripts detect the architecture from tensor shapes, so a ViT-L/14 checkpoint also loads.

**Use these exact labels.** Every downstream file name and LaTeX key is derived from the label (`"FairCLIP (Race)"` → `fairclip-race`), and the paired tests look for the reference model by the label `CLIP baseline`.

## 3. Sanity check before any long run

Run the smoke test on 64 images. It takes a couple of minutes and prints a time estimate for the full run:

```
python fairclip_full_eval.py --dataset_dir "<DATA>" ^
  --checkpoints "<CKPT>\clip.pth" --labels "CLIP baseline" ^
  --n_samples 64 --quantize --gray --out_dir results_smoke
```

(`^` continues a line in Windows `cmd`; use `` ` `` in PowerShell and `\` on Linux.)

Check two lines of the output:

- `prompt geometry: ... P(glaucoma) is confined to [0.46, 0.54]` (roughly). This is the bound from the report's Eq. (1).
- `quantized-clean max |dp|` close to 0. This confirms clean images already sit on the 8-bit grid. If it is not near 0, your preprocessing differs from ours and the 8-bit results will not reproduce.

## 4. GPU runs

Let `CKPTS` be the four checkpoint paths, comma-separated in the order of `LABELS = "CLIP baseline,FairCLIP (Race),FairCLIP (Gender),FairCLIP (Ethnicity)"`.

### 4.1 Per-image attacks, grayscale (main results)

```
python fairclip_full_eval.py --dataset_dir "<DATA>" --checkpoints "<CKPTS>" --labels "<LABELS>" ^
  --quantize --gray --out_dir results_gray
```

Noise, FGSM and 10-step PGD at budgets 0.0001 to 4/255 on all 1,883 test images, each adversarial image scored both as a float tensor and after rounding to 8 bits.

### 4.2 Per-image attacks, per-channel (ablation)

Same command without `--gray`, into `results_rgb`. AutoAttack (§4.3) reads this folder.

### 4.3 APGD-CE (strength check)

```
python fairclip_autoattack.py --dataset_dir "<DATA>" --checkpoints "<CKPTS>" --labels "<LABELS>" ^
  --from_dir results_rgb --out_dir results_apgd --epsilons 0.0002,0.0005 --n_iter 100
```

**Measured time: about 37 minutes per budget per model on an RTX 5060, so about 5 hours for the full run.** Use `--n_iter 50` to halve it. Progress and a time estimate print every five batches.

### 4.4 Universal attacks

```
python fairclip_uap.py --dataset_dir "<DATA>" --checkpoints "<CKPTS>" --labels "<LABELS>" ^
  --epsilons 1/255,2/255,4/255,8/255,16/255 --out_dir results_uap
```

Learns one grayscale perturbation per model and budget on the 7,000 training images (5 epochs), chooses the epoch on validation, applies it once to the test set, and evaluates every model's perturbation on every other model (transfer). Learned perturbations are saved in `results_uap/deltas/`; `--skip_learn` re-evaluates them without retraining.

## 5. CPU analysis

```
python fairclip_analyze.py --results_dir results_gray --out_dir analysis_gray
python fairclip_paired_tests.py --results_dir results_gray --out_dir analysis_gray

python fairclip_analyze.py --results_dir results_rgb --out_dir analysis_rgb

python fairclip_analyze.py --results_dir results_apgd --out_dir analysis_apgd
python fairclip_paired_tests.py --results_dir results_apgd --out_dir analysis_apgd --attack apgd

python fairclip_analyze.py --results_dir results_uap --out_dir analysis_uap
python fairclip_paired_tests.py --results_dir results_uap --out_dir paired_uap    --attack uap    --eps 1/255,2/255
python fairclip_paired_tests.py --results_dir results_uap --out_dir paired_uap_cc --attack uap_cc --eps 1/255
python -c "import pandas as pd; pd.concat([pd.read_csv('paired_uap/paired_tests.csv'), pd.read_csv('paired_uap_cc/paired_tests.csv')]).to_csv('analysis_uap/paired_tests.csv', index=False)"
```

`fairclip_paired_tests.py` writes `paired_tests.csv` into its output folder, so the two universal-attack runs go to separate folders and are then concatenated. Holm correction is applied within each run, which is what the report states.

The report also uses five extra universal-attack CSVs in `analysis_uap/` (`exploratory_tests_uap.csv`, `shift_uap.csv`, `transfer_uap.csv`, `delta_stats_uap.csv`, `subgroup_shift_uap.csv`). **TODO: add the post-processing script that writes these to the repository and give its command here.**

## 6. Numbers, figures and the report

From the `course_report/` folder:

```
sh scripts/make_all_numbers.sh <path>/analysis_gray <path>/analysis_rgb <path>/analysis_apgd <path>/analysis_uap
latexmk -pdf report_course.tex
```

This writes `numbers*.tex` and `figures/fig_course_auc.pdf`, then compiles. Every experimental number in the text comes from these generated files; a missing one prints as a red **TBD** box, so an incomplete run is visible in the PDF.

## 7. Caveats that change results

**Use fp32.** Our scripts load the model in fp32. The repository's `evaluate_CLIP.py` runs fp16 on GPU, and because all predicted probabilities lie within 0.01 of 0.5, fp16 collapses the 1,883 test predictions onto about 32 distinct values. That creates ties and shifts AUC slightly (0.721 vs. 0.722 for the gender checkpoint; up to about 0.01 for small groups).

**Don't judge these models at the 0.5 threshold.** Three of the four checkpoints predict glaucoma for almost no one at 0.5 (the gender checkpoint for no one), so accuracy, flip rate, demographic parity and equalized odds mostly reflect calibration. The report uses AUC.

**Don't use AutoAttack's standard wrapper here.** `AutoAttack.run_standard_evaluation` only attacks images that are correctly classified at 0.5 and returns the clean image for any it cannot flip. On these models it left CLIP's AUC exactly at its clean value at ε = 0.0001, where PGD gives 0.572, and took 90 minutes per budget per model to do so. `fairclip_autoattack.py` instead calls APGD-CE directly on every image and keeps the highest-loss point. The rest of the AutoAttack suite doesn't apply: APGD-DLR needs at least three classes, and APGD-T / FAB-T have a single target in a two-class problem.

**8-bit budgets.** Clean images are already on the 8-bit grid, so any perturbation below half a gray level (ε < 0.5/255 ≈ 0.00196) vanishes when rounded. That is why the 8-bit APGD rows equal clean AUC. A budget just above half a level (0.002) rounds up to a full level.

**Grayscale vs. per-channel.** SLO scans are single-channel, replicated to RGB. `--gray` shares one perturbation across channels so the image stays a plausible scan; the per-channel run is stronger below one gray level and identical at one level and above.

**Test set size.** The repository's data loader drops evaluation images with any unknown demographic attribute, leaving 1,883 of 2,000 test images (and 940 of 1,000 validation images). This is expected.

**One checkpoint per objective.** Model-to-model differences (for example, the race checkpoint degrading more slowly) can't be separated from run-to-run training variation without retraining several seeds.

**Pre-registered vs. exploratory.** The PGD subgroup budgets (0.0002, 0.0005) were fixed before any subgroup result was seen. The universal-attack subgroup budgets were chosen after seeing overall AUC, so those subgroup results are exploratory.

**Randomness.** All scripts default to seed 0. Each model's random stream is reset to the same seed, so running checkpoints separately gives the same noise and PGD random starts as one combined run. Bootstrap CIs use their own seed (`--seed` in the analysis scripts).

## 8. Windows notes

- Quote every path that contains spaces.
- If data loading hangs or crashes, run with `--num_workers 0`.
- Output files are named from the label with non-alphanumerics replaced by `_` (`FairCLIP__Race.npz`); don't rename them, since later scripts find files by label.

## 9. Repository layout

```
attacks/                      copy these into FairCLIP/FairCLIP/
  fairclip_full_eval.py       noise / FGSM / PGD
  fairclip_autoattack.py      APGD-CE
  fairclip_uap.py             universal, class-paired, transfer
  fairclip_analyze.py         CSVs and figures from any results folder
  fairclip_paired_tests.py    paired bootstrap tests with Holm correction
requirements.txt
course_report/                LaTeX report, number generators, figure script
REPRODUCE.md                  this file
```
