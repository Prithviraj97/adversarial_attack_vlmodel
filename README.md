# Adversarial Attacks on Fairness-Aware Glaucoma Screening with FairCLIP

How robust is a vision-language model trained to be *fair*? We perform simple attacks on [FairCLIP](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP) (Luo et al., CVPR 2024), a CLIP model fine-tuned for glaucoma detection from SLO fundus scans with an optimal-transport penalty that equalizes performance across demographic groups. We evaluate the CLIP baseline and the three released FairCLIP checkpoints (race, gender, ethnicity) on the full Harvard-FairVLMed test set.

Every adversarial image is kept **grayscale**, like a real SLO scan, and **rounded to 8 bits**, so it could be saved as an ordinary image file.

## Key findings

- **One gray level is enough.** A PGD perturbation of a single 8-bit gray level, the smallest change a stored image can carry, drives every checkpoint's AUC from about 0.7 to **0.000** (a fully inverted ranking). One FGSM step leaves at most 0.003. Random noise four times larger changes AUC by less than 0.01.
- **The attack is not an artifact of a weak optimizer.** APGD-CE, the main attack in AutoAttack, with ten times the iterations, lowers AUC only 0.001–0.012 below our 10-step PGD.
- **No model access is needed at test time.** A single universal perturbation, learned once on training images and added unchanged to every scan, lowers AUC to 0.48–0.61 at two gray levels and below chance at four.
- **Thresholded metrics are unreliable for this model family.** All predicted probabilities lie within 0.01 of 0.5; three of the four checkpoints almost never predict glaucoma at the 0.5 threshold, and the gender checkpoint's demographic parity and equalized-odds differences are exactly zero because it predicts no positives. Standard accuracy-based AutoAttack would therefore report these models as robust.
- **Fairness training doesn't protect against attacks.** No checkpoint narrows the robustness gap on the attribute it was trained to equalize (smallest Holm-corrected *p* = 1.000 over 15 tests). Under a partial, label-aware universal attack, Black patients end with lower AUC than White patients in three of the four models; this is an exploratory result.

| Model | Clean | Noise (4 levels) | FGSM (1) | PGD (1) | Universal δ (2) | Class-paired δ (1) |
|---|---|---|---|---|---|---|
| CLIP (baseline) | 0.691 | 0.684 | 0.001 | 0.000 | 0.481 | 0.106 |
| FairCLIP (Race) | 0.713 | 0.708 | 0.003 | 0.000 | 0.612 | 0.241 |
| FairCLIP (Gender) | 0.722 | 0.716 | <0.001 | 0.000 | 0.514 | 0.183 |
| FairCLIP (Ethnicity) | 0.728 | 0.723 | 0.001 | 0.000 | 0.506 | 0.181 |

*Test AUC (n = 1,883), grayscale perturbations, after 8-bit rounding. Numbers in parentheses are budgets in gray levels.*

## Repository contents

```
attacks/
  fairclip_full_eval.py      noise, FGSM, PGD (grayscale or per-channel; float and 8-bit scoring)
  fairclip_autoattack.py     APGD-CE on every image
  fairclip_uap.py            universal, class-paired and transfer attacks
  fairclip_analyze.py        AUC with CIs, subgroup AUC, ES-AUC, calibration, figures
  fairclip_paired_tests.py   paired bootstrap tests with Holm correction
course_report/               LaTeX report, number generators, figure scripts
requirements.txt
REPRODUCE.md                 full step-by-step reproduction guide
```

## Quick start

The scripts run from inside a clone of the FairCLIP repository, using its data loader and released checkpoints.

```
git clone https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP.git
cp attacks/*.py FairCLIP/FairCLIP/
cd FairCLIP/FairCLIP
pip install -r <this repo>/requirements_attacks.txt

python fairclip_full_eval.py --dataset_dir <DATA> --checkpoints <clip.pth> \
  --labels "CLIP baseline" --n_samples 64 --quantize --gray --out_dir results_smoke
```

That is a 64-image smoke test. For the dataset, checkpoints, GPU setup (including RTX 50-series), the full set of runs and the caveats that affect results, see **[REPRODUCE.md](REPRODUCE.md)**.

## Data and models

This repository contains no data or model weights. Download Harvard-FairVLMed and the released checkpoints from the links in the [FairCLIP README](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP), and follow their license terms.

## Limitations

All attacks are white-box against undefended models, and the class-paired universal attack uses the true label. Each fairness objective is represented by one released checkpoint, so differences between models can't be separated from run-to-run training variation. Some demographic groups are small (e.g. 76 Hispanic and 30 Spanish-speaking test patients).

## Citation
Please also cite FairCLIP:

```bibtex
@inproceedings{luo2024fairclip,
  title     = {FairCLIP: Harnessing Fairness in Vision-Language Learning},
  author    = {Luo, Yan and Shi, Min and Khan, Muhammad Osama and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```

## Acknowledgements

Built on the [FairCLIP](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP) code, data and checkpoints by the Harvard Ophthalmology AI Lab, and on [AutoAttack](https://github.com/fra31/auto-attack) by Croce and Hein.
