# FairCLIP attack analysis: summary

## Clean-score calibration

| model                |    n |   prevalence |   text_cos |   delta_norm |   p_bound_lo |   p_bound_hi |   margin_mean_pos |   margin_mean_neg |   margin_gap |   margin_dprime |   clean_auc |   acc_at_0p5 |   frac_pred_pos |   p_min |   p_q01 |   p_q25 |   p_median |   p_q75 |   p_q99 |   p_max |   frac_within_0p001 |   frac_within_0p01 |   margin_mean |   margin_std |   cos_glaucoma_mean |   cos_nonglaucoma_mean |
|:---------------------|-----:|-------------:|-----------:|-------------:|-------------:|-------------:|------------------:|------------------:|-------------:|----------------:|------------:|-------------:|----------------:|--------:|--------:|--------:|-----------:|--------:|--------:|--------:|--------------------:|-------------------:|--------------:|-------------:|--------------------:|-----------------------:|
| CLIP baseline        | 1883 |       0.5098 |     0.9879 |       0.1553 |       0.4612 |       0.5388 |           -0.0072 |           -0.01   |       0.0028 |          0.6894 |      0.6908 |       0.4997 |          0.0106 |  0.4955 |  0.4958 |  0.497  |     0.4979 |  0.4987 |  0.5    |  0.5005 |              0.1747 |                  1 |       -0.0085 |       0.0043 |              0.2336 |                 0.2421 |
| FairCLIP (Ethnicity) | 1883 |       0.5098 |     0.9805 |       0.1976 |       0.4508 |       0.5492 |           -0.014  |           -0.0204 |       0.0064 |          0.8662 |      0.7277 |       0.5019 |          0.0138 |  0.4912 |  0.4919 |  0.4941 |     0.4955 |  0.4972 |  0.5002 |  0.5008 |              0.0632 |                  1 |       -0.0171 |       0.008  |              0.0267 |                 0.0439 |
| FairCLIP (Gender)    | 1883 |       0.5098 |     0.9798 |       0.201  |       0.4499 |       0.5501 |           -0.0142 |           -0.0188 |       0.0046 |          0.8101 |      0.7219 |       0.4902 |          0      |  0.492  |  0.4928 |  0.4947 |     0.4959 |  0.4971 |  0.4988 |  0.4994 |              0.0042 |                  1 |       -0.0165 |       0.0061 |              0.0174 |                 0.0339 |
| FairCLIP (Race)      | 1883 |       0.5098 |     0.9932 |       0.1167 |       0.4708 |       0.5292 |            0.0005 |           -0.0021 |       0.0026 |          0.7884 |      0.7131 |       0.6766 |          0.4041 |  0.4977 |  0.4981 |  0.4991 |     0.4998 |  0.5004 |  0.5017 |  0.5021 |              0.6872 |                  1 |       -0.0008 |       0.0035 |              0.0178 |                 0.0186 |

If P(glaucoma) sits in a very narrow band around 0.5, the 0.5 threshold is fragile and flip rate mostly reflects clean accuracy, not robustness. Saturated flip rate under a strong attack should approximately equal clean accuracy at 0.5 (the attack can only flip correct predictions).

## Overall

| model                | pixels   | attack   |   eps_x255 |    auc |   auc_lo |   auc_hi |    acc |   flip_rate |    asr |   es_auc_race |   es_auc_gender |   es_auc_ethnicity |   es_auc_language |
|:---------------------|:---------|:---------|-----------:|-------:|---------:|---------:|-------:|------------:|-------:|--------------:|----------------:|-------------------:|------------------:|
| CLIP baseline        | float    | none     |     0      | 0.6908 |   0.6644 |   0.7152 | 0.4997 |      0      | 0      |        0.6477 |          0.6501 |             0.6192 |            0.5792 |
| CLIP baseline        | float    | apgd     |     0.051  | 0.4387 |   0.4117 |   0.465  | 0.4891 |      0.0106 | 0.0213 |        0.4096 |          0.4153 |             0.3963 |            0.348  |
| CLIP baseline        | float    | apgd     |     0.1275 | 0.1074 |   0.0941 |   0.1216 | 0.4456 |      0.0542 | 0.1084 |        0.104  |          0.1055 |             0.1026 |            0.0947 |
| CLIP baseline        | 8bit     | apgd     |     0.051  | 0.6907 |   0.6644 |   0.7152 | 0.4997 |      0      | 0      |        0.6477 |          0.6501 |             0.6192 |            0.5792 |
| CLIP baseline        | 8bit     | apgd     |     0.1275 | 0.6907 |   0.6644 |   0.7152 | 0.4997 |      0      | 0      |        0.6477 |          0.6501 |             0.6192 |            0.5792 |
| FairCLIP (Ethnicity) | float    | none     |     0      | 0.7277 |   0.704  |   0.7512 | 0.5019 |      0      | 0      |        0.6664 |          0.6788 |             0.6546 |            0.574  |
| FairCLIP (Ethnicity) | float    | apgd     |     0.051  | 0.4867 |   0.4604 |   0.5156 | 0.4875 |      0.0143 | 0.0286 |        0.4527 |          0.4525 |             0.4282 |            0.4002 |
| FairCLIP (Ethnicity) | float    | apgd     |     0.1275 | 0.1386 |   0.1226 |   0.1553 | 0.4663 |      0.0356 | 0.0709 |        0.1297 |          0.1356 |             0.1283 |            0.1278 |
| FairCLIP (Ethnicity) | 8bit     | apgd     |     0.051  | 0.7277 |   0.704  |   0.7512 | 0.5019 |      0      | 0      |        0.6664 |          0.6788 |             0.6546 |            0.574  |
| FairCLIP (Ethnicity) | 8bit     | apgd     |     0.1275 | 0.7277 |   0.704  |   0.7512 | 0.5019 |      0      | 0      |        0.6664 |          0.6788 |             0.6546 |            0.574  |
| FairCLIP (Gender)    | float    | none     |     0      | 0.7219 |   0.6985 |   0.7446 | 0.4902 |      0      | 0      |        0.6679 |          0.6764 |             0.649  |            0.6388 |
| FairCLIP (Gender)    | float    | apgd     |     0.051  | 0.4808 |   0.4531 |   0.509  | 0.4902 |      0      | 0      |        0.4421 |          0.4489 |             0.4204 |            0.3982 |
| FairCLIP (Gender)    | float    | apgd     |     0.1275 | 0.1278 |   0.1122 |   0.1438 | 0.4891 |      0.0011 | 0.0022 |        0.1216 |          0.1244 |             0.1182 |            0.1152 |
| FairCLIP (Gender)    | 8bit     | apgd     |     0.051  | 0.7219 |   0.6985 |   0.7446 | 0.4902 |      0      | 0      |        0.6679 |          0.6764 |             0.6489 |            0.6388 |
| FairCLIP (Gender)    | 8bit     | apgd     |     0.1275 | 0.7219 |   0.6985 |   0.7446 | 0.4902 |      0      | 0      |        0.6679 |          0.6764 |             0.6489 |            0.6388 |
| FairCLIP (Race)      | float    | none     |     0      | 0.7131 |   0.6887 |   0.7363 | 0.6766 |      0      | 0      |        0.6758 |          0.6636 |             0.6237 |            0.5845 |
| FairCLIP (Race)      | float    | apgd     |     0.051  | 0.5145 |   0.4874 |   0.5415 | 0.5231 |      0.1535 | 0.2268 |        0.4962 |          0.4786 |             0.4377 |            0.4014 |
| FairCLIP (Race)      | float    | apgd     |     0.1275 | 0.1955 |   0.1775 |   0.2153 | 0.2926 |      0.384  | 0.5675 |        0.1881 |          0.1905 |             0.1749 |            0.1687 |
| FairCLIP (Race)      | 8bit     | apgd     |     0.051  | 0.7131 |   0.6887 |   0.7363 | 0.6766 |      0      | 0      |        0.6758 |          0.6636 |             0.6237 |            0.5845 |
| FairCLIP (Race)      | 8bit     | apgd     |     0.1275 | 0.7131 |   0.6887 |   0.7363 | 0.6766 |      0      | 0      |        0.6758 |          0.6636 |             0.6237 |            0.5845 |

## Subgroups: AUC drop with paired 95% CI (apgd)

| model                |   epsilon | attribute   | group        |    n |   n_pos |   auc_clean |   auc_adv |   auc_drop |   drop_lo |   drop_hi |    asr |
|:---------------------|----------:|:------------|:-------------|-----:|--------:|------------:|----------:|-----------:|----------:|----------:|-------:|
| CLIP baseline        |    0.0002 | race        | Asian        |  152 |      75 |      0.7243 |    0.471  |     0.2533 |    0.2154 |    0.2965 | 0.0128 |
| CLIP baseline        |    0.0005 | race        | Asian        |  152 |      75 |      0.7243 |    0.1025 |     0.6218 |    0.5548 |    0.6863 | 0.1282 |
| CLIP baseline        |    0.0002 | race        | Black        |  285 |     184 |      0.707  |    0.4064 |     0.3006 |    0.2699 |    0.3351 | 0.0291 |
| CLIP baseline        |    0.0005 | race        | Black        |  285 |     184 |      0.707  |    0.0816 |     0.6254 |    0.5759 |    0.6723 | 0.1165 |
| CLIP baseline        |    0.0002 | race        | White        | 1446 |     701 |      0.6742 |    0.4323 |     0.2419 |    0.2314 |    0.2535 | 0.0211 |
| CLIP baseline        |    0.0005 | race        | White        | 1446 |     701 |      0.6742 |    0.1096 |     0.5646 |    0.5432 |    0.5849 | 0.1053 |
| CLIP baseline        |    0.0002 | gender      | Female       | 1034 |     534 |      0.6632 |    0.4138 |     0.2494 |    0.2363 |    0.2631 | 0.0235 |
| CLIP baseline        |    0.0005 | gender      | Female       | 1034 |     534 |      0.6632 |    0.0995 |     0.5638 |    0.5345 |    0.5894 | 0.1118 |
| CLIP baseline        |    0.0002 | gender      | Male         |  849 |     426 |      0.7257 |    0.4701 |     0.2556 |    0.2414 |    0.2717 | 0.0186 |
| CLIP baseline        |    0.0005 | gender      | Male         |  849 |     426 |      0.7257 |    0.1177 |     0.6081 |    0.5826 |    0.6369 | 0.1044 |
| CLIP baseline        |    0.0002 | ethnicity   | non-Hispanic | 1807 |     924 |      0.6948 |    0.4423 |     0.2525 |    0.2422 |    0.2621 | 0.0222 |
| CLIP baseline        |    0.0005 | ethnicity   | non-Hispanic | 1807 |     924 |      0.6948 |    0.1088 |     0.586  |    0.5673 |    0.6041 | 0.111  |
| CLIP baseline        |    0.0002 | ethnicity   | Hispanic     |   76 |      36 |      0.5792 |    0.3354 |     0.2437 |    0.1861 |    0.2993 | 0      |
| CLIP baseline        |    0.0005 | ethnicity   | Hispanic     |   76 |      36 |      0.5792 |    0.0618 |     0.5174 |    0.4035 |    0.6278 | 0.05   |
| CLIP baseline        |    0.0002 | language    | English      | 1767 |     875 |      0.6933 |    0.4441 |     0.2492 |    0.2388 |    0.2598 | 0.022  |
| CLIP baseline        |    0.0005 | language    | English      | 1767 |     875 |      0.6933 |    0.1105 |     0.5829 |    0.5637 |    0.6022 | 0.1055 |
| CLIP baseline        |    0.0002 | language    | Spanish      |   30 |      22 |      0.6193 |    0.3295 |     0.2898 |    0.1761 |    0.4318 | 0      |
| CLIP baseline        |    0.0005 | language    | Spanish      |   30 |      22 |      0.6193 |    0.0227 |     0.5966 |    0.3977 |    0.7727 | 0.125  |
| CLIP baseline        |    0.0002 | language    | Other        |   86 |      63 |      0.5721 |    0.2926 |     0.2795 |    0.2202 |    0.3437 | 0      |
| CLIP baseline        |    0.0005 | language    | Other        |   86 |      63 |      0.5721 |    0.0614 |     0.5107 |    0.4043 |    0.6128 | 0.2174 |
| FairCLIP (Ethnicity) |    0.0002 | race        | Asian        |  152 |      75 |      0.7704 |    0.5453 |     0.2251 |    0.1905 |    0.2622 | 0.0385 |
| FairCLIP (Ethnicity) |    0.0005 | race        | Asian        |  152 |      75 |      0.7704 |    0.1777 |     0.5927 |    0.5377 |    0.653  | 0.0897 |
| FairCLIP (Ethnicity) |    0.0002 | race        | Black        |  285 |     184 |      0.7623 |    0.49   |     0.2722 |    0.2425 |    0.3021 | 0.0194 |
| FairCLIP (Ethnicity) |    0.0005 | race        | Black        |  285 |     184 |      0.7623 |    0.1116 |     0.6507 |    0.606  |    0.6972 | 0.0388 |
| FairCLIP (Ethnicity) |    0.0002 | race        | White        | 1446 |     701 |      0.713  |    0.4735 |     0.2394 |    0.2285 |    0.2512 | 0.0288 |
| FairCLIP (Ethnicity) |    0.0005 | race        | White        | 1446 |     701 |      0.713  |    0.1364 |     0.5766 |    0.5575 |    0.5987 | 0.0733 |
| FairCLIP (Ethnicity) |    0.0002 | gender      | Female       | 1034 |     534 |      0.6956 |    0.4537 |     0.2419 |    0.2293 |    0.2535 | 0.0273 |
| FairCLIP (Ethnicity) |    0.0005 | gender      | Female       | 1034 |     534 |      0.6956 |    0.1287 |     0.5669 |    0.5408 |    0.5902 | 0.0762 |
| FairCLIP (Ethnicity) |    0.0002 | gender      | Male         |  849 |     426 |      0.7676 |    0.5293 |     0.2384 |    0.2216 |    0.2543 | 0.03   |
| FairCLIP (Ethnicity) |    0.0005 | gender      | Male         |  849 |     426 |      0.7676 |    0.1509 |     0.6167 |    0.5907 |    0.6451 | 0.0647 |
| FairCLIP (Ethnicity) |    0.0002 | ethnicity   | non-Hispanic | 1807 |     924 |      0.7318 |    0.4915 |     0.2403 |    0.231  |    0.2501 | 0.0298 |
| FairCLIP (Ethnicity) |    0.0005 | ethnicity   | non-Hispanic | 1807 |     924 |      0.7318 |    0.1414 |     0.5904 |    0.5725 |    0.6072 | 0.0718 |
| FairCLIP (Ethnicity) |    0.0002 | ethnicity   | Hispanic     |   76 |      36 |      0.6201 |    0.3549 |     0.2653 |    0.2146 |    0.3222 | 0      |
| FairCLIP (Ethnicity) |    0.0005 | ethnicity   | Hispanic     |   76 |      36 |      0.6201 |    0.0611 |     0.559  |    0.4409 |    0.6715 | 0.05   |
| FairCLIP (Ethnicity) |    0.0002 | language    | English      | 1767 |     875 |      0.7312 |    0.4917 |     0.2395 |    0.2292 |    0.2499 | 0.0263 |
| FairCLIP (Ethnicity) |    0.0005 | language    | English      | 1767 |     875 |      0.7312 |    0.14   |     0.5912 |    0.5733 |    0.6098 | 0.0658 |
| FairCLIP (Ethnicity) |    0.0002 | language    | Spanish      |   30 |      22 |      0.5852 |    0.3523 |     0.233  |    0.1307 |    0.3354 | 0.1111 |
| FairCLIP (Ethnicity) |    0.0005 | language    | Spanish      |   30 |      22 |      0.5852 |    0.0625 |     0.5227 |    0.3408 |    0.6989 | 0.2222 |
| FairCLIP (Ethnicity) |    0.0002 | language    | Other        |   86 |      63 |      0.6059 |    0.4099 |     0.196  |    0.1463 |    0.2416 | 0.0833 |
| FairCLIP (Ethnicity) |    0.0005 | language    | Other        |   86 |      63 |      0.6059 |    0.1318 |     0.4741 |    0.3706 |    0.5715 | 0.2083 |
| FairCLIP (Gender)    |    0.0002 | race        | Asian        |  152 |      75 |      0.7732 |    0.5474 |     0.2258 |    0.1898 |    0.2618 | 0      |
| FairCLIP (Gender)    |    0.0005 | race        | Asian        |  152 |      75 |      0.7732 |    0.1668 |     0.6064 |    0.5463 |    0.667  | 0.013  |
| FairCLIP (Gender)    |    0.0002 | race        | Black        |  285 |     184 |      0.7392 |    0.49   |     0.2492 |    0.2242 |    0.2788 | 0      |
| FairCLIP (Gender)    |    0.0005 | race        | Black        |  285 |     184 |      0.7392 |    0.1348 |     0.6044 |    0.5613 |    0.6463 | 0      |
| FairCLIP (Gender)    |    0.0002 | race        | White        | 1446 |     701 |      0.7096 |    0.4692 |     0.2404 |    0.2298 |    0.2508 | 0      |
| FairCLIP (Gender)    |    0.0005 | race        | White        | 1446 |     701 |      0.7096 |    0.1223 |     0.5873 |    0.567  |    0.6072 | 0.0013 |
| FairCLIP (Gender)    |    0.0002 | gender      | Female       | 1034 |     534 |      0.6918 |    0.4494 |     0.2424 |    0.2296 |    0.2548 | 0      |
| FairCLIP (Gender)    |    0.0005 | gender      | Female       | 1034 |     534 |      0.6918 |    0.1161 |     0.5757 |    0.5487 |    0.5986 | 0.004  |
| FairCLIP (Gender)    |    0.0002 | gender      | Male         |  849 |     426 |      0.759  |    0.5204 |     0.2386 |    0.2237 |    0.2549 | 0      |
| FairCLIP (Gender)    |    0.0005 | gender      | Male         |  849 |     426 |      0.759  |    0.1436 |     0.6154 |    0.5905 |    0.6425 | 0      |
| FairCLIP (Gender)    |    0.0002 | ethnicity   | non-Hispanic | 1807 |     924 |      0.7256 |    0.4853 |     0.2403 |    0.2307 |    0.2501 | 0      |
| FairCLIP (Gender)    |    0.0005 | ethnicity   | non-Hispanic | 1807 |     924 |      0.7256 |    0.1305 |     0.5951 |    0.5773 |    0.6137 | 0.0023 |
| FairCLIP (Gender)    |    0.0002 | ethnicity   | Hispanic     |   76 |      36 |      0.6132 |    0.3417 |     0.2715 |    0.2194 |    0.3215 | 0      |
| FairCLIP (Gender)    |    0.0005 | ethnicity   | Hispanic     |   76 |      36 |      0.6132 |    0.0493 |     0.5639 |    0.4472 |    0.6715 | 0      |
| FairCLIP (Gender)    |    0.0002 | language    | English      | 1767 |     875 |      0.7232 |    0.4838 |     0.2394 |    0.2301 |    0.2494 | 0      |
| FairCLIP (Gender)    |    0.0005 | language    | English      | 1767 |     875 |      0.7232 |    0.1298 |     0.5934 |    0.5744 |    0.6128 | 0.0011 |
| FairCLIP (Gender)    |    0.0002 | language    | Spanish      |   30 |      22 |      0.625  |    0.3011 |     0.3239 |    0.2102 |    0.4545 | 0      |
| FairCLIP (Gender)    |    0.0005 | language    | Spanish      |   30 |      22 |      0.625  |    0.0227 |     0.6023 |    0.3919 |    0.7786 | 0      |
| FairCLIP (Gender)    |    0.0002 | language    | Other        |   86 |      63 |      0.6901 |    0.4562 |     0.234  |    0.1843 |    0.2871 | 0      |
| FairCLIP (Gender)    |    0.0005 | language    | Other        |   86 |      63 |      0.6901 |    0.1256 |     0.5645 |    0.4726 |    0.6632 | 0.0435 |
| FairCLIP (Race)      |    0.0002 | race        | Asian        |  152 |      75 |      0.7345 |    0.5245 |     0.21   |    0.1784 |    0.2426 | 0.1782 |
| FairCLIP (Race)      |    0.0005 | race        | Asian        |  152 |      75 |      0.7345 |    0.1945 |     0.5401 |    0.4847 |    0.6019 | 0.5644 |
| FairCLIP (Race)      |    0.0002 | race        | Black        |  285 |     184 |      0.7335 |    0.4978 |     0.2357 |    0.2079 |    0.2639 | 0.2698 |
| FairCLIP (Race)      |    0.0005 | race        | Black        |  285 |     184 |      0.7335 |    0.1602 |     0.5733 |    0.5301 |    0.6174 | 0.6243 |
| FairCLIP (Race)      |    0.0002 | race        | White        | 1446 |     701 |      0.6998 |    0.5042 |     0.1956 |    0.1864 |    0.2056 | 0.2236 |
| FairCLIP (Race)      |    0.0005 | race        | White        | 1446 |     701 |      0.6998 |    0.1923 |     0.5075 |    0.49   |    0.5257 | 0.5569 |
| FairCLIP (Race)      |    0.0002 | gender      | Female       | 1034 |     534 |      0.6804 |    0.4822 |     0.1982 |    0.1871 |    0.2089 | 0.2207 |
| FairCLIP (Race)      |    0.0005 | gender      | Female       | 1034 |     534 |      0.6804 |    0.184  |     0.4964 |    0.4731 |    0.5186 | 0.5676 |
| FairCLIP (Race)      |    0.0002 | gender      | Male         |  849 |     426 |      0.755  |    0.5572 |     0.1978 |    0.183  |    0.2121 | 0.2336 |
| FairCLIP (Race)      |    0.0005 | gender      | Male         |  849 |     426 |      0.755  |    0.2103 |     0.5448 |    0.5185 |    0.5726 | 0.5674 |
| FairCLIP (Race)      |    0.0002 | ethnicity   | non-Hispanic | 1807 |     924 |      0.7177 |    0.5205 |     0.1972 |    0.1891 |    0.2052 | 0.2243 |
| FairCLIP (Race)      |    0.0005 | ethnicity   | non-Hispanic | 1807 |     924 |      0.7177 |    0.2002 |     0.5176 |    0.5014 |    0.5338 | 0.5628 |
| FairCLIP (Race)      |    0.0002 | ethnicity   | Hispanic     |   76 |      36 |      0.5743 |    0.3451 |     0.2292 |    0.1729 |    0.2931 | 0.2917 |
| FairCLIP (Race)      |    0.0005 | ethnicity   | Hispanic     |   76 |      36 |      0.5743 |    0.0819 |     0.4924 |    0.3826 |    0.5986 | 0.6875 |
| FairCLIP (Race)      |    0.0002 | language    | English      | 1767 |     875 |      0.716  |    0.5188 |     0.1972 |    0.1883 |    0.2066 | 0.225  |
| FairCLIP (Race)      |    0.0005 | language    | English      | 1767 |     875 |      0.716  |    0.1972 |     0.5188 |    0.5011 |    0.5367 | 0.5649 |
| FairCLIP (Race)      |    0.0002 | language    | Spanish      |   30 |      22 |      0.6364 |    0.3693 |     0.267  |    0.1364 |    0.4034 | 0.3889 |
| FairCLIP (Race)      |    0.0005 | language    | Spanish      |   30 |      22 |      0.6364 |    0.0909 |     0.5455 |    0.3578 |    0.7047 | 0.7222 |
| FairCLIP (Race)      |    0.0002 | language    | Other        |   86 |      63 |      0.5728 |    0.3823 |     0.1905 |    0.147  |    0.2367 | 0.2128 |
| FairCLIP (Race)      |    0.0005 | language    | Other        |   86 |      63 |      0.5728 |    0.1429 |     0.43   |    0.3354 |    0.5273 | 0.5745 |

Read the drop CIs before claiming a disparity: groups whose CIs overlap are not distinguishable at this sample size.

## Flip budget by group (initially-correct samples)

| model                | attack   | attribute   | group        |   n_correct |   median_flip_eps |   flipped@0.00020 |   flipped@0.00050 |   kruskal_p |
|:---------------------|:---------|:------------|:-------------|------------:|------------------:|------------------:|------------------:|------------:|
| CLIP baseline        | apgd     | race        | Asian        |          78 |          inf      |            0.0128 |            0.1282 |      0.806  |
| CLIP baseline        | apgd     | race        | Black        |         103 |          inf      |            0.0291 |            0.1165 |      0.806  |
| CLIP baseline        | apgd     | race        | White        |         760 |          inf      |            0.0211 |            0.1053 |      0.806  |
| CLIP baseline        | apgd     | gender      | Female       |         510 |          inf      |            0.0235 |            0.1118 |      0.7041 |
| CLIP baseline        | apgd     | gender      | Male         |         431 |          inf      |            0.0186 |            0.1044 |      0.7041 |
| CLIP baseline        | apgd     | ethnicity   | non-Hispanic |         901 |          inf      |            0.0222 |            0.111  |      0.2171 |
| CLIP baseline        | apgd     | ethnicity   | Hispanic     |          40 |          inf      |            0      |            0.05   |      0.2171 |
| CLIP baseline        | apgd     | language    | English      |         910 |          inf      |            0.022  |            0.1055 |      0.103  |
| CLIP baseline        | apgd     | language    | Other        |          23 |          inf      |            0      |            0.2174 |      0.103  |
| FairCLIP (Ethnicity) | apgd     | race        | Asian        |          78 |          inf      |            0.0385 |            0.0897 |      0.3556 |
| FairCLIP (Ethnicity) | apgd     | race        | Black        |         103 |          inf      |            0.0194 |            0.0388 |      0.3556 |
| FairCLIP (Ethnicity) | apgd     | race        | White        |         764 |          inf      |            0.0288 |            0.0733 |      0.3556 |
| FairCLIP (Ethnicity) | apgd     | gender      | Female       |         512 |          inf      |            0.0273 |            0.0762 |      0.5125 |
| FairCLIP (Ethnicity) | apgd     | gender      | Male         |         433 |          inf      |            0.03   |            0.0647 |      0.5125 |
| FairCLIP (Ethnicity) | apgd     | ethnicity   | non-Hispanic |         905 |          inf      |            0.0298 |            0.0718 |      0.5744 |
| FairCLIP (Ethnicity) | apgd     | ethnicity   | Hispanic     |          40 |          inf      |            0      |            0.05   |      0.5744 |
| FairCLIP (Ethnicity) | apgd     | language    | English      |         912 |          inf      |            0.0263 |            0.0658 |      0.0068 |
| FairCLIP (Ethnicity) | apgd     | language    | Other        |          24 |          inf      |            0.0833 |            0.2083 |      0.0068 |
| FairCLIP (Gender)    | apgd     | race        | Asian        |          77 |          inf      |            0      |            0.013  |      0.0994 |
| FairCLIP (Gender)    | apgd     | race        | Black        |         101 |          inf      |            0      |            0      |      0.0994 |
| FairCLIP (Gender)    | apgd     | race        | White        |         745 |          inf      |            0      |            0.0013 |      0.0994 |
| FairCLIP (Gender)    | apgd     | gender      | Female       |         500 |          inf      |            0      |            0.004  |      0.1931 |
| FairCLIP (Gender)    | apgd     | gender      | Male         |         423 |          inf      |            0      |            0      |      0.1931 |
| FairCLIP (Gender)    | apgd     | ethnicity   | non-Hispanic |         883 |          inf      |            0      |            0.0023 |      0.7633 |
| FairCLIP (Gender)    | apgd     | ethnicity   | Hispanic     |          40 |          inf      |            0      |            0      |      0.7633 |
| FairCLIP (Gender)    | apgd     | language    | English      |         892 |          inf      |            0      |            0.0011 |      0      |
| FairCLIP (Gender)    | apgd     | language    | Other        |          23 |          inf      |            0      |            0.0435 |      0      |
| FairCLIP (Race)      | apgd     | race        | Asian        |         101 |            0.0005 |            0.1782 |            0.5644 |      0.1567 |
| FairCLIP (Race)      | apgd     | race        | Black        |         189 |            0.0005 |            0.2698 |            0.6243 |      0.1567 |
| FairCLIP (Race)      | apgd     | race        | White        |         984 |            0.0005 |            0.2236 |            0.5569 |      0.1567 |
| FairCLIP (Race)      | apgd     | gender      | Female       |         666 |            0.0005 |            0.2207 |            0.5676 |      0.812  |
| FairCLIP (Race)      | apgd     | gender      | Male         |         608 |            0.0005 |            0.2336 |            0.5674 |      0.812  |
| FairCLIP (Race)      | apgd     | ethnicity   | non-Hispanic |        1226 |            0.0005 |            0.2243 |            0.5628 |      0.089  |
| FairCLIP (Race)      | apgd     | ethnicity   | Hispanic     |          48 |            0.0005 |            0.2917 |            0.6875 |      0.089  |
| FairCLIP (Race)      | apgd     | language    | English      |        1209 |            0.0005 |            0.225  |            0.5649 |      0.2438 |
| FairCLIP (Race)      | apgd     | language    | Spanish      |          18 |            0.0005 |            0.3889 |            0.7222 |      0.2438 |
| FairCLIP (Race)      | apgd     | language    | Other        |          47 |            0.0005 |            0.2128 |            0.5745 |      0.2438 |