# Frontier matrix (quick tier)

**Run notes.** Pilot on OpenML-CC18 (suite 99) and OpenML-CTR23 (suite 353): first 3 official folds per task, every library at its defaults, 10 threads. Machine: Apple M4 MacBook Air (10 cores: 4 performance + 6 efficiency, 24 GB), macOS 26.6.2, Python 3.12.10. Alloy was built from source. The commit `02cc63f` above is a local, unpushed merge of this branch at `622e255` with `claude/fix-categorical-predict` at `0ff894b` (the target-encoded categorical predict fix, PR #151). Four CC18 image tasks were left out for time: mnist_784 (3573), Fashion-MNIST (146825), CIFAR_10 (167124) and Devnagari-Script (167121). So 103 of 107 tasks ran. One task (brazilian_houses) failed on the Alloy arm in every fold and is left out of the ratios, which cover 102 tasks.

Arms: alloy 1.0.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10. Seeds: 3 (OpenML cells: official folds, 103 tasks). Threads: 10. Commit: `02cc63f607c9`.

Loss ratios compare `alloy` with the best peer in each cell (below 1.0 means alloy is better). Win/tie/loss is per cell, paired by seed; a difference inside two seed standard errors or under 0.5% is a tie.

## Overall

| Slice | Cells | Loss vs best peer (geomean) | Best or tied | W/T/L vs lightgbm | W/T/L vs xgboost | W/T/L vs catboost | Fit time vs fastest peer | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all cells | 102 | 1.035 | 73/102 | 54/40/8 | 65/31/6 | 20/58/24 | 2.80x | 1 |
| task: regression | 34 | 0.992 | 23/34 | 16/15/3 | 23/8/3 | 7/18/9 | 5.09x | 1 |
| task: binary | 35 | 1.120 | 23/35 | 16/15/4 | 20/13/2 | 5/21/9 | 1.97x | 0 |
| task: multiclass | 33 | 0.996 | 27/33 | 22/10/1 | 22/10/1 | 8/19/6 | 3.96x | 0 |

## Data axes, `defaults` regime (every library at its own defaults (Alloy: auto policy, auto rounds))

| Slice | Cells | Loss vs best peer (geomean) | Best or tied | W/T/L vs lightgbm | W/T/L vs xgboost | W/T/L vs catboost | Fit time vs fastest peer | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real data = openml | 102 | 1.035 | 73/102 | 54/40/8 | 65/31/6 | 20/58/24 | 2.80x | 1 |

## Largest gaps (alloy vs best peer)

| Cell | Task | Loss ratio | Best peer | Outcome |
|---|---|---:|---|---|
| `openml_10093_banknote-authentication|defaults` | binary | 2.232 | catboost | tie |
| `openml_361253_wave_energy|defaults` | regression | 1.949 | catboost | loss |
| `openml_9976_madelon|defaults` | binary | 1.870 | catboost | loss |
| `openml_9960_wall-robot-navigation|defaults` | multiclass | 1.719 | catboost | tie |
| `openml_49_tic-tac-toe|defaults` | binary | 1.564 | lightgbm | loss |
| `openml_11_balance-scale|defaults` | multiclass | 1.546 | catboost | loss |
| `openml_9946_wdbc|defaults` | binary | 1.504 | catboost | tie |
| `openml_146824_mfeat-pixel|defaults` | multiclass | 1.476 | catboost | loss |
| `openml_3022_vowel|defaults` | multiclass | 1.452 | catboost | loss |
| `openml_14954_cylinder-bands|defaults` | binary | 1.375 | lightgbm | loss |
| `openml_12_mfeat-factors|defaults` | multiclass | 1.370 | catboost | loss |
| `openml_29_credit-approval|defaults` | binary | 1.295 | catboost | loss |
| `openml_32_pendigits|defaults` | multiclass | 1.241 | catboost | tie |
| `openml_361622_cars|defaults` | regression | 1.239 | catboost | loss |
| `openml_28_optdigits|defaults` | multiclass | 1.236 | catboost | tie |

## Largest leads (alloy vs best peer)

| Cell | Task | Loss ratio | Best peer | Outcome |
|---|---|---:|---|---|
| `openml_146821_car|defaults` | multiclass | 0.214 | xgboost | win |
| `openml_361268_fps_benchmark|defaults` | regression | 0.354 | lightgbm | win |
| `openml_3549_analcatdata_authorship|defaults` | multiclass | 0.583 | lightgbm | tie |
| `openml_361252_video_transcoding|defaults` | regression | 0.641 | catboost | win |
| `openml_219_electricity|defaults` | binary | 0.738 | xgboost | win |
| `openml_14970_har|defaults` | multiclass | 0.771 | lightgbm | tie |
| `openml_146195_connect-4|defaults` | multiclass | 0.794 | xgboost | win |
| `openml_361254_sarcos|defaults` | regression | 0.833 | catboost | win |
| `openml_3481_isolet|defaults` | multiclass | 0.842 | catboost | tie |
| `openml_361236_auction_verification|defaults` | regression | 0.850 | xgboost | tie |
| `openml_23_cmc|defaults` | multiclass | 0.894 | catboost | win |
| `openml_361618_forest_fires|defaults` | regression | 0.896 | catboost | tie |
| `openml_6_letter|defaults` | multiclass | 0.899 | lightgbm | win |
| `openml_125922_texture|defaults` | multiclass | 0.905 | catboost | tie |
| `openml_361241_physiochemical_protein|defaults` | regression | 0.928 | xgboost | win |

## Failed fits

| Cell | Arm | Error |
|---|---|---|
| `openml_361267_brazilian_houses|defaults` | alloy | ValueError: value at index 21131 exceeds max supported bin 65535 |
