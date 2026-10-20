# PR #132 MorphBoost Formula Trials

## Raw information gradients under regularization

Status: **rejected**. Production retains effective L1/DRO gradients for both the Newton-gain and
information channels.

The calibration compared the optimized effective-gradient control with a candidate that supplied
raw parent/child gradients only to the information term. It covered `lambda_l1` values `0.1` and
`0.5` across all nine acceptance fixtures, plus representative Morph+DRO regression, binary,
multiclass, and ranking fixtures. Seeds `0`, `1`, and `2` produced 78 paired fits.

| Gate statistic | Result | Required |
|---|---:|---:|
| Equal-dataset mean improvement | -1.529% | >= +0.250% |
| Paired median improvement | -0.295% | >= 0% |
| Practical wins/ties | 43.6% | >= 60% |
| Bootstrap lower bound | -3.313% | > -0.250% |
| Worst paired change | -21.987% | >= -3.000% |

Family means were regression `-3.168%`, binary `-0.055%`, multiclass `-0.604%`, and ranking
`+0.148%`. The candidate failed calibration, so confirmation seeds were not run. The production
assignments and separation-specific L1 test were removed after recording the result.

Evidence:

- `pr132_morph_regularized_control.json`: optimized effective-gradient control.
- `pr132_morph_raw_info.json`: raw-information candidate.

## Balance and information-weight defaults

Status: **no default change**. Calibration used seeds `0`, `1`, and `2` on the full nine-fixture
matrix. No candidate passed every predeclared veto, so confirmation seeds and public-report datasets
were not run.

| Arm | Mean | Median | Win/tie | Worst pair | Decision |
|---|---:|---:|---:|---:|---|
| No balance | -3.482% | -1.418% | 29.6% | -10.353% | Reject |
| Info weight 0.05 | +2.579% | +0.594% | 70.4% | -4.402% | Reject |
| Info weight 0.075 | +1.784% | +0.743% | 77.8% | -3.166% | Reject |
| Info weight 0.10 | 0.000% | 0.000% | 100.0% | 0.000% | Current control |
| Info weight 0.15 | -4.553% | -2.763% | 22.2% | -14.682% | Reject |

Disabling balance particularly hurt regression (`-6.186%` family mean), so the enabled default is
retained. Weight `0.05` failed binary, multiclass, and ranking family vetoes as well as the worst-pair
limit. Weight `0.075` was the strongest candidate but missed the declared `-3.0%` worst-pair limit
with `-3.166%`; it was rejected without using held-out data. Full records and gate summaries are in
`pr132_morph_calibration.json`.
