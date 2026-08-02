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
