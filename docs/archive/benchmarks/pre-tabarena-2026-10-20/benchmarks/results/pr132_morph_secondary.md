# PR #132 MorphBoost Secondary Cost Profile

All measurements were run in release mode on Apple Silicon with seven measured batches after
warmup. No secondary implementation crossed its predeclared threshold.

## EMA preparation

`benchmark_morph_ema_preparation` copies and updates EMA statistics for 8,192 gradient pairs. The
median was **4.60 us per round**. The matching tall/narrow Morph control took 0.1710 s for 80 rounds,
or 2.14 ms per round, so EMA preparation accounts for approximately **0.22%**. Even eliminating the
work entirely cannot approach the 3% end-to-end threshold; a direct-pair prototype was therefore
not warranted.

Decision: retain the reusable gradient scratch and existing `GradientEmaStats::update` path.

## Categorical scanning

`benchmark_morph_categorical_scan` exhaustively Fisher-sorts and scans 64 categories plus missing
mass. The median was **5.15 us per feature/node scan**. A representative 8,192-row, 16-feature,
80-round native-categorical Morph fit had a seven-run median of **216.54 ms**. Assuming every tree
reaches all 31 internal nodes and the categorical feature is scanned at every node gives a
conservative 12.77 ms, or **5.9%** of fit time; actual share is lower when trees stop early.

Decision: keep categorical Morph on the scalar Fisher scanner because the upper bound is below the
10% vectorization threshold.

## Joint-output counts

`benchmark_joint_morph_counts` used five fixed-seed 64-bin fixtures where every fractional Hessian
mapped to proxy count one while exact bin counts ranged from 2 to 20. Proxy and exact counts chose
the same threshold in **5/5** fixtures. Current proxy scoring took a median **0.380 us per 64-bin
one-output scan**.

A shared `u32` count plane adds `1 / (2K)` to the grad/hess histogram payload for `K` outputs: 25%
at two outputs, 16.7% at three, and 10% at five. It therefore exceeds the 10% representative-memory
limit for common low-output joint models without evidence of a candidate-ordering benefit.

Decision: defer exact joint counts and retain the documented Hessian-derived proxy.
