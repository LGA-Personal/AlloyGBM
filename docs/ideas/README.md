# Ideas And Research

This directory category is for design notes, external inspiration, and future experiments.

Active work:

- [`accuracy-at-depth-board.md`](accuracy-at-depth-board.md) — **open for
  contribution.** Multi-author board for closing AlloyGBM's accuracy gap at high
  depth. We are at parity with LightGBM and XGBoost at depth 6 but degrade ~22% on
  log-loss objectives by depth 12, where they stay flat. The problem statement,
  the code audit, and ideas 1–9 are in the document itself; every idea has a
  reviewer field, and there is space to add new ideas from 10 onward.

Closed:

- [`single-thread-optimization-board.md`](single-thread-optimization-board.md) —
  **closed 2026-09-16**, all 18 ideas resolved: six landed (every one
  bit-identical) and twelve rejected on measurement. Took a 2,000-row fit from
  0.292 s to 0.074 s with byte-identical models. Seeded with one measured win and
  seven hypotheses; Codex added feedback and experiments (9–12); Antigravity added
  commentary, hypotheses (13–16), and structural comparison. Read
  [the problem statement](../reviews/2026-09-06-single-thread-throughput.md) for
  the method, which the accuracy board reuses.

Useful starting points:

- `docs/plans/perpetual_inspiration_for_alloygbm.md`
- `docs/archive/ideas/performance_and_training_ideas.md`
- `docs/archive/ideas/performance_and_training_ideas_2.md`

The files under `docs/archive/ideas/` are legacy notes from the previous planning system. They remain useful as raw research material, but they are not current execution plans.
