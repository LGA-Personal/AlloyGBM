# Sampled Prediction Delta Benchmark

## Runtime Identity

| Arm | Source commit | Python | Workdir |
| --- | --- | --- | --- |
| baseline | `6076af71d1d21b503f359af7101f9345bd43e112` | `/private/tmp/alloygbm-pr129-baseline/.venv/bin/python` | `/private/tmp/alloygbm-pr129-baseline` |
| candidate | `3538e3db633bf7f806111b95eddc93cf1c1c951a` | `/Users/lashby/Projects/AlloyGBM/.worktrees/pr-129-sampled-prediction-deltas/.venv/bin/python` | `/Users/lashby/Projects/AlloyGBM/.worktrees/pr-129-sampled-prediction-deltas` |

## Gates

- Performance gated: True
- Delta-sensitive native-time ratio: 0.7439
- All-eligible native-time ratio: 0.7179
- Aggregate RSS ratio: 0.9410
- Failures: 0
- Worst eligible case: `scalar_small_wide_leaf_subsample_050` slowed by 3.76%
  (1.0376x), within the 1.08x per-case limit.
- DART and quantile timing are descriptive fallback sentinels.

This full profile compares separate, manifest-attested baseline and candidate runtimes.
Every arm receives one unmeasured warmup subprocess per case.

## Case Medians

| Case | Reps | Baseline native s | Candidate native s | Ratio | Baseline RSS MiB | Candidate RSS MiB | Ratio | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fallback_scalar_dart_subsample_050` | 5 | 0.262560 | 0.258186 | 0.9833 | 9.69 | 8.73 | 0.9016 | dart_full_replay |
| `fallback_scalar_quantile_subsample_050` | 5 | 0.311461 | 0.286551 | 0.9200 | 9.97 | 9.06 | 0.9091 | quantile_full_replay |
| `multiclass_medium_wide_leaf_goss` | 5 | 0.196767 | 0.180014 | 0.9149 | 25.42 | 23.94 | 0.9416 |  |
| `multiclass_tall_narrow_level_subsample_050` | 5 | 0.307511 | 0.208310 | 0.6774 | 13.89 | 13.27 | 0.9550 |  |
| `scalar_medium_wide_level_goss` | 5 | 0.112232 | 0.111728 | 0.9955 | 24.47 | 24.38 | 0.9962 |  |
| `scalar_shallow_tall_leaf_subsample_050` | 5 | 0.191558 | 0.144761 | 0.7557 | 18.47 | 18.23 | 0.9873 |  |
| `scalar_small_wide_leaf_subsample_050` | 5 | 0.123686 | 0.128342 | 1.0376 | 18.09 | 17.36 | 0.9594 |  |
| `scalar_tall_narrow_level_full` | 5 | 0.205312 | 0.082360 | 0.4011 | 13.39 | 12.11 | 0.9043 |  |
| `scalar_tall_narrow_level_subsample_050` | 5 | 0.182783 | 0.127454 | 0.6973 | 13.22 | 12.25 | 0.9267 |  |
| `scalar_tall_narrow_level_subsample_080` | 5 | 0.214570 | 0.111869 | 0.5214 | 13.50 | 12.61 | 0.9340 |  |

## Exact Equivalence

| Case | Rep | Metric | Baseline quality | Candidate quality | Artifact SHA-256 | Prediction SHA-256 | Rounds | Stop reason |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: | --- |
| `scalar_tall_narrow_level_full` | 0 | rmse | 0.603686920217 | 0.603686920217 | `154c56e8f359667eb05f2d94c20c94fd24d2f42c45cb92c1ce6ebc115dfe1827` | `bcc0a39ee91255659b0a30685077d828e54ae3adb5bbcc8b23baef4225078c8c` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_full` | 1 | rmse | 0.603686920217 | 0.603686920217 | `154c56e8f359667eb05f2d94c20c94fd24d2f42c45cb92c1ce6ebc115dfe1827` | `bcc0a39ee91255659b0a30685077d828e54ae3adb5bbcc8b23baef4225078c8c` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_full` | 2 | rmse | 0.603686920217 | 0.603686920217 | `154c56e8f359667eb05f2d94c20c94fd24d2f42c45cb92c1ce6ebc115dfe1827` | `bcc0a39ee91255659b0a30685077d828e54ae3adb5bbcc8b23baef4225078c8c` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_full` | 3 | rmse | 0.603686920217 | 0.603686920217 | `154c56e8f359667eb05f2d94c20c94fd24d2f42c45cb92c1ce6ebc115dfe1827` | `bcc0a39ee91255659b0a30685077d828e54ae3adb5bbcc8b23baef4225078c8c` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_full` | 4 | rmse | 0.603686920217 | 0.603686920217 | `154c56e8f359667eb05f2d94c20c94fd24d2f42c45cb92c1ce6ebc115dfe1827` | `bcc0a39ee91255659b0a30685077d828e54ae3adb5bbcc8b23baef4225078c8c` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_subsample_080` | 0 | rmse | 0.51129884639 | 0.51129884639 | `f3570d52480889d2d108c8d5a26056435949d2127da934e4048049e4521d00a1` | `d670767b7eef072815fa1f792c6a2dc886fd236249f109a10de6751769bb5a9d` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_subsample_080` | 1 | rmse | 0.51129884639 | 0.51129884639 | `f3570d52480889d2d108c8d5a26056435949d2127da934e4048049e4521d00a1` | `d670767b7eef072815fa1f792c6a2dc886fd236249f109a10de6751769bb5a9d` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_subsample_080` | 2 | rmse | 0.51129884639 | 0.51129884639 | `f3570d52480889d2d108c8d5a26056435949d2127da934e4048049e4521d00a1` | `d670767b7eef072815fa1f792c6a2dc886fd236249f109a10de6751769bb5a9d` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_subsample_080` | 3 | rmse | 0.51129884639 | 0.51129884639 | `f3570d52480889d2d108c8d5a26056435949d2127da934e4048049e4521d00a1` | `d670767b7eef072815fa1f792c6a2dc886fd236249f109a10de6751769bb5a9d` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_subsample_080` | 4 | rmse | 0.51129884639 | 0.51129884639 | `f3570d52480889d2d108c8d5a26056435949d2127da934e4048049e4521d00a1` | `d670767b7eef072815fa1f792c6a2dc886fd236249f109a10de6751769bb5a9d` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_subsample_050` | 0 | rmse | 0.62448429007 | 0.62448429007 | `d149829f58a05956b7e8dffeda94d1a3ee12bdbd2c80ed940b599e5f82c83348` | `ba156faa47acd4f242672d710022636cd0f683ac7c2ef8fb8b7b196d8f2e7efa` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_subsample_050` | 1 | rmse | 0.62448429007 | 0.62448429007 | `d149829f58a05956b7e8dffeda94d1a3ee12bdbd2c80ed940b599e5f82c83348` | `ba156faa47acd4f242672d710022636cd0f683ac7c2ef8fb8b7b196d8f2e7efa` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_subsample_050` | 2 | rmse | 0.62448429007 | 0.62448429007 | `d149829f58a05956b7e8dffeda94d1a3ee12bdbd2c80ed940b599e5f82c83348` | `ba156faa47acd4f242672d710022636cd0f683ac7c2ef8fb8b7b196d8f2e7efa` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_subsample_050` | 3 | rmse | 0.62448429007 | 0.62448429007 | `d149829f58a05956b7e8dffeda94d1a3ee12bdbd2c80ed940b599e5f82c83348` | `ba156faa47acd4f242672d710022636cd0f683ac7c2ef8fb8b7b196d8f2e7efa` | 24 | CompletedRequestedRounds |
| `scalar_tall_narrow_level_subsample_050` | 4 | rmse | 0.62448429007 | 0.62448429007 | `d149829f58a05956b7e8dffeda94d1a3ee12bdbd2c80ed940b599e5f82c83348` | `ba156faa47acd4f242672d710022636cd0f683ac7c2ef8fb8b7b196d8f2e7efa` | 24 | CompletedRequestedRounds |
| `scalar_shallow_tall_leaf_subsample_050` | 0 | rmse | 0.729853148225 | 0.729853148225 | `82d1b7975b6b91cfd4e06be224a2bb506b6839f61007690c4c6042cb3c157013` | `55a6cf490f9c84c662a235698a162d14a5fc94c0caf69facb2cbde3a385e8608` | 24 | CompletedRequestedRounds |
| `scalar_shallow_tall_leaf_subsample_050` | 1 | rmse | 0.729853148225 | 0.729853148225 | `82d1b7975b6b91cfd4e06be224a2bb506b6839f61007690c4c6042cb3c157013` | `55a6cf490f9c84c662a235698a162d14a5fc94c0caf69facb2cbde3a385e8608` | 24 | CompletedRequestedRounds |
| `scalar_shallow_tall_leaf_subsample_050` | 2 | rmse | 0.729853148225 | 0.729853148225 | `82d1b7975b6b91cfd4e06be224a2bb506b6839f61007690c4c6042cb3c157013` | `55a6cf490f9c84c662a235698a162d14a5fc94c0caf69facb2cbde3a385e8608` | 24 | CompletedRequestedRounds |
| `scalar_shallow_tall_leaf_subsample_050` | 3 | rmse | 0.729853148225 | 0.729853148225 | `82d1b7975b6b91cfd4e06be224a2bb506b6839f61007690c4c6042cb3c157013` | `55a6cf490f9c84c662a235698a162d14a5fc94c0caf69facb2cbde3a385e8608` | 24 | CompletedRequestedRounds |
| `scalar_shallow_tall_leaf_subsample_050` | 4 | rmse | 0.729853148225 | 0.729853148225 | `82d1b7975b6b91cfd4e06be224a2bb506b6839f61007690c4c6042cb3c157013` | `55a6cf490f9c84c662a235698a162d14a5fc94c0caf69facb2cbde3a385e8608` | 24 | CompletedRequestedRounds |
| `scalar_medium_wide_level_goss` | 0 | rmse | 0.887792787086 | 0.887792787086 | `8a2281663e736bfe39358a6a1baa2c7e708141979c3c054759d954107f26ca12` | `36027014040be49d905c10630cd8acfcd5647f3f539dc7b4dfb7570a3477c6b4` | 24 | CompletedRequestedRounds |
| `scalar_medium_wide_level_goss` | 1 | rmse | 0.887792787086 | 0.887792787086 | `8a2281663e736bfe39358a6a1baa2c7e708141979c3c054759d954107f26ca12` | `36027014040be49d905c10630cd8acfcd5647f3f539dc7b4dfb7570a3477c6b4` | 24 | CompletedRequestedRounds |
| `scalar_medium_wide_level_goss` | 2 | rmse | 0.887792787086 | 0.887792787086 | `8a2281663e736bfe39358a6a1baa2c7e708141979c3c054759d954107f26ca12` | `36027014040be49d905c10630cd8acfcd5647f3f539dc7b4dfb7570a3477c6b4` | 24 | CompletedRequestedRounds |
| `scalar_medium_wide_level_goss` | 3 | rmse | 0.887792787086 | 0.887792787086 | `8a2281663e736bfe39358a6a1baa2c7e708141979c3c054759d954107f26ca12` | `36027014040be49d905c10630cd8acfcd5647f3f539dc7b4dfb7570a3477c6b4` | 24 | CompletedRequestedRounds |
| `scalar_medium_wide_level_goss` | 4 | rmse | 0.887792787086 | 0.887792787086 | `8a2281663e736bfe39358a6a1baa2c7e708141979c3c054759d954107f26ca12` | `36027014040be49d905c10630cd8acfcd5647f3f539dc7b4dfb7570a3477c6b4` | 24 | CompletedRequestedRounds |
| `scalar_small_wide_leaf_subsample_050` | 0 | rmse | 0.979100393483 | 0.979100393483 | `5ad038419d680c5c7f2f6a1cf0bb65f9a8cc9fec476f0d9175fc430361b1ec4f` | `d4ee829b531575fbb43073f75373bcf84bd7fae4212efea3af10b41a137b6125` | 24 | CompletedRequestedRounds |
| `scalar_small_wide_leaf_subsample_050` | 1 | rmse | 0.979100393483 | 0.979100393483 | `5ad038419d680c5c7f2f6a1cf0bb65f9a8cc9fec476f0d9175fc430361b1ec4f` | `d4ee829b531575fbb43073f75373bcf84bd7fae4212efea3af10b41a137b6125` | 24 | CompletedRequestedRounds |
| `scalar_small_wide_leaf_subsample_050` | 2 | rmse | 0.979100393483 | 0.979100393483 | `5ad038419d680c5c7f2f6a1cf0bb65f9a8cc9fec476f0d9175fc430361b1ec4f` | `d4ee829b531575fbb43073f75373bcf84bd7fae4212efea3af10b41a137b6125` | 24 | CompletedRequestedRounds |
| `scalar_small_wide_leaf_subsample_050` | 3 | rmse | 0.979100393483 | 0.979100393483 | `5ad038419d680c5c7f2f6a1cf0bb65f9a8cc9fec476f0d9175fc430361b1ec4f` | `d4ee829b531575fbb43073f75373bcf84bd7fae4212efea3af10b41a137b6125` | 24 | CompletedRequestedRounds |
| `scalar_small_wide_leaf_subsample_050` | 4 | rmse | 0.979100393483 | 0.979100393483 | `5ad038419d680c5c7f2f6a1cf0bb65f9a8cc9fec476f0d9175fc430361b1ec4f` | `d4ee829b531575fbb43073f75373bcf84bd7fae4212efea3af10b41a137b6125` | 24 | CompletedRequestedRounds |
| `multiclass_tall_narrow_level_subsample_050` | 0 | log_loss | 0.462458959471 | 0.462458959471 | `d77dfc6a8e8cd17430a8e70f8cce6be4ab31091359322f083fd74372da71e894` | `e991220d8904d4a2af7d57b38acd7d5925d2ab76fd925bfe904aed7613f89abb` | 24 | CompletedRequestedRounds |
| `multiclass_tall_narrow_level_subsample_050` | 1 | log_loss | 0.462458959471 | 0.462458959471 | `d77dfc6a8e8cd17430a8e70f8cce6be4ab31091359322f083fd74372da71e894` | `e991220d8904d4a2af7d57b38acd7d5925d2ab76fd925bfe904aed7613f89abb` | 24 | CompletedRequestedRounds |
| `multiclass_tall_narrow_level_subsample_050` | 2 | log_loss | 0.462458959471 | 0.462458959471 | `d77dfc6a8e8cd17430a8e70f8cce6be4ab31091359322f083fd74372da71e894` | `e991220d8904d4a2af7d57b38acd7d5925d2ab76fd925bfe904aed7613f89abb` | 24 | CompletedRequestedRounds |
| `multiclass_tall_narrow_level_subsample_050` | 3 | log_loss | 0.462458959471 | 0.462458959471 | `d77dfc6a8e8cd17430a8e70f8cce6be4ab31091359322f083fd74372da71e894` | `e991220d8904d4a2af7d57b38acd7d5925d2ab76fd925bfe904aed7613f89abb` | 24 | CompletedRequestedRounds |
| `multiclass_tall_narrow_level_subsample_050` | 4 | log_loss | 0.462458959471 | 0.462458959471 | `d77dfc6a8e8cd17430a8e70f8cce6be4ab31091359322f083fd74372da71e894` | `e991220d8904d4a2af7d57b38acd7d5925d2ab76fd925bfe904aed7613f89abb` | 24 | CompletedRequestedRounds |
| `multiclass_medium_wide_leaf_goss` | 0 | log_loss | 0.796069426138 | 0.796069426138 | `3f824fb52a506e943324124c660780ecbb68515e1cd4e4e41b08414fef6a7b96` | `063f3e8734118a14d9d50ae1b706f34ac59c504e429f155c890974b488decb59` | 24 | CompletedRequestedRounds |
| `multiclass_medium_wide_leaf_goss` | 1 | log_loss | 0.796069426138 | 0.796069426138 | `3f824fb52a506e943324124c660780ecbb68515e1cd4e4e41b08414fef6a7b96` | `063f3e8734118a14d9d50ae1b706f34ac59c504e429f155c890974b488decb59` | 24 | CompletedRequestedRounds |
| `multiclass_medium_wide_leaf_goss` | 2 | log_loss | 0.796069426138 | 0.796069426138 | `3f824fb52a506e943324124c660780ecbb68515e1cd4e4e41b08414fef6a7b96` | `063f3e8734118a14d9d50ae1b706f34ac59c504e429f155c890974b488decb59` | 24 | CompletedRequestedRounds |
| `multiclass_medium_wide_leaf_goss` | 3 | log_loss | 0.796069426138 | 0.796069426138 | `3f824fb52a506e943324124c660780ecbb68515e1cd4e4e41b08414fef6a7b96` | `063f3e8734118a14d9d50ae1b706f34ac59c504e429f155c890974b488decb59` | 24 | CompletedRequestedRounds |
| `multiclass_medium_wide_leaf_goss` | 4 | log_loss | 0.796069426138 | 0.796069426138 | `3f824fb52a506e943324124c660780ecbb68515e1cd4e4e41b08414fef6a7b96` | `063f3e8734118a14d9d50ae1b706f34ac59c504e429f155c890974b488decb59` | 24 | CompletedRequestedRounds |
| `fallback_scalar_dart_subsample_050` | 0 | rmse | 0.878289610392 | 0.878289610392 | `94c5bc6d63b51f2948435c7540270cccb71f543373bd10f2427301577bb91c38` | `345454d03e081fe03d8fd4a85d19e9de4707fcdbfcda4b6adb27ef1210cca6cf` | 24 | CompletedRequestedRounds |
| `fallback_scalar_dart_subsample_050` | 1 | rmse | 0.878289610392 | 0.878289610392 | `94c5bc6d63b51f2948435c7540270cccb71f543373bd10f2427301577bb91c38` | `345454d03e081fe03d8fd4a85d19e9de4707fcdbfcda4b6adb27ef1210cca6cf` | 24 | CompletedRequestedRounds |
| `fallback_scalar_dart_subsample_050` | 2 | rmse | 0.878289610392 | 0.878289610392 | `94c5bc6d63b51f2948435c7540270cccb71f543373bd10f2427301577bb91c38` | `345454d03e081fe03d8fd4a85d19e9de4707fcdbfcda4b6adb27ef1210cca6cf` | 24 | CompletedRequestedRounds |
| `fallback_scalar_dart_subsample_050` | 3 | rmse | 0.878289610392 | 0.878289610392 | `94c5bc6d63b51f2948435c7540270cccb71f543373bd10f2427301577bb91c38` | `345454d03e081fe03d8fd4a85d19e9de4707fcdbfcda4b6adb27ef1210cca6cf` | 24 | CompletedRequestedRounds |
| `fallback_scalar_dart_subsample_050` | 4 | rmse | 0.878289610392 | 0.878289610392 | `94c5bc6d63b51f2948435c7540270cccb71f543373bd10f2427301577bb91c38` | `345454d03e081fe03d8fd4a85d19e9de4707fcdbfcda4b6adb27ef1210cca6cf` | 24 | CompletedRequestedRounds |
| `fallback_scalar_quantile_subsample_050` | 0 | rmse | 0.688216233665 | 0.688216233665 | `e62272aafe5280413098fd6e93e35fd2048ade1f201492340efe8df83131e428` | `724e6ce3e1628090548739aeed9dd55fe7a3d2af6c78538c6d221dac2904f678` | 24 | CompletedRequestedRounds |
| `fallback_scalar_quantile_subsample_050` | 1 | rmse | 0.688216233665 | 0.688216233665 | `e62272aafe5280413098fd6e93e35fd2048ade1f201492340efe8df83131e428` | `724e6ce3e1628090548739aeed9dd55fe7a3d2af6c78538c6d221dac2904f678` | 24 | CompletedRequestedRounds |
| `fallback_scalar_quantile_subsample_050` | 2 | rmse | 0.688216233665 | 0.688216233665 | `e62272aafe5280413098fd6e93e35fd2048ade1f201492340efe8df83131e428` | `724e6ce3e1628090548739aeed9dd55fe7a3d2af6c78538c6d221dac2904f678` | 24 | CompletedRequestedRounds |
| `fallback_scalar_quantile_subsample_050` | 3 | rmse | 0.688216233665 | 0.688216233665 | `e62272aafe5280413098fd6e93e35fd2048ade1f201492340efe8df83131e428` | `724e6ce3e1628090548739aeed9dd55fe7a3d2af6c78538c6d221dac2904f678` | 24 | CompletedRequestedRounds |
| `fallback_scalar_quantile_subsample_050` | 4 | rmse | 0.688216233665 | 0.688216233665 | `e62272aafe5280413098fd6e93e35fd2048ade1f201492340efe8df83131e428` | `724e6ce3e1628090548739aeed9dd55fe7a3d2af6c78538c6d221dac2904f678` | 24 | CompletedRequestedRounds |
