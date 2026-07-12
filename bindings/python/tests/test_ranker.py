"""Tests for GBMRanker and NDCG evaluation metric."""

from __future__ import annotations

import pickle
import unittest

import numpy as np

from alloygbm import GBMRanker, ndcg


def _make_ranking_dataset(
    n_queries: int = 10,
    docs_per_query: int = 15,
    n_features: int = 4,
    seed: int = 42,
) -> tuple:
    """Create a synthetic ranking dataset with linearly separable relevance."""
    rng = np.random.RandomState(seed)
    n_total = n_queries * docs_per_query
    group = np.repeat(np.arange(n_queries, dtype=np.uint32), docs_per_query)
    X = rng.randn(n_total, n_features).astype(np.float32)
    # Relevance labels: correlated with first feature.
    raw_rel = X[:, 0] + 0.5 * X[:, 1]
    # Bin into 0-4 graded relevance.
    y = np.clip(np.digitize(raw_rel, [-1.5, -0.5, 0.5, 1.5]), 0, 4).astype(np.float32)
    return X, y, group


class GBMRankerObjectiveTests(unittest.TestCase):
    """Test that all ranking objectives can train and predict."""

    _OBJECTIVES = [
        "rank:ndcg",
        "rank:pairwise",
        "rank:xendcg",
        "queryrmse",
        "yetirank",
    ]

    def test_all_objectives_train_without_error(self) -> None:
        X, y, group = _make_ranking_dataset()
        for obj in self._OBJECTIVES:
            with self.subTest(objective=obj):
                ranker = GBMRanker(
                    ranking_objective=obj,
                    n_estimators=5,
                    seed=42,
                    training_policy="manual",
                    min_split_gain=0.0,
                )
                ranker.fit(X, y, group=group)
                preds = ranker.predict(X)
                self.assertIsInstance(preds, np.ndarray)
                self.assertEqual(len(preds), len(y))
                self.assertTrue(np.issubdtype(preds.dtype, np.floating))

    def test_objective_name_mapping(self) -> None:
        self.assertEqual(GBMRanker(ranking_objective="rank:ndcg")._objective_name(), "rank_ndcg")
        self.assertEqual(GBMRanker(ranking_objective="rank:pairwise")._objective_name(), "rank_pairwise")
        self.assertEqual(GBMRanker(ranking_objective="rank:xendcg")._objective_name(), "rank_xendcg")
        self.assertEqual(GBMRanker(ranking_objective="queryrmse")._objective_name(), "queryrmse")
        self.assertEqual(GBMRanker(ranking_objective="yetirank")._objective_name(), "yetirank")

    def test_invalid_objective_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "ranking_objective"):
            GBMRanker(ranking_objective="invalid")

    def test_group_required(self) -> None:
        X, y, _group = _make_ranking_dataset()
        ranker = GBMRanker(n_estimators=3, seed=42)
        with self.assertRaisesRegex(ValueError, "group"):
            ranker.fit(X, y, group=None)

    def test_sorts_by_group(self) -> None:
        """Data with shuffled group IDs should still train correctly."""
        X, y, group = _make_ranking_dataset(n_queries=3, docs_per_query=5)
        # Shuffle the data.
        rng = np.random.RandomState(123)
        perm = rng.permutation(len(y))
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        group_shuffled = group[perm]

        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=5,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        # Should not raise — internal sorting handles it.
        ranker.fit(X_shuffled, y_shuffled, group=group_shuffled)
        preds = ranker.predict(X_shuffled)
        self.assertEqual(len(preds), len(y))

    def test_predict_returns_numpy_array(self) -> None:
        x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        group = [0, 0, 1, 1]
        model = GBMRanker(n_estimators=3, max_depth=2).fit(x, y, group=group)

        predictions = model.predict(x)

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (4,))
        self.assertTrue(np.issubdtype(predictions.dtype, np.floating))


class GBMRankerEarlyStoppingTests(unittest.TestCase):
    """Test ranking with validation and early stopping."""

    def test_early_stopping_with_eval_set(self) -> None:
        X, y, group = _make_ranking_dataset(n_queries=15, docs_per_query=20, seed=42)
        X_val, y_val, group_val = _make_ranking_dataset(
            n_queries=5, docs_per_query=20, seed=99
        )
        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=100,
            early_stopping_rounds=5,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        ranker.fit(X, y, group=group, eval_set=(X_val, y_val), eval_group=group_val)
        # Early stopping should have kicked in.
        self.assertLess(ranker.n_estimators_, 100)
        self.assertIsNotNone(ranker.best_iteration_)

    def test_eval_group_required_with_eval_set(self) -> None:
        X, y, group = _make_ranking_dataset(n_queries=5, docs_per_query=5)
        ranker = GBMRanker(n_estimators=3, seed=42)
        with self.assertRaisesRegex(ValueError, "eval_group"):
            ranker.fit(
                X, y, group=group,
                eval_set=(X[:10], y[:10]),
                # eval_group missing
            )

    def test_evals_result_has_ndcg_for_ranking(self) -> None:
        X, y, group = _make_ranking_dataset(n_queries=10, docs_per_query=10)
        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=5,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        ranker.fit(X, y, group=group)
        er = ranker.evals_result_
        self.assertIn("ndcg", er["train"])
        self.assertEqual(len(er["train"]["ndcg"]), 5)

    def test_evals_result_has_queryrmse_for_queryrmse(self) -> None:
        X, y, group = _make_ranking_dataset(n_queries=10, docs_per_query=10)
        ranker = GBMRanker(
            ranking_objective="queryrmse",
            n_estimators=5,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        ranker.fit(X, y, group=group)
        er = ranker.evals_result_
        self.assertIn("queryrmse", er["train"])


class GBMRankerSerializationTests(unittest.TestCase):
    """Test pickle and get_params/set_params."""

    def test_pickle_roundtrip(self) -> None:
        X, y, group = _make_ranking_dataset(n_queries=5, docs_per_query=10)
        ranker = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=5,
            seed=42,
            training_policy="manual",
            min_split_gain=0.0,
        )
        ranker.fit(X, y, group=group)
        preds1 = ranker.predict(X)

        restored = pickle.loads(pickle.dumps(ranker))
        preds2 = restored.predict(X)
        for a, b in zip(preds1, preds2):
            self.assertAlmostEqual(a, b, places=5)

    def test_get_params_includes_ranking_objective(self) -> None:
        ranker = GBMRanker(ranking_objective="yetirank")
        params = ranker.get_params()
        self.assertEqual(params["ranking_objective"], "yetirank")

    def test_ranking_sigma_param_roundtrip(self) -> None:
        import inspect

        ranker = GBMRanker(ranking_sigma=0.75)

        self.assertEqual(ranker.ranking_sigma, 0.75)
        self.assertEqual(ranker.get_params()["ranking_sigma"], 0.75)
        self.assertIn("ranking_sigma", inspect.signature(GBMRanker.__init__).parameters)
        self.assertIn("ranking_sigma=0.75", repr(ranker))

        ranker.set_params(ranking_sigma=1.5)
        self.assertEqual(ranker.ranking_sigma, 1.5)

    def test_lambdarank_truncation_level_param_roundtrip(self) -> None:
        import inspect

        ranker = GBMRanker(lambdarank_truncation_level=3, lambdarank_normalize=True)

        self.assertEqual(ranker.lambdarank_truncation_level, 3)
        self.assertTrue(ranker.lambdarank_normalize)
        self.assertEqual(ranker.get_params()["lambdarank_truncation_level"], 3)
        self.assertTrue(ranker.get_params()["lambdarank_normalize"])
        self.assertIn(
            "lambdarank_truncation_level",
            inspect.signature(GBMRanker.__init__).parameters,
        )
        self.assertIn(
            "lambdarank_normalize",
            inspect.signature(GBMRanker.__init__).parameters,
        )
        self.assertIn("lambdarank_truncation_level=3", repr(ranker))
        self.assertIn("lambdarank_normalize=True", repr(ranker))

        ranker.set_params(lambdarank_truncation_level=None)
        self.assertIsNone(ranker.lambdarank_truncation_level)
        ranker.set_params(lambdarank_normalize=False)
        self.assertFalse(ranker.lambdarank_normalize)

    def test_lambdarank_truncation_level_validation(self) -> None:
        for invalid in (0, -1, 1.5, float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "lambdarank_truncation_level"):
                    GBMRanker(lambdarank_truncation_level=invalid)
        for invalid in (0, 1, "yes"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "lambdarank_normalize"):
                    GBMRanker(lambdarank_normalize=invalid)

    def test_ranking_sigma_validation(self) -> None:
        for invalid in (0.0, -1.0, float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "ranking_sigma"):
                    GBMRanker(ranking_sigma=invalid)

    def test_set_params_ranking_objective(self) -> None:
        ranker = GBMRanker(ranking_objective="rank:ndcg")
        ranker.set_params(ranking_objective="rank:pairwise")
        self.assertEqual(ranker.ranking_objective, "rank:pairwise")

    def test_repr(self) -> None:
        ranker = GBMRanker(ranking_objective="rank:ndcg")
        r = repr(ranker)
        self.assertIn("GBMRanker(", r)
        self.assertIn("ranking_objective='rank:ndcg'", r)

    def test_ranking_sigma_changes_pairwise_fit(self) -> None:
        X, y, group = _make_ranking_dataset(n_queries=6, docs_per_query=5, seed=11)
        common = dict(
            ranking_objective="rank:pairwise",
            n_estimators=6,
            learning_rate=0.2,
            max_depth=3,
            training_policy="manual",
            seed=11,
        )
        low_sigma = GBMRanker(**common, ranking_sigma=0.5).fit(X, y, group=group)
        high_sigma = GBMRanker(**common, ranking_sigma=2.0).fit(X, y, group=group)

        diff = np.abs(np.asarray(low_sigma.predict(X)) - np.asarray(high_sigma.predict(X)))
        self.assertGreater(float(diff.mean()), 1e-6)

    def test_lambdarank_truncation_level_changes_ndcg_fit(self) -> None:
        X, y, group = _make_ranking_dataset(n_queries=8, docs_per_query=6, seed=9)
        common = dict(
            ranking_objective="rank:ndcg",
            n_estimators=8,
            learning_rate=0.2,
            max_depth=3,
            training_policy="manual",
            seed=9,
        )
        full = GBMRanker(**common).fit(X, y, group=group)
        top2 = GBMRanker(**common, lambdarank_truncation_level=2).fit(X, y, group=group)

        diff = np.abs(np.asarray(full.predict(X)) - np.asarray(top2.predict(X)))
        self.assertGreater(float(diff.mean()), 1e-6)

    def test_lambdarank_normalize_changes_ndcg_fit(self) -> None:
        X, y, group = _make_ranking_dataset(n_queries=8, docs_per_query=6, seed=11)
        common = dict(
            ranking_objective="rank:ndcg",
            n_estimators=8,
            learning_rate=0.2,
            max_depth=3,
            training_policy="manual",
            seed=11,
        )
        unnormalized = GBMRanker(**common).fit(X, y, group=group)
        normalized = GBMRanker(**common, lambdarank_normalize=True).fit(X, y, group=group)

        diff = np.abs(np.asarray(unnormalized.predict(X)) - np.asarray(normalized.predict(X)))
        self.assertGreater(float(diff.mean()), 1e-6)


class GBMRankerAutoPolicyRegressionTests(unittest.TestCase):
    """Regression tests for the auto-policy ranking bug.

    Before the fix, GBMRanker under the default ``training_policy="auto"``
    exited after only 4 rounds with ``LossImprovementBelowThreshold`` for
    ``rank:ndcg`` on a 5000-row ranking dataset (because the auto-policy's
    ``min_loss_improvement`` threshold and the unconditional "training loss
    went up" early-exit were tuned for regression losses, not
    NDCG-normalized ranking losses). The result was bit-identical NDCG across
    all seeds/profiles and ~15 ms fit times regardless of ``n_estimators``.
    """

    def _bench_shaped_dataset(self) -> tuple:
        # Large enough to hit row_count >= 4096 auto-policy branches that
        # set min_loss_improvement and max_consecutive_weak_improvements.
        return _make_ranking_dataset(n_queries=200, docs_per_query=25, n_features=16, seed=7)

    def test_ranker_auto_policy_commits_trees(self) -> None:
        X, y, group = self._bench_shaped_dataset()
        model = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=50,
            learning_rate=0.05,
            max_depth=6,
            row_subsample=0.8,
            col_subsample=0.8,
            seed=7,
        )
        model.fit(X, y, group=group)
        self.assertGreaterEqual(model.n_estimators_, 45)
        self.assertEqual(model.stop_reason_, "CompletedRequestedRounds")

    def test_ranker_predictions_not_constant(self) -> None:
        X, y, group = self._bench_shaped_dataset()
        model = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=50,
            learning_rate=0.05,
            max_depth=6,
            row_subsample=0.8,
            col_subsample=0.8,
            seed=7,
        )
        model.fit(X, y, group=group)
        preds = np.asarray(model.predict(X))
        self.assertGreater(float(preds.std()), 0.0)
        # Fewer than 5% unique values would suggest degenerate prediction.
        self.assertGreater(int(np.unique(preds).size), len(preds) // 20)

    def test_ranker_depth_matters(self) -> None:
        X, y, group = self._bench_shaped_dataset()
        shallow = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=30,
            max_depth=2,
            row_subsample=0.8,
            col_subsample=0.8,
            seed=7,
        )
        deep = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=30,
            max_depth=8,
            row_subsample=0.8,
            col_subsample=0.8,
            seed=7,
        )
        shallow.fit(X, y, group=group)
        deep.fit(X, y, group=group)
        shallow_preds = np.asarray(shallow.predict(X))
        deep_preds = np.asarray(deep.predict(X))
        # Depth must actually influence the predictions (not a zero-tree exit).
        self.assertGreater(float(np.abs(shallow_preds - deep_preds).mean()), 0.0)

    def test_ranker_rounds_matter(self) -> None:
        X, y, group = self._bench_shaped_dataset()
        short = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=10,
            learning_rate=0.05,
            max_depth=6,
            row_subsample=0.8,
            col_subsample=0.8,
            seed=7,
        )
        long_ = GBMRanker(
            ranking_objective="rank:ndcg",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            row_subsample=0.8,
            col_subsample=0.8,
            seed=7,
        )
        short.fit(X, y, group=group)
        long_.fit(X, y, group=group)
        self.assertEqual(short.n_estimators_, 10)
        self.assertEqual(long_.n_estimators_, 200)
        short_preds = np.asarray(short.predict(X))
        long_preds = np.asarray(long_.predict(X))
        self.assertGreater(float(np.abs(short_preds - long_preds).mean()), 0.0)

    def test_init_signature_exposes_regressor_params(self) -> None:
        """inspect.signature(GBMRanker.__init__) must expose the full
        GBMRegressor parameter surface — benchmarks and sklearn clone build
        kwargs by introspecting the signature, and silently dropping
        ``n_estimators`` / ``learning_rate`` is how the ranker ended up
        training with default ``n_estimators=6`` on 4 profiles × 5 seeds.
        """
        import inspect
        from alloygbm import GBMRegressor

        ranker_sig = inspect.signature(GBMRanker.__init__)
        regressor_sig = inspect.signature(GBMRegressor.__init__)
        for required in (
            "learning_rate",
            "max_depth",
            "n_estimators",
            "row_subsample",
            "col_subsample",
            "seed",
            "min_split_gain",
            "lambda_l2",
            "training_policy",
        ):
            self.assertIn(
                required,
                ranker_sig.parameters,
                f"GBMRanker.__init__ signature must expose {required!r}",
            )
        # ranking_objective is ranker-only; every other GBMRegressor param
        # should be surfaced identically.
        self.assertIn("ranking_objective", ranker_sig.parameters)
        for name, param in regressor_sig.parameters.items():
            if name in ("self",) or param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            self.assertIn(name, ranker_sig.parameters, f"missing {name}")

    def test_ranker_stop_reason_exposed(self) -> None:
        X, y, group = _make_ranking_dataset(n_queries=5, docs_per_query=5, seed=7)
        model = GBMRanker(n_estimators=5, seed=7)
        model.fit(X, y, group=group)
        self.assertTrue(hasattr(model, "stop_reason_"))
        self.assertIsInstance(model.stop_reason_, str)
        self.assertGreater(len(model.stop_reason_), 0)
        self.assertTrue(hasattr(model, "rounds_completed_"))
        self.assertIsInstance(model.rounds_completed_, int)


class NDCGMetricTests(unittest.TestCase):
    """Test the ndcg evaluation metric."""

    def test_perfect_ranking(self) -> None:
        score = ndcg([3, 2, 1, 0], [3, 2, 1, 0], group=[0, 0, 0, 0])
        self.assertAlmostEqual(score, 1.0)

    def test_reversed_ranking(self) -> None:
        score = ndcg([3, 2, 1, 0], [0, 1, 2, 3], group=[0, 0, 0, 0])
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_all_same_labels(self) -> None:
        score = ndcg([1, 1, 1], [3, 1, 2], group=[0, 0, 0])
        self.assertAlmostEqual(score, 1.0)

    def test_multi_group(self) -> None:
        # Two groups, both perfectly ranked.
        score = ndcg(
            [2, 1, 0, 2, 1, 0],
            [3, 2, 1, 3, 2, 1],
            group=[0, 0, 0, 1, 1, 1],
        )
        self.assertAlmostEqual(score, 1.0)

    def test_single_document_group(self) -> None:
        score = ndcg([1], [0.5], group=[0])
        self.assertAlmostEqual(score, 1.0)

    def test_k_truncation(self) -> None:
        # With k=1, only the top-ranked document matters.
        # Perfect: doc with label 3 ranked first.
        perfect_at_1 = ndcg([3, 0, 0], [3, 0, 0], group=[0, 0, 0], k=1)
        self.assertAlmostEqual(perfect_at_1, 1.0)
        # Worst at 1: doc with label 0 ranked first.
        worst_at_1 = ndcg([3, 0, 0], [0, 0, 3], group=[0, 0, 0], k=1)
        self.assertAlmostEqual(worst_at_1, 0.0)

    def test_validates_same_length(self) -> None:
        with self.assertRaises(ValueError):
            ndcg([1, 2], [1, 2, 3], group=[0, 0])

    def test_validates_group_length(self) -> None:
        with self.assertRaises(ValueError):
            ndcg([1, 2], [1, 2], group=[0])


if __name__ == "__main__":
    unittest.main()
