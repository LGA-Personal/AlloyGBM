"""GBMRanker: learning-to-rank with LambdaMART on synthetic queries.

Generates a small grouped dataset (queries × documents with graded
relevance) and trains a ranker with the NDCG-weighted LambdaMART
objective.  Evaluates with NDCG@10.
"""

from __future__ import annotations

import numpy as np

from alloygbm import GBMRanker, ndcg


def _make_synthetic_ranking_dataset(
    n_queries: int = 50,
    docs_per_query: int = 20,
    n_features: int = 10,
    seed: int = 7,
):
    rng = np.random.RandomState(seed)
    rows = n_queries * docs_per_query

    X = rng.randn(rows, n_features).astype(np.float32)
    # True relevance is a noisy linear function of the first 3 features
    # bucketed into 5 levels (0–4).  Within each query, AlloyGBM's
    # LambdaMART objective will learn to rank docs by this score.
    raw = X[:, 0] * 2.0 + X[:, 1] - X[:, 2] + rng.randn(rows) * 0.3
    y = np.digitize(raw, np.quantile(raw, [0.2, 0.4, 0.6, 0.8])).astype(np.int32)

    group = np.repeat(np.arange(n_queries), docs_per_query).astype(np.int32)
    return X, y, group


def main() -> None:
    X, y, group = _make_synthetic_ranking_dataset()

    # 70/30 train/test split that keeps whole queries on one side.
    split = int(group.max() * 0.7)
    train_mask = group <= split
    X_train, y_train, group_train = X[train_mask], y[train_mask], group[train_mask]
    X_test, y_test, group_test = X[~train_mask], y[~train_mask], group[~train_mask]

    model = GBMRanker(
        ranking_objective="rank:ndcg",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=300,
        deterministic=True,
        seed=7,
    )
    model.fit(X_train, y_train, group=group_train)

    scores = model.predict(X_test)
    score = ndcg(y_test, scores, group=group_test, k=10)

    print(f"rounds trained:   {model.n_estimators_}")
    print(f"NDCG@10:          {score:.4f}")


if __name__ == "__main__":
    main()
