"""MultiLabelGBMRanker: multi-output ranking with per-label objectives.

Demonstrates a 2-label ranking scenario where each label may have its
own ranking objective (here: ``rank:ndcg`` for clicks, ``rank:pairwise``
for conversions).  ``y`` is shaped ``(n_rows, n_labels)``; ``predict``
returns scores with the same column layout.

Useful for e.g. recommender systems where you want to jointly model
multiple engagement signals against a shared candidate set.
"""

from __future__ import annotations

import numpy as np

from alloygbm import MultiLabelGBMRanker, ndcg


def _make_two_label_dataset(seed: int = 7):
    rng = np.random.RandomState(seed)
    n_queries, docs_per_query, n_features = 40, 25, 8
    rows = n_queries * docs_per_query

    X = rng.randn(rows, n_features).astype(np.float32)
    # Label 0 (clicks) primarily depends on features 0–2.
    # Label 1 (conversions) primarily on features 3–5.
    clicks_raw = X[:, 0] * 1.5 + X[:, 1] - X[:, 2] + rng.randn(rows) * 0.4
    convs_raw = X[:, 3] * 1.2 - X[:, 4] + X[:, 5] * 0.8 + rng.randn(rows) * 0.4

    clicks = np.digitize(clicks_raw, np.quantile(clicks_raw, [0.25, 0.5, 0.75])).astype(np.int32)
    convs = np.digitize(convs_raw, np.quantile(convs_raw, [0.5])).astype(np.int32)

    y = np.column_stack([clicks, convs])
    group = np.repeat(np.arange(n_queries), docs_per_query).astype(np.int32)
    return X, y, group


def main() -> None:
    X, y, group = _make_two_label_dataset()

    split = int(group.max() * 0.75)
    train_mask = group <= split
    X_train, y_train, group_train = X[train_mask], y[train_mask], group[train_mask]
    X_test, y_test, group_test = X[~train_mask], y[~train_mask], group[~train_mask]

    # Per-label ranking_objective list: clicks use NDCG-weighted
    # LambdaMART; conversions use pairwise (binary, no graded
    # relevance).
    model = MultiLabelGBMRanker(
        ranking_objective=["rank:ndcg", "rank:pairwise"],
        learning_rate=0.05,
        max_depth=6,
        n_estimators=300,
        deterministic=True,
        seed=7,
    )
    model.fit(X_train, y_train, group=group_train)

    scores = model.predict(X_test)  # shape (n_rows, n_labels)
    ndcg_clicks = ndcg(y_test[:, 0], scores[:, 0], group=group_test, k=10)
    ndcg_convs = ndcg(y_test[:, 1], scores[:, 1], group=group_test, k=10)

    print(f"prediction shape:   {scores.shape}")
    print(f"NDCG@10 clicks:     {ndcg_clicks:.4f}")
    print(f"NDCG@10 conversions:{ndcg_convs:.4f}")


if __name__ == "__main__":
    main()
