"""GBMClassifier: binary classification on Wisconsin Breast Cancer.

Demonstrates predict / predict_proba, sklearn ClassifierMixin
compatibility (`.score` returns accuracy), and probabilistic log-loss
evaluation.
"""

from __future__ import annotations

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from alloygbm import GBMClassifier, accuracy, log_loss


def main() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=7, stratify=data.target
    )

    model = GBMClassifier(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        early_stopping_rounds=30,
        deterministic=True,
        seed=7,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    labels = model.predict(X_test)
    probas = model.predict_proba(X_test)

    print(f"rounds trained:   {model.n_estimators_}")
    print(f"test accuracy:    {accuracy(y_test, labels):.4f}")
    # log_loss expects P(y=1) — column 1 of predict_proba.
    print(f"test log-loss:    {log_loss(y_test, probas[:, 1]):.4f}")
    print(f"sklearn .score(): {model.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
