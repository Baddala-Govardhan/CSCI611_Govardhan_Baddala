from __future__ import annotations


def resolve_boost_backend(requested: str) -> str:
    requested = requested.lower().strip()
    if requested in {"xgboost", "histgb"}:
        return requested
    if requested == "auto":
        try:
            from xgboost import XGBClassifier  # noqa: F401

            return "xgboost"
        except Exception:
            return "histgb"
    raise ValueError(f"Unknown backend: {requested}")


def make_gradient_booster(
    backend: str,
    *,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
):
    bk = backend.lower().strip()
    if bk == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed,
            verbose=1,
        )

    if bk == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=2,
        )

    raise ValueError(f"Unknown backend: {backend}")
