from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    metric: str
    layer: int
    pool: str
    r2_train: float
    r2_test: float
    n_train: int
    n_test: int
    weights: np.ndarray   # [d_model] float32
    intercept: float


def fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: str,
    layer: int,
    pool: str,
    alpha: float = 1.0,
) -> ProbeResult:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.astype(np.float32))
    X_test_s = scaler.transform(X_test.astype(np.float32))

    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    model.fit(X_train_s, y_train.astype(np.float32))

    return ProbeResult(
        metric=metric,
        layer=layer,
        pool=pool,
        r2_train=float(model.score(X_train_s, y_train)),
        r2_test=float(model.score(X_test_s, y_test)),
        n_train=len(y_train),
        n_test=len(y_test),
        weights=model.coef_.astype(np.float32),
        intercept=float(model.intercept_),
    )
