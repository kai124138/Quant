"""Small classifiers and validation temperature scaling; logits are unitless."""

from dataclasses import dataclass
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from lob_latency.features import QI_COLUMNS
from lob_latency.metrics import entropy_bits, logit_losses_bits

HGB_ITERATIONS = (50, 100, 200, 400)
SEEDS = (0, 1, 2, 3, 4)


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit positive T by validation cross-entropy; optimize log(T) in [-8, 8]."""
    result = minimize_scalar(lambda log_t: logit_losses_bits(labels, logits / np.exp(log_t)).mean(),
                             bounds=(-8.0, 8.0), method="bounded",
                             options={"xatol": 1e-8})
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    return float(np.exp(result.x))


@dataclass
class FittedModel:
    """Trained estimator with fixed validation-selected iterations and temperature."""

    estimator: object
    columns: list[int] | None
    temperature: float
    prior: float
    validation_skill_bits: float
    seed: int
    max_iter: int | None
    validation_candidates: dict

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Return temperature-scaled class-1 logits for feature rows."""
        if self.columns is not None:
            x = x[:, self.columns]
        if isinstance(self.estimator, LogisticRegression):
            scores = self.estimator.decision_function(x)
        else:
            scores = logit(np.clip(self.estimator.predict_proba(x)[:, 1], 1e-15, 1 - 1e-15))
        return scores / self.temperature

    def metadata(self) -> dict:
        return {"seed": self.seed, "temperature": self.temperature, "train_prior_up": self.prior,
                "validation_skill_bits": self.validation_skill_bits, "max_iter": self.max_iter,
                "validation_candidate_loss_bits": self.validation_candidates}


def fit_model(name: str, x_train: np.ndarray, y_train: np.ndarray,
              x_validation: np.ndarray, y_validation: np.ndarray, seed: int = 0) -> FittedModel:
    """Select HGB iteration count with raw validation loss, then calibrate once."""
    if len(np.unique(y_train)) != 2 or not np.isin(y_train, [0, 1]).all():
        raise ValueError("Training labels must contain both classes")
    columns = QI_COLUMNS if name == "qi_logit" else None
    xt = x_train[:, columns] if columns is not None else x_train
    xv = x_validation[:, columns] if columns is not None else x_validation
    candidates = {}
    best_iter = None
    if name in ("qi_logit", "ofi_logit"):
        estimator = LogisticRegression(random_state=seed, max_iter=1000)
        estimator.fit(xt, y_train)
        raw_logits = estimator.decision_function(xv)
    elif name == "hgb":
        estimator = HistGradientBoostingClassifier(learning_rate=0.1, early_stopping=False,
                                                   random_state=seed, warm_start=True)
        best_loss = np.inf
        best_estimator = None
        # Continuing the same boosting trajectory avoids four independent fits.
        for iterations in HGB_ITERATIONS:
            estimator.set_params(max_iter=iterations)
            estimator.fit(xt, y_train)
            scores = logit(np.clip(estimator.predict_proba(xv)[:, 1], 1e-15, 1 - 1e-15))
            loss = float(logit_losses_bits(y_validation, scores).mean())
            candidates[str(iterations)] = loss
            if loss < best_loss:
                best_loss, best_iter = loss, iterations
                best_estimator = deepcopy(estimator)
        estimator = best_estimator
        estimator.set_params(warm_start=False)
        raw_logits = logit(np.clip(estimator.predict_proba(xv)[:, 1], 1e-15, 1 - 1e-15))
    else:
        raise ValueError(f"Unknown model: {name}")
    temperature = temperature_scale(raw_logits, y_validation)
    prior = float(y_train.mean())
    validation_skill = entropy_bits(prior) - float(
        logit_losses_bits(y_validation, raw_logits / temperature).mean())
    return FittedModel(estimator, columns, temperature, prior, validation_skill,
                       seed, best_iter, candidates)
