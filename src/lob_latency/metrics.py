"""Skill and paired moving-time-block bootstrap, in bits/prediction and ns."""

import numpy as np
from scipy.special import xlog1py, xlogy
from sklearn.metrics import matthews_corrcoef

LN2 = np.log(2.0)


def entropy_bits(prior: float) -> float:
    """Binary Shannon entropy in bits."""
    return float(-(xlogy(prior, prior) + xlog1py(1 - prior, -prior)) / LN2)


def probability_losses_bits(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Cross-entropy in bits, including exact zero loss for perfect predictions."""
    return -(xlogy(labels, probabilities) + xlog1py(1 - labels, -probabilities)) / LN2


def logit_losses_bits(labels: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Stable Bernoulli cross-entropy in bits per prediction."""
    # Signed softplus avoids cancellation for highly confident correct logits.
    return np.logaddexp(0.0, (1 - 2 * np.asarray(labels, dtype=float)) * logits) / LN2


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    """Accuracy and MCC at probability >= 0.5."""
    prediction = logits >= 0
    return {"accuracy": float(np.mean(prediction == labels)),
            "mcc": float(matthews_corrcoef(labels, prediction))}


def skill_bits(labels: np.ndarray, probabilities: np.ndarray, prior: float) -> float:
    return entropy_bits(prior) - float(probability_losses_bits(labels, probabilities).mean())


def interval(values: np.ndarray) -> list[float] | None:
    """Finite 95% percentile interval; undefined subgroups return None."""
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return None
    return np.percentile(finite, [2.5, 97.5]).tolist()


def moving_time_blocks(time_ns: np.ndarray, resamples: int = 1000,
                       block_seconds: int = 300, seed: int = 0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Sample overlapping complete 5-minute blocks from observed test anchors.

    Starts are sampled uniformly from eligible anchor indices. Each replicate
    draws ceil(observed time span / block duration) blocks; variable event
    counts are retained and denominators recomputed, with replacement.
    """
    block_ns = block_seconds * 1_000_000_000
    eligible = np.flatnonzero(time_ns <= time_ns[-1] - block_ns)
    if not len(eligible):
        raise ValueError("Test window is shorter than one bootstrap block")
    n_blocks = int(np.ceil((time_ns[-1] - time_ns[0]) / block_ns))
    rng = np.random.default_rng(seed)
    left = rng.choice(eligible, (resamples, n_blocks), replace=True)
    right = np.searchsorted(time_ns, time_ns[left] + block_ns, side="left")
    return left, right


def block_means(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Recompute per-anchor means from the same resampled blocks for all columns."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    prefix = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
    total = (prefix[right] - prefix[left]).sum(axis=1)
    return total / (right - left).sum(axis=1)[:, None]


def masked_skill(losses: np.ndarray, mask: np.ndarray, prior_entropy: np.ndarray | float,
                 left: np.ndarray, right: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Point and bootstrap skill on a subgroup, preserving paired block draws."""
    if losses.ndim == 1:
        losses = losses[:, None]
    numerator = block_means(np.where(mask[:, None], losses, 0.0), left, right)
    denominator = block_means(mask.astype(float), left, right)
    point = prior_entropy - np.mean(losses[mask], axis=0) if np.any(mask) else np.full(
        losses.shape[1], np.nan)
    boot = prior_entropy - np.divide(numerator, denominator,
                                     out=np.full_like(numerator, np.nan), where=denominator > 0)
    return point, boot
