"""Shared Gate experiments: calibrated losses in bits at delays in events/ns."""

from collections.abc import Callable

import numpy as np

from lob_latency.labels import TAU_GRID
from lob_latency.metrics import classification_metrics, entropy_bits, interval, logit_losses_bits
from lob_latency.models import SEEDS, fit_model

MODEL_NAMES = ["qi_logit", "ofi_logit", "hgb"]


def curve_statistics(point: np.ndarray, boot: np.ndarray) -> dict:
    """Skill and paired differences in bits per prediction."""
    return {"S_bits": point.tolist(),
            "S_ci95_bits": [interval(boot[:, j]) for j in range(len(point))],
            "S_zero_minus_tau_bits": (point[0] - point).tolist(),
            "S_zero_minus_tau_ci95_bits": [interval(boot[:, 0] - boot[:, j])
                                           for j in range(len(point))]}


def latency_statistics(point: np.ndarray, boot: np.ndarray) -> dict:
    """Adjusted skill and paired differences in bits per prediction."""
    return {"A_bits": point.tolist(),
            "A_ci95_bits": [interval(boot[:, j]) for j in range(len(point))],
            "A_fast_minus_L_bits": (point[0] - point).tolist(),
            "A_fast_minus_L_ci95_bits": [interval(boot[:, 0] - boot[:, j])
                                         for j in range(len(point))]}


def fit_grid(xs: dict, ys: dict, evaluation: tuple[str, ...] = ("test",), *,
             hgb_seeds: tuple[int, ...] = SEEDS,
             progress: Callable[[int], None] | None = None) -> dict:
    """Fit once per event delay, score all held-out days without refitting.

    Match Gate 0's seed-mean losses and metrics, including frozen zero-delay
    logits for naive scoring. Training priors and validation choices are shared
    by every evaluation population. Input features are already standardized.
    """
    entropy = np.array([entropy_bits(p) for p in ys["train"].mean(axis=0)])
    metadata = {name: {} for name in MODEL_NAMES}
    validation_skill = {name: [] for name in MODEL_NAMES}
    evaluated = {}
    for split in evaluation:
        shape = (len(xs[split]), len(TAU_GRID))
        evaluated[split] = {
            "losses": {name: np.empty(shape) for name in MODEL_NAMES},
            "naive_losses": {name: np.empty(shape) for name in MODEL_NAMES},
            "scores": {name: {"accuracy": [], "mcc": []} for name in MODEL_NAMES},
            "naive_scores": {name: {"accuracy": [], "mcc": []} for name in MODEL_NAMES},
            "seed_spread": [],
        }
    zero_logits = {split: {} for split in evaluation}
    for j, tau in enumerate(TAU_GRID):
        if progress:
            progress(int(tau))
        for name in MODEL_NAMES:
            seeds = hgb_seeds if name == "hgb" else (0,)
            fitted = [fit_model(name, xs["train"], ys["train"][:, j],
                                xs["validation"], ys["validation"][:, j], seed)
                      for seed in seeds]
            fitted_metadata = [model.metadata() for model in fitted]
            metadata[name][str(tau)] = fitted_metadata
            validation_skill[name].append(float(np.mean([
                m["validation_skill_bits"] for m in fitted_metadata])))
            for split, result in evaluated.items():
                logits = [model.logits(xs[split]) for model in fitted]
                losses = [logit_losses_bits(ys[split][:, j], z) for z in logits]
                scores = [classification_metrics(ys[split][:, j], z) for z in logits]
                result["losses"][name][:, j] = np.mean(losses, axis=0)
                for metric in ("accuracy", "mcc"):
                    result["scores"][name][metric].append(float(np.mean([s[metric] for s in scores])))
                if tau == 0:
                    zero_logits[split][name] = logits
                old_losses = [logit_losses_bits(ys[split][:, j], z) for z in zero_logits[split][name]]
                old_scores = [classification_metrics(ys[split][:, j], z) for z in zero_logits[split][name]]
                result["naive_losses"][name][:, j] = np.mean(old_losses, axis=0)
                for metric in ("accuracy", "mcc"):
                    result["naive_scores"][name][metric].append(float(np.mean([s[metric] for s in old_scores])))
                if name == "hgb":
                    skills = [float(entropy[j] - loss.mean()) for loss in losses]
                    result["seed_spread"].append({
                        "tau_events": int(tau), "S_bits_by_seed": skills,
                        "S_min_max_bits": [min(skills), max(skills)],
                        "S_std_bits": float(np.std(skills)),
                        "accuracy_by_seed": [s["accuracy"] for s in scores],
                        "mcc_by_seed": [s["mcc"] for s in scores],
                        "naive_S_bits_by_seed": [float(entropy[j] - loss.mean()) for loss in old_losses]})
    return {"evaluation": evaluated, "fit_metadata": metadata,
            "validation_skill": validation_skill, "entropy": entropy}
