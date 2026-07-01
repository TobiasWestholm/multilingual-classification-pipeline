"""
Generic abstention threshold optimizer for frozen-embedding classifiers.

Reads OOF (Out-Of-Fold) artifacts (oof_probs.npy, oof_labels.npy) from a checkpoint
directory and writes an optimized thresholds.json.

Supports two task types:

  multilabel   — Multi-label classification tasks.
                 oof_probs shape (N, K) sigmoid outputs.
                 Per-class independent threshold sweep → Dict[str, float].
                 Abstention = all classes below threshold (all-zero prediction).

  multiclass   — Multi-class classification tasks.
                 oof_probs shape (N, K) softmax outputs (abstain class excluded
                 from model output).
                 Single threshold sweep: predicted = argmax if max_prob >= t else abstain.
                 Abstain index = K (one past last trainable class).

Usage examples:

  # Multilabel
  python3 training/optimize_abstention_thresholds.py \
    --checkpoint-dir checkpoints_multilabel_frozen/ \
    --abstention-label abstain \
    --categories class_A class_B class_C \
    --task-type multilabel \
    --metric macro_f1

  # Multiclass
  python3 training/optimize_abstention_thresholds.py \
    --checkpoint-dir checkpoints_multiclass_frozen/ \
    --abstention-label abstain \
    --categories class_A class_B class_C \
    --task-type multiclass \
    --metric macro_f1

Output thresholds.json format:

  Multilabel:
    {
      "task_type": "multilabel",
      "per_class": {"class_A": 0.35, "class_B": 0.50, ...},
      ...
    }

  Multiclass:
    {
      "task_type": "multiclass",
      "threshold": 0.62,
      ...
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
)

TEMPERATURE_FIT_OBJECTIVE = "confidence_f1_spearman"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize abstention thresholds from OOF predictions."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing oof_probs.npy and oof_labels.npy.",
    )
    parser.add_argument(
        "--abstention-label",
        required=True,
        help="Name of the abstention label (e.g. abstain, unknown).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        required=True,
        help="Ordered list of trainable label names (abstention excluded).",
    )
    parser.add_argument(
        "--task-type",
        choices=["multilabel", "multiclass"],
        required=True,
        help="multilabel: per-class sigmoid thresholds. multiclass: single argmax threshold.",
    )
    parser.add_argument(
        "--metric",
        choices=["macro_f1", "macro_pr_auc", "exact_match_accuracy"],
        default="macro_f1",
        help="Metric to maximise when selecting thresholds (default: macro_f1).",
    )
    parser.add_argument(
        "--threshold-grid-min",
        type=float,
        default=None,
        help="Minimum threshold value to sweep (default: 0.05). Uses fixed step size 0.01.",
    )
    parser.add_argument(
        "--threshold-grid-max",
        type=float,
        default=None,
        help="Maximum threshold value to sweep (default: 0.95). Uses fixed step size 0.01.",
    )
    parser.add_argument(
        "--threshold-grid-steps",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility knob. Threshold sweep now enforces 0.01 spacing; "
            "if provided, must match implied step count."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for output thresholds.json (default: <checkpoint-dir>/thresholds.json).",
    )
    parser.add_argument(
        "--use-temperature-scaling",
        dest="use_temperature_scaling",
        action="store_true",
        help="Enable per-class temperature scaling on multilabel OOF probabilities before threshold sweep.",
    )
    parser.add_argument(
        "--no-use-temperature-scaling",
        dest="use_temperature_scaling",
        action="store_false",
        help="Disable temperature scaling (default).",
    )
    parser.set_defaults(use_temperature_scaling=None)
    parser.add_argument(
        "--temperature-grid-min",
        type=float,
        default=None,
        help="Minimum temperature value to sweep when temperature scaling is enabled (default: 0.50).",
    )
    parser.add_argument(
        "--temperature-grid-max",
        type=float,
        default=None,
        help="Maximum temperature value to sweep when temperature scaling is enabled (default: 2.00).",
    )
    parser.add_argument(
        "--temperature-grid-steps",
        type=int,
        default=None,
        help="Number of evenly-spaced temperature candidates (default: 31).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_oof_artifacts(checkpoint_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    probs_path = os.path.join(checkpoint_dir, "oof_probs.npy")
    labels_path = os.path.join(checkpoint_dir, "oof_labels.npy")
    for p in (probs_path, labels_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")
    oof_probs = np.load(probs_path)
    oof_labels = np.load(labels_path)
    logging.info("Loaded oof_probs %s  oof_labels %s", oof_probs.shape, oof_labels.shape)
    return oof_probs, oof_labels


def _tiebreak_closer_to_half(current_best: float, candidate: float) -> bool:
    """Return True if candidate is strictly closer to 0.5 than current_best."""
    return abs(candidate - 0.5) < abs(current_best - 0.5)


def _safe_logit(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probs, eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def apply_multilabel_temperature_scaling(
    probs: np.ndarray,
    categories: List[str],
    per_class_temperatures: Dict[str, float],
) -> np.ndarray:
    logits = _safe_logit(probs.astype(np.float64))
    temp_arr = np.array(
        [float(per_class_temperatures.get(cat, 1.0)) for cat in categories],
        dtype=np.float64,
    )
    temp_arr = np.clip(temp_arr, 1e-6, None)
    scaled_logits = logits / temp_arr[None, :]
    scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return scaled_probs.astype(np.float32)


def _compute_multilabel_confidence(
    probs: np.ndarray,
    threshold_arr: np.ndarray,
) -> np.ndarray:
    above = probs >= threshold_arr[None, :]
    pos_space = np.where(
        above,
        (probs - threshold_arr[None, :]) / np.clip(1.0 - threshold_arr[None, :], 1e-9, None),
        0.0,
    )
    neg_space = np.where(
        ~above,
        (threshold_arr[None, :] - probs) / np.clip(threshold_arr[None, :], 1e-9, None),
        0.0,
    )
    return (pos_space + neg_space).mean(axis=1).astype(np.float32)


def _compute_per_document_f1(
    labels: np.ndarray,
    preds: np.ndarray,
) -> np.ndarray:
    labels_i = labels.astype(np.int32)
    preds_i = preds.astype(np.int32)
    tp = (labels_i & preds_i).sum(axis=1).astype(np.float32)
    n_pred = preds_i.sum(axis=1).astype(np.float32)
    n_true = labels_i.sum(axis=1).astype(np.float32)
    denom = n_pred + n_true

    doc_f1 = np.zeros(labels_i.shape[0], dtype=np.float32)
    nonzero = denom > 0
    doc_f1[nonzero] = (2.0 * tp[nonzero]) / denom[nonzero]
    doc_f1[(n_pred == 0) & (n_true == 0)] = 1.0
    return doc_f1


def _compute_confidence_f1_spearman_score(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Dict[str, float],
    categories: List[str],
) -> float:
    threshold_arr = np.array([float(thresholds[cat]) for cat in categories], dtype=np.float32)
    confidence = _compute_multilabel_confidence(probs, threshold_arr)
    preds = (probs >= threshold_arr[None, :]).astype(np.int32)
    doc_f1 = _compute_per_document_f1(labels, preds)
    corr, _ = spearmanr(confidence, doc_f1)
    if corr is None or not np.isfinite(corr):
        return -1.0
    return float(corr)


def _select_per_class_thresholds_only(
    oof_probs: np.ndarray,
    oof_labels: np.ndarray,
    categories: List[str],
    metric: str,
    threshold_grid: np.ndarray,
) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    effective_metric = "macro_f1" if metric == "macro_pr_auc" else metric
    for i, cat in enumerate(categories):
        y_true = oof_labels[:, i].astype(int)
        y_score = oof_probs[:, i]
        thresholds[cat] = _sweep_single_class(y_true, y_score, threshold_grid, effective_metric)
    return thresholds


def fit_multilabel_per_class_temperatures(
    oof_probs: np.ndarray,
    oof_labels: np.ndarray,
    categories: List[str],
    threshold_grid: np.ndarray,
    temperature_grid: np.ndarray,
    metric: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    per_class_temperatures: Dict[str, float] = {cat: 1.0 for cat in categories}
    fit_summary: List[Dict[str, Any]] = []

    for i, cat in enumerate(categories):
        y_true = oof_labels[:, i].astype(np.int64)

        if int(y_true.sum()) == 0 or int(y_true.sum()) == int(len(y_true)):
            per_class_temperatures[cat] = 1.0
            fit_summary.append(
                {
                    "label": cat,
                    "selected_temperature": 1.0,
                    "selection_score": None,
                    "reason": "degenerate_label_distribution",
                }
            )
            continue

        best_t = 1.0
        best_score = -float("inf")

        for temp in temperature_grid:
            t_float = float(temp)
            candidate_temps = dict(per_class_temperatures)
            candidate_temps[cat] = t_float
            scaled_probs = apply_multilabel_temperature_scaling(
                oof_probs,
                categories,
                candidate_temps,
            )
            candidate_thresholds = _select_per_class_thresholds_only(
                scaled_probs,
                oof_labels,
                categories,
                metric,
                threshold_grid,
            )
            score = _compute_confidence_f1_spearman_score(
                scaled_probs,
                oof_labels,
                candidate_thresholds,
                categories,
            )

            if score > best_score or (
                abs(score - best_score) <= 1e-12 and abs(t_float - 1.0) < abs(best_t - 1.0)
            ):
                best_t = t_float
                best_score = score

        per_class_temperatures[cat] = float(best_t)
        fit_summary.append(
            {
                "label": cat,
                "selected_temperature": float(best_t),
                "selection_score": float(best_score),
                "reason": "optimized_confidence_f1_spearman",
            }
        )

    return per_class_temperatures, fit_summary


# ---------------------------------------------------------------------------
# Multilabel: per-class independent sweep
# ---------------------------------------------------------------------------

def _sweep_single_class(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_grid: np.ndarray,
    metric: str,
) -> float:
    """Sweep thresholds for a single binary class and return the best threshold."""
    if int(y_true.sum()) == 0:
        # No support for this class in OOF; keep neutral default threshold.
        return 0.5

    best_t = 0.5
    best_score = -float("inf")

    # For macro_pr_auc we use the raw per-class AP (score doesn't depend on threshold).
    # Still sweep to return a threshold, but optimise F1 per class in that case.
    effective_metric = "macro_f1" if metric == "macro_pr_auc" else metric

    for t in threshold_grid:
        preds = (y_score >= float(t)).astype(int)

        if effective_metric == "macro_f1":
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, preds, average="binary", zero_division=0
            )
            score = float(f1)
        else:  # exact_match_accuracy treated as accuracy for binary class
            score = float(accuracy_score(y_true, preds))

        if score > best_score:
            best_score = score
            best_t = float(t)
        elif abs(score - best_score) <= 1e-12:
            if _tiebreak_closer_to_half(best_t, float(t)):
                best_t = float(t)

    return best_t


def optimize_multilabel(
    oof_probs: np.ndarray,
    oof_labels: np.ndarray,
    categories: List[str],
    metric: str,
    threshold_grid: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Per-class independent threshold sweep.

    Returns (per_class_thresholds, calibrated_metrics).
    """
    K = len(categories)
    if oof_probs.ndim != 2 or oof_probs.shape[1] != K:
        raise ValueError(
            f"Expected oof_probs shape (N, {K}), got {oof_probs.shape}. "
            "Verify --categories matches the training label order."
        )
    if oof_labels.ndim != 2 or oof_labels.shape[1] != K:
        raise ValueError(
            f"Expected oof_labels shape (N, {K}), got {oof_labels.shape}."
        )

    per_class_thresholds: Dict[str, float] = {}
    for i, cat in enumerate(categories):
        y_true = oof_labels[:, i].astype(int)
        y_score = oof_probs[:, i]
        best_t = _sweep_single_class(y_true, y_score, threshold_grid, metric)
        per_class_thresholds[cat] = best_t
        logging.info("  [%s] best_threshold=%.3f", cat, best_t)

    # Compute overall calibrated metrics with selected per-class thresholds.
    threshold_arr = np.array(
        [per_class_thresholds[cat] for cat in categories], dtype=np.float32
    )
    preds = (oof_probs >= threshold_arr).astype(np.float32)

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        oof_labels, preds, average=None, zero_division=0
    )
    supported_mask = per_support > 0
    if np.any(supported_mask):
        macro_p = float(np.mean(per_p[supported_mask]))
        macro_r = float(np.mean(per_r[supported_mask]))
        macro_f1 = float(np.mean(per_f1[supported_mask]))
    else:
        macro_p = 0.0
        macro_r = 0.0
        macro_f1 = 0.0
    exact_match = float(accuracy_score(oof_labels, preds))
    abstention_rate = float((preds.sum(axis=1) == 0).mean())

    per_class_ap: List[float] = []
    per_class_pr_auc: Dict[str, Optional[float]] = {}
    for i, cat in enumerate(categories):
        if oof_labels[:, i].sum() > 0:
            ap = float(average_precision_score(oof_labels[:, i], oof_probs[:, i]))
            per_class_pr_auc[cat] = ap
            per_class_ap.append(ap)
        else:
            per_class_pr_auc[cat] = None

    macro_pr_auc = float(np.mean(per_class_ap)) if per_class_ap else 0.0

    calibrated_metrics: Dict[str, Any] = {
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "exact_match_accuracy": exact_match,
        "abstention_rate": abstention_rate,
        "macro_pr_auc": macro_pr_auc,
        "per_class_pr_auc": per_class_pr_auc,
    }

    return per_class_thresholds, calibrated_metrics


# ---------------------------------------------------------------------------
# Multiclass: single threshold sweep
# ---------------------------------------------------------------------------

def optimize_multiclass(
    oof_probs: np.ndarray,
    oof_labels: np.ndarray,
    categories: List[str],
    abstention_label: str,
    metric: str,
    threshold_grid: np.ndarray,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Single threshold sweep for multiclass-with-abstention.

    oof_probs: (N, K)   softmax outputs over trainable classes (abstain excluded from model)
    oof_labels: (N,)    integer class indices, where K = abstain index

    Returns (best_threshold, best_metrics, sweep_rows).
    """
    K = len(categories)
    abstain_idx = K  # abstain is one past last trainable index

    if oof_probs.ndim != 2 or oof_probs.shape[1] != K:
        raise ValueError(
            f"Expected oof_probs shape (N, {K}), got {oof_probs.shape}."
        )
    if oof_labels.ndim != 1 or oof_labels.shape[0] != oof_probs.shape[0]:
        raise ValueError(
            f"Expected oof_labels shape ({oof_probs.shape[0]},), got {oof_labels.shape}."
        )

    all_label_indices = list(range(K + 1))  # including abstain

    # Keep parity with multilabel behavior: PR-AUC is threshold-free, so use
    # macro_f1 for selecting a threshold when macro_pr_auc is requested.
    effective_metric = "macro_f1" if metric == "macro_pr_auc" else metric
    if metric == "macro_pr_auc":
        logging.info(
            "Multiclass threshold selection requested macro_pr_auc; "
            "falling back to macro_f1 for threshold sweep."
        )

    best_t = 0.5
    best_score = -float("inf")
    best_metrics: Dict[str, Any] = {}
    sweep: List[Dict[str, Any]] = []

    for t in threshold_grid:
        preds = np.argmax(oof_probs, axis=1).astype(np.int64)
        max_probs = oof_probs.max(axis=1)
        preds[max_probs < float(t)] = abstain_idx

        per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
            oof_labels,
            preds,
            labels=all_label_indices,
            average=None,
            zero_division=0,
        )
        supported_mask = per_support > 0
        if np.any(supported_mask):
            macro_p = float(np.mean(per_p[supported_mask]))
            macro_r = float(np.mean(per_r[supported_mask]))
            macro_f1 = float(np.mean(per_f1[supported_mask]))
        else:
            macro_p = 0.0
            macro_r = 0.0
            macro_f1 = 0.0
        exact_match = float(accuracy_score(oof_labels, preds))
        abstention_rate = float((preds == abstain_idx).mean())

        if effective_metric == "macro_f1":
            score = float(macro_f1)
        elif effective_metric == "exact_match_accuracy":
            score = float(exact_match)
        else:
            raise ValueError(f"Unsupported metric for multiclass threshold sweep: {metric}")

        row: Dict[str, Any] = {
            "threshold": float(t),
            "score": score,
            "macro_f1": float(macro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "exact_match_accuracy": exact_match,
            "abstention_rate": abstention_rate,
        }
        sweep.append(row)

        if score > best_score:
            best_score = score
            best_t = float(t)
            best_metrics = row
        elif abs(score - best_score) <= 1e-12:
            if _tiebreak_closer_to_half(best_t, float(t)):
                best_t = float(t)
                best_metrics = row

    logging.info("Multiclass best_threshold=%.3f  score(%.6f)=%.4f", best_t, best_t, best_score)
    return best_t, best_metrics, sweep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    args = parse_args()

    categories = list(args.categories)
    checkpoint_dir = str(args.checkpoint_dir)
    output_path = args.output_path or os.path.join(checkpoint_dir, "thresholds.json")

    logging.info("=" * 70)
    logging.info("THRESHOLD OPTIMIZER")
    logging.info("task_type=%s  metric=%s  abstention_label=%s", args.task_type, args.metric, args.abstention_label)
    logging.info("categories(%d): %s", len(categories), categories)
    logging.info("=" * 70)

    oof_probs, oof_labels = load_oof_artifacts(checkpoint_dir)

    grid_min = float(args.threshold_grid_min) if args.threshold_grid_min is not None else 0.05
    grid_max = float(args.threshold_grid_max) if args.threshold_grid_max is not None else 0.95
    if grid_max < grid_min:
        raise ValueError(f"threshold-grid-max ({grid_max}) must be >= threshold-grid-min ({grid_min})")
    threshold_step = 0.01
    ratio = (grid_max - grid_min) / threshold_step
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"Threshold grid must align to 0.01 increments. Received min={grid_min}, max={grid_max}."
        )
    threshold_grid = np.arange(grid_min, grid_max + 1e-12, threshold_step, dtype=np.float64)
    threshold_grid = np.round(threshold_grid, 6)
    implied_steps = int(len(threshold_grid))
    if args.threshold_grid_steps is not None and int(args.threshold_grid_steps) != implied_steps:
        raise ValueError(
            f"--threshold-grid-steps={args.threshold_grid_steps} does not match implied 0.01-step grid "
            f"count ({implied_steps}) for min={grid_min}, max={grid_max}."
        )

    grid_steps = implied_steps
    logging.info(
        "Threshold grid: %.2f–%.2f in %d steps (step=%.2f)",
        threshold_grid[0],
        threshold_grid[-1],
        len(threshold_grid),
        threshold_step,
    )

    grid_meta = {
        "min": float(grid_min),
        "max": float(grid_max),
        "steps": int(grid_steps),
        "values": [float(t) for t in threshold_grid],
    }

    use_temperature_scaling = bool(args.use_temperature_scaling) if args.use_temperature_scaling is not None else False

    if args.task_type == "multilabel":
        temp_grid_min = float(args.temperature_grid_min) if args.temperature_grid_min is not None else 0.50
        temp_grid_max = float(args.temperature_grid_max) if args.temperature_grid_max is not None else 2.00
        temp_grid_steps = int(args.temperature_grid_steps) if args.temperature_grid_steps is not None else 31
        temperature_grid = np.linspace(temp_grid_min, temp_grid_max, temp_grid_steps)
        probs_for_thresholding = oof_probs
        temperature_scaling: Dict[str, Any] = {
            "enabled": use_temperature_scaling,
            "mode": "per_class",
            "fit_objective": TEMPERATURE_FIT_OBJECTIVE,
            "temperature_grid": [float(t) for t in temperature_grid],
            "per_class": {},
            "fit_summary": [],
        }
        if use_temperature_scaling:
            per_class_temperatures, fit_summary = fit_multilabel_per_class_temperatures(
                oof_probs=oof_probs,
                oof_labels=oof_labels,
                categories=categories,
                threshold_grid=threshold_grid,
                temperature_grid=temperature_grid,
                metric=args.metric,
            )
            probs_for_thresholding = apply_multilabel_temperature_scaling(
                oof_probs,
                categories,
                per_class_temperatures,
            )
            temperature_scaling["per_class"] = per_class_temperatures
            temperature_scaling["fit_summary"] = fit_summary
            logging.info("Applied per-class temperature scaling before threshold sweep.")
        else:
            logging.info("Temperature scaling disabled.")

        per_class_thresholds, calibrated_metrics = optimize_multilabel(
            probs_for_thresholding, oof_labels, categories, args.metric, threshold_grid
        )
        payload: Dict[str, Any] = {
            "task_type": "multilabel",
            "abstention_label": args.abstention_label,
            "categories": categories,
            "selection_metric": args.metric,
            "per_class": per_class_thresholds,
            "threshold_grid": grid_meta,
            "temperature_scaling": temperature_scaling,
            "optimized_metrics": calibrated_metrics,
        }
        logging.info("Per-class thresholds: %s", per_class_thresholds)
        logging.info(
            "Calibrated macro_f1=%.4f  abstention_rate=%.4f  macro_pr_auc=%.4f",
            calibrated_metrics["macro_f1"],
            calibrated_metrics["abstention_rate"],
            calibrated_metrics["macro_pr_auc"],
        )

    else:  # multiclass
        best_t, best_metrics, sweep = optimize_multiclass(
            oof_probs, oof_labels, categories, args.abstention_label,
            args.metric, threshold_grid
        )
        payload = {
            "task_type": "multiclass",
            "abstention_label": args.abstention_label,
            "categories": categories,
            "selection_metric": args.metric,
            "threshold": float(best_t),
            # Alias for backwards-compat with code that reads "selected_threshold"
            "selected_threshold": float(best_t),
            "threshold_grid": grid_meta,
            "optimized_metrics": best_metrics,
            "sweep": sweep,
        }
        logging.info(
            "Single threshold=%.3f  macro_f1=%.4f  exact_match=%.4f  abstention=%.4f",
            best_t,
            best_metrics.get("macro_f1", 0.0),
            best_metrics.get("exact_match_accuracy", 0.0),
            best_metrics.get("abstention_rate", 0.0),
        )

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logging.info("Thresholds written to: %s", output_path)
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
