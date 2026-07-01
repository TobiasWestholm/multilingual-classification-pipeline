"""
Unified HPO orchestrator for multilingual classifiers.

Supported workflows:
- multilabel_frozen: Frozen multilabel classifier HPO flow
- multilabel_lora: LoRA multilabel classifier HPO flow
- multiclass_frozen: Frozen multiclass classifier HPO flow
- multiclass_lora: LoRA multiclass classifier HPO flow

Run with --workflow to select the model target. Workflow-specific defaults
(trainer script, dataset, abstention bounds) are set automatically but can
be overridden via CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    hamming_loss,
    precision_recall_fscore_support,
)


ACTIVE_SELECTION_METRIC = "macro_f1"
SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_script_path(script_path: str) -> Path:
    candidate = Path(str(script_path))
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (SCRIPT_DIR / candidate).resolve()


def _resolve_existing_script_path(script_path: str, role: str) -> str:
    resolved = _resolve_script_path(script_path)
    if not resolved.exists():
        raise FileNotFoundError(f"{role} not found: {resolved}")
    return str(resolved)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepared_signature_from_items(prepared_items: List[Dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for item in prepared_items:
        hasher.update(str(item.get("file_name", "")).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_hash_text(str(item.get("text", ""))).encode("utf-8"))
        hasher.update(b"\0")
        labels = item.get("labels", []) or []
        hasher.update("|".join(sorted(str(x) for x in labels)).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()



@dataclass
class RunSpec:
    run_id: str
    phase: str
    stage: str
    params: Dict[str, Any]
    notes: str = ""


class StageASHAState:
    """
    Lightweight ASHA-style state for sequential runs in one stage.
    Uses rung epochs (default: 2 and 3) and compares run score vs prior-stage history.
    """

    def __init__(
        self,
        stage_name: str,
        rung_epochs: Sequence[int] = (2, 3),
        min_history: int = 2,
        median_margin: float = 0.03,
        best_margin: float = 0.05,
        min_improvement: float = 0.002,
    ) -> None:
        self.stage_name = stage_name
        self.rung_epochs = tuple(int(e) for e in rung_epochs)
        self.min_history = int(min_history)
        self.median_margin = float(median_margin)
        self.best_margin = float(best_margin)
        self.min_improvement = float(min_improvement)
        self.rung_scores: Dict[int, List[float]] = {int(e): [] for e in self.rung_epochs}

    @staticmethod
    def combined_score(macro_f1: float, macro_pr_auc: float) -> float:
        return 0.5 * (float(macro_f1) + float(macro_pr_auc))

    def should_prune(self, epoch: int, score: float, previous_epoch_score: Optional[float]) -> Tuple[bool, str]:
        epoch = int(epoch)
        if epoch not in self.rung_scores:
            return False, ""

        hist = self.rung_scores.get(epoch, [])
        if len(hist) < self.min_history:
            return False, ""

        median_hist = float(np.median(hist))
        best_hist = float(max(hist))
        behind_median = score < (median_hist - self.median_margin)
        behind_best = score < (best_hist - self.best_margin)
        non_improving = previous_epoch_score is not None and score < (previous_epoch_score + self.min_improvement)

        if behind_median and behind_best and non_improving:
            reason = (
                f"ASHA prune at epoch={epoch}: score={score:.4f}, "
                f"median_hist={median_hist:.4f}, best_hist={best_hist:.4f}, "
                f"prev_epoch_score={previous_epoch_score:.4f}"
            )
            return True, reason

        return False, ""

    def update_from_cv_results(self, cv_results: Dict[str, Any]) -> None:
        fold_results = cv_results.get("fold_results") or []
        if not isinstance(fold_results, list):
            return

        for epoch in self.rung_epochs:
            vals: List[float] = []
            for fold in fold_results:
                hist = fold.get("epoch_history") or []
                for rec in hist:
                    if int(rec.get("epoch", -1)) != int(epoch):
                        continue
                    f1 = float(rec.get("val_macro_f1", 0.0))
                    pr = float(rec.get("val_macro_pr_auc", 0.0))
                    vals.append(self.combined_score(f1, pr))
            if vals:
                self.rung_scores[int(epoch)].append(float(np.mean(vals)))


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_campaign_signature(args: argparse.Namespace, trainer_path: Path) -> str:
    optimizer_path = Path(__file__).resolve()
    payload = {
        "optimizer_sha256": sha256_file(optimizer_path),
        "trainer_sha256": sha256_file(trainer_path.resolve()),
        "dataset_name": args.dataset_name,
        "checkpoint_selection_metric": args.checkpoint_selection_metric,
        "selection_metric": ACTIVE_SELECTION_METRIC,
        "abstention_min": args.abstention_min,
        "abstention_max": args.abstention_max,
        "use_temperature_scaling": bool(getattr(args, "use_temperature_scaling", False))
        if _is_multilabel_frozen_workflow(str(args.workflow))
        else False,
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) <= 1e-12
        except (TypeError, ValueError):
            return False
    return expected == actual


def expected_resume_fields(
    args: argparse.Namespace,
    spec: RunSpec,
) -> Dict[str, Any]:
    expected = {
        "dataset_name": args.dataset_name,
        "random_state": args.seed,
        "checkpoint_selection_metric": args.checkpoint_selection_metric,
        "campaign_signature": args.campaign_signature,
    }
    for key, value in spec.params.items():
        expected[key] = value
    if _is_multilabel_frozen_workflow(str(args.workflow)):
        expected["use_temperature_scaling"] = bool(getattr(args, "use_temperature_scaling", False))
    snapshot = getattr(args, "_expected_data_snapshot", None)
    if isinstance(snapshot, dict):
        prepared_signature = str(snapshot.get("prepared_signature", "") or "")
        if prepared_signature:
            expected["prepared_signature"] = prepared_signature
    return expected


def resume_config_mismatches(
    run_cfg: Dict[str, Any],
    expected: Dict[str, Any],
) -> List[str]:
    mismatches: List[str] = []
    for key, exp_val in expected.items():
        if key not in run_cfg:
            mismatches.append(f"{key}: missing in run_config")
            continue
        act_val = run_cfg[key]
        if not values_match(exp_val, act_val):
            mismatches.append(f"{key}: expected={exp_val} actual={act_val}")
    return mismatches


def load_training_config() -> Dict[str, Any]:
    config_path = Path("training_config.json")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if not line.strip().startswith(("//", "#"))]
            return json.loads("".join(lines))
        except Exception as exc:
            print(f"[CONFIG_ERROR] Failed to read training_config.json: {exc}")
    return {}


def parse_args() -> argparse.Namespace:
    cfg = load_training_config()
    parser = argparse.ArgumentParser(description="Hyperparameter optimization orchestrator")
    parser.add_argument("--trainer-script", default=None)
    parser.add_argument(
        "--workflow",
        choices=[
            "multiclass_frozen",
            "multiclass_lora",
            "multilabel_frozen",
            "multilabel_lora",
        ],
        default="multiclass_frozen" if cfg.get("task_type") == "multiclass" else "multilabel_lora",
        help="Optimization workflow.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dataset-name", default=cfg.get("dataset_name"))
    parser.add_argument(
        "--task",
        default=cfg.get("task_name"),
        help=(
            "Optional task override passed to trainer (e.g. policy_classification). "
            "Also used to enforce task-specific output-root naming."
        ),
    )
    parser.add_argument("--prompt-name", default=cfg.get("prompt_name"))
    parser.add_argument("--abstain-label", default=cfg.get("abstention_label"))
    parser.add_argument("--embeddings-path", default="data_preparation/train_test_document_embeddings.npy")
    parser.add_argument("--metadata-path", default="training/document_train_test_split.csv")
    parser.add_argument("--raw-documents-dir", default=cfg.get("raw_documents_dir"))
    parser.add_argument("--model-name", default=cfg.get("backbone_model_name"))
    parser.add_argument("--dataset-cache-path", default=None)
    parser.add_argument("--tokenized-cache-path", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-selection-metric",
        default="macro_pr_auc",
        choices=[
            "macro_pr_auc",
            "macro_f1",
            "macro_f1_0_5",
            "exact_match_accuracy",
            "val_loss",
        ],
        help=(
            "Metric used by trainer to select best epoch checkpoint per fold. "
            "For multilabel workflows, macro_f1 is normalized to legacy macro_f1_0_5."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "exact_match_accuracy", "macro_pr_auc"],
        default="macro_f1",
        help="Metric used to rank/promote candidates during HPO.",
    )
    parser.add_argument("--abstention-min", type=float, default=None)
    parser.add_argument("--abstention-max", type=float, default=None)
    scaling_group = parser.add_mutually_exclusive_group()
    scaling_group.add_argument(
        "--use-temperature-scaling",
        dest="use_temperature_scaling",
        action="store_true",
        default=False,
        help="Enable per-label temperature scaling for frozen multilabel trainer runs.",
    )
    scaling_group.add_argument(
        "--no-use-temperature-scaling",
        dest="use_temperature_scaling",
        action="store_false",
        help="Disable temperature scaling for frozen multilabel trainer runs (default).",
    )
    parser.add_argument(
        "--stage-b-clear-winner-prauc-gap",
        type=float,
        default=0.03,
        help=(
            "If Stage A top PR-AUC exceeds every other Stage A candidate by this gap, "
            "Stage B runs only from that single base config."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    post_group = parser.add_mutually_exclusive_group()
    post_group.add_argument(
        "--post-oof-to-langfuse",
        dest="post_oof_to_langfuse",
        action="store_true",
        default=False,
        help="Enable OOF posting to Langfuse for trainer runs.",
    )
    post_group.add_argument(
        "--no-post-oof-to-langfuse",
        dest="post_oof_to_langfuse",
        action="store_false",
        help="Disable OOF posting to Langfuse for trainer runs (default).",
    )
    winner_post_group = parser.add_mutually_exclusive_group()
    winner_post_group.add_argument(
        "--post-winner-to-langfuse",
        dest="post_winner_to_langfuse",
        action="store_true",
        default=True,
        help="After HPO, post only the final winner OOF results to Langfuse from saved artifacts (default).",
    )
    winner_post_group.add_argument(
        "--no-post-winner-to-langfuse",
        dest="post_winner_to_langfuse",
        action="store_false",
        help="Disable winner-only post to Langfuse after HPO.",
    )
    parser.add_argument(
        "--fail-on-langfuse-post-error",
        action="store_true",
        help="Pass --fail-on-langfuse-post-error to trainer runs.",
    )
    parser.add_argument(
        "--stop-after-phase",
        choices=[
            "phase0",
            "phase1",
            "phase2",
            "phase3",
            "phase4",
            "lora_a",
            "lora_b",
            "lora_c",
            "lora_d",
        ],
        default=None,
        help="Stop once this phase is completed. For LoRA workflows use: lora_a/lora_b/lora_c/lora_d.",
    )
    # Post-HPO sequence flags
    human_eval_group = parser.add_mutually_exclusive_group()
    human_eval_group.add_argument(
        "--run-human-eval",
        dest="run_human_eval",
        action="store_true",
        default=True,
        help="Run test_model.py on the winner checkpoint after HPO (default: enabled).",
    )
    human_eval_group.add_argument(
        "--no-run-human-eval",
        dest="run_human_eval",
        action="store_false",
        help="Skip human-eval post-HPO.",
    )
    parser.add_argument(
        "--human-eval-dataset",
        default=cfg.get("testset_name"),
        help="Langfuse test dataset name for post-HPO human evaluation.",
    )
    parser.add_argument(
        "--human-eval-script",
        default="test_model.py",
        help="Path to the human evaluation script.",
    )
    parser.add_argument(
        "--threshold-optimizer-script",
        default="optimize_abstention_thresholds.py",
        help="Path to standalone threshold optimizer script used in post-HPO winner flow.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cleanup_fresh_start(
    output_root: Path,
    workflow: str,
    dataset_cache_path: str,
    tokenized_cache_path: str,
    cleanup_dataset_cache: bool,
    cleanup_tokenized_cache: bool,
) -> None:
    removed: List[str] = []
    phase_dirs = sorted({p for spec in WORKFLOW_REGISTRY.values() for p in spec["valid_stop_phases"]})
    for phase in phase_dirs:
        phase_path = output_root / phase
        if phase_path.exists():
            shutil.rmtree(phase_path)
            removed.append(str(phase_path))

    for filename in ["hpo_summary.json"]:
        file_path = output_root / filename
        if file_path.exists():
            file_path.unlink()
            removed.append(str(file_path))

    if cleanup_dataset_cache:
        cache_path = Path(dataset_cache_path)
        if cache_path.exists():
            cache_path.unlink()
            removed.append(str(cache_path))

    if cleanup_tokenized_cache:
        token_cache_path = Path(tokenized_cache_path)
        if token_cache_path.exists():
            token_cache_path.unlink()
            removed.append(str(token_cache_path))

    print(
        "[FRESH_START] workflow=%s | resume=False | removed=%d artifacts"
        % (workflow, len(removed))
    )
    if removed:
        print("[FRESH_START] Removed paths:")
        for p in removed:
            print(f"  - {p}")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_command(
    python_exec: str,
    trainer_script: str,
    checkpoint_dir: Path,
    dataset_name: str,
    dataset_cache_path: str,
    campaign_signature: str,
    seed: int,
    checkpoint_selection_metric: str,
    post_oof_to_langfuse: bool,
    fail_on_langfuse_post_error: bool,
    params: Dict[str, Any],
) -> List[str]:
    cmd = [
        python_exec,
        trainer_script,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--dataset-name",
        dataset_name,
        "--dataset-cache-path",
        dataset_cache_path,
        "--campaign-signature",
        campaign_signature,
        "--seed",
        str(seed),
        "--checkpoint-selection-metric",
        checkpoint_selection_metric,
    ]

    if post_oof_to_langfuse:
        cmd.append("--post-oof-to-langfuse")
    else:
        cmd.append("--no-post-oof-to-langfuse")
    if fail_on_langfuse_post_error:
        cmd.append("--fail-on-langfuse-post-error")

    arg_map = {
        "n_splits": "--n-splits",
        "max_folds": "--max-folds",
        "num_epochs": "--num-epochs",
        "batch_size": "--batch-size",
        "learning_rate": "--learning-rate",
        "learning_rate_lora": "--learning-rate-lora",
        "learning_rate_head": "--learning-rate-head",
        "weight_decay": "--weight-decay",
        "dropout": "--dropout",
        "intermediate_size": "--intermediate-size",
        "max_grad_norm": "--max-grad-norm",
        "early_stopping_patience": "--early-stopping-patience",
        "early_stopping_min_delta": "--early-stopping-min-delta",
        "gradient_accumulation_steps": "--gradient-accumulation-steps",
        "max_seq_length": "--max-seq-length",
        "chunk_overlap": "--chunk-overlap",
        "max_chunks_per_doc": "--max-chunks-per-doc",
        "chunk_micro_batch_size": "--chunk-micro-batch-size",
        "tokenized_cache_path": "--tokenized-cache-path",
        "threshold_selection_metric": "--threshold-selection-metric",
        "lora_r": "--lora-r",
        "lora_alpha": "--lora-alpha",
        "lora_dropout": "--lora-dropout",
        "lora_target_set": "--lora-target-set",
        "pos_weight_clip_max": "--pos-weight-clip-max",
        "pos_weight_mode": "--pos-weight-mode",
        "sampler_strength": "--sampler-strength",
        "chunk_pooling": "--chunk-pooling",
        "focal_gamma": "--focal-gamma",
        "max_items": "--max-items",
        "embeddings_path": "--embeddings-path",
        "metadata_path": "--metadata-path",
        "task": "--task",
        "task_type": "--task-type",
        "prompt_name": "--prompt-name",
        "abstain_label": "--abstain-label",
        "model_name": "--model-name",
    }
    bool_arg_map = {
        "use_temperature_scaling": ("--use-temperature-scaling", "--no-use-temperature-scaling"),
    }

    for key, value in params.items():
        if key in bool_arg_map and value is not None:
            true_flag, false_flag = bool_arg_map[key]
            cmd.append(true_flag if bool(value) else false_flag)
            continue
        if key in arg_map and value is not None:
            cmd.extend([arg_map[key], str(value)])

    return cmd


def _extract_multilabel_metrics(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics for all multilabel classifiers (LoRA and all frozen multilabel)."""
    calibrated = cv_results["oof_metrics_calibrated"]
    per_class = calibrated.get("per_class_metrics", {})
    system_summary = cv_results.get("system_metrics_summary", {})
    process_summary = system_summary.get("process", {})
    device_summary = system_summary.get("device", {})

    minority_classes: List[str] = cv_results.get("minority_classes") or []

    fold_results = cv_results.get("fold_results") or []
    fold_praucs: List[float] = []
    if isinstance(fold_results, list):
        for fold in fold_results:
            if isinstance(fold, dict) and fold.get("best_val_macro_pr_auc") is not None:
                fold_praucs.append(float(fold.get("best_val_macro_pr_auc", 0.0)))
    cv_macro_pr_auc_mean = float(np.mean(fold_praucs)) if fold_praucs else float(calibrated.get("macro_pr_auc", 0.0))

    per_class_pr_auc = calibrated.get("per_class_pr_auc", {})
    minority_f1_values: List[float] = []
    minority_precision_values: List[float] = []
    minority_recall_values: List[float] = []
    minority_pr_auc_values: List[float] = []
    for cls in minority_classes:
        if cls in per_class:
            minority_f1_values.append(float(per_class[cls].get("f1", 0.0)))
            minority_precision_values.append(float(per_class[cls].get("precision", 0.0)))
            minority_recall_values.append(float(per_class[cls].get("recall", 0.0)))
        if cls in per_class_pr_auc and per_class_pr_auc[cls] is not None:
            minority_pr_auc_values.append(float(per_class_pr_auc[cls]))

    return {
        "macro_f1": float(calibrated.get("macro_f1", 0.0)),
        "macro_precision": float(calibrated.get("macro_precision", 0.0)),
        "macro_recall": float(calibrated.get("macro_recall", 0.0)),
        "cv_macro_pr_auc_mean": cv_macro_pr_auc_mean,
        "macro_pr_auc": float(calibrated.get("macro_pr_auc", 0.0)),
        "minority_macro_pr_auc": float(np.mean(minority_pr_auc_values)) if minority_pr_auc_values else 0.0,
        "exact_match_accuracy": float(calibrated.get("exact_match_accuracy", 0.0)),
        "abstention_rate": float(calibrated.get("abstention_rate", 0.0)),
        "minority_mean_f1": float(np.mean(minority_f1_values)) if minority_f1_values else 0.0,
        "minority_mean_precision": float(np.mean(minority_precision_values)) if minority_precision_values else 0.0,
        "minority_mean_recall": float(np.mean(minority_recall_values)) if minority_recall_values else 0.0,
        "peak_process_rss_mb": float(process_summary.get("peak_process_rss_mb", 0.0) or 0.0),
        "peak_system_cpu_percent": float(process_summary.get("peak_system_cpu_percent", 0.0) or 0.0),
        "peak_system_ram_percent": float(process_summary.get("peak_system_ram_percent", 0.0) or 0.0),
        "peak_device_memory_mb": float(
            device_summary.get("peak_cuda_memory_reserved_mb", 0.0)
            or device_summary.get("peak_mps_driver_allocated_mb", 0.0)
            or 0.0
        ),
        "system_metrics_file": str(system_summary.get("system_metrics_file", "")),
    }


def _extract_multiclass_frozen_metrics(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    calibrated = cv_results.get("oof_metrics_calibrated", {})
    per_class = calibrated.get("per_class_metrics", {})
    system_summary = cv_results.get("system_metrics_summary", {})
    process_summary = system_summary.get("process", {})
    device_summary = system_summary.get("device", {})

    minority_classes: List[str] = cv_results.get("minority_classes") or []
    fold_results = cv_results.get("fold_results") or []
    fold_praucs: List[float] = []
    if isinstance(fold_results, list):
        for fold in fold_results:
            if isinstance(fold, dict) and fold.get("best_val_macro_pr_auc") is not None:
                fold_praucs.append(float(fold.get("best_val_macro_pr_auc", 0.0)))
    cv_macro_pr_auc_mean = float(np.mean(fold_praucs)) if fold_praucs else float(calibrated.get("macro_pr_auc", 0.0))

    per_class_pr_auc = calibrated.get("per_class_pr_auc", {})
    minority_f1_values: List[float] = []
    minority_precision_values: List[float] = []
    minority_recall_values: List[float] = []
    minority_pr_auc_values: List[float] = []
    for cls in minority_classes:
        if cls in per_class:
            minority_f1_values.append(float(per_class[cls].get("f1", 0.0)))
            minority_precision_values.append(float(per_class[cls].get("precision", 0.0)))
            minority_recall_values.append(float(per_class[cls].get("recall", 0.0)))
        if cls in per_class_pr_auc and per_class_pr_auc[cls] is not None:
            minority_pr_auc_values.append(float(per_class_pr_auc[cls]))

    return {
        "macro_f1": float(calibrated.get("macro_f1", 0.0)),
        "macro_precision": float(calibrated.get("macro_precision", 0.0)),
        "macro_recall": float(calibrated.get("macro_recall", 0.0)),
        "cv_macro_pr_auc_mean": cv_macro_pr_auc_mean,
        "macro_pr_auc": float(calibrated.get("macro_pr_auc", 0.0)),
        "minority_macro_pr_auc": float(np.mean(minority_pr_auc_values)) if minority_pr_auc_values else 0.0,
        "exact_match_accuracy": float(calibrated.get("exact_match_accuracy", 0.0)),
        "abstention_rate": float(calibrated.get("abstention_rate", 0.0)),
        "minority_mean_f1": float(np.mean(minority_f1_values)) if minority_f1_values else 0.0,
        "minority_mean_precision": float(np.mean(minority_precision_values)) if minority_precision_values else 0.0,
        "minority_mean_recall": float(np.mean(minority_recall_values)) if minority_recall_values else 0.0,
        "peak_process_rss_mb": float(process_summary.get("peak_process_rss_mb", 0.0) or 0.0),
        "peak_system_cpu_percent": float(process_summary.get("peak_system_cpu_percent", 0.0) or 0.0),
        "peak_system_ram_percent": float(process_summary.get("peak_system_ram_percent", 0.0) or 0.0),
        "peak_device_memory_mb": float(
            device_summary.get("peak_cuda_memory_reserved_mb", 0.0)
            or device_summary.get("peak_mps_driver_allocated_mb", 0.0)
            or 0.0
        ),
        "system_metrics_file": str(system_summary.get("system_metrics_file", "")),
    }


def extract_metrics(cv_results: Dict[str, Any], workflow: str) -> Dict[str, Any]:
    return WORKFLOW_REGISTRY[workflow]["metrics_fn"](cv_results)


def compute_multilabel_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
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
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    exact_match = float(accuracy_score(labels, preds))
    ham_loss = float(hamming_loss(labels, preds))
    abstention_rate = float((preds.sum(axis=1) == 0).mean())
    return {
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "exact_match_accuracy": exact_match,
        "hamming_loss": ham_loss,
        "abstention_rate": abstention_rate,
    }


def compute_singlelabel_frozen_predictions(oof_probs: np.ndarray, threshold: float) -> np.ndarray:
    preds = np.argmax(oof_probs, axis=1).astype(np.int64)
    max_probs = oof_probs.max(axis=1)
    # abstain index = number of train-class output neurons (always the next integer)
    preds[max_probs < float(threshold)] = oof_probs.shape[1]
    return preds


def compute_singlelabel_frozen_pred_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    gold_labels: List[str],
    abstain_idx: int,
    abstain_label: str = "verification_unknown",
) -> Dict[str, Any]:
    n_classes = max(int(abstain_idx) + 1, len(gold_labels))
    eval_labels = list(range(n_classes))
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        labels,
        preds,
        labels=eval_labels,
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
    class_names = list(gold_labels)
    if abstain_idx == len(class_names):
        class_names.append(str(abstain_label))
    per_class = {}
    for idx in eval_labels:
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        per_class[name] = {
            "precision": float(per_p[idx]),
            "recall": float(per_r[idx]),
            "f1": float(per_f1[idx]),
            "support": int(per_support[idx]),
        }
    return {
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "exact_match_accuracy": float(accuracy_score(labels, preds)),
        "abstention_rate": float((preds == abstain_idx).mean()),
        "per_class_metrics": per_class,
    }


def compute_singlelabel_frozen_pr_auc(
    oof_probs: np.ndarray,
    labels: np.ndarray,
    train_labels: List[str],
    abstain_idx: int,
    minority_classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    labeled_mask = labels != abstain_idx
    if not np.any(labeled_mask):
        return {
            "macro_pr_auc": 0.0,
            "minority_macro_pr_auc": 0.0,
            "per_class_pr_auc": {name: None for name in train_labels},
        }

    minority_set = set(minority_classes or [])
    probs = oof_probs[labeled_mask]
    y = labels[labeled_mask]
    class_vals: List[float] = []
    minority_vals: List[float] = []
    per_class: Dict[str, Optional[float]] = {}
    for idx, name in enumerate(train_labels):
        y_true = (y == idx).astype(np.int32)
        if y_true.sum() == 0:
            per_class[name] = None
            continue
        ap = float(average_precision_score(y_true, probs[:, idx]))
        per_class[name] = ap
        class_vals.append(ap)
        if name in minority_set:
            minority_vals.append(ap)
    return {
        "macro_pr_auc": float(np.mean(class_vals)) if class_vals else 0.0,
        "minority_macro_pr_auc": float(np.mean(minority_vals)) if minority_vals else 0.0,
        "per_class_pr_auc": per_class,
    }


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")


def _load_thresholds(thresholds_path: Path) -> Dict[str, float]:
    payload = read_json(thresholds_path)
    thresholds = payload.get("per_class") or payload.get("per_class_thresholds") or payload.get("calibrated_thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError(f"Invalid or empty thresholds format in {thresholds_path}")
    return {cat: float(val) for cat, val in thresholds.items()}


def _load_multilabel_temperature_scaling(thresholds_path: Path) -> Dict[str, Any]:
    payload = read_json(thresholds_path)
    ts = payload.get("temperature_scaling")
    if ts is None:
        return {"enabled": False, "mode": "per_class", "per_class": {}}
    if not isinstance(ts, dict):
        raise ValueError(f"Invalid temperature_scaling format in {thresholds_path}")
    return ts


def _apply_multilabel_temperature_scaling(
    oof_probs: np.ndarray,
    train_labels: List[str],
    temperature_scaling: Dict[str, Any],
) -> np.ndarray:
    if not bool(temperature_scaling.get("enabled", False)):
        return oof_probs
    per_class = temperature_scaling.get("per_class") or {}
    if not isinstance(per_class, dict):
        return oof_probs
    clipped = np.clip(oof_probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    temp_arr = np.array([float(per_class.get(c, 1.0)) for c in train_labels], dtype=np.float64)
    temp_arr = np.clip(temp_arr, 1e-6, None)
    scaled_logits = logits / temp_arr[None, :]
    scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return scaled_probs.astype(np.float32)


def _load_singlelabel_frozen_threshold(thresholds_path: Path) -> float:
    payload = read_json(thresholds_path)
    selected = payload.get("selected_threshold")
    if selected is None:
        selected = payload.get("calibrated_threshold")
    if selected is None:
        selected = payload.get("threshold")
    if selected is None:
        raise ValueError(f"Missing selected_threshold in {thresholds_path}")
    return float(selected)


def _validate_multiclass_frozen_artifacts(run_dir: Path, cv_results: Dict[str, Any], tol: float = 1e-8) -> None:
    for fname in [
        "cv_results.json",
        "run_config.json",
        "oof_probs.npy",
        "oof_labels.npy",
        "oof_fold_indices.npy",
        "thresholds.json",
    ]:
        _require_file(run_dir / fname)

    oof_item_indices_path = run_dir / "oof_item_indices.npy"
    if oof_item_indices_path.exists():
        _require_file(oof_item_indices_path)

    oof_probs = np.load(run_dir / "oof_probs.npy")
    oof_labels = np.load(run_dir / "oof_labels.npy")
    oof_fold_indices = np.load(run_dir / "oof_fold_indices.npy")

    label_mapping = cv_results.get("label_mapping") or {}
    train_labels: List[str] = list(label_mapping.get("train_labels") or [])
    abstain_idx = int(label_mapping.get("abstain_index", len(train_labels)))
    k = len(train_labels)

    if k and (oof_probs.ndim != 2 or oof_probs.shape[1] != k):
        raise ValueError(f"Unexpected OOF shape for multiclass frozen: {oof_probs.shape}, expected (N, {k})")
    if oof_labels.ndim != 1 or oof_labels.shape[0] != oof_probs.shape[0]:
        raise ValueError(f"Invalid multiclass oof_labels shape: {oof_labels.shape}")
    if oof_fold_indices.ndim != 1 or oof_fold_indices.shape[0] != oof_probs.shape[0]:
        raise ValueError(f"Invalid oof_fold_indices shape: {oof_fold_indices.shape}")
    if oof_item_indices_path.exists():
        oof_item_indices = np.load(oof_item_indices_path)
        if oof_item_indices.ndim != 1 or oof_item_indices.shape[0] != oof_probs.shape[0]:
            raise ValueError(f"Invalid oof_item_indices shape: {oof_item_indices.shape}")
    if np.isnan(oof_probs).any():
        raise ValueError("OOF probabilities contain NaN values")
    if (oof_fold_indices < 0).any():
        raise ValueError("oof_fold_indices has uncovered samples (<0)")

    threshold = _load_singlelabel_frozen_threshold(run_dir / "thresholds.json")
    preds = compute_singlelabel_frozen_predictions(oof_probs, threshold)

    pred_metrics = compute_singlelabel_frozen_pred_metrics(
        labels=oof_labels.astype(np.int64),
        preds=preds.astype(np.int64),
        gold_labels=train_labels,
        abstain_idx=abstain_idx,
        abstain_label=str(label_mapping.get("abstain_label") or label_mapping.get("raw_abstain_label") or "verification_unknown"),
    )
    prauc_metrics = compute_singlelabel_frozen_pr_auc(
        oof_probs=oof_probs.astype(np.float32),
        labels=oof_labels.astype(np.int64),
        train_labels=train_labels,
        abstain_idx=abstain_idx,
        minority_classes=cv_results.get("minority_classes") or [],
    )
    recomputed = {**pred_metrics, **prauc_metrics}

    reported = cv_results.get("oof_metrics_calibrated", {})
    required_keys = [
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "exact_match_accuracy",
        "abstention_rate",
        "macro_pr_auc",
        "minority_macro_pr_auc",
    ]
    for key in required_keys:
        if key not in reported:
            raise ValueError(f"Missing calibrated metric '{key}' in cv_results.json")
        diff = abs(float(reported[key]) - float(recomputed[key]))
        if diff > tol:
            raise ValueError(
                f"Metric mismatch for {key}: reported={reported[key]} recomputed={recomputed[key]} diff={diff:.3e}"
            )


def _validate_multilabel_artifacts(run_dir: Path, cv_results: Dict[str, Any], tol: float = 1e-8) -> None:
    """Validate OOF artifacts for all multilabel classifiers (LoRA and frozen multilabel)."""
    for fname in ["cv_results.json", "run_config.json", "oof_probs.npy", "oof_labels.npy",
                  "oof_fold_indices.npy", "thresholds.json"]:
        _require_file(run_dir / fname)

    oof_item_indices_path = run_dir / "oof_item_indices.npy"
    if not oof_item_indices_path.exists():
        logging.warning("oof_item_indices.npy not found in %s (older checkpoint); skipping item-index check.", run_dir)
    else:
        _require_file(oof_item_indices_path)

    oof_probs = np.load(run_dir / "oof_probs.npy")
    oof_labels = np.load(run_dir / "oof_labels.npy")
    oof_fold_indices = np.load(run_dir / "oof_fold_indices.npy")

    train_labels: List[str] = list(
        (cv_results.get("label_mapping") or {}).get("train_labels") or []
    )
    K = len(train_labels)
    if K and (oof_probs.ndim != 2 or oof_probs.shape[1] != K):
        raise ValueError(f"Unexpected OOF shape for multilabel: {oof_probs.shape}, expected (N, {K})")
    if K and (oof_labels.ndim != 2 or oof_labels.shape[1] != K):
        raise ValueError(f"Invalid multilabel oof_labels shape: {oof_labels.shape}")
    if oof_fold_indices.ndim != 1 or oof_fold_indices.shape[0] != oof_probs.shape[0]:
        raise ValueError(f"Invalid oof_fold_indices shape: {oof_fold_indices.shape}")
    if oof_item_indices_path.exists():
        oof_item_indices = np.load(oof_item_indices_path)
        if oof_item_indices.ndim != 1 or oof_item_indices.shape[0] != oof_probs.shape[0]:
            raise ValueError(f"Invalid oof_item_indices shape: {oof_item_indices.shape}")
    if np.isnan(oof_probs).any() or np.isnan(oof_labels).any():
        raise ValueError("OOF arrays contain NaN values")
    if (oof_fold_indices < 0).any():
        raise ValueError("oof_fold_indices has uncovered samples (<0)")

    thresholds_path = run_dir / "thresholds.json"
    thresholds = _load_thresholds(thresholds_path)
    temperature_scaling = _load_multilabel_temperature_scaling(thresholds_path)
    probs_for_thresholds = _apply_multilabel_temperature_scaling(
        oof_probs.astype(np.float32),
        train_labels,
        temperature_scaling,
    )
    threshold_array = np.array([thresholds.get(c, 0.5) for c in train_labels], dtype=np.float32)
    preds = (probs_for_thresholds >= threshold_array).astype(np.float32)
    recomputed = compute_multilabel_metrics(oof_labels.astype(np.float32), preds)

    reported = cv_results.get("oof_metrics_calibrated", {})
    required_keys = [
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "micro_f1",
        "micro_precision",
        "micro_recall",
        "exact_match_accuracy",
        "hamming_loss",
        "abstention_rate",
    ]
    for key in required_keys:
        if key not in reported:
            raise ValueError(f"Missing calibrated metric '{key}' in cv_results.json")
        diff = abs(float(reported[key]) - float(recomputed[key]))
        if diff > tol:
            raise ValueError(
                f"Metric mismatch for {key}: reported={reported[key]} recomputed={recomputed[key]} diff={diff:.3e}"
            )


def validate_artifacts_and_metrics(
    run_dir: Path,
    cv_results: Dict[str, Any],
    workflow: str,
    tol: float = 1e-8,
) -> None:
    WORKFLOW_REGISTRY[workflow]["validate_fn"](run_dir, cv_results, tol=tol)


def _snapshot_from_dataset_cache_multilabel(cache_path: Path) -> Dict[str, Any]:
    """Dataset cache snapshot for all multilabel classifiers (LoRA and frozen)."""
    payload = read_json(cache_path)
    prepared_items = payload.get("prepared_items")
    if not isinstance(prepared_items, list):
        raise ValueError(f"Invalid cache format: missing prepared_items in {cache_path}")

    categories: List[str] = list(
        (payload.get("label_mapping") or {}).get("train_labels") or []
    )
    counts = {cat: 0 for cat in categories}
    all_zero = 0
    for item in prepared_items:
        labels = item.get("labels") or []
        if not labels:
            all_zero += 1
        else:
            for lbl in labels:
                if lbl in counts:
                    counts[lbl] += 1

    prepared_signature = str(payload.get("prepared_signature") or "")
    if not prepared_signature:
        hasher = hashlib.sha256()
        for item in prepared_items:
            hasher.update(str(item.get("file_name", "")).encode("utf-8"))
            hasher.update(b"\0")
            labels = item.get("labels", []) or []
            hasher.update("|".join(sorted(str(x) for x in labels)).encode("utf-8"))
            hasher.update(b"\n")
        prepared_signature = hasher.hexdigest()

    return {
        "total_items": int(len(prepared_items)),
        "all_zero_target_docs": int(all_zero),
        "per_class_counts": counts,
        "class_keys": categories,
        "prepared_signature": prepared_signature,
        "population_scope": "prepared_items",
        "compare_per_class": True,
    }


def _snapshot_from_dataset_cache_multiclass(cache_path: Path) -> Dict[str, Any]:
    payload = read_json(cache_path)
    prepared_items = payload.get("prepared_items")
    if not isinstance(prepared_items, list):
        raise ValueError(f"Invalid cache format: missing prepared_items in {cache_path}")

    label_mapping = payload.get("label_mapping") or {}
    categories: List[str] = list(label_mapping.get("train_labels") or [])
    abstain_label = str(label_mapping.get("abstain_label") or "verification_unknown")
    counts = {cat: 0 for cat in categories}
    abstain_count = 0
    for item in prepared_items:
        if "gold_label" in item:
            gold = str(item.get("gold_label") or "")
            if gold == abstain_label:
                abstain_count += 1
            elif gold in counts:
                counts[gold] += 1
            continue
        labels = item.get("labels") or []
        if not labels:
            abstain_count += 1
        elif labels:
            lbl = str(labels[0])
            if lbl in counts:
                counts[lbl] += 1

    prepared_signature = str(payload.get("prepared_signature") or "")
    if not prepared_signature:
        hasher = hashlib.sha256()
        for item in prepared_items:
            hasher.update(str(item.get("file_name", "")).encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(str(item.get("gold_label", "")).encode("utf-8"))
            hasher.update(b"\n")
        prepared_signature = hasher.hexdigest()

    return {
        "total_items": int(len(prepared_items)),
        "all_zero_target_docs": int(abstain_count),
        "per_class_counts": counts,
        "class_keys": categories,
        "prepared_signature": prepared_signature,
        "population_scope": "prepared_items",
        "compare_per_class": True,
    }


def snapshot_from_dataset_cache(cache_path: Path, workflow: str) -> Dict[str, Any]:
    return WORKFLOW_REGISTRY[workflow]["snapshot_cache_fn"](cache_path)


def _snapshot_from_cv_results_multilabel(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    """CV results snapshot for all multilabel classifiers (LoRA and frozen)."""
    categories: List[str] = list(
        (cv_results.get("label_mapping") or {}).get("train_labels") or []
    )
    dataset_stats = cv_results.get("dataset_stats", {})
    label_stats = cv_results.get("label_stats", {})
    distribution = label_stats.get("distribution", {})

    total_items = int(dataset_stats.get("total_items", label_stats.get("num_items", 0)))
    label_num_items = int(label_stats.get("num_items", total_items))
    all_zero = int(dataset_stats.get("all_zero_target_docs", dataset_stats.get("abstain_docs", label_stats.get("all_zero_target_docs", 0))))

    compare_per_class = label_num_items == total_items
    counts: Dict[str, int] = {}
    if compare_per_class:
        for cat in categories:
            v = distribution.get(cat, {})
            if isinstance(v, dict):
                counts[cat] = int(v.get("count", 0))
            else:
                counts[cat] = int(v or 0)

    return {
        "total_items": total_items,
        "all_zero_target_docs": all_zero,
        "per_class_counts": counts,
        "class_keys": list(categories),
        "prepared_signature": str(cv_results.get("prepared_signature") or ""),
        "population_scope": "prepared_items" if compare_per_class else "trainable_subset",
        "compare_per_class": compare_per_class,
        "label_num_items": label_num_items,
    }


def _snapshot_from_cv_results_multiclass(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    label_mapping = cv_results.get("label_mapping") or {}
    categories: List[str] = list(label_mapping.get("train_labels") or [])
    abstain_label = str(label_mapping.get("abstain_label") or label_mapping.get("raw_abstain_label") or "verification_unknown")
    dataset_stats = cv_results.get("dataset_stats", {})
    label_stats = cv_results.get("label_stats", {})
    distribution = label_stats.get("distribution", {})

    total_items = int(dataset_stats.get("total_items", label_stats.get("num_items", 0)))
    label_num_items = int(label_stats.get("num_items", total_items))
    abstain_from_distribution = 0
    if isinstance(distribution.get(abstain_label), dict):
        abstain_from_distribution = int(distribution.get(abstain_label, {}).get("count", 0))
    all_zero = int(dataset_stats.get("all_zero_target_docs", dataset_stats.get("abstain_docs", abstain_from_distribution)))

    counts: Dict[str, int] = {}
    for cat in categories:
        v = distribution.get(cat, {})
        if isinstance(v, dict):
            counts[cat] = int(v.get("count", 0))
        else:
            counts[cat] = int(v or 0)

    return {
        "total_items": total_items,
        "all_zero_target_docs": all_zero,
        "per_class_counts": counts,
        "class_keys": list(categories),
        "prepared_signature": str(cv_results.get("prepared_signature") or ""),
        "population_scope": "prepared_items" if label_num_items == total_items else "trainable_subset",
        "compare_per_class": label_num_items == total_items,
        "label_num_items": label_num_items,
    }


def snapshot_from_cv_results(cv_results: Dict[str, Any], workflow: str) -> Dict[str, Any]:
    return WORKFLOW_REGISTRY[workflow]["snapshot_cv_fn"](cv_results)


def snapshot_mismatches(expected: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    mismatches: List[str] = []
    if int(expected["total_items"]) != int(current["total_items"]):
        mismatches.append(
            f"total_items expected={expected['total_items']} actual={current['total_items']}"
        )
    if int(expected["all_zero_target_docs"]) != int(current["all_zero_target_docs"]):
        mismatches.append(
            f"all_zero_target_docs expected={expected['all_zero_target_docs']} actual={current['all_zero_target_docs']}"
        )

    # Strongest integrity check when both sides provide signatures.
    exp_sig = str(expected.get("prepared_signature", "") or "")
    cur_sig = str(current.get("prepared_signature", "") or "")
    if exp_sig and cur_sig and exp_sig != cur_sig:
        mismatches.append(
            f"prepared_signature expected={exp_sig[:12]}... actual={cur_sig[:12]}..."
        )

    expected_compare_per_class = bool(expected.get("compare_per_class", True))
    current_compare_per_class = bool(current.get("compare_per_class", True))
    if expected_compare_per_class and current_compare_per_class:
        e_counts = expected.get("per_class_counts", {})
        c_counts = current.get("per_class_counts", {})
        class_keys = expected.get("class_keys") or current.get("class_keys") or []
        for cat in class_keys:
            e_val = int(e_counts.get(cat, 0))
            c_val = int(c_counts.get(cat, 0))
            if e_val != c_val:
                mismatches.append(f"class[{cat}] expected={e_val} actual={c_val}")

    return mismatches


def ensure_data_integrity_gate(args: argparse.Namespace, cv_results: Optional[Dict[str, Any]] = None) -> None:
    expected = getattr(args, "_expected_data_snapshot", None)

    if expected is None and args.dataset_cache_path and Path(args.dataset_cache_path).exists():
        expected = snapshot_from_dataset_cache(Path(args.dataset_cache_path), workflow=str(args.workflow))
        args._expected_data_snapshot = expected
        args._expected_data_snapshot_source = f"cache:{args.dataset_cache_path}"
        print(
            "[DATA_GATE] initialized from cache | "
            f"items={expected['total_items']} all_zero={expected['all_zero_target_docs']}"
        )

    if expected is not None and args.dataset_cache_path and Path(args.dataset_cache_path).exists():
        current_cache = snapshot_from_dataset_cache(Path(args.dataset_cache_path), workflow=str(args.workflow))
        mismatches = snapshot_mismatches(expected, current_cache)
        if mismatches:
            raise RuntimeError(
                "Data integrity gate failed against dataset cache before run: "
                + " | ".join(mismatches[:20])
            )

    if cv_results is None:
        return

    current_cv = snapshot_from_cv_results(cv_results, workflow=str(args.workflow))
    if expected is None:
        args._expected_data_snapshot = current_cv
        args._expected_data_snapshot_source = "first_successful_run"
        print(
            "[DATA_GATE] initialized from first successful run | "
            f"items={current_cv['total_items']} all_zero={current_cv['all_zero_target_docs']}"
        )
        return

    mismatches = snapshot_mismatches(expected, current_cv)
    if mismatches:
        source = getattr(args, "_expected_data_snapshot_source", "unknown")
        raise RuntimeError(
            f"Data integrity gate failed against expected snapshot ({source}): "
            + " | ".join(mismatches[:20])
        )


def _summarize_metric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _load_run_cv_results(run_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    run_dir = Path(str(run_result.get("run_dir", "")))
    cv_path = run_dir / "cv_results.json"
    if not cv_path.exists():
        return None
    return read_json(cv_path)


def _minority_breakdown_from_cv_results(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    calibrated = cv_results.get("oof_metrics_calibrated", {})
    per_class = calibrated.get("per_class_metrics", {})
    classes: List[str] = cv_results.get("minority_classes") or []

    class_metrics: Dict[str, Dict[str, float]] = {}
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []
    supports: Dict[str, int] = {}

    for cls in classes:
        m = per_class.get(cls, {})
        p = float(m.get("precision", 0.0))
        r = float(m.get("recall", 0.0))
        f = float(m.get("f1", 0.0))
        s = int(m.get("support", 0))
        class_metrics[cls] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": s,
        }
        supports[cls] = s
        precision_values.append(p)
        recall_values.append(r)
        f1_values.append(f)

    return {
        "means": {
            "precision": float(np.mean(precision_values)) if precision_values else 0.0,
            "recall": float(np.mean(recall_values)) if recall_values else 0.0,
            "f1": float(np.mean(f1_values)) if f1_values else 0.0,
        },
        "class_metrics": class_metrics,
        "supports": supports,
    }


def _minority_breakdown_from_run_result(run_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if run_result.get("status") == "dry_run":
        return None
    cv_results = _load_run_cv_results(run_result)
    if cv_results is None:
        return None
    return _minority_breakdown_from_cv_results(cv_results)


def _minority_delta(
    current_breakdown: Optional[Dict[str, Any]],
    previous_breakdown: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if current_breakdown is None or previous_breakdown is None:
        return None
    curr_cls = current_breakdown.get("class_metrics", {})
    prev_cls = previous_breakdown.get("class_metrics", {})
    classes_delta: Dict[str, Dict[str, float]] = {}
    class_keys = sorted(set(curr_cls.keys()) | set(prev_cls.keys()))
    for cls in class_keys:
        c = curr_cls.get(cls, {})
        p = prev_cls.get(cls, {})
        classes_delta[cls] = {
            "precision_delta": float(c.get("precision", 0.0) - p.get("precision", 0.0)),
            "recall_delta": float(c.get("recall", 0.0) - p.get("recall", 0.0)),
            "f1_delta": float(c.get("f1", 0.0) - p.get("f1", 0.0)),
        }
    return {
        "means": {
            "precision_delta": float(
                current_breakdown["means"]["precision"] - previous_breakdown["means"]["precision"]
            ),
            "recall_delta": float(
                current_breakdown["means"]["recall"] - previous_breakdown["means"]["recall"]
            ),
            "f1_delta": float(current_breakdown["means"]["f1"] - previous_breakdown["means"]["f1"]),
        },
        "classes": classes_delta,
    }


def build_stage_health(
    stage_name: str,
    results: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    selected: Optional[Dict[str, Any]],
    previous_selected: Optional[Dict[str, Any]],
    abstention_min: float,
    abstention_max: float,
    min_macro_gain: float,
    max_minority_drop: float,
    precision_collapse_delta: float,
    recall_spike_delta: float,
    enforce_gate: bool,
) -> Dict[str, Any]:
    successful = [r for r in results if r["status"] in {"ok", "resumed", "dry_run"}]
    status_counts: Dict[str, int] = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    macro_values = [float(r["macro_f1"]) for r in successful]
    minority_values = [float(r["minority_mean_f1"]) for r in successful]
    abstention_values = [float(r["abstention_rate"]) for r in successful]
    peak_rss_values = [float(r.get("peak_process_rss_mb", 0.0)) for r in successful]
    peak_cpu_values = [float(r.get("peak_system_cpu_percent", 0.0)) for r in successful]
    peak_ram_values = [float(r.get("peak_system_ram_percent", 0.0)) for r in successful]
    peak_devmem_values = [float(r.get("peak_device_memory_mb", 0.0)) for r in successful]
    guardrail_pass_count = int(sum(1 for r in results if bool(r.get("guardrail_pass", False))))

    selected_summary = None
    if selected is not None:
        selected_summary = {
            "run_id": selected["run_id"],
            "status": selected["status"],
            "guardrail_pass": bool(selected.get("guardrail_pass", False)),
            "macro_f1": float(selected.get("macro_f1", 0.0)),
            "macro_precision": float(selected.get("macro_precision", 0.0)),
            "macro_recall": float(selected.get("macro_recall", 0.0)),
            "minority_mean_f1": float(selected.get("minority_mean_f1", 0.0)),
            "minority_mean_precision": float(selected.get("minority_mean_precision", 0.0)),
            "minority_mean_recall": float(selected.get("minority_mean_recall", 0.0)),
            "exact_match_accuracy": float(selected.get("exact_match_accuracy", 0.0)),
            "abstention_rate": float(selected.get("abstention_rate", 0.0)),
            "peak_process_rss_mb": float(selected.get("peak_process_rss_mb", 0.0)),
            "peak_system_cpu_percent": float(selected.get("peak_system_cpu_percent", 0.0)),
            "peak_system_ram_percent": float(selected.get("peak_system_ram_percent", 0.0)),
            "peak_device_memory_mb": float(selected.get("peak_device_memory_mb", 0.0)),
            "system_metrics_file": str(selected.get("system_metrics_file", "")),
            "runtime_sec": float(selected.get("runtime_sec", 0.0)),
            "run_dir": selected.get("run_dir"),
            "params": selected.get("params", {}),
        }

    previous_summary = None
    if previous_selected is not None:
        previous_summary = {
            "run_id": previous_selected["run_id"],
            "macro_f1": float(previous_selected.get("macro_f1", 0.0)),
            "minority_mean_f1": float(previous_selected.get("minority_mean_f1", 0.0)),
            "abstention_rate": float(previous_selected.get("abstention_rate", 0.0)),
            "run_dir": previous_selected.get("run_dir"),
        }

    delta_vs_previous = None
    if selected is not None and previous_selected is not None:
        delta_vs_previous = {
            "macro_f1_delta": float(selected.get("macro_f1", 0.0) - previous_selected.get("macro_f1", 0.0)),
            "minority_mean_f1_delta": float(
                selected.get("minority_mean_f1", 0.0) - previous_selected.get("minority_mean_f1", 0.0)
            ),
            "exact_match_delta": float(
                selected.get("exact_match_accuracy", 0.0) - previous_selected.get("exact_match_accuracy", 0.0)
            ),
            "abstention_rate_delta": float(
                selected.get("abstention_rate", 0.0) - previous_selected.get("abstention_rate", 0.0)
            ),
        }

    selected_minority_breakdown = (
        _minority_breakdown_from_run_result(selected) if selected is not None else None
    )
    previous_minority_breakdown = (
        _minority_breakdown_from_run_result(previous_selected) if previous_selected is not None else None
    )
    minority_delta = _minority_delta(selected_minority_breakdown, previous_minority_breakdown)

    alerts: List[Dict[str, Any]] = []
    if len(successful) == 0:
        alerts.append({"code": "no_successful_runs", "severity": "critical"})
    if guardrail_pass_count == 0:
        alerts.append({"code": "no_guardrail_passing_runs", "severity": "critical"})

    if selected is not None:
        abst = float(selected.get("abstention_rate", 0.0))
        if not (abstention_min <= abst <= abstention_max):
            alerts.append(
                {
                    "code": "selected_abstention_outside_guardrail",
                    "severity": "diagnostic",
                    "abstention_rate": abst,
                    "guardrail": {"min": abstention_min, "max": abstention_max},
                }
            )

    if delta_vs_previous is not None:
        if delta_vs_previous["macro_f1_delta"] < min_macro_gain:
            alerts.append(
                {
                    "code": "stage_gain_below_threshold",
                    "severity": "diagnostic",
                    "required_min_delta": min_macro_gain,
                    "observed_delta": delta_vs_previous["macro_f1_delta"],
                }
            )
        if delta_vs_previous["minority_mean_f1_delta"] < -max_minority_drop:
            alerts.append(
                {
                    "code": "minority_mean_f1_drop_exceeds_threshold",
                    "severity": "diagnostic",
                    "allowed_drop": -max_minority_drop,
                    "observed_delta": delta_vs_previous["minority_mean_f1_delta"],
                }
            )

    precision_collapse_classes: List[str] = []
    recall_spike_classes: List[str] = []
    if minority_delta is not None:
        for cls, delta in minority_delta["classes"].items():
            if float(delta["precision_delta"]) <= -precision_collapse_delta:
                precision_collapse_classes.append(cls)
            if float(delta["recall_delta"]) >= recall_spike_delta:
                recall_spike_classes.append(cls)

    if precision_collapse_classes:
        alerts.append(
            {
                "code": "rare_precision_collapse",
                "severity": "diagnostic",
                "classes": precision_collapse_classes,
                "threshold": precision_collapse_delta,
            }
        )
    if recall_spike_classes:
        alerts.append(
            {
                "code": "rare_recall_spike",
                "severity": "diagnostic",
                "classes": recall_spike_classes,
                "threshold": recall_spike_delta,
            }
        )

    should_block = False
    if enforce_gate and delta_vs_previous is not None and selected is not None:
        gain_ok = delta_vs_previous["macro_f1_delta"] >= min_macro_gain
        minority_ok = delta_vs_previous["minority_mean_f1_delta"] >= -max_minority_drop
        guardrail_ok = bool(selected.get("guardrail_pass", False))
        should_block = not (gain_ok and minority_ok and guardrail_ok)

    return {
        "stage_name": stage_name,
        "num_runs": len(results),
        "num_successful_runs": len(successful),
        "num_guardrail_passing_runs": guardrail_pass_count,
        "status_counts": status_counts,
        "metrics_distribution": {
            "macro_f1": _summarize_metric(macro_values),
            "minority_mean_f1": _summarize_metric(minority_values),
            "abstention_rate": _summarize_metric(abstention_values),
            "peak_process_rss_mb": _summarize_metric(peak_rss_values),
            "peak_system_cpu_percent": _summarize_metric(peak_cpu_values),
            "peak_system_ram_percent": _summarize_metric(peak_ram_values),
            "peak_device_memory_mb": _summarize_metric(peak_devmem_values),
        },
        "selected": selected_summary,
        "previous_selected": previous_summary,
        "deltas_vs_previous": delta_vs_previous,
        "minority_breakdown_selected": selected_minority_breakdown,
        "minority_breakdown_previous": previous_minority_breakdown,
        "minority_deltas_vs_previous": minority_delta,
        "alerts": alerts,
        "gate": {
            "enforced": bool(enforce_gate),
            "min_macro_gain": float(min_macro_gain),
            "max_minority_f1_drop": float(max_minority_drop),
            "abstention_guardrail": {"min": float(abstention_min), "max": float(abstention_max)},
            "should_block": bool(should_block),
        },
        "leaderboard_head": ranked[:5],
    }


def write_stage_health(phase_root: Path, phase_name: str, health: Dict[str, Any]) -> str:
    ensure_dir(phase_root)
    path = phase_root / f"{phase_name}_health.json"
    write_json(path, health)
    return str(path)


def top_successful_candidate(ranked: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in ranked:
        if r.get("status") in {"ok", "resumed", "dry_run"}:
            return r
    return None


def enforce_stage_gate(stage_name: str, health: Dict[str, Any]) -> None:
    gate = health.get("gate", {})
    if not bool(gate.get("enforced", False)):
        return
    if not bool(gate.get("should_block", False)):
        return

    selected = health.get("selected") or {}
    delta = health.get("deltas_vs_previous") or {}
    raise RuntimeError(
        f"{stage_name}: stage gate failed. "
        f"selected_run={selected.get('run_id')} "
        f"macro_f1={selected.get('macro_f1')} "
        f"delta_macro_f1={delta.get('macro_f1_delta')} "
        f"delta_minority_mean_f1={delta.get('minority_mean_f1_delta')}. "
        "Inspect stage health JSON for details."
    )


def _prauc_sanity_pass(result: Dict[str, Any]) -> bool:
    macro_pr_auc = float(result.get("macro_pr_auc", result.get("cv_macro_pr_auc_mean", 0.0)))
    minority_pr_auc = float(result.get("minority_macro_pr_auc", 0.0))
    floor = max(0.05, 0.35 * macro_pr_auc)
    return minority_pr_auc >= floor


def _selection_value(result: Dict[str, Any]) -> float:
    if ACTIVE_SELECTION_METRIC == "exact_match_accuracy":
        return float(result.get("exact_match_accuracy", 0.0))
    if ACTIVE_SELECTION_METRIC == "macro_pr_auc":
        return float(result.get("macro_pr_auc", result.get("cv_macro_pr_auc_mean", 0.0)))
    return float(result.get("macro_f1", 0.0))


def result_sort_key(result: Dict[str, Any]) -> Tuple[Any, ...]:
    prauc_ok = True
    if ACTIVE_SELECTION_METRIC == "macro_pr_auc":
        prauc_ok = bool(result.get("prauc_sanity_pass", _prauc_sanity_pass(result)))
    return (
        int(result.get("guardrail_pass", False)),
        int(prauc_ok),
        _selection_value(result),
        float(result.get("macro_f1", 0.0)),
        float(result.get("exact_match_accuracy", 0.0)),
        float(result.get("cv_macro_pr_auc_mean", 0.0)),
        float(result.get("minority_mean_f1", 0.0)),
        -float(result.get("runtime_sec", 0.0)),
    )


def run_single_experiment(
    spec: RunSpec,
    phase_root: Path,
    args: argparse.Namespace,
    asha_state: Optional[StageASHAState] = None,
) -> Dict[str, Any]:
    run_dir = phase_root / spec.run_id
    ensure_dir(run_dir)

    cv_results_path = run_dir / "cv_results.json"
    run_cfg_path = run_dir / "run_config.json"

    if not args.dry_run:
        ensure_data_integrity_gate(args)

    if args.resume and cv_results_path.exists() and run_cfg_path.exists():
        cv_results = read_json(cv_results_path)
        run_cfg = read_json(run_cfg_path)
        expected = expected_resume_fields(args, spec)
        mismatches = resume_config_mismatches(run_cfg, expected)
        if not mismatches:
            try:
                validate_artifacts_and_metrics(run_dir, cv_results, workflow=str(args.workflow))
                ensure_data_integrity_gate(args, cv_results)
                metrics = extract_metrics(cv_results, workflow=str(args.workflow))
                result = {
                    "run_id": spec.run_id,
                    "phase": spec.phase,
                    "stage": spec.stage,
                    "status": "resumed",
                    "runtime_sec": float(cv_results.get("runtime_sec", 0.0)),
                    **metrics,
                    "params": spec.params,
                    "notes": spec.notes,
                    "run_dir": str(run_dir),
                }
                if asha_state is not None:
                    asha_state.update_from_cv_results(cv_results)
                return result
            except Exception as exc:
                print(f"[RESUME_INVALID] {spec.run_id}: {exc}. Rerunning.")
        print(
            f"[RESUME_MISMATCH] {spec.run_id} has stale config; rerunning. "
            f"Mismatches: {' | '.join(mismatches)}"
        )

    cmd_params = dict(spec.params)
    cmd_params["task_type"] = WORKFLOW_REGISTRY[args.workflow]["task_type"]
    fixed_params = WORKFLOW_REGISTRY[args.workflow].get("fixed_hpo_params") or {}
    for _key, _val in fixed_params.items():
        if _key == "task" and getattr(args, "task", None):
            cmd_params[_key] = str(getattr(args, "task"))
            continue
        if _key not in cmd_params:
            cmd_params[_key] = _val
    if getattr(args, "task", None):
        cmd_params["task"] = str(getattr(args, "task"))
    if getattr(args, "prompt_name", None):
        cmd_params["prompt_name"] = str(getattr(args, "prompt_name"))
    if getattr(args, "abstain_label", None):
        cmd_params["abstain_label"] = str(getattr(args, "abstain_label"))
    if getattr(args, "embeddings_path", None):
        cmd_params["embeddings_path"] = str(getattr(args, "embeddings_path"))
    if getattr(args, "metadata_path", None):
        cmd_params["metadata_path"] = str(getattr(args, "metadata_path"))
    if getattr(args, "model_name", None):
        cmd_params["model_name"] = str(getattr(args, "model_name"))
    if _is_multilabel_frozen_workflow(str(args.workflow)):
        cmd_params["use_temperature_scaling"] = bool(getattr(args, "use_temperature_scaling", False))

    allowed_threshold_metrics = set(_allowed_threshold_metrics_for_workflow(str(args.workflow)))
    auto_threshold_metric = bool(WORKFLOW_REGISTRY[str(args.workflow)].get("auto_threshold_metric", True))
    if auto_threshold_metric and "threshold_selection_metric" not in cmd_params:
        proposed_metric = str(args.selection_metric) if str(args.selection_metric) in {"macro_f1", "exact_match_accuracy"} else None
        if proposed_metric in allowed_threshold_metrics:
            cmd_params["threshold_selection_metric"] = str(proposed_metric)
        elif "macro_f1" in allowed_threshold_metrics:
            # Safe default across trainers when selection metric is not directly supported.
            cmd_params["threshold_selection_metric"] = "macro_f1"
    if "threshold_selection_metric" in cmd_params:
        threshold_metric = str(cmd_params["threshold_selection_metric"])
        if threshold_metric not in allowed_threshold_metrics:
            raise ValueError(
                f"Unsupported threshold_selection_metric='{threshold_metric}' for workflow={args.workflow}. "
                f"Allowed: {sorted(allowed_threshold_metrics)}"
            )

    cmd = build_command(
        python_exec=args.python,
        trainer_script=args.trainer_script,
        checkpoint_dir=run_dir,
        dataset_name=args.dataset_name,
        dataset_cache_path=args.dataset_cache_path,
        campaign_signature=args.campaign_signature,
        seed=args.seed,
        checkpoint_selection_metric=args.checkpoint_selection_metric,
        post_oof_to_langfuse=args.post_oof_to_langfuse,
        fail_on_langfuse_post_error=args.fail_on_langfuse_post_error,
        params=cmd_params,
    )

    if args.dry_run:
        return {
            "run_id": spec.run_id,
            "phase": spec.phase,
            "stage": spec.stage,
            "status": "dry_run",
            "runtime_sec": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "cv_macro_pr_auc_mean": 0.0,
            "macro_pr_auc": 0.0,
            "minority_macro_pr_auc": 0.0,
            "minority_mean_f1": 0.0,
            "minority_mean_precision": 0.0,
            "minority_mean_recall": 0.0,
            "exact_match_accuracy": 0.0,
            "abstention_rate": 0.0,
            "peak_process_rss_mb": 0.0,
            "peak_system_cpu_percent": 0.0,
            "peak_system_ram_percent": 0.0,
            "peak_device_memory_mb": 0.0,
            "system_metrics_file": "",
            "params": spec.params,
            "notes": spec.notes,
            "run_dir": str(run_dir),
            "command": cmd,
        }

    epoch_line_re = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+\|.*val_macro_f1=(?P<f1>[0-9]*\.?[0-9]+).*val_macro_pr_auc=(?P<prauc>[0-9]*\.?[0-9]+)"
    )
    asha_enabled = asha_state is not None
    seen_epoch_scores: Dict[int, List[float]] = {}
    pruned_reason = ""

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    out_lines: List[str] = []

    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line)
        m = epoch_line_re.search(line)
        if asha_enabled and m is not None:
            epoch = int(m.group("epoch"))
            f1 = float(m.group("f1"))
            pr = float(m.group("prauc"))
            score = StageASHAState.combined_score(f1, pr)
            seen_epoch_scores.setdefault(epoch, []).append(score)

            prev_epoch_score: Optional[float] = None
            prev_epochs = [e for e in seen_epoch_scores.keys() if e < epoch and seen_epoch_scores[e]]
            if prev_epochs:
                prev_epoch = max(prev_epochs)
                prev_epoch_score = float(np.mean(seen_epoch_scores[prev_epoch]))

            should_prune, reason = asha_state.should_prune(
                epoch=epoch,
                score=float(np.mean(seen_epoch_scores[epoch])),
                previous_epoch_score=prev_epoch_score,
            )
            if should_prune:
                pruned_reason = reason
                print(f"[ASHA_PRUNE] {spec.run_id}: {reason}")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break

    proc.wait()
    runtime_sec = time.time() - start

    combined_out = "".join(out_lines)
    (run_dir / "stdout.log").write_text(combined_out, encoding="utf-8")
    # stderr is merged into stdout for streaming parse; mirror content for tooling
    # that expects diagnostics in stderr.log.
    (run_dir / "stderr.log").write_text(combined_out, encoding="utf-8")

    if pruned_reason:
        return {
            "run_id": spec.run_id,
            "phase": spec.phase,
            "stage": spec.stage,
            "status": "pruned(asha)",
            "runtime_sec": runtime_sec,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "cv_macro_pr_auc_mean": 0.0,
            "macro_pr_auc": 0.0,
            "minority_macro_pr_auc": 0.0,
            "minority_mean_f1": 0.0,
            "minority_mean_precision": 0.0,
            "minority_mean_recall": 0.0,
            "exact_match_accuracy": 0.0,
            "abstention_rate": 0.0,
            "peak_process_rss_mb": 0.0,
            "peak_system_cpu_percent": 0.0,
            "peak_system_ram_percent": 0.0,
            "peak_device_memory_mb": 0.0,
            "system_metrics_file": "",
            "params": spec.params,
            "notes": f"{spec.notes} | {pruned_reason}",
            "run_dir": str(run_dir),
        }

    if proc.returncode != 0:
        return {
            "run_id": spec.run_id,
            "phase": spec.phase,
            "stage": spec.stage,
            "status": f"failed({proc.returncode})",
            "runtime_sec": runtime_sec,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "cv_macro_pr_auc_mean": 0.0,
            "macro_pr_auc": 0.0,
            "minority_macro_pr_auc": 0.0,
            "minority_mean_f1": 0.0,
            "minority_mean_precision": 0.0,
            "minority_mean_recall": 0.0,
            "exact_match_accuracy": 0.0,
            "abstention_rate": 0.0,
            "peak_process_rss_mb": 0.0,
            "peak_system_cpu_percent": 0.0,
            "peak_system_ram_percent": 0.0,
            "peak_device_memory_mb": 0.0,
            "system_metrics_file": "",
            "params": spec.params,
            "notes": spec.notes,
            "run_dir": str(run_dir),
        }

    if not cv_results_path.exists():
        return {
            "run_id": spec.run_id,
            "phase": spec.phase,
            "stage": spec.stage,
            "status": "failed(no_cv_results)",
            "runtime_sec": runtime_sec,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "cv_macro_pr_auc_mean": 0.0,
            "macro_pr_auc": 0.0,
            "minority_macro_pr_auc": 0.0,
            "minority_mean_f1": 0.0,
            "minority_mean_precision": 0.0,
            "minority_mean_recall": 0.0,
            "exact_match_accuracy": 0.0,
            "abstention_rate": 0.0,
            "peak_process_rss_mb": 0.0,
            "peak_system_cpu_percent": 0.0,
            "peak_system_ram_percent": 0.0,
            "peak_device_memory_mb": 0.0,
            "system_metrics_file": "",
            "params": spec.params,
            "notes": spec.notes,
            "run_dir": str(run_dir),
        }

    cv_results = read_json(cv_results_path)
    validate_artifacts_and_metrics(run_dir, cv_results, workflow=str(args.workflow))
    ensure_data_integrity_gate(args, cv_results)
    cv_results["runtime_sec"] = runtime_sec
    write_json(cv_results_path, cv_results)
    if asha_state is not None:
        asha_state.update_from_cv_results(cv_results)

    metrics = extract_metrics(cv_results, workflow=str(args.workflow))

    return {
        "run_id": spec.run_id,
        "phase": spec.phase,
        "stage": spec.stage,
        "status": "ok",
        "runtime_sec": runtime_sec,
        **metrics,
        "params": spec.params,
        "notes": spec.notes,
        "run_dir": str(run_dir),
    }


def apply_guardrail(results: List[Dict[str, Any]], abstention_min: float, abstention_max: float) -> None:
    for r in results:
        r["prauc_sanity_pass"] = bool(_prauc_sanity_pass(r))
        if r["status"] == "dry_run":
            pass_status = True
        else:
            pass_status = (
                r["status"] in {"ok", "resumed"}
                and abstention_min <= r["abstention_rate"] <= abstention_max
            )
            if ACTIVE_SELECTION_METRIC == "macro_pr_auc":
                pass_status = pass_status and bool(r["prauc_sanity_pass"])
        r["guardrail_pass"] = pass_status


def rank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(results, key=result_sort_key, reverse=True)


def write_phase_outputs(phase_root: Path, phase_name: str, ranked: List[Dict[str, Any]]) -> None:
    ensure_dir(phase_root)

    json_payload = {"phase": phase_name, "results": ranked}
    write_json(phase_root / f"{phase_name}_leaderboard.json", json_payload)

    csv_path = phase_root / f"{phase_name}_leaderboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank",
            "run_id",
            "status",
            "guardrail_pass",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "cv_macro_pr_auc_mean",
            "macro_pr_auc",
            "minority_macro_pr_auc",
            "minority_mean_f1",
            "minority_mean_precision",
            "minority_mean_recall",
            "exact_match_accuracy",
            "abstention_rate",
            "peak_process_rss_mb",
            "peak_system_cpu_percent",
            "peak_system_ram_percent",
            "peak_device_memory_mb",
            "runtime_sec",
            "phase",
            "stage",
            "params",
            "run_dir",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": i,
                    "run_id": row["run_id"],
                    "status": row["status"],
                    "guardrail_pass": row["guardrail_pass"],
                    "macro_f1": row["macro_f1"],
                    "macro_precision": row.get("macro_precision", 0.0),
                    "macro_recall": row.get("macro_recall", 0.0),
                    "cv_macro_pr_auc_mean": row.get("cv_macro_pr_auc_mean", 0.0),
                    "macro_pr_auc": row.get("macro_pr_auc", row.get("cv_macro_pr_auc_mean", 0.0)),
                    "minority_macro_pr_auc": row.get("minority_macro_pr_auc", 0.0),
                    "minority_mean_f1": row["minority_mean_f1"],
                    "minority_mean_precision": row.get("minority_mean_precision", 0.0),
                    "minority_mean_recall": row.get("minority_mean_recall", 0.0),
                    "exact_match_accuracy": row["exact_match_accuracy"],
                    "abstention_rate": row["abstention_rate"],
                    "peak_process_rss_mb": row.get("peak_process_rss_mb", 0.0),
                    "peak_system_cpu_percent": row.get("peak_system_cpu_percent", 0.0),
                    "peak_system_ram_percent": row.get("peak_system_ram_percent", 0.0),
                    "peak_device_memory_mb": row.get("peak_device_memory_mb", 0.0),
                    "runtime_sec": row["runtime_sec"],
                    "phase": row["phase"],
                    "stage": row["stage"],
                    "params": json.dumps(row["params"], sort_keys=True),
                    "run_dir": row["run_dir"],
                    "notes": row.get("notes", ""),
                }
            )


def run_batch(
    specs: Sequence[RunSpec],
    phase_root: Path,
    args: argparse.Namespace,
    asha_state: Optional[StageASHAState] = None,
) -> List[Dict[str, Any]]:
    results = []
    for spec in specs:
        print(f"[RUN] {spec.run_id} | phase={spec.phase} stage={spec.stage} params={spec.params}")
        result = run_single_experiment(spec, phase_root, args, asha_state=asha_state)
        results.append(result)
    return results


def run_batch_with_stage_early_stop(
    specs: Sequence[RunSpec],
    phase_root: Path,
    args: argparse.Namespace,
    stage_name: str,
    improvement_delta: float = 0.005,
    early_stop_patience: int = 0,
    asha_state: Optional[StageASHAState] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not specs:
        return results

    total = len(specs)
    half_point = (total + 1) // 2
    best_selection_metric = -float("inf")
    best_at_half: Optional[float] = None
    stage_no_improve_checks = 0
    successful_statuses = {"ok", "resumed", "dry_run"}

    for i, spec in enumerate(specs, start=1):
        print(f"[RUN] {spec.run_id} | phase={spec.phase} stage={spec.stage} params={spec.params}")
        result = run_single_experiment(spec, phase_root, args, asha_state=asha_state)
        results.append(result)

        if result["status"] in successful_statuses:
            best_selection_metric = max(best_selection_metric, float(_selection_value(result)))

        if i == half_point:
            best_at_half = best_selection_metric

        if i > half_point and i < total and best_at_half is not None and not args.dry_run:
            if best_selection_metric < (best_at_half + improvement_delta):
                stage_no_improve_checks += 1
                if stage_no_improve_checks > int(max(0, early_stop_patience)):
                    print(
                        f"[STAGE_EARLY_STOP] {stage_name}: stopping after {i}/{total} runs "
                        f"(best_{ACTIVE_SELECTION_METRIC}={best_selection_metric:.4f}, "
                        f"needed >= {best_at_half + improvement_delta:.4f}, "
                        f"patience={early_stop_patience})"
                    )
                    break
            else:
                stage_no_improve_checks = 0

    return results


def pick_top(results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    candidates = [r for r in results if r.get("guardrail_pass", False)]
    if not candidates:
        candidates = [r for r in results if r["status"] in {"ok", "resumed", "dry_run"}]
    return rank_results(candidates)[:k]


def strict_pick_top(results: List[Dict[str, Any]], k: int, phase_name: str) -> List[Dict[str, Any]]:
    candidates = [r for r in results if r.get("guardrail_pass", False)]
    if candidates:
        return rank_results(candidates)[:k]

    successful = [r for r in results if r["status"] in {"ok", "resumed", "dry_run"}]
    if successful:
        abstentions = [float(r["abstention_rate"]) for r in successful]
        best_sel = max(float(_selection_value(r)) for r in successful)
        raise RuntimeError(
            f"{phase_name}: no guardrail-passing runs. "
            f"Best {ACTIVE_SELECTION_METRIC} among successful runs={best_sel:.4f}; "
            f"abstention_rate range=[{min(abstentions):.4f}, {max(abstentions):.4f}]. "
            "Adjust abstention guardrail or inspect threshold behavior."
        )

    raise RuntimeError(f"{phase_name}: no successful runs to select from.")


def dedupe_ranked_by_run_id(ranked: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep highest-ranked instance per run_id and return (kept, dropped_duplicates)."""
    seen: set = set()
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for row in ranked:
        run_id = str(row.get("run_id", ""))
        if run_id in seen:
            dropped.append(row)
            continue
        seen.add(run_id)
        kept.append(row)
    return kept, dropped


def prune_by_selection_metric_gap(results: List[Dict[str, Any]], max_gap: float) -> List[Dict[str, Any]]:
    successful = [r for r in results if r["status"] in {"ok", "resumed", "dry_run"}]
    if not successful:
        return results
    best = max(float(_selection_value(r)) for r in successful)
    cutoff = best - float(max_gap)
    return [r for r in results if r["status"] not in {"ok", "resumed", "dry_run"} or float(_selection_value(r)) >= cutoff]


def phase0_specs() -> List[RunSpec]:
    specs: List[RunSpec] = []
    for bs in [8, 16, 32]:
        specs.append(
            RunSpec(
                run_id=f"p0_bs{bs}",
                phase="phase0",
                stage="throughput_envelope",
                params={
                    "n_splits": 3,
                    "num_epochs": 20,
                    "early_stopping_patience": 6,
                    "batch_size": bs,
                },
                notes="Throughput envelope short run",
            )
        )
    return specs


def phase1_specs(batch_size: int, n_splits: int) -> List[RunSpec]:
    specs: List[RunSpec] = []
    idx = 0
    for mode in ["ratio", "sqrt_ratio"]:
        for clip_max in [2, 4, 8, 16]:
            for sampler_strength in ["off", "medium", "high", "full"]:
                idx += 1
                specs.append(
                    RunSpec(
                        run_id=(
                            f"p1_{idx:02d}_mode_{mode}_pwmax{clip_max}"
                            f"_sampler_{sampler_strength}"
                        ),
                        phase="phase1",
                        stage="phase1_imbalance",
                        params={
                            "batch_size": batch_size,
                            "n_splits": n_splits,
                            "num_epochs": 50,
                            "early_stopping_patience": 12,
                            "pos_weight_mode": mode,
                            "pos_weight_clip_max": clip_max,
                            "sampler_strength": sampler_strength,
                        },
                        notes="Imbalance mini-search",
                    )
                )
    return specs


def phase2_specs(batch_size: int, n_splits: int, imbalance_params: Dict[str, Any]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    # Log-spaced grid spanning ~40× in LR and ~48× in WD.
    # Covers the AdamW sweet-spot for a small frozen-embedding MLP (1e-4 to 1e-2).
    # WD upper bound 0.48 needed for large architectures (256/512 hidden) on ~400 docs.
    lrs = [2e-4, 5e-4, 1e-3, 3e-3, 8e-3]
    wds = [0.01, 0.04, 0.16, 0.48]

    idx = 0
    for lr in lrs:
        for wd in wds:
            idx += 1
            specs.append(
                RunSpec(
                    run_id=f"p2_{idx:02d}_lr{lr}_wd{wd}",
                    phase="phase2",
                    stage="phase2_lr_wd",
                    params={
                        "batch_size": batch_size,
                        "n_splits": n_splits,
                        "num_epochs": 80,
                        "early_stopping_patience": 20,
                        "learning_rate": lr,
                        "weight_decay": wd,
                        "dropout": 0.3,
                        "intermediate_size": 128,
                        "pos_weight_mode": imbalance_params["pos_weight_mode"],
                        "pos_weight_clip_max": imbalance_params["pos_weight_clip_max"],
                        "sampler_strength": imbalance_params["sampler_strength"],
                    },
                    notes="Core quality stage1",
                )
            )
    return specs


def phase3_specs(
    batch_size: int,
    n_splits: int,
    best_lr: float,
    best_wd: float,
    imbalance_params: Dict[str, Any],
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    # Architecture search: intermediate_size=0 is direct linear (768→K), >0 adds a hidden layer.
    dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    mids = [0, 64, 128, 256, 512]

    idx = 0
    for d in dropouts:
        for m in mids:
            idx += 1
            specs.append(
                RunSpec(
                    run_id=f"p3_{idx:02d}_do{d}_mid{m}",
                    phase="phase3",
                    stage="phase3_architecture",
                    params={
                        "batch_size": batch_size,
                        "n_splits": n_splits,
                        "num_epochs": 80,
                        "early_stopping_patience": 20,
                        "learning_rate": best_lr,
                        "weight_decay": best_wd,
                        "dropout": d,
                        "intermediate_size": m,
                        "pos_weight_mode": imbalance_params["pos_weight_mode"],
                        "pos_weight_clip_max": imbalance_params["pos_weight_clip_max"],
                        "sampler_strength": imbalance_params["sampler_strength"],
                    },
                    notes="Core quality stage2",
                )
            )
    return specs


def phase4_specs(top2: List[Dict[str, Any]]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for i, base in enumerate(top2, start=1):
        p = dict(base["params"])
        p["n_splits"] = 5
        p["num_epochs"] = 150
        p["early_stopping_patience"] = 30
        specs.append(
            RunSpec(
                run_id=f"p4_finalist_{i}",
                phase="phase4",
                stage="full_fidelity_confirmation",
                params=p,
                notes=f"Full-fidelity confirmation from {base['run_id']}",
            )
        )
    return specs


def write_summary(output_root: Path, payload: Dict[str, Any]) -> None:
    write_json(output_root / "hpo_summary.json", payload)


def _langfuse_post_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["LANGFUSE_TRACING_ENABLED"] = "true"
    return env


def post_winner_from_artifacts(
    winner: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    winner_dir = Path(winner["run_dir"]).resolve()
    workflow = str(args.workflow)
    prefix = WORKFLOW_REGISTRY[workflow]["oof_run_name_prefix"]
    task_name = (WORKFLOW_REGISTRY[workflow].get("fixed_hpo_params") or {}).get("task")
    run_name = f"{prefix}_oof_hpo_winner_{Path(args.output_root).name}"
    cmd = [
        args.python,
        args.trainer_script,
        "--checkpoint-dir",
        str(winner_dir),
        "--dataset-name",
        args.dataset_name,
        "--dataset-cache-path",
        args.dataset_cache_path,
        "--post-oof-from-artifacts",
        "--post-oof-to-langfuse",
        "--langfuse-oof-run-name",
        run_name,
    ]
    if task_name:
        cmd.extend(["--task", str(task_name)])
    if args.fail_on_langfuse_post_error:
        cmd.append("--fail-on-langfuse-post-error")

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_langfuse_post_env())
    runtime_sec = time.time() - start

    (winner_dir / "winner_post_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (winner_dir / "winner_post_stderr.log").write_text(proc.stderr, encoding="utf-8")

    status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
    return {
        "status": status,
        "runtime_sec": runtime_sec,
        "run_name": run_name,
        "command": cmd,
        "stdout_log": str(winner_dir / "winner_post_stdout.log"),
        "stderr_log": str(winner_dir / "winner_post_stderr.log"),
    }


def resolve_workflow(args: argparse.Namespace) -> str:
    return str(args.workflow)


def _infer_effective_task_slug(
    args: argparse.Namespace,
    workflow: str,
    workflow_spec: Optional[Dict[str, Any]] = None,
) -> str:
    # Highest priority: explicit CLI override.
    cli_task = getattr(args, "task", None)
    if cli_task:
        return str(cli_task)

    spec = workflow_spec or (WORKFLOW_REGISTRY.get(workflow) if isinstance(WORKFLOW_REGISTRY, dict) else None) or {}

    # Next: infer from dataset name, if provided.
    dataset_name = str(getattr(args, "dataset_name", None) or spec.get("dataset") or "").strip().lower()
    if dataset_name:
        if "golden_set_" in dataset_name:
            return dataset_name.split("golden_set_")[-1]
        return dataset_name

    # Fallback: workflow-fixed task.
    fixed_params = spec.get("fixed_hpo_params") or {}
    task = fixed_params.get("task")
    if task:
        return str(task)
    return "generic_task"


def _infer_output_root_task_slug(
    args: argparse.Namespace,
    workflow: str,
    workflow_spec: Optional[Dict[str, Any]] = None,
) -> str:
    return _infer_effective_task_slug(args, workflow, workflow_spec)


def _ensure_output_root_contains_task(output_root: str, task_slug: str) -> str:
    root = str(output_root).strip()
    task = str(task_slug).strip()
    if not root or not task:
        return root
    if task.lower() in Path(root).name.lower():
        return root
    return f"{root}_{task}"


def _workflow_task_type(workflow: str) -> str:
    spec = WORKFLOW_REGISTRY.get(workflow) if isinstance(WORKFLOW_REGISTRY, dict) else None
    if isinstance(spec, dict):
        task_type = str(spec.get("task_type", "")).strip().lower()
        if task_type in {"multilabel", "multiclass"}:
            return task_type
    return "multilabel"


def _is_multilabel_workflow(workflow: str) -> bool:
    return _workflow_task_type(workflow) == "multilabel"


def _is_multilabel_frozen_workflow(workflow: str) -> bool:
    return _workflow_task_type(workflow) == "multilabel" and "frozen" in workflow


def _allowed_checkpoint_metrics_for_workflow(workflow: str) -> Sequence[str]:
    task_type = _workflow_task_type(workflow)
    if task_type == "multilabel":
        return ("macro_pr_auc", "val_loss", "macro_f1_0_5")
    return ("macro_pr_auc", "macro_f1", "exact_match_accuracy", "val_loss")


def _allowed_threshold_metrics_for_workflow(workflow: str) -> Sequence[str]:
    task_type = _workflow_task_type(workflow)
    if task_type == "multilabel":
        return ("macro_f1", "macro_pr_auc")
    return ("macro_f1", "exact_match_accuracy")


def _normalize_checkpoint_metric_for_workflow(workflow: str, metric: str) -> str:
    normalized = str(metric)
    aliases = WORKFLOW_REGISTRY.get(workflow, {}).get("checkpoint_metric_aliases") or {}
    if normalized in aliases:
        return str(aliases[normalized])
    # Multilabel trainers use/expect fixed-threshold naming for checkpoint F1.
    if _is_multilabel_workflow(workflow) and normalized == "macro_f1":
        return "macro_f1_0_5"
    # Multiclass trainer uses macro_f1 naming.
    if (not _is_multilabel_workflow(workflow)) and normalized == "macro_f1_0_5":
        return "macro_f1"
    return normalized


def normalize_and_validate_metric_overrides(args: argparse.Namespace) -> None:
    workflow = str(args.workflow)
    if bool(getattr(args, "use_temperature_scaling", False)) and not _is_multilabel_frozen_workflow(workflow):
        raise ValueError(
            "--use-temperature-scaling is only supported for frozen multilabel workflows."
        )

    ck_metric = _normalize_checkpoint_metric_for_workflow(workflow, str(args.checkpoint_selection_metric))
    allowed_ck = set(_allowed_checkpoint_metrics_for_workflow(workflow))
    if ck_metric not in allowed_ck:
        raise ValueError(
            f"Unsupported --checkpoint-selection-metric='{args.checkpoint_selection_metric}' "
            f"for workflow={workflow}. Allowed: {sorted(allowed_ck)}"
        )
    args.checkpoint_selection_metric = ck_metric

    # selection_metric is a global HPO ranking metric and is validated by argparse choices.
    # We only ensure we never forward an incompatible threshold metric later.


def lora_base_params(args: argparse.Namespace) -> Dict[str, Any]:
    base = {
        "n_splits": 3,
        "max_folds": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "chunk_micro_batch_size": 24,
        "num_epochs": 9,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.0,
        "learning_rate_head": 8e-4,
        "weight_decay": 0.05,
        "dropout": 0.15,
        "intermediate_size": 128,
        "max_seq_length": 512,
        "chunk_overlap": 0,
        "max_chunks_per_doc": 144,
        "chunk_pooling": "max",
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "pos_weight_mode": "ratio",
        "pos_weight_clip_max": 4.0,
        "sampler_strength": "low",
    }
    if args.tokenized_cache_path:
        base["tokenized_cache_path"] = args.tokenized_cache_path
    return base


def lora_stage_a_specs(base: Dict[str, Any], seed: int, n_samples: int = 10) -> List[RunSpec]:
    # Random exploration over high-impact LoRA/training knobs.
    rng = np.random.default_rng(int(seed))
    space: List[Dict[str, Any]] = []
    base_head_lr = float(base.get("learning_rate_head", 8e-4))
    head_lr_values = [base_head_lr, base_head_lr * 0.5]
    for target_set in ["attn"]:
        for lora_r in [8, 16]:
            for lr_lora in [1e-4, 2e-4]:
                for wd in [0.03]:
                    for do in [0.1]:
                        for intermediate_size in [128]:
                            for lr_head in head_lr_values:
                                p = dict(base)
                                p.update(
                                    {
                                        "lora_target_set": target_set,
                                        "lora_r": lora_r,
                                        "learning_rate_lora": lr_lora,
                                        "learning_rate_head": lr_head,
                                        "weight_decay": wd,
                                        "dropout": do,
                                        "intermediate_size": intermediate_size,
                                        "chunk_micro_batch_size": 24,
                                        "early_stopping_min_delta": 0.01,
                                    }
                                )
                                space.append(p)

    if not space:
        return []
    take = int(max(1, min(n_samples, len(space))))
    chosen_idx = rng.choice(len(space), size=take, replace=False)
    specs: List[RunSpec] = []
    for idx, ci in enumerate(chosen_idx, start=1):
        specs.append(
            RunSpec(
                run_id=f"lora_a_{idx:02d}",
                phase="lora_a",
                stage="random_explore_lora_core",
                params=space[int(ci)],
                notes="LoRA Stage A random search",
            )
        )
    return specs


def lora_stage_b_specs(
    base_from_a: Dict[str, Any],
    seed: int,
    n_samples: int = 4,
    run_prefix: str = "lora_b",
) -> List[RunSpec]:
    # Random imbalance exploration under fixed short budget.
    rng = np.random.default_rng(int(seed) + 1009)
    combos: List[Tuple[str, float, str]] = []
    for mode in ["ratio"]:
        for clip_max in [2.0, 4.0]:
            for sampler in ["off", "low", "medium"]:
                combos.append((mode, clip_max, sampler))

    take = int(max(1, min(n_samples, len(combos))))
    chosen_idx = rng.choice(len(combos), size=take, replace=False)
    specs: List[RunSpec] = []
    for idx, ci in enumerate(chosen_idx, start=1):
        mode, clip_max, sampler = combos[int(ci)]
        p = dict(base_from_a)
        p.update(
            {
                "pos_weight_mode": mode,
                "pos_weight_clip_max": clip_max,
                "sampler_strength": sampler,
            }
        )
        specs.append(
            RunSpec(
                run_id=f"{run_prefix}_{idx:02d}",
                phase="lora_b",
                stage="random_explore_imbalance",
                params=p,
                notes="LoRA Stage B random search",
            )
        )
    return specs


def lora_stage_c_specs(promoted_from_b: List[Dict[str, Any]]) -> List[RunSpec]:
    # Multi-fidelity promotion stage:
    # Re-run top 2-3 shortlisted configs at longer budget (12/3).
    specs: List[RunSpec] = []
    for idx, base in enumerate(promoted_from_b, start=1):
        p = dict(base["params"])
        p["n_splits"] = 2
        p["max_folds"] = None
        p["num_epochs"] = 12
        p["early_stopping_patience"] = 3
        specs.append(
            RunSpec(
                run_id=f"lora_c_{idx:02d}",
                phase="lora_c",
                stage="multi_fidelity_promotion_12ep",
                params=p,
                notes=f"LoRA Stage C promotion from {base['run_id']}",
            )
        )
    return specs


def lora_stage_d_specs(finalists: List[Dict[str, Any]]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for i, finalist in enumerate(finalists, start=1):
        p = dict(finalist["params"])
        p["n_splits"] = 3
        p["max_folds"] = None
        p["early_stopping_min_delta"] = 0.0
        specs.append(
            RunSpec(
                run_id=f"lora_d_finalist_{i}",
                phase="lora_d",
                stage="full_fidelity_confirmation",
                params=p,
                notes=f"LoRA Stage D confirmation from {finalist['run_id']}",
            )
        )
    return specs


def run_lora_parity_workflow(
    args: argparse.Namespace,
    output_root: Path,
    summary: Dict[str, Any],
) -> None:
    """LoRA workflow aligned with frozen orchestration style (no ASHA/clear-winner heuristics)."""
    base = lora_base_params(args)
    # Keep Stage A defaults explicit for parity workflows.
    base["pos_weight_mode"] = "ratio"
    base["pos_weight_clip_max"] = 4.0
    base["sampler_strength"] = "low"
    base["early_stopping_patience"] = 3
    base["chunk_pooling"] = "max"
    stage_gate_min_gain = 0.005
    stage_gate_max_minority_drop = 0.02
    rare_precision_collapse_delta = 0.05
    rare_recall_spike_delta = 0.05
    summary["monitoring"] = {
        "stage_gate_min_macro_f1_gain": stage_gate_min_gain,
        "stage_gate_max_minority_mean_f1_drop": stage_gate_max_minority_drop,
        "rare_precision_collapse_delta": rare_precision_collapse_delta,
        "rare_recall_spike_delta": rare_recall_spike_delta,
    }

    # Stage A (frozen-style initial shortlist)
    stage_a_root = output_root / "lora_a"
    ensure_dir(stage_a_root)
    a_results = run_batch(lora_stage_a_specs(base, seed=args.seed, n_samples=5), stage_a_root, args)
    apply_guardrail(a_results, args.abstention_min, args.abstention_max)
    a_ranked = rank_results(a_results)
    write_phase_outputs(stage_a_root, "lora_a", a_ranked)
    a_selected = top_successful_candidate(a_ranked)
    a_health = build_stage_health(
        stage_name="lora_a",
        results=a_results,
        ranked=a_ranked,
        selected=a_selected,
        previous_selected=None,
        abstention_min=args.abstention_min,
        abstention_max=args.abstention_max,
        min_macro_gain=stage_gate_min_gain,
        max_minority_drop=stage_gate_max_minority_drop,
        precision_collapse_delta=rare_precision_collapse_delta,
        recall_spike_delta=rare_recall_spike_delta,
        enforce_gate=False,
    )
    a_health_path = write_stage_health(stage_a_root, "lora_a", a_health)
    a_top2 = strict_pick_top(a_ranked, 2, "LoRA Stage A")
    summary["phases"]["lora_a"] = {
        "top": a_ranked[:5],
        "selected_for_stage_b": a_top2,
        "health_file": a_health_path,
        "health_overview": {
            "alerts": [a["code"] for a in a_health.get("alerts", [])],
            "gate": a_health.get("gate", {}),
        },
    }
    if args.stop_after_phase == "lora_a":
        write_summary(output_root, summary)
        return

    # Stage B (imbalance exploration from A finalists)
    stage_b_root = output_root / "lora_b"
    ensure_dir(stage_b_root)
    b_results: List[Dict[str, Any]] = []
    for branch_idx, a_base in enumerate(a_top2, start=1):
        branch_specs = lora_stage_b_specs(
            a_base["params"],
            seed=args.seed + (branch_idx * 1000),
            n_samples=3,
            run_prefix=f"lora_b_b{branch_idx}",
        )
        b_results.extend(run_batch(branch_specs, stage_b_root, args))
    apply_guardrail(b_results, args.abstention_min, args.abstention_max)
    b_ranked = rank_results(b_results)
    write_phase_outputs(stage_b_root, "lora_b", b_ranked)
    b_selected = top_successful_candidate(b_ranked)
    b_health = build_stage_health(
        stage_name="lora_b",
        results=b_results,
        ranked=b_ranked,
        selected=b_selected,
        previous_selected=a_selected,
        abstention_min=args.abstention_min,
        abstention_max=args.abstention_max,
        min_macro_gain=stage_gate_min_gain,
        max_minority_drop=stage_gate_max_minority_drop,
        precision_collapse_delta=rare_precision_collapse_delta,
        recall_spike_delta=rare_recall_spike_delta,
        enforce_gate=False,
    )
    b_health_path = write_stage_health(stage_b_root, "lora_b", b_health)
    b_top2 = strict_pick_top(b_ranked, 2, "LoRA Stage B")
    summary["phases"]["lora_b"] = {
        "top": b_ranked[:5],
        "promoted_to_stage_c": b_top2,
        "health_file": b_health_path,
        "health_overview": {
            "alerts": [a["code"] for a in b_health.get("alerts", [])],
            "gate": b_health.get("gate", {}),
        },
    }
    if args.stop_after_phase == "lora_b":
        write_summary(output_root, summary)
        return

    # Stage C (multi-fidelity promotion)
    stage_c_root = output_root / "lora_c"
    ensure_dir(stage_c_root)
    c_results = run_batch(lora_stage_c_specs(b_top2), stage_c_root, args)
    apply_guardrail(c_results, args.abstention_min, args.abstention_max)
    c_ranked = rank_results(c_results)
    write_phase_outputs(stage_c_root, "lora_c", c_ranked)
    c_selected = top_successful_candidate(c_ranked)
    c_health = build_stage_health(
        stage_name="lora_c",
        results=c_results,
        ranked=c_ranked,
        selected=c_selected,
        previous_selected=b_selected,
        abstention_min=args.abstention_min,
        abstention_max=args.abstention_max,
        min_macro_gain=stage_gate_min_gain,
        max_minority_drop=stage_gate_max_minority_drop,
        precision_collapse_delta=rare_precision_collapse_delta,
        recall_spike_delta=rare_recall_spike_delta,
        enforce_gate=False,
    )
    c_health_path = write_stage_health(stage_c_root, "lora_c", c_health)
    c_top1 = strict_pick_top(c_ranked, 1, "LoRA Stage C")
    summary["phases"]["lora_c"] = {
        "top": c_ranked[:5],
        "selected_for_final": c_top1,
        "health_file": c_health_path,
        "health_overview": {
            "alerts": [a["code"] for a in c_health.get("alerts", [])],
            "gate": c_health.get("gate", {}),
        },
    }
    if args.stop_after_phase == "lora_c":
        write_summary(output_root, summary)
        return

    # Stage D (full-fidelity confirmation)
    stage_d_root = output_root / "lora_d"
    ensure_dir(stage_d_root)
    d_results = run_batch(lora_stage_d_specs(c_top1), stage_d_root, args)
    apply_guardrail(d_results, args.abstention_min, args.abstention_max)
    d_ranked = rank_results(d_results)
    write_phase_outputs(stage_d_root, "lora_d", d_ranked)
    d_selected = top_successful_candidate(d_ranked)
    d_health = build_stage_health(
        stage_name="lora_d",
        results=d_results,
        ranked=d_ranked,
        selected=d_selected,
        previous_selected=c_selected,
        abstention_min=args.abstention_min,
        abstention_max=args.abstention_max,
        min_macro_gain=0.0,
        max_minority_drop=1.0,
        precision_collapse_delta=rare_precision_collapse_delta,
        recall_spike_delta=rare_recall_spike_delta,
        enforce_gate=False,
    )
    d_health_path = write_stage_health(stage_d_root, "lora_d", d_health)
    winner = strict_pick_top(d_ranked, 1, "LoRA Stage D")[0]
    summary["phases"]["lora_d"] = {
        "leaderboard": d_ranked,
        "winner": winner,
        "health_file": d_health_path,
        "health_overview": {
            "alerts": [a["code"] for a in d_health.get("alerts", [])],
            "gate": d_health.get("gate", {}),
        },
    }
    summary["winner"] = {
        "run_id": winner["run_id"],
        "run_dir": winner["run_dir"],
        "params": winner["params"],
        "macro_f1": winner["macro_f1"],
        "macro_precision": winner.get("macro_precision", 0.0),
        "macro_recall": winner.get("macro_recall", 0.0),
        "macro_pr_auc": winner.get("macro_pr_auc", winner.get("cv_macro_pr_auc_mean", 0.0)),
        "minority_macro_pr_auc": winner.get("minority_macro_pr_auc", 0.0),
        "minority_mean_f1": winner["minority_mean_f1"],
        "minority_mean_precision": winner.get("minority_mean_precision", 0.0),
        "minority_mean_recall": winner.get("minority_mean_recall", 0.0),
        "exact_match_accuracy": winner["exact_match_accuracy"],
        "abstention_rate": winner["abstention_rate"],
        "peak_process_rss_mb": winner.get("peak_process_rss_mb", 0.0),
        "peak_system_cpu_percent": winner.get("peak_system_cpu_percent", 0.0),
        "peak_system_ram_percent": winner.get("peak_system_ram_percent", 0.0),
        "peak_device_memory_mb": winner.get("peak_device_memory_mb", 0.0),
        "system_metrics_file": winner.get("system_metrics_file", ""),
        "runtime_sec": winner["runtime_sec"],
    }

    post_hpo_results = run_post_hpo_sequence(winner["run_dir"], str(args.workflow), args)
    summary["post_hpo_sequence"] = post_hpo_results
    if args.post_winner_to_langfuse and not args.dry_run:
        post_result = post_winner_from_artifacts(winner, args)
        summary["winner_post_to_langfuse"] = post_result
        if post_result["status"] != "ok":
            print(
                "[WARN] Winner post-to-Langfuse failed. "
                f"See logs: {post_result['stdout_log']} | {post_result['stderr_log']}"
            )
    else:
        summary["winner_post_to_langfuse"] = {
            "status": "skipped",
            "reason": "disabled_or_dry_run",
        }
    write_summary(output_root, summary)

    print("\n=== LORA HPO PARITY COMPLETE ===")
    print(f"Winner run_id: {summary['winner']['run_id']}")
    print(f"Winner dir: {summary['winner']['run_dir']}")
    print(f"Winner params: {summary['winner']['params']}")


def run_post_hpo_sequence(
    winner_dir: str,
    workflow: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run post-HPO steps: threshold optimization, human eval, routing threshold."""
    results: Dict[str, Any] = {}

    # Read winner artifacts once so we never duplicate class lists/grid settings in code.
    _winner_cv_path = Path(winner_dir) / "cv_results.json"
    _winner_cv_payload: Dict[str, Any] = {}
    if _winner_cv_path.exists():
        try:
            _winner_cv_payload = read_json(_winner_cv_path)
        except Exception:
            _winner_cv_payload = {}

    _winner_label_mapping: Dict[str, Any] = _winner_cv_payload.get("label_mapping") or {}
    _task_type_from_mapping = str(_winner_label_mapping.get("task_type") or "")
    _train_labels_from_mapping: List[str] = list(_winner_label_mapping.get("train_labels") or [])
    _abstain_from_mapping = str(
        _winner_label_mapping.get("abstain_label")
        or _winner_label_mapping.get("raw_abstain_label")
        or ""
    )
    if not _train_labels_from_mapping:
        _train_labels_from_mapping = list(WORKFLOW_REGISTRY.get(workflow, {}).get("train_labels") or [])
    if not _abstain_from_mapping:
        _abstain_from_mapping = str(WORKFLOW_REGISTRY.get(workflow, {}).get("abstention_label") or "")

    _task_name = str((WORKFLOW_REGISTRY.get(workflow, {}).get("fixed_hpo_params") or {}).get("task") or "")
    _task_type = _task_type_from_mapping or WORKFLOW_REGISTRY[workflow]["task_type"]
    _categories = _train_labels_from_mapping
    _abstention_label = getattr(args, "abstain_label", None) or _abstain_from_mapping or "abstain"
    _winner_path = Path(winner_dir)

    # Winner-only threshold optimization for multilabel and multiclass.
    if _task_type in {"multilabel", "multiclass"} and not args.dry_run:
        if not _categories:
            results["threshold_optimization"] = {"status": "skipped", "reason": "no_categories"}
        else:
            metric = "macro_f1"
            run_cfg_path = _winner_path / "run_config.json"
            if run_cfg_path.exists():
                try:
                    run_cfg_payload = read_json(run_cfg_path)
                    metric = str(run_cfg_payload.get("threshold_selection_metric") or "macro_f1")
                except Exception:
                    metric = "macro_f1"
            threshold_optimizer_script = _resolve_existing_script_path(
                str(getattr(args, "threshold_optimizer_script", "optimize_abstention_thresholds.py")),
                "Threshold optimizer script",
            )
            cmd = [
                args.python,
                threshold_optimizer_script,
                "--checkpoint-dir", str(winner_dir),
                "--abstention-label", _abstention_label,
                "--categories", *_categories,
                "--task-type", _task_type,
                "--metric", metric,
            ]
            if _task_type == "multilabel":
                cmd.append("--use-temperature-scaling")
                print(f"\n[POST_HPO] Running winner threshold optimization with temperature scaling: {winner_dir}")
            else:
                print(f"\n[POST_HPO] Running winner threshold optimization: {winner_dir}")
            start = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, env=_langfuse_post_env())
            runtime = time.time() - start
            (_winner_path / "threshold_opt_stdout.log").write_text(proc.stdout, encoding="utf-8")
            (_winner_path / "threshold_opt_stderr.log").write_text(proc.stderr, encoding="utf-8")
            status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
            results["threshold_optimization"] = {
                "status": status,
                "runtime_sec": runtime,
                "command": cmd,
                "stdout_log": str(_winner_path / "threshold_opt_stdout.log"),
                "stderr_log": str(_winner_path / "threshold_opt_stderr.log"),
                "temperature_scaling_enabled": bool(_task_type == "multilabel"),
                "task_type": _task_type,
            }
            if status != "ok":
                print(
                    f"[POST_HPO][WARN] Threshold optimization failed (exit={proc.returncode}). "
                    f"See {results['threshold_optimization']['stderr_log']}"
                )
    else:
        results["threshold_optimization"] = {
            "status": "skipped",
            "reason": "unsupported_task_type_or_dry_run",
        }

    # Human evaluation
    routing_llm_dataset_name: Optional[str] = None
    routing_llm_run_name: Optional[str] = None
    if getattr(args, "run_human_eval", True) and not args.dry_run:
        human_eval_dataset = getattr(args, "human_eval_dataset", None) or WORKFLOW_REGISTRY[workflow].get("human_eval_dataset")
        if not human_eval_dataset:
            print("[POST_HPO][WARN] --human-eval-dataset not set; skipping human eval.")
            results["human_eval"] = {"status": "skipped", "reason": "no_dataset_specified"}
        else:
            if not _categories:
                print("[POST_HPO][WARN] Missing --categories for human eval; skipping human eval.")
                results["human_eval"] = {"status": "skipped", "reason": "no_categories"}
                return results

            human_eval_script = _resolve_existing_script_path(
                str(getattr(args, "human_eval_script", "test_model.py")),
                "Human evaluation script",
            )
            cmd = [
                args.python,
                human_eval_script,
                "--checkpoint-dir", str(winner_dir),
                "--task-type", _task_type,
                "--categories", *_categories,
                "--abstention-label", _abstention_label,
                "--dataset-name", human_eval_dataset,
            ]
            print(f"\n[POST_HPO] Running human evaluation on winner: {winner_dir}")
            start = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, env=_langfuse_post_env())
            runtime = time.time() - start
            (_winner_path / "human_eval_stdout.log").write_text(proc.stdout, encoding="utf-8")
            (_winner_path / "human_eval_stderr.log").write_text(proc.stderr, encoding="utf-8")
            status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
            results["human_eval"] = {
                "status": status,
                "runtime_sec": runtime,
                "dataset": human_eval_dataset,
                "command": cmd,
                "stdout_log": str(_winner_path / "human_eval_stdout.log"),
                "stderr_log": str(_winner_path / "human_eval_stderr.log"),
            }
            if status == "ok":
                human_eval_results_path = _winner_path / "human_eval_results.json"
                if human_eval_results_path.exists():
                    try:
                        human_eval_payload = read_json(human_eval_results_path)
                        routing_llm_dataset_name = str(
                            human_eval_payload.get("test_dataset")
                            or human_eval_dataset
                            or ""
                        )
                        _langfuse_run_name = human_eval_payload.get("langfuse_run_name")
                        routing_llm_run_name = str(_langfuse_run_name) if _langfuse_run_name else None
                        results["human_eval"]["results_file"] = str(human_eval_results_path)
                        results["human_eval"]["langfuse_run_name"] = routing_llm_run_name
                    except Exception as exc:
                        print(f"[POST_HPO][WARN] Failed to parse human_eval_results.json: {exc}")
                print(f"[POST_HPO] Human evaluation complete.")
            else:
                print(f"[POST_HPO][WARN] Human evaluation failed (exit={proc.returncode}). "
                      f"See {results['human_eval']['stderr_log']}")
    else:
        results["human_eval"] = {"status": "skipped"}

    if (not routing_llm_dataset_name or not routing_llm_run_name):
        human_eval_results_path = _winner_path / "human_eval_results.json"
        if human_eval_results_path.exists():
            try:
                human_eval_payload = read_json(human_eval_results_path)
                routing_llm_dataset_name = str(
                    human_eval_payload.get("test_dataset")
                    or routing_llm_dataset_name
                    or ""
                )
                _langfuse_run_name = human_eval_payload.get("langfuse_run_name")
                if _langfuse_run_name:
                    routing_llm_run_name = str(_langfuse_run_name)
            except Exception:
                pass

    return results


def run_frozen_workflow(args: argparse.Namespace, output_root: Path, summary: Dict[str, Any]) -> None:
    stage_gate_min_gain = 0.005
    stage_gate_max_minority_drop = 0.02
    rare_precision_collapse_delta = 0.05
    rare_recall_spike_delta = 0.05
    summary["monitoring"] = {
        "stage_gate_min_macro_f1_gain": stage_gate_min_gain,
        "stage_gate_max_minority_mean_f1_drop": stage_gate_max_minority_drop,
        "rare_precision_collapse_delta": rare_precision_collapse_delta,
        "rare_recall_spike_delta": rare_recall_spike_delta,
    }

    # Phase 0
    phase0_root = output_root / "phase0"
    ensure_dir(phase0_root)
    p0_results = run_batch(phase0_specs(), phase0_root, args)
    apply_guardrail(p0_results, args.abstention_min, args.abstention_max)
    p0_ranked = rank_results(p0_results)
    write_phase_outputs(phase0_root, "phase0", p0_ranked)
    p0_selected = top_successful_candidate(p0_ranked)
    p0_health = build_stage_health(
        stage_name="phase0",
        results=p0_results,
        ranked=p0_ranked,
        selected=p0_selected,
        previous_selected=None,
        abstention_min=args.abstention_min,
        abstention_max=args.abstention_max,
        min_macro_gain=stage_gate_min_gain,
        max_minority_drop=stage_gate_max_minority_drop,
        precision_collapse_delta=rare_precision_collapse_delta,
        recall_spike_delta=rare_recall_spike_delta,
        enforce_gate=False,
    )
    p0_health_path = write_stage_health(phase0_root, "phase0", p0_health)

    success_statuses = {"ok", "resumed"}
    if args.dry_run:
        success_statuses.add("dry_run")

    p0_ok = [r for r in p0_ranked if r["status"] in success_statuses]
    if not p0_ok:
        raise RuntimeError("Phase 0 produced no successful runs.")

    p0_guardrail = [r for r in p0_ok if r["guardrail_pass"]]
    if not p0_guardrail:
        raise RuntimeError(
            "Phase 0 produced no guardrail-passing runs. "
            "Adjust --abstention-min/--abstention-max or inspect data/threshold behavior."
        )

    best_p0_metric = max(_selection_value(r) for r in p0_guardrail)
    feasible = [r for r in p0_guardrail if _selection_value(r) >= (best_p0_metric - 0.01)]
    if not feasible:
        feasible = p0_guardrail

    locked_batch = max(feasible, key=lambda x: int(x["params"]["batch_size"]))["params"]["batch_size"]

    search_n_splits = 3
    summary["phases"]["phase0"] = {
        "locked_batch_size": locked_batch,
        "search_n_splits": search_n_splits,
        "top": p0_ranked[:3],
        "health_file": p0_health_path,
        "health_overview": {
            "alerts": [a["code"] for a in p0_health.get("alerts", [])],
            "gate": p0_health.get("gate", {}),
        },
    }

    if args.stop_after_phase == "phase0":
        write_summary(output_root, summary)
        return

    # Phase 1
    phase1_root = output_root / "phase1"
    ensure_dir(phase1_root)
    p1_specs_list = phase1_specs(locked_batch, search_n_splits)
    p1_results = run_batch(p1_specs_list, phase1_root, args)
    apply_guardrail(p1_results, args.abstention_min, args.abstention_max)
    p1_ranked = rank_results(p1_results)
    write_phase_outputs(phase1_root, "phase1", p1_ranked)
    p1_selected = top_successful_candidate(p1_ranked)
    p1_health = build_stage_health(
        stage_name="phase1",
        results=p1_results,
        ranked=p1_ranked,
        selected=p1_selected,
        previous_selected=p0_selected,
        abstention_min=args.abstention_min,
        abstention_max=args.abstention_max,
        min_macro_gain=stage_gate_min_gain,
        max_minority_drop=stage_gate_max_minority_drop,
        precision_collapse_delta=rare_precision_collapse_delta,
        recall_spike_delta=rare_recall_spike_delta,
        enforce_gate=False,
    )
    p1_health_path = write_stage_health(phase1_root, "phase1", p1_health)

    p1_top2 = strict_pick_top(p1_ranked, 2, "Phase 1")

    selected_imbalance = [
        {
            "pos_weight_mode": str(r["params"]["pos_weight_mode"]),
            "pos_weight_clip_max": float(r["params"]["pos_weight_clip_max"]),
            "sampler_strength": str(r["params"]["sampler_strength"]),
            "source_run_id": r["run_id"],
            "macro_f1": r["macro_f1"],
        }
        for r in p1_top2
    ]
    summary["phases"]["phase1"] = {
        "top": p1_ranked[:5],
        "selected": selected_imbalance,
        "health_file": p1_health_path,
        "health_overview": {
            "alerts": [a["code"] for a in p1_health.get("alerts", [])],
            "gate": p1_health.get("gate", {}),
        },
    }

    if args.stop_after_phase == "phase1":
        write_summary(output_root, summary)
        return

    # Phase 2/3 (branch per selected imbalance config)
    phase23_branches: List[Dict[str, Any]] = []
    for idx, imbalance_cfg in enumerate(selected_imbalance, start=1):
        branch_id = (
            f"branch{idx}_mode_{imbalance_cfg['pos_weight_mode']}"
            f"_sampler_{imbalance_cfg['sampler_strength']}"
            f"_pwmax{str(imbalance_cfg['pos_weight_clip_max']).replace('.', 'p')}"
        )
        phase23_branch_root = output_root / "phase2_3" / branch_id
        ensure_dir(phase23_branch_root)

        p2_specs_list = phase2_specs(locked_batch, search_n_splits, imbalance_cfg)
        p2_results = run_batch(
            p2_specs_list,
            phase23_branch_root,
            args,
        )
        apply_guardrail(p2_results, args.abstention_min, args.abstention_max)
        p2_ranked = rank_results(p2_results)
        write_phase_outputs(phase23_branch_root, "phase2", p2_ranked)
        p1_source_run = next(
            (r for r in p1_ranked if r["run_id"] == imbalance_cfg.get("source_run_id")),
            p1_selected,
        )
        p2_selected = top_successful_candidate(p2_ranked)
        p2_health = build_stage_health(
            stage_name=f"phase2_{branch_id}",
            results=p2_results,
            ranked=p2_ranked,
            selected=p2_selected,
            previous_selected=p1_source_run,
            abstention_min=args.abstention_min,
            abstention_max=args.abstention_max,
            min_macro_gain=stage_gate_min_gain,
            max_minority_drop=stage_gate_max_minority_drop,
            precision_collapse_delta=rare_precision_collapse_delta,
            recall_spike_delta=rare_recall_spike_delta,
            enforce_gate=False,
        )
        p2_health_path = write_stage_health(phase23_branch_root, "phase2", p2_health)

        p2_top = strict_pick_top(p2_ranked, 4, f"Phase 2 ({branch_id})")

        if args.stop_after_phase == "phase2":
            phase23_branches.append(
                {
                    "branch_id": branch_id,
                    "imbalance": imbalance_cfg,
                    "stage1_top": p2_ranked[:4],
                    "stage2_subbranches": [],
                    "phase3_top3": [],
                    "stage1_health_file": p2_health_path,
                    "stage1_health_overview": {
                        "alerts": [a["code"] for a in p2_health.get("alerts", [])],
                        "gate": p2_health.get("gate", {}),
                    },
                    "phase3_skipped": True,
                    "phase3_skip_reason": "stop_after_phase_phase2",
                }
            )
            continue

        # Run phase 3 independently for each of the top-2 LR/WD combos so that
        # the architecture search is not biased toward the single best LR found
        # at the fixed intermediate_size=128 used in phase 2.
        p3_subbranches: List[Dict[str, Any]] = []
        all_p3_results: List[Dict[str, Any]] = []

        for lrwd_idx, lrwd_cand in enumerate(p2_top[:2], start=1):
            sub_lr = float(lrwd_cand["params"]["learning_rate"])
            sub_wd = float(lrwd_cand["params"]["weight_decay"])
            sub_id = f"lrwd{lrwd_idx}_lr{sub_lr}_wd{sub_wd}"
            p3_sub_root = phase23_branch_root / f"phase3_{sub_id}"
            ensure_dir(p3_sub_root)

            p3_sub_specs = phase3_specs(locked_batch, search_n_splits, sub_lr, sub_wd, imbalance_cfg)
            p3_sub_results = run_batch(p3_sub_specs, p3_sub_root, args)
            apply_guardrail(p3_sub_results, args.abstention_min, args.abstention_max)
            p3_sub_ranked = rank_results(p3_sub_results)
            write_phase_outputs(p3_sub_root, f"phase3_{sub_id}", p3_sub_ranked)

            p3_sub_selected = top_successful_candidate(p3_sub_ranked)
            p3_sub_health = build_stage_health(
                stage_name=f"phase3_{branch_id}_{sub_id}",
                results=p3_sub_results,
                ranked=p3_sub_ranked,
                selected=p3_sub_selected,
                previous_selected=p2_selected,
                abstention_min=args.abstention_min,
                abstention_max=args.abstention_max,
                min_macro_gain=stage_gate_min_gain,
                max_minority_drop=stage_gate_max_minority_drop,
                precision_collapse_delta=rare_precision_collapse_delta,
                recall_spike_delta=rare_recall_spike_delta,
                enforce_gate=False,
            )
            p3_sub_health_path = write_stage_health(p3_sub_root, f"phase3_{sub_id}", p3_sub_health)

            p3_subbranches.append({
                "sub_id": sub_id,
                "lr": sub_lr,
                "wd": sub_wd,
                "top": p3_sub_ranked[:3],
                "health_file": p3_sub_health_path,
                "health_overview": {
                    "alerts": [a["code"] for a in p3_sub_health.get("alerts", [])],
                    "gate": p3_sub_health.get("gate", {}),
                },
            })
            all_p3_results.extend(p3_sub_results)

        # Global ranking across both LR/WD sub-branches.
        p3_ranked = rank_results(all_p3_results)
        p3_top3 = strict_pick_top(p3_ranked, 3, f"Phase 3 ({branch_id})")

        phase23_branches.append(
            {
                "branch_id": branch_id,
                "imbalance": imbalance_cfg,
                "stage1_top": p2_ranked[:4],
                "stage2_subbranches": p3_subbranches,
                "phase3_top3": p3_top3,
                "stage1_health_file": p2_health_path,
                "stage1_health_overview": {
                    "alerts": [a["code"] for a in p2_health.get("alerts", [])],
                    "gate": p2_health.get("gate", {}),
                },
            }
        )

    summary["phases"]["phase2_3"] = {"branches": phase23_branches}

    if args.stop_after_phase == "phase2":
        write_summary(output_root, summary)
        return

    if args.stop_after_phase == "phase3":
        write_summary(output_root, summary)
        return

    # Select top 2 from phase 3 across all branches for final confirmation.
    all_phase3_candidates = [r for branch in phase23_branches for r in branch["phase3_top3"]]
    phase3_all_ranked = rank_results(all_phase3_candidates)
    phase3_unique_ranked, phase3_duplicate_rows = dedupe_ranked_by_run_id(phase3_all_ranked)
    if len(phase3_unique_ranked) < 2:
        # Fallback keeps legacy behavior when uniqueness would underflow finalists.
        phase3_unique_ranked = phase3_all_ranked
        phase3_duplicate_rows = []
    if len(phase3_unique_ranked) < 2:
        raise RuntimeError("Need at least two Phase 3 candidates for final confirmation.")
    phase4_candidates = phase3_unique_ranked[:2]
    summary["phases"]["phase3_finalist_pool"] = {
        "pool_size_total": len(phase3_all_ranked),
        "pool_size_unique_run_id": len(phase3_unique_ranked),
        "dropped_duplicates_count": len(phase3_duplicate_rows),
        "selected_phase4_candidates": phase4_candidates,
    }

    # Phase 4
    phase4_root = output_root / "phase4"
    ensure_dir(phase4_root)
    p4_results = run_batch(phase4_specs(phase4_candidates), phase4_root, args)
    apply_guardrail(p4_results, args.abstention_min, args.abstention_max)
    p4_ranked = rank_results(p4_results)
    write_phase_outputs(phase4_root, "phase4", p4_ranked)
    p4_selected = top_successful_candidate(p4_ranked)
    p4_health = build_stage_health(
        stage_name="phase4",
        results=p4_results,
        ranked=p4_ranked,
        selected=p4_selected,
        previous_selected=phase4_candidates[0] if phase4_candidates else None,
        abstention_min=args.abstention_min,
        abstention_max=args.abstention_max,
        min_macro_gain=0.0,
        max_minority_drop=1.0,
        precision_collapse_delta=rare_precision_collapse_delta,
        recall_spike_delta=rare_recall_spike_delta,
        enforce_gate=False,
    )
    p4_health_path = write_stage_health(phase4_root, "phase4", p4_health)

    winner = strict_pick_top(p4_ranked, 1, "Phase 4")

    summary["phases"]["phase4"] = {
        "leaderboard": p4_ranked,
        "winner": winner[0],
        "health_file": p4_health_path,
        "health_overview": {
            "alerts": [a["code"] for a in p4_health.get("alerts", [])],
            "gate": p4_health.get("gate", {}),
        },
    }

    summary["winner"] = {
        "run_id": winner[0]["run_id"],
        "run_dir": winner[0]["run_dir"],
        "params": winner[0]["params"],
        "macro_f1": winner[0]["macro_f1"],
        "macro_precision": winner[0].get("macro_precision", 0.0),
        "macro_recall": winner[0].get("macro_recall", 0.0),
        "macro_pr_auc": winner[0].get("macro_pr_auc", winner[0].get("cv_macro_pr_auc_mean", 0.0)),
        "minority_macro_pr_auc": winner[0].get("minority_macro_pr_auc", 0.0),
        "minority_mean_f1": winner[0]["minority_mean_f1"],
        "minority_mean_precision": winner[0].get("minority_mean_precision", 0.0),
        "minority_mean_recall": winner[0].get("minority_mean_recall", 0.0),
        "exact_match_accuracy": winner[0]["exact_match_accuracy"],
        "abstention_rate": winner[0]["abstention_rate"],
        "peak_process_rss_mb": winner[0].get("peak_process_rss_mb", 0.0),
        "peak_system_cpu_percent": winner[0].get("peak_system_cpu_percent", 0.0),
        "peak_system_ram_percent": winner[0].get("peak_system_ram_percent", 0.0),
        "peak_device_memory_mb": winner[0].get("peak_device_memory_mb", 0.0),
        "system_metrics_file": winner[0].get("system_metrics_file", ""),
        "runtime_sec": winner[0]["runtime_sec"],
    }

    # Post-HPO sequence: optional human eval
    post_hpo_results = run_post_hpo_sequence(winner[0]["run_dir"], str(args.workflow), args)
    summary["post_hpo_sequence"] = post_hpo_results
    if args.post_winner_to_langfuse and not args.dry_run:
        post_result = post_winner_from_artifacts(winner[0], args)
        summary["winner_post_to_langfuse"] = post_result
        if post_result["status"] != "ok":
            print(
                "[WARN] Winner post-to-Langfuse failed. "
                f"See logs: {post_result['stdout_log']} | {post_result['stderr_log']}"
            )
    else:
        summary["winner_post_to_langfuse"] = {
            "status": "skipped",
            "reason": "disabled_or_dry_run",
        }

    write_summary(output_root, summary)

    print("\n=== HPO COMPLETE ===")
    print(f"Winner run_id: {summary['winner']['run_id']}")
    print(f"Winner dir: {summary['winner']['run_dir']}")
    print(f"Winner params: {summary['winner']['params']}")


WORKFLOW_REGISTRY: Dict[str, Dict[str, Any]] = {
    "multilabel_frozen": {
        "trainer": "train_head_only.py",
        "dataset": "generic_dataset",
        "human_eval_dataset": "generic_dataset_human_eval",
        "output_root_default": "outputs/checkpoints_multilabel_frozen",
        "output_root_timestamped": True,
        "abstention_min": 0.0,
        "abstention_max": 0.50,
        "force_selection_metric": None,
        "task_type": "multilabel",
        "valid_stop_phases": {"phase0", "phase1", "phase2", "phase3", "phase4"},
        "workflow_fn": run_frozen_workflow,
        "metrics_fn": _extract_multilabel_metrics,
        "validate_fn": _validate_multilabel_artifacts,
        "snapshot_cache_fn": _snapshot_from_dataset_cache_multilabel,
        "snapshot_cv_fn": _snapshot_from_cv_results_multilabel,
        "oof_run_name_prefix": "multilingual_multilabel_frozen",
        "fixed_hpo_params": {"task": "generic_task"},
        "auto_threshold_metric": True,
    },
    "multilabel_lora": {
        "trainer": "train_lora.py",
        "dataset": "generic_dataset",
        "human_eval_dataset": "generic_dataset_human_eval",
        "output_root_default": "outputs/hpo_multilabel_lora",
        "output_root_timestamped": True,
        "abstention_min": 0.0,
        "abstention_max": 0.50,
        "force_selection_metric": None,
        "task_type": "multilabel",
        "valid_stop_phases": {"lora_a", "lora_b", "lora_c", "lora_d"},
        "workflow_fn": run_lora_parity_workflow,
        "metrics_fn": _extract_multilabel_metrics,
        "validate_fn": _validate_multilabel_artifacts,
        "snapshot_cache_fn": _snapshot_from_dataset_cache_multilabel,
        "snapshot_cv_fn": _snapshot_from_cv_results_multilabel,
        "oof_run_name_prefix": "multilingual_multilabel_lora",
        "fixed_hpo_params": {"task": "generic_task"},
        "auto_threshold_metric": True,
    },
    "multiclass_lora": {
        "trainer": "train_lora.py",
        "dataset": "generic_dataset",
        "human_eval_dataset": "generic_dataset_human_eval",
        "output_root_default": "outputs/hpo_multiclass_lora",
        "output_root_timestamped": True,
        "abstention_min": 0.0,
        "abstention_max": 0.50,
        "force_selection_metric": None,
        "task_type": "multiclass",
        "valid_stop_phases": {"lora_a", "lora_b", "lora_c", "lora_d"},
        "workflow_fn": run_lora_parity_workflow,
        "metrics_fn": _extract_multiclass_frozen_metrics,
        "validate_fn": _validate_multiclass_frozen_artifacts,
        "snapshot_cache_fn": _snapshot_from_dataset_cache_multiclass,
        "snapshot_cv_fn": _snapshot_from_cv_results_multiclass,
        "oof_run_name_prefix": "multilingual_multiclass_lora",
        "abstention_label": "abstain",
        "fixed_hpo_params": {"task": "generic_task"},
        "auto_threshold_metric": True,
    },
    "multiclass_frozen": {
        "trainer": "train_head_only.py",
        "dataset": "generic_dataset",
        "human_eval_dataset": "generic_dataset_human_eval",
        "output_root_default": "outputs/checkpoints_multiclass_frozen",
        "output_root_timestamped": True,
        "abstention_min": 0.0,
        "abstention_max": 0.50,
        "force_selection_metric": None,
        "task_type": "multiclass",
        "valid_stop_phases": {"phase0", "phase1", "phase2", "phase3", "phase4"},
        "workflow_fn": run_frozen_workflow,
        "metrics_fn": _extract_multiclass_frozen_metrics,
        "validate_fn": _validate_multiclass_frozen_artifacts,
        "snapshot_cache_fn": _snapshot_from_dataset_cache_multiclass,
        "snapshot_cv_fn": _snapshot_from_cv_results_multiclass,
        "oof_run_name_prefix": "multilingual_multiclass_frozen",
        "abstention_label": "abstain",
        "fixed_hpo_params": {"task": "generic_task"},
        "auto_threshold_metric": True,
    },
}


def main() -> None:
    global ACTIVE_SELECTION_METRIC
    args = parse_args()
    workflow = resolve_workflow(args)
    args.workflow = workflow
    normalize_and_validate_metric_overrides(args)
    _spec = WORKFLOW_REGISTRY[workflow]
    ACTIVE_SELECTION_METRIC = _spec["force_selection_metric"] or str(args.selection_metric)

    _ts = time.strftime("%Y%m%d_%H%M%S")
    _task_slug = _infer_output_root_task_slug(args, workflow, _spec)
    if args.trainer_script is None:
        args.trainer_script = _spec["trainer"]
    args.trainer_script = _resolve_existing_script_path(str(args.trainer_script), "Trainer script")
    args.human_eval_script = str(_resolve_script_path(str(args.human_eval_script)))
    args.threshold_optimizer_script = str(_resolve_script_path(str(args.threshold_optimizer_script)))

    if args.output_root is None:
        base = _ensure_output_root_contains_task(str(_spec["output_root_default"]), _task_slug)
        args.output_root = f"{base}_{_ts}" if _spec["output_root_timestamped"] else base
    else:
        args.output_root = _ensure_output_root_contains_task(str(args.output_root), _task_slug)
    if args.dataset_name is None:
        args.dataset_name = _spec["dataset"]
    if args.abstention_min is None:
        args.abstention_min = _spec["abstention_min"]
    if args.abstention_max is None:
        args.abstention_max = _spec["abstention_max"]

    trainer_path = Path(str(args.trainer_script))
    args.campaign_signature = build_campaign_signature(args, trainer_path)

    output_root = Path(str(args.output_root))
    ensure_dir(output_root)
    if args.dataset_cache_path is None:
        args.dataset_cache_path = str(output_root / "langfuse_dataset_cache.json")
    if args.tokenized_cache_path is None:
        args.tokenized_cache_path = str(output_root / "tokenized_dataset_cache.pt")
    if not args.resume and not args.dry_run:
        cleanup_fresh_start(
            output_root=output_root,
            workflow=workflow,
            dataset_cache_path=args.dataset_cache_path,
            tokenized_cache_path=args.tokenized_cache_path,
            cleanup_dataset_cache=bool(args.dataset_cache_path),
            cleanup_tokenized_cache=bool(args.tokenized_cache_path),
        )
    elif not args.resume and args.dry_run:
        print("[FRESH_START] skipped cleanup in --dry-run mode (non-destructive).")

    summary: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "dataset_cache_path": args.dataset_cache_path,
        "tokenized_cache_path": args.tokenized_cache_path,
        "campaign_signature": args.campaign_signature,
        "seed": args.seed,
        "checkpoint_selection_metric": args.checkpoint_selection_metric,
        "use_temperature_scaling": bool(getattr(args, "use_temperature_scaling", False))
        if _is_multilabel_frozen_workflow(workflow)
        else False,
        "selection_metric": ACTIVE_SELECTION_METRIC,
        "abstention_guardrail": {
            "min": args.abstention_min,
            "max": args.abstention_max,
        },
        "phases": {},
    }

    summary["workflow"] = workflow

    if args.stop_after_phase and args.stop_after_phase not in _spec["valid_stop_phases"]:
        valid = ", ".join(sorted(_spec["valid_stop_phases"]))
        raise ValueError(
            f"stop-after-phase for workflow={workflow} must be one of: {valid}"
        )

    _spec["workflow_fn"](args, output_root, summary)


if __name__ == "__main__":
    main()
