"""
Evaluate a frozen or LoRA checkpoint on a Langfuse human-annotated test dataset.

Supports multilabel and multiclass classification. Ensembles all fold checkpoints.

Langfuse posting matches the HPO OOF structure exactly:
- Span name: "oof_validation_prediction"
- Metadata structure mirrors _build_langfuse_run_metadata
- Per-item scores: precision, recall, f_score via score_trace

Usage:
    python3 training/test_model.py \
        --checkpoint-dir checkpoints_multilabel_frozen/ \
        --dataset-name generic_human_eval_dataset \
        --task-type multilabel \
        --abstention-label abstain
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from scipy.stats import spearmanr

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from langfuse import Langfuse
from langfuse.api.resources.dataset_run_items.types.create_dataset_run_item_request import (
    CreateDatasetRunItemRequest,
)
from langfuse.api.resources.score.types.create_score_request import CreateScoreRequest
from langfuse_fetch_utils import fetch_run_traces
from langfuse_post_timing import resolve_langfuse_post_timing_from_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Human-eval default for LoRA inference.
LORA_EVAL_MAX_CHUNKS_PER_DOC = 500
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_default_data_file(*relative_candidates: str) -> str:
    for rel in relative_candidates:
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return str(candidate)
    return str(PROJECT_ROOT / relative_candidates[0])


DEFAULT_METADATA_PATH = _resolve_default_data_file(
    "training/document_train_test_split.csv",
    "document_train_test_split.csv",
)
DEFAULT_EMBEDDINGS_PATH = _resolve_default_data_file(
    "data_preparation/train_test_document_embeddings.npy",
    "train_test_document_embeddings.npy",
)


# ---------------------------------------------------------------------------
# Model (layer names match all frozen trainer checkpoints exactly)
# ---------------------------------------------------------------------------


class FrozenMLPHead(nn.Module):
    """
    intermediate_size > 0: dropout1 -> dense -> relu -> layer_norm -> dropout2 -> classifier
    intermediate_size == 0: dropout1 -> classifier
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 128,
        num_labels: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.use_intermediate = int(intermediate_size) > 0
        self.dropout1 = nn.Dropout(float(dropout))
        if self.use_intermediate:
            self.dense = nn.Linear(int(hidden_size), int(intermediate_size))
            self.layer_norm = nn.LayerNorm(int(intermediate_size))
            self.dropout2 = nn.Dropout(float(dropout))
            self.classifier = nn.Linear(int(intermediate_size), int(num_labels))
        else:
            self.classifier = nn.Linear(int(hidden_size), int(num_labels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(x)
        if self.use_intermediate:
            x = self.dense(x)
            x = torch.relu(x)
            x = self.layer_norm(x)
            x = self.dropout2(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    import json
    cfg = {}
    config_path = Path("training_config.json")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if not line.strip().startswith(("//", "#"))]
            cfg = json.loads("".join(lines))
        except Exception as exc:
            print(f"[CONFIG_ERROR] Failed to read training_config.json: {exc}")

    parser = argparse.ArgumentParser(
        description="Evaluate a frozen MLP checkpoint on a Langfuse human-annotated test dataset."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing fold*_best.pt, run_config.json, thresholds.json.",
    )
    parser.add_argument(
        "--dataset-name",
        required=(cfg.get("testset_name") is None),
        default=cfg.get("testset_name"),
        help="Langfuse test dataset name.",
    )
    parser.add_argument(
        "--task-type",
        required=(cfg.get("task_type") is None),
        default=cfg.get("task_type"),
        choices=["multilabel", "multiclass"],
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Ordered trainable label names (abstention excluded). "
             "Reads from cv_results.json label_mapping if omitted.",
    )
    parser.add_argument(
        "--abstention-label",
        default=cfg.get("abstention_label", "abstain"),
        help="Label name for abstention ground truth.",
    )
    parser.add_argument(
        "--thresholds-file",
        default=None,
        help="Path to thresholds.json. Default: <checkpoint-dir>/thresholds.json.",
    )
    parser.add_argument(
        "--embeddings-path",
        default=DEFAULT_EMBEDDINGS_PATH,
    )
    parser.add_argument(
        "--metadata-path",
        default=DEFAULT_METADATA_PATH,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write human_eval_results.json. Default: checkpoint-dir.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Langfuse dataset run name (auto-generated if not set).",
    )
    parser.add_argument(
        "--no-post-to-langfuse",
        action="store_true",
        default=False,
        help="Skip posting results to Langfuse (still fetches the dataset).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Debug: limit number of test items processed.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def resolve_categories(args: argparse.Namespace, cv_results: Dict[str, Any]) -> List[str]:
    if args.categories:
        return list(args.categories)
    label_mapping = cv_results.get("label_mapping") or {}
    cats = label_mapping.get("train_labels") or []
    if not cats:
        raise ValueError(
            "--categories not supplied and cv_results.json has no label_mapping.train_labels"
        )
    return list(cats)


def load_thresholds(
    thresholds_file: str,
    task_type: str,
    fallback_multiclass_threshold: float = 0.5,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    with open(thresholds_file) as f:
        data = json.load(f)
    if task_type == "multilabel":
        per_class = data.get("per_class")
        if not isinstance(per_class, dict):
            raise ValueError(f"thresholds.json missing 'per_class' for multilabel: {thresholds_file}")
        temperature_scaling = data.get("temperature_scaling")
        if temperature_scaling is not None and not isinstance(temperature_scaling, dict):
            raise ValueError(f"thresholds.json has invalid 'temperature_scaling' payload: {thresholds_file}")
        return per_class, temperature_scaling
    else:
        t = data.get("selected_threshold")
        if t is None:
            t = data.get("threshold")
        if t is None:
            t = data.get("calibrated_threshold")
        if t is None:
            logging.warning(
                "thresholds.json missing multiclass threshold key (selected_threshold/threshold/calibrated_threshold). "
                "Using fallback threshold=%.4f.",
                float(fallback_multiclass_threshold),
            )
            return float(fallback_multiclass_threshold), None
        return float(t), None


def _safe_logit_vec(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probs.astype(np.float64), eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def apply_multilabel_temperature_scaling(
    probs: np.ndarray,
    categories: List[str],
    temperature_scaling: Optional[Dict[str, Any]],
) -> np.ndarray:
    if not temperature_scaling or not bool(temperature_scaling.get("enabled", False)):
        return probs
    per_class = temperature_scaling.get("per_class") or {}
    if not isinstance(per_class, dict):
        return probs
    logits = _safe_logit_vec(probs)
    temp_arr = np.array([float(per_class.get(cat, 1.0)) for cat in categories], dtype=np.float64)
    temp_arr = np.clip(temp_arr, 1e-6, None)
    scaled_logits = logits / temp_arr
    scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return scaled_probs.astype(np.float32)


def load_fold_models_frozen(
    checkpoint_dir: str,
    hidden_size: int,
    intermediate_size: int,
    num_labels: int,
    dropout: float,
    device: torch.device,
) -> List[nn.Module]:
    models = []
    for path in sorted(Path(checkpoint_dir).glob("fold*_best.pt")):
        model = FrozenMLPHead(hidden_size, intermediate_size, num_labels, dropout)
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
        logging.info("Loaded %s", path.name)
    if not models:
        raise FileNotFoundError(f"No fold*_best.pt checkpoints found in {checkpoint_dir}")
    return models


def has_frozen_artifacts(checkpoint_dir: str) -> bool:
    return any(Path(checkpoint_dir).glob("fold*_best.pt"))


def has_lora_artifacts(checkpoint_dir: str) -> bool:
    return any(Path(checkpoint_dir).glob("best_model_fold_*.pt"))


def detect_checkpoint_mode(checkpoint_dir: str) -> str:
    if has_frozen_artifacts(checkpoint_dir):
        return "frozen"
    if has_lora_artifacts(checkpoint_dir):
        return "lora"
    raise FileNotFoundError(
        f"No supported checkpoint artifacts found in {checkpoint_dir}. "
        "Expected fold*_best.pt (frozen) or best_model_fold_*.pt (LoRA)."
    )


def load_fold_models_lora(
    checkpoint_dir: str,
    run_config: Dict[str, Any],
    num_labels: int,
    device: torch.device,
) -> List[nn.Module]:
    # Lazy imports to avoid heavy imports in frozen-only usage
    from train_lora import LoRAClassifier, lora_target_modules_from_set
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel

    models: List[nn.Module] = []
    model_name = run_config.get("model_name") or run_config.get("mmbert_model_name") or "jhu-clsp/mmBERT-base"

    logging.info("Loading transformers backbone: %s", model_name)
    raw_backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    lora_target = lora_target_modules_from_set(run_config.get("lora_target_set", "all"))
    lora_cfg = LoraConfig(
        r=int(run_config.get("lora_r", 8)),
        lora_alpha=int(run_config.get("lora_alpha", 16)),
        target_modules=lora_target,
        lora_dropout=float(run_config.get("lora_dropout", 0.1)),
        bias="none"
    )
    backbone_lora = get_peft_model(raw_backbone, lora_cfg)

    paths = sorted(Path(checkpoint_dir).glob("best_model_fold_*.pt"))
    for path in paths:
        model = LoRAClassifier(
            backbone=backbone_lora,
            hidden_size=int(run_config.get("hidden_size", 768)),
            intermediate_size=int(run_config.get("intermediate_size", 128)),
            num_labels=int(num_labels),
            dropout=float(run_config.get("dropout", 0.15))
        )
        model.load_state_dict(torch.load(str(path), map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        models.append(model)
        logging.info("Loaded LoRA fold %s", path.name)

    if not models:
        raise FileNotFoundError(f"No best_model_fold_*.pt pairs found in {checkpoint_dir}")

    return models


# ---------------------------------------------------------------------------
# Gold label extraction
# ---------------------------------------------------------------------------


def gold_labels_multilabel(
    expected_output: Any,
    categories: List[str],
    abstention_label: str,
) -> List[str]:
    if not isinstance(expected_output, dict):
        return []
    results = expected_output.get("results") or []
    raw = [r.get("category", "") for r in results if isinstance(r, dict)]
    if raw and all(c == abstention_label for c in raw if c):
        return []
    return [c for c in raw if c in categories]


def gold_label_multiclass(
    expected_output: Any,
    categories: List[str],
    abstention_label: str,
) -> str:
    if not isinstance(expected_output, dict):
        return abstention_label
    results = expected_output.get("results") or []
    if not results or not isinstance(results[0], dict):
        return abstention_label
    cat = str(results[0].get("category", ""))
    return cat if (cat in categories or cat == abstention_label) else abstention_label


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def infer_multilabel(
    models: List[nn.Module],
    emb: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        x = emb.unsqueeze(0).to(device)
        return np.mean([torch.sigmoid(m(x)).cpu().numpy() for m in models], axis=0).squeeze(0)


def infer_multiclass(
    models: List[nn.Module],
    emb: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        x = emb.unsqueeze(0).to(device)
        return np.mean([F.softmax(m(x), dim=1).cpu().numpy() for m in models], axis=0).squeeze(0)


def _uniform_indices(total: int, keep: int) -> List[int]:
    if keep >= total:
        return list(range(total))
    if keep <= 1:
        return [0]
    raw = np.linspace(0, total - 1, num=keep)
    idx = [int(round(x)) for x in raw]
    seen = set()
    uniq: List[int] = []
    for x in idx:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    if len(uniq) < keep:
        for x in range(total):
            if x not in seen:
                uniq.append(x)
                if len(uniq) == keep:
                    break
    return sorted(uniq[:keep])


def _chunk_tokens_for_lora(tokens: List[int], tokenizer: Any, max_seq_length: int, overlap: int) -> List[List[int]]:
    if max_seq_length <= 2:
        raise ValueError("max_seq_length must be > 2")
    body_len = max_seq_length - 2
    stride = body_len - overlap
    if stride <= 0:
        raise ValueError("chunk_overlap must be less than max_seq_length - 2")

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    if len(tokens) <= body_len:
        core = tokens
        if cls_id is not None and sep_id is not None:
            return [[cls_id] + core + [sep_id]]
        return [core]

    chunks: List[List[int]] = []
    start = 0
    while start < len(tokens):
        core = tokens[start : start + body_len]
        if cls_id is not None and sep_id is not None:
            chunks.append([cls_id] + core + [sep_id])
        else:
            chunks.append(core)
        start += stride
        if start >= len(tokens):
            break
    return chunks


def build_lora_inputs_from_text(
    text: str,
    tokenizer: Any,
    max_seq_length: int,
    chunk_overlap: int,
    max_chunks_per_doc: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_tokens = tokenizer.encode(text, add_special_tokens=False)
    max_token_budget = int(max_seq_length) * int(max_chunks_per_doc)
    if len(raw_tokens) > max_token_budget:
        raise ValueError(
            f"Document exceeds token budget ({len(raw_tokens)} > {max_token_budget}); "
            "it would have been omitted by LoRA training budget filters."
        )
    chunks = _chunk_tokens_for_lora(raw_tokens, tokenizer, int(max_seq_length), int(chunk_overlap))
    if len(chunks) > int(max_chunks_per_doc):
        keep_idx = _uniform_indices(len(chunks), int(max_chunks_per_doc))
        chunks = [chunks[i] for i in keep_idx]
    max_chunk_len = max(len(c) for c in chunks)
    input_ids: List[List[int]] = []
    attention_masks: List[List[int]] = []
    for chunk in chunks:
        actual_len = len(chunk)
        if actual_len < max_chunk_len:
            pad = max_chunk_len - actual_len
            input_ids.append(chunk + [0] * pad)
            attention_masks.append([1] * actual_len + [0] * pad)
        else:
            input_ids.append(chunk)
            attention_masks.append([1] * actual_len)
    ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)          # [1, num_chunks, seq_len]
    mask = torch.tensor(attention_masks, dtype=torch.long).unsqueeze(0)   # [1, num_chunks, seq_len]
    return ids, mask


def infer_lora_multilabel(
    models: List[nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        ids = input_ids.to(device)
        mask = attention_mask.to(device)
        preds = []
        for model in models:
            logits = model(input_ids=ids, attention_mask=mask)
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.mean(preds, axis=0).squeeze(0)


def infer_lora_multiclass(
    models: List[nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        ids = input_ids.to(device)
        mask = attention_mask.to(device)
        preds = []
        for model in models:
            logits = model(input_ids=ids, attention_mask=mask)
            preds.append(F.softmax(logits, dim=1).detach().cpu().numpy())
        return np.mean(preds, axis=0).squeeze(0)


# ---------------------------------------------------------------------------
# Per-item metrics (matching _set_metrics_from_rows in trainer scripts)
# ---------------------------------------------------------------------------


def item_metrics_from_binary_rows(labels_row: np.ndarray, preds_row: np.ndarray) -> Dict[str, float]:
    """Compute precision/recall/f_score for one item (multilabel binary vectors)."""
    labels_bin = labels_row.astype(np.int32)
    preds_bin = preds_row.astype(np.int32)
    tp = int(np.sum((labels_bin == 1) & (preds_bin == 1)))
    pred_pos = int(np.sum(preds_bin == 1))
    true_pos = int(np.sum(labels_bin == 1))
    precision = float(tp / pred_pos) if pred_pos > 0 else 0.0
    recall = float(tp / true_pos) if true_pos > 0 else 0.0
    f_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f_score": f_score}


def item_metrics_multiclass(gold: str, pred: str, abstention_label: str) -> Dict[str, float]:
    """Compute precision/recall/f_score for one multiclass item."""
    tp = 1 if (pred == gold and pred != abstention_label) else 0
    pred_pos = 1 if pred != abstention_label else 0
    true_pos = 1 if gold != abstention_label else 0
    precision = float(tp / pred_pos) if pred_pos > 0 else 0.0
    recall = float(tp / true_pos) if true_pos > 0 else 0.0
    f_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f_score": f_score}


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def compute_multilabel_metrics(
    gold: np.ndarray,
    pred: np.ndarray,
    categories: List[str],
) -> Dict[str, Any]:
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        gold, pred, average=None, zero_division=0
    )
    supported_mask = per_sup > 0
    if np.any(supported_mask):
        macro_p = float(np.mean(per_p[supported_mask]))
        macro_r = float(np.mean(per_r[supported_mask]))
        macro_f1 = float(np.mean(per_f1[supported_mask]))
    else:
        macro_p = 0.0
        macro_r = 0.0
        macro_f1 = 0.0
    per_class = {
        cat: {
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f1[i]),
            "support": int(per_sup[i]),
        }
        for i, cat in enumerate(categories)
    }
    return {
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "exact_match": float(accuracy_score(gold, pred)),
        "hamming_loss": float(hamming_loss(gold, pred)),
        "abstention_rate": float(np.mean(pred.sum(axis=1) == 0)),
        "per_class_metrics": per_class,
    }


def compute_multiclass_metrics(
    gold_list: List[str],
    pred_list: List[str],
    categories: List[str],
    abstention_label: str,
) -> Dict[str, Any]:
    all_classes = categories + ([abstention_label] if abstention_label else [])
    to_idx = {c: i for i, c in enumerate(all_classes)}
    g_idx = [to_idx.get(g, len(all_classes)) for g in gold_list]
    p_idx = [to_idx.get(p, len(all_classes)) for p in pred_list]
    class_range = list(range(len(all_classes)))
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        g_idx, p_idx, average=None, zero_division=0, labels=class_range
    )
    supported_mask = per_sup > 0
    if np.any(supported_mask):
        macro_p = float(np.mean(per_p[supported_mask]))
        macro_r = float(np.mean(per_r[supported_mask]))
        macro_f1 = float(np.mean(per_f1[supported_mask]))
    else:
        macro_p = 0.0
        macro_r = 0.0
        macro_f1 = 0.0
    per_class = {
        cls: {
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f1[i]),
            "support": int(per_sup[i]),
        }
        for i, cls in enumerate(all_classes)
    }
    return {
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "exact_match": float(accuracy_score(g_idx, p_idx)),
        "hamming_loss": None,
        "abstention_rate": float(sum(p == abstention_label for p in pred_list) / max(len(pred_list), 1)),
        "num_correct": int(sum(g == p for g, p in zip(gold_list, pred_list))),
        "num_total": len(gold_list),
        "per_class_metrics": per_class,
    }


def compute_minority_mean_f1(
    per_class_metrics: Dict[str, Any],
    minority_classes: List[str],
) -> Optional[float]:
    values = [
        float(per_class_metrics[c]["f1"])
        for c in minority_classes
        if c in per_class_metrics and per_class_metrics[c].get("f1") is not None
    ]
    return float(np.mean(values)) if values else None


def compute_temperature_scaling_confidence_stats(
    checkpoint_dir: str,
    task_type: str,
    categories: List[str],
    thresholds: Any,
    temperature_scaling: Optional[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """
    Compute Spearman(confidence, per-document F1) on OOF data using the
    currently active thresholds + temperature scaling configuration.
    """
    empty = {
        "confidence_spearman_r": None,
        "confidence_spearman_p": None,
        "confidence_spearman_n": None,
    }
    if task_type != "multilabel":
        return empty
    if not isinstance(thresholds, dict):
        return empty
    try:
        oof_probs = np.load(str(Path(checkpoint_dir) / "oof_probs.npy"))
        oof_labels = np.load(str(Path(checkpoint_dir) / "oof_labels.npy"))
    except Exception:
        return empty
    if oof_probs.ndim != 2 or oof_labels.ndim != 2 or oof_probs.shape != oof_labels.shape:
        return empty
    try:
        threshold_arr = np.array([float(thresholds[cat]) for cat in categories], dtype=np.float32)
    except Exception:
        return empty

    probs = apply_multilabel_temperature_scaling(oof_probs, categories, temperature_scaling)

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
    confidence = (pos_space + neg_space).mean(axis=1)

    preds = (probs >= threshold_arr[None, :]).astype(np.int32)
    labels = oof_labels.astype(np.int32)
    tp = (labels & preds).sum(axis=1).astype(np.float32)
    n_pred = preds.sum(axis=1).astype(np.float32)
    n_true = labels.sum(axis=1).astype(np.float32)
    denom = n_pred + n_true
    doc_f1 = np.zeros(labels.shape[0], dtype=np.float32)
    nz = denom > 0
    doc_f1[nz] = (2.0 * tp[nz]) / denom[nz]
    doc_f1[(n_pred == 0) & (n_true == 0)] = 1.0

    corr, p_value = spearmanr(confidence, doc_f1)
    corr_val = float(corr) if corr is not None and np.isfinite(corr) else None
    p_val = float(p_value) if p_value is not None and np.isfinite(p_value) else None
    return {
        "confidence_spearman_r": corr_val,
        "confidence_spearman_p": p_val,
        "confidence_spearman_n": int(labels.shape[0]),
    }


# ---------------------------------------------------------------------------
# Langfuse metadata (mirrors _build_langfuse_run_metadata exactly)
# ---------------------------------------------------------------------------


def build_run_metadata(
    run_config: Dict[str, Any],
    cv_results: Dict[str, Any],
    test_metrics: Dict[str, Any],
    minority_classes: List[str],
    test_dataset_name: str,
    temperature_scaling: Optional[Dict[str, Any]] = None,
    temperature_scaling_confidence_stats: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, Any]:
    oof_metrics = cv_results.get("oof_metrics_calibrated") or {}
    script_name = run_config.get("training_script_name") or run_config.get("script_name") or ""
    # Derive model_name from script: strip "train_" prefix and ".py" suffix
    model_name = script_name
    if model_name.startswith("train_"):
        model_name = model_name[len("train_"):]
    if model_name.endswith(".py"):
        model_name = model_name[:-3]

    minority_mean_f1_oof: Optional[float] = None
    oof_per_class = (oof_metrics.get("per_class_metrics") or {})
    if minority_classes and oof_per_class:
        minority_mean_f1_oof = compute_minority_mean_f1(oof_per_class, minority_classes)

    test_per_class = test_metrics.get("per_class_metrics") or {}
    minority_mean_f1_test = compute_minority_mean_f1(test_per_class, minority_classes) if minority_classes else None

    ts = temperature_scaling if isinstance(temperature_scaling, dict) else {}
    ts_conf = (
        temperature_scaling_confidence_stats
        if isinstance(temperature_scaling_confidence_stats, dict)
        else {}
    )

    return {
        # --- Fields matching _build_langfuse_run_metadata ---
        "model_name": model_name,
        "architecture": "frozen_embeddings_mlp",
        "key_hyperparameters": {
            "learning_rate": run_config.get("learning_rate"),
            "weight_decay": run_config.get("weight_decay"),
            "dropout": run_config.get("dropout"),
            "intermediate_size": run_config.get("intermediate_size"),
            "batch_size": run_config.get("batch_size"),
            "num_epochs": run_config.get("num_epochs"),
            "early_stopping_patience": run_config.get("early_stopping_patience"),
            "max_grad_norm": run_config.get("max_grad_norm"),
            "eval_threshold": run_config.get("eval_threshold"),
            "threshold_selection_metric": run_config.get("threshold_selection_metric"),
        },
        "imbalance_strategy": {
            "pos_weight_mode": run_config.get("pos_weight_mode"),
            "pos_weight_clip_min": run_config.get("pos_weight_clip_min"),
            "pos_weight_clip_max": run_config.get("pos_weight_clip_max"),
            "sampler_strength": run_config.get("sampler_strength"),
            "sampler_reduced_alpha": run_config.get("sampler_reduced_alpha"),
        },
        "script_name": script_name,
        "training_script_path": run_config.get("training_script_path"),
        "posting_script_name": "test_model.py",
        "oof_macro_f1": float(oof_metrics["macro_f1"]) if oof_metrics.get("macro_f1") is not None else None,
        "oof_macro_precision": float(oof_metrics["macro_precision"]) if oof_metrics.get("macro_precision") is not None else None,
        "oof_macro_recall": float(oof_metrics["macro_recall"]) if oof_metrics.get("macro_recall") is not None else None,
        "oof_macro_pr_auc": float(oof_metrics["macro_pr_auc"]) if oof_metrics.get("macro_pr_auc") is not None else None,
        "minority_mean_f1": minority_mean_f1_oof,
        "checkpoint_selection_metric": run_config.get("checkpoint_selection_metric"),
        "cv_splits": run_config.get("n_splits"),
        "seed": run_config.get("random_state"),
        "checkpoint_dir": run_config.get("checkpoint_dir"),
        "dataset_name": run_config.get("dataset_name"),
        "campaign_signature": run_config.get("campaign_signature"),
        # --- Test-specific additions ---
        "evaluation_type": "human_test_set",
        "test_dataset_name": test_dataset_name,
        "test_macro_f1": test_metrics.get("macro_f1"),
        "test_macro_precision": test_metrics.get("macro_precision"),
        "test_macro_recall": test_metrics.get("macro_recall"),
        "test_minority_mean_f1": minority_mean_f1_test,
        "temperature_scaling.per_class": (
            ts.get("per_class") if isinstance(ts.get("per_class"), dict) else {}
        ),
        "temperature_scaling.enabled": bool(ts.get("enabled", False)),
        "confidence_spearman_r": ts_conf.get("confidence_spearman_r"),
        "confidence_spearman_p": ts_conf.get("confidence_spearman_p"),
        "confidence_spearman_n": ts_conf.get("confidence_spearman_n"),
    }


def make_trace_id(run_name: str, dataset_item_id: str, trace_namespace: str = "") -> str:
    key = f"{run_name}|{trace_namespace}|{dataset_item_id}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:32]


def make_score_id(trace_id: str, score_name: str) -> str:
    key = f"{trace_id}|{score_name}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:32]


def create_trace_scores(
    langfuse: Langfuse,
    trace_id: str,
    item_metrics: Dict[str, float],
) -> None:
    for score_name in ("precision", "recall", "f_score"):
        langfuse.api.score.create(
            request=CreateScoreRequest(
                id=make_score_id(trace_id, score_name),
                trace_id=trace_id,
                name=score_name,
                value=float(item_metrics[score_name]),
                metadata={"source": "oof_validation_prediction"},
            )
        )


def make_dataset_run_score_id(run_name: str, score_name: str) -> str:
    key = f"{run_name}|dataset_run|{score_name}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:32]


def post_run_level_macro_scores(
    langfuse: Langfuse,
    dataset_name: str,
    run_name: str,
    macro_precision: float,
    macro_recall: float,
    macro_f1: float,
    source: str = "human_test_set_prediction",
) -> None:
    run = langfuse.api.datasets.get_run(dataset_name=dataset_name, run_name=run_name)
    run_id = getattr(run, "id", None)
    if not run_id:
        raise RuntimeError(
            f"Could not resolve Langfuse dataset run id for dataset={dataset_name} run={run_name}"
        )

    payload = [
        ("macro_precision", float(macro_precision)),
        ("macro_recall", float(macro_recall)),
        ("macro_f1", float(macro_f1)),
    ]
    for score_name, score_value in payload:
        langfuse.api.score.create(
            request=CreateScoreRequest(
                id=make_dataset_run_score_id(run_name, score_name),
                datasetRunId=str(run_id),
                name=score_name,
                value=score_value,
                metadata={"source": source},
            )
        )


def _is_langfuse_not_found_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return ("notfound" in name) or ("not found" in msg) or ("status_code: 404" in msg)


def _ensure_langfuse_tracing_enabled() -> None:
    prev = os.getenv("LANGFUSE_TRACING_ENABLED")
    if prev and prev.lower() not in {"1", "true", "yes", "on"}:
        logging.warning(
            "Overriding LANGFUSE_TRACING_ENABLED=%s -> true for reliable trace persistence.",
            prev,
        )
    os.environ["LANGFUSE_TRACING_ENABLED"] = "true"


def _delete_dataset_run_and_wait(
    langfuse: Langfuse,
    dataset_name: str,
    run_name: str,
    timeout_sec: float,
    poll_sec: float,
) -> None:
    try:
        langfuse.api.datasets.delete_run(dataset_name=dataset_name, run_name=run_name)
        logging.info("Deleted existing Langfuse run before repost: %s", run_name)
    except Exception as exc:
        if not _is_langfuse_not_found_error(exc):
            raise
        return

    deadline = time.time() + max(1.0, float(timeout_sec))
    sleep_s = max(0.1, float(poll_sec))
    while time.time() < deadline:
        try:
            langfuse.api.datasets.get_run(dataset_name=dataset_name, run_name=run_name)
        except Exception as exc:
            if _is_langfuse_not_found_error(exc):
                return
            raise
        time.sleep(sleep_s)

    raise TimeoutError(
        f"Timed out waiting for Langfuse run deletion to complete: dataset={dataset_name} run={run_name}"
    )


def _validate_langfuse_post_integrity(
    langfuse: Langfuse,
    dataset_name: str,
    run_name: str,
    expected_trace_ids: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    run_items, traces_by_id = fetch_run_traces(
        langfuse=langfuse,
        dataset_name=dataset_name,
        run_name=run_name,
        retries=1,
    )
    linked_trace_ids = {
        str(getattr(item, "trace_id"))
        for item in run_items
        if getattr(item, "trace_id", None) is not None
    }
    missing_links = [trace_id for trace_id in expected_trace_ids if trace_id not in linked_trace_ids]

    missing_traces: List[str] = []
    missing_scores: List[str] = []
    required_score_names = {"precision", "recall", "f_score"}
    for trace_id in expected_trace_ids:
        trace = traces_by_id.get(trace_id)
        if trace is None:
            missing_traces.append(trace_id)
            continue
        score_names = {
            str(getattr(score, "name"))
            for score in (getattr(trace, "scores", []) or [])
            if getattr(score, "name", None) is not None
        }
        if not required_score_names.issubset(score_names):
            missing_scores.append(trace_id)

    return missing_links, missing_traces, missing_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    checkpoint_dir = str(args.checkpoint_dir)

    run_config = load_json(Path(checkpoint_dir) / "run_config.json")
    cv_results = load_json(Path(checkpoint_dir) / "cv_results.json")

    categories = resolve_categories(args, cv_results)
    num_labels = len(categories)
    hidden_size = int(run_config.get("hidden_size", 768))
    intermediate_size = int(run_config.get("intermediate_size", 128))
    dropout = float(run_config.get("dropout", 0.3))
    minority_classes: List[str] = list(cv_results.get("minority_classes") or [])

    thresholds_file = args.thresholds_file or str(Path(checkpoint_dir) / "thresholds.json")
    fallback_multiclass_threshold = float(run_config.get("eval_threshold", 0.5))
    thresholds, temperature_scaling = load_thresholds(
        thresholds_file,
        args.task_type,
        fallback_multiclass_threshold=fallback_multiclass_threshold,
    )

    output_dir = Path(args.output_dir or checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = run_config.get("training_script_name") or run_config.get("script_name") or ""
    model_prefix = script_name.replace("train_", "").replace(".py", "") or "frozen"
    run_name = args.run_name or f"{model_prefix}_human_eval_{timestamp}"

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logging.info("Device: %s", device)

    checkpoint_mode = detect_checkpoint_mode(checkpoint_dir)
    logging.info("Checkpoint mode detected: %s", checkpoint_mode)
    if checkpoint_mode == "lora":
        models = load_fold_models_lora(
            checkpoint_dir=checkpoint_dir,
            run_config=run_config,
            num_labels=num_labels,
            device=device,
        )
        # Lazy import only when LoRA inference path is used.
        from transformers import AutoTokenizer

        lora_tokenizer = AutoTokenizer.from_pretrained(run_config.get("mmbert_model_name", "jhu-clsp/mmBERT-base"))
        lora_max_seq_length = int(run_config.get("max_seq_length", 512))
        lora_chunk_overlap = int(run_config.get("chunk_overlap", 0))
        lora_max_chunks_per_doc = int(LORA_EVAL_MAX_CHUNKS_PER_DOC)
    else:
        models = load_fold_models_frozen(checkpoint_dir, hidden_size, intermediate_size, num_labels, dropout, device)
        lora_tokenizer = None
        lora_max_seq_length = 0
        lora_chunk_overlap = 0
        lora_max_chunks_per_doc = 0
    logging.info("Loaded %d fold model(s)", len(models))

    if checkpoint_mode == "frozen":
        embeddings_all = np.load(args.embeddings_path)
        metadata_df = pd.read_csv(args.metadata_path)
        file_to_idx: Dict[str, int] = {str(row["file_name"]): i for i, row in metadata_df.iterrows()}
        logging.info("Embeddings: %s | Metadata rows: %d", embeddings_all.shape, len(file_to_idx))
    else:
        embeddings_all = None
        file_to_idx = {}
        logging.info(
            "LoRA inference mode uses text chunking from dataset input "
            "(max_seq_length=%d, chunk_overlap=%d, max_chunks_per_doc=%d).",
            lora_max_seq_length,
            lora_chunk_overlap,
            lora_max_chunks_per_doc,
        )

    _ensure_langfuse_tracing_enabled()
    # Langfuse (required to load dataset)
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://langfuse.devops.positiongreen.com"),
    )
    dataset = langfuse.get_dataset(args.dataset_name)
    dataset_items = list(dataset.items)
    if args.max_items:
        dataset_items = dataset_items[: args.max_items]
    logging.info("Test dataset '%s': %d items", args.dataset_name, len(dataset_items))

    # -----------------------------------------------------------------------
    # Pass 1: inference — accumulate predictions and gold labels
    # -----------------------------------------------------------------------

    # Each entry: (ds_item, file_name, probs_np, pred_vec_or_class, gold_vec_or_class, item_metrics)
    results_buffer: List[Dict[str, Any]] = []
    skipped = 0

    for ds_item in dataset_items:
        item_meta = ds_item.metadata or {}
        file_name = str(item_meta.get("file_name", ""))
        if checkpoint_mode == "frozen":
            if not file_name or file_name not in file_to_idx:
                logging.warning("Skipping id=%s: file_name '%s' not in embeddings", ds_item.id, file_name)
                skipped += 1
                continue
            emb_tensor = torch.tensor(
                embeddings_all[file_to_idx[file_name]].astype(np.float32), dtype=torch.float32
            )
        else:
            emb_tensor = None
            text = ""
            if isinstance(ds_item.input, dict):
                text = str(ds_item.input.get("text", "") or "")
            if not text:
                text = str(ds_item.input or "")
            try:
                lora_input_ids, lora_attention_mask = build_lora_inputs_from_text(
                    text=text,
                    tokenizer=lora_tokenizer,
                    max_seq_length=lora_max_seq_length,
                    chunk_overlap=lora_chunk_overlap,
                    max_chunks_per_doc=lora_max_chunks_per_doc,
                )
            except Exception as exc:
                logging.warning("Skipping id=%s: failed to build LoRA chunks (%s)", ds_item.id, exc)
                skipped += 1
                continue

        if args.task_type == "multilabel":
            if checkpoint_mode == "frozen":
                probs = infer_multilabel(models, emb_tensor, device)
            else:
                probs = infer_lora_multilabel(models, lora_input_ids, lora_attention_mask, device)
            probs = apply_multilabel_temperature_scaling(
                probs,
                categories,
                temperature_scaling,
            )
            pred_vec = np.array(
                [1.0 if probs[i] >= float(thresholds.get(cat, 0.5)) else 0.0
                 for i, cat in enumerate(categories)],
                dtype=np.float32,
            )
            gold_cats = gold_labels_multilabel(ds_item.expected_output, categories, args.abstention_label)
            gold_vec = np.zeros(num_labels, dtype=np.float32)
            for cat in gold_cats:
                gold_vec[categories.index(cat)] = 1.0

            pred_cats = [categories[i] for i in range(num_labels) if pred_vec[i] > 0.5]
            im = item_metrics_from_binary_rows(gold_vec, pred_vec)
            output_results = [
                {"category": cat, "confidence_score": round(float(probs[i]), 6)}
                for i, cat in enumerate(categories)
                if pred_vec[i] > 0.5
            ]
            results_buffer.append({
                "ds_item": ds_item,
                "file_name": file_name,
                "probs": probs,
                "gold_vec": gold_vec,
                "pred_vec": pred_vec,
                "gold_cats": gold_cats,
                "pred_cats": pred_cats,
                "item_metrics": im,
                "output_results": output_results,
            })

        else:  # multiclass
            if checkpoint_mode == "frozen":
                probs = infer_multiclass(models, emb_tensor, device)
            else:
                probs = infer_lora_multiclass(models, lora_input_ids, lora_attention_mask, device)
            max_prob = float(probs.max())
            pred_class = (
                categories[int(np.argmax(probs))]
                if max_prob >= float(thresholds)
                else (args.abstention_label or "abstain")
            )
            gold_class = gold_label_multiclass(ds_item.expected_output, categories, args.abstention_label)
            im = item_metrics_multiclass(gold_class, pred_class, args.abstention_label)
            output_results = [{"category": pred_class, "confidence_score": round(max_prob, 6)}]
            results_buffer.append({
                "ds_item": ds_item,
                "file_name": file_name,
                "probs": probs,
                "gold_class": gold_class,
                "pred_class": pred_class,
                "item_metrics": im,
                "output_results": output_results,
            })

    n_evaluated = len(results_buffer)
    logging.info("Inference done: %d evaluated, %d skipped", n_evaluated, skipped)


    # -----------------------------------------------------------------------
    # Compute aggregate metrics
    # -----------------------------------------------------------------------

    if args.task_type == "multilabel":
        if not results_buffer:
            raise RuntimeError("No items evaluated.")
        gold_mat = np.stack([r["gold_vec"] for r in results_buffer])
        pred_mat = np.stack([r["pred_vec"] for r in results_buffer])
        test_metrics = compute_multilabel_metrics(gold_mat, pred_mat, categories)
    else:
        if not results_buffer:
            raise RuntimeError("No items evaluated.")
        test_metrics = compute_multiclass_metrics(
            [r["gold_class"] for r in results_buffer],
            [r["pred_class"] for r in results_buffer],
            categories,
            args.abstention_label,
        )

    # -----------------------------------------------------------------------
    # Pass 2: Langfuse posting (with full aggregate context in metadata)
    # -----------------------------------------------------------------------

    post_to_langfuse = not args.no_post_to_langfuse
    if post_to_langfuse:
        ts_conf_stats = compute_temperature_scaling_confidence_stats(
            checkpoint_dir=checkpoint_dir,
            task_type=args.task_type,
            categories=categories,
            thresholds=thresholds,
            temperature_scaling=temperature_scaling,
        )
        base_metadata = build_run_metadata(
            run_config,
            cv_results,
            test_metrics,
            minority_classes,
            args.dataset_name,
            temperature_scaling=temperature_scaling,
            temperature_scaling_confidence_stats=ts_conf_stats,
        )
        per_class_thresholds = thresholds if args.task_type == "multilabel" else {"threshold": float(thresholds)}
        run_description = f"Human test set evaluation — {base_metadata['model_name']}"
        post_timing = resolve_langfuse_post_timing_from_env()
        max_attempts = int(post_timing.max_attempts)
        delete_wait_timeout_sec = float(post_timing.delete_wait_timeout_sec)
        delete_wait_poll_sec = float(post_timing.delete_wait_poll_sec)
        verify_timeout_sec = float(post_timing.verify_timeout_sec)
        verify_poll_sec = float(post_timing.verify_poll_sec)
        verify_stable_sec = float(post_timing.verify_stable_sec)
        last_error: Optional[Exception] = None
        posted = 0

        for attempt in range(1, max_attempts + 1):
            trace_namespace = f"post_attempt_{attempt}_{time.time_ns()}"
            _delete_dataset_run_and_wait(
                langfuse=langfuse,
                dataset_name=args.dataset_name,
                run_name=run_name,
                timeout_sec=delete_wait_timeout_sec,
                poll_sec=delete_wait_poll_sec,
            )

            posted = 0
            expected_trace_ids: List[str] = []
            for r in results_buffer:
                ds_item = r["ds_item"]
                dataset_item_id = str(ds_item.id)
                trace_id = make_trace_id(run_name, dataset_item_id, trace_namespace=trace_namespace)
                im = r["item_metrics"]

                run_metadata = dict(base_metadata)
                run_metadata["per_class_thresholds"] = per_class_thresholds
                run_metadata["trace_namespace"] = trace_namespace
                if args.task_type == "multilabel":
                    run_metadata["gold_labels"] = r["gold_cats"]
                    run_metadata["predicted_labels"] = r["pred_cats"]
                    span_input = {"file_name": r["file_name"], "evaluation_type": "human_test_set"}
                else:
                    run_metadata["gold_labels"] = [r["gold_class"]]
                    run_metadata["predicted_labels"] = [r["pred_class"]]
                    span_input = {"file_name": r["file_name"], "evaluation_type": "human_test_set"}

                expected_trace_ids.append(trace_id)
                with langfuse.start_as_current_span(
                    name="oof_validation_prediction",
                    trace_context={"trace_id": trace_id},
                    input=span_input,
                    output={"results": r["output_results"]},
                    metadata=run_metadata,
                ):
                    pass
                create_trace_scores(langfuse, trace_id, im)

                langfuse.api.dataset_run_items.create(
                    request=CreateDatasetRunItemRequest(
                        run_name=run_name,
                        run_description=run_description,
                        metadata=run_metadata,
                        dataset_item_id=dataset_item_id,
                        trace_id=trace_id,
                    )
                )
                posted += 1

            langfuse.flush()
            missing_links: List[str] = []
            missing_traces: List[str] = []
            missing_scores: List[str] = []
            verify_deadline = time.time() + max(1.0, verify_timeout_sec)
            stable_since: Optional[float] = None
            while True:
                missing_links, missing_traces, missing_scores = _validate_langfuse_post_integrity(
                    langfuse=langfuse,
                    dataset_name=args.dataset_name,
                    run_name=run_name,
                    expected_trace_ids=expected_trace_ids,
                )
                now = time.time()
                if not missing_links and not missing_traces and not missing_scores:
                    if stable_since is None:
                        stable_since = now
                    if (now - stable_since) >= max(0.0, verify_stable_sec):
                        break
                else:
                    stable_since = None
                if now >= verify_deadline:
                    break
                time.sleep(max(0.1, verify_poll_sec))
            if not missing_links and not missing_traces and not missing_scores:
                post_run_level_macro_scores(
                    langfuse=langfuse,
                    dataset_name=args.dataset_name,
                    run_name=run_name,
                    macro_precision=float(test_metrics["macro_precision"]),
                    macro_recall=float(test_metrics["macro_recall"]),
                    macro_f1=float(test_metrics["macro_f1"]),
                )
                logging.info(
                    "Posted %d traces to Langfuse | run_name=%s | attempt=%d/%d",
                    posted,
                    run_name,
                    attempt,
                    max_attempts,
                )
                break

            last_error = RuntimeError(
                "Langfuse post integrity failed for run=%s (missing_links=%d missing_traces=%d missing_scores=%d)"
                % (run_name, len(missing_links), len(missing_traces), len(missing_scores))
            )
            logging.warning("%s", last_error)
            if attempt < max_attempts:
                time.sleep(float(attempt))
        else:
            raise RuntimeError(
                f"Failed to persist Langfuse human-eval traces for run={run_name} after {max_attempts} attempts"
            ) from last_error

    # -----------------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------------

    result = {
        "checkpoint_dir": checkpoint_dir,
        "test_dataset": args.dataset_name,
        "task_type": args.task_type,
        "categories": categories,
        "abstention_label": args.abstention_label,
        "num_folds": len(models),
        "num_items_evaluated": n_evaluated,
        "num_items_skipped": skipped,
        "langfuse_run_name": run_name if post_to_langfuse else None,
        **test_metrics,
    }

    out_path = output_dir / "human_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info("Results written to %s", out_path)

    print("\n=== HUMAN EVAL RESULTS ===")
    print(f"Dataset:        {args.dataset_name}")
    print(f"Task type:      {args.task_type}")
    print(f"Items:          {n_evaluated}  (skipped: {skipped})")
    print(f"Macro F1:       {test_metrics['macro_f1']:.4f}")
    print(f"Macro P:        {test_metrics['macro_precision']:.4f}")
    print(f"Macro R:        {test_metrics['macro_recall']:.4f}")
    print(f"Exact match:    {test_metrics['exact_match']:.4f}")
    if test_metrics.get("hamming_loss") is not None:
        print(f"Hamming loss:   {test_metrics['hamming_loss']:.4f}")
    print(f"Abstention:     {test_metrics.get('abstention_rate', 0.0):.4f}")
    print("\nPer-class F1:")
    for cat, m in test_metrics.get("per_class_metrics", {}).items():
        print(f"  {cat:<35} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  sup={m['support']}")


if __name__ == "__main__":
    main()
