"""
Generic LoRA Fine-Tuning Trainer for Transformer Backbone Models.

Supports both multiclass and multilabel classification tasks dynamically using command-line arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:
    raise ImportError("peft is required for LoRA training. Install with: pip install peft") from exc

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

@dataclass
class TaskSpec:
    task_type: str  # "multiclass" | "multilabel"
    dataset_name: str
    train_labels: List[str]
    abstain_label: str
    checkpoint_dir: str

def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

class LoRAClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        num_labels: int,
        dropout: float,
        chunk_pooling: str = "max"
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.chunk_pooling = chunk_pooling
        self.dropout1 = nn.Dropout(dropout)
        self.use_intermediate = intermediate_size is not None and int(intermediate_size) > 0
        if self.use_intermediate:
            self.dense = nn.Linear(hidden_size, intermediate_size)
            self.layer_norm = nn.LayerNorm(intermediate_size)
            self.dropout2 = nn.Dropout(dropout)
            self.classifier = nn.Linear(intermediate_size, num_labels)
        else:
            self.classifier = nn.Linear(hidden_size, num_labels)

    def _pool_chunks(self, chunk_embeddings: torch.Tensor, chunk_valid: torch.Tensor) -> torch.Tensor:
        if self.chunk_pooling == "mean":
            masked = chunk_embeddings * chunk_valid.to(chunk_embeddings.dtype)
            denom = chunk_valid.to(chunk_embeddings.dtype).sum(dim=1).clamp(min=1e-6)
            return masked.sum(dim=1) / denom

        fill_val = torch.finfo(chunk_embeddings.dtype).min
        masked = chunk_embeddings.masked_fill(~chunk_valid.expand_as(chunk_embeddings), fill_val)
        return masked.max(dim=1).values

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, num_chunks, seq_len]
        batch_size, num_chunks, seq_len = input_ids.shape
        hidden_size = self.dense.in_features if self.use_intermediate else self.classifier.in_features
        
        # Flatten chunks to feed through backbone
        ids_flat = input_ids.view(batch_size * num_chunks, seq_len)
        mask_flat = attention_mask.view(batch_size * num_chunks, seq_len)
        
        valid_rows = (mask_flat.sum(dim=1) > 0)
        emb_flat = ids_flat.new_zeros((ids_flat.shape[0], hidden_size), dtype=torch.float32)
        
        if valid_rows.any():
            ids_valid = ids_flat[valid_rows]
            mask_valid = mask_flat[valid_rows]
            
            # Trim sequence length to local max
            max_len_local = int(mask_valid.sum(dim=1).max().item())
            ids_valid = ids_valid[:, :max_len_local]
            mask_valid = mask_valid[:, :max_len_local]
            
            outputs = self.backbone(input_ids=ids_valid, attention_mask=mask_valid)
            emb_valid = masked_mean_pool(outputs.last_hidden_state, mask_valid)
            emb_flat[valid_rows] = emb_valid.to(dtype=emb_flat.dtype)
            
        chunk_emb = emb_flat.view(batch_size, num_chunks, hidden_size)
        chunk_valid = (attention_mask.sum(dim=2) > 0).unsqueeze(-1)
        doc_emb = self._pool_chunks(chunk_emb, chunk_valid)
        
        x = self.dropout1(doc_emb)
        if self.use_intermediate:
            x = self.dense(x)
            x = torch.relu(x)
            x = self.layer_norm(x)
            x = self.dropout2(x)
        return self.classifier(x)

class LoRADataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        train_labels: List[str],
        task_type: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 512,
        chunk_overlap: int = 0
    ) -> None:
        self.items = items
        self.train_labels = train_labels
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chunk_overlap = chunk_overlap
        self.label_to_idx = {name: idx for idx, name in enumerate(train_labels)}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        
        # Tokenize dynamically if chunks not in cache
        if "chunks" in item:
            chunks = item["chunks"]
        else:
            text = item.get("text", "")
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            chunks = []
            if not tokens:
                chunks = [[self.tokenizer.pad_token_id or 0]]
            else:
                step = self.max_seq_length - self.chunk_overlap
                for i in range(0, len(tokens), step):
                    chunks.append(tokens[i : i + self.max_seq_length])
                    if i + self.max_seq_length >= len(tokens):
                        break
                        
        chunk_lengths = [len(c) for c in chunks]
        max_chunk_len = max(chunk_lengths)

        chunk_input_ids = []
        chunk_attention_masks = []

        for chunk_tokens in chunks:
            actual_len = len(chunk_tokens)
            pad_id = self.tokenizer.pad_token_id or 0
            if actual_len < max_chunk_len:
                pad = max_chunk_len - actual_len
                chunk_tokens = chunk_tokens + [pad_id] * pad
                chunk_mask = [1] * actual_len + [0] * pad
            else:
                chunk_mask = [1] * actual_len
            chunk_input_ids.append(chunk_tokens)
            chunk_attention_masks.append(chunk_mask)

        if self.task_type == "multilabel":
            label_vec = np.zeros(len(self.train_labels), dtype=np.float32)
            for cat in item.get("labels", []):
                if cat in self.label_to_idx:
                    label_vec[self.label_to_idx[cat]] = 1.0
            target = torch.tensor(label_vec, dtype=torch.float32)
        else:
            lbl_idx = -1
            for cat in item.get("labels", []):
                if cat in self.label_to_idx:
                    lbl_idx = self.label_to_idx[cat]
                    break
            target = torch.tensor(lbl_idx, dtype=torch.long)

        return {
            "input_ids": torch.tensor(chunk_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(chunk_attention_masks, dtype=torch.long),
            "labels": target,
            "file_name": item.get("file_name", f"doc_{idx}"),
            "item_idx": int(item.get("item_idx", idx)),
            "num_chunks": int(len(chunks)),
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_chunks = max(int(item["num_chunks"]) for item in batch)
    max_seq_len = max(int(item["input_ids"].shape[1]) for item in batch)

    batch_input_ids = []
    batch_attention_masks = []
    batch_labels = []

    for item in batch:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        num_chunks = int(item["num_chunks"])
        seq_len = int(input_ids.shape[1])

        if seq_len < max_seq_len:
            seq_pad = max_seq_len - seq_len
            input_ids = torch.cat([input_ids, torch.zeros(num_chunks, seq_pad, dtype=torch.long)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros(num_chunks, seq_pad, dtype=torch.long)], dim=1)

        if num_chunks < max_chunks:
            chunk_pad = max_chunks - num_chunks
            input_ids = torch.cat([input_ids, torch.zeros(chunk_pad, max_seq_len, dtype=torch.long)], dim=0)
            attention_mask = torch.cat([attention_mask, torch.zeros(chunk_pad, max_seq_len, dtype=torch.long)], dim=0)

        batch_input_ids.append(input_ids)
        batch_attention_masks.append(attention_mask)
        batch_labels.append(item["labels"])

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_masks),
        "labels": torch.stack(batch_labels),
    }

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def lora_target_modules_from_set(target_set: str) -> List[str]:
    if target_set == "attn":
        return ["attn.Wqkv", "attn.Wo"]
    if target_set == "attn_mlp":
        return ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
    return ["query", "value", "key"]  # default fallback

def load_or_prepare_dataset_items(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    try:
        from training.dataset_resolver import resolve_dataset
    except ImportError:
        from dataset_resolver import resolve_dataset
    return resolve_dataset(config)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    val_loss = total_loss / len(loader)
    
    if task_type == "multilabel":
        probs = 1 / (1 + np.exp(-all_logits))
        preds = (probs >= 0.5).astype(float)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, preds, average="macro", zero_division=0)
        acc = accuracy_score(all_targets, preds)
        
        return {
            "val_loss": val_loss,
            "macro_f1": f1,
            "macro_precision": prec,
            "macro_recall": rec,
            "exact_match_accuracy": acc,
            "probs": probs,
            "targets": all_targets
        }
    else:
        probs = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True)
        preds = np.argmax(all_logits, axis=1)
        
        valid = all_targets != -1
        if np.sum(valid) == 0:
            return {"val_loss": val_loss, "macro_f1": 0.0, "accuracy": 0.0, "probs": probs, "targets": all_targets}
            
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets[valid], preds[valid], average="macro", zero_division=0)
        acc = accuracy_score(all_targets[valid], preds[valid])
        
        return {
            "val_loss": val_loss,
            "macro_f1": f1,
            "macro_precision": prec,
            "macro_recall": rec,
            "exact_match_accuracy": acc,
            "probs": probs,
            "targets": all_targets
        }

def run_cross_validation(
    items: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    task_spec: TaskSpec,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    n_splits = config["n_splits"]
    random_state = config.get("random_state", config.get("seed", 42))
    num_labels = len(task_spec.train_labels)
    
    use_ml_strat = False
    if task_spec.task_type == "multilabel":
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
            use_ml_strat = True
        except ImportError:
            logging.warning("iterstrat not available. Falling back to standard KFold for multi-label splits.")
            from sklearn.model_selection import KFold
            
    if task_spec.task_type == "multiclass":
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        label_to_idx = {name: idx for idx, name in enumerate(task_spec.train_labels)}
        targets_split = []
        for item in items:
            lbl_idx = -1
            for lbl in item.get("labels", []):
                if lbl in label_to_idx:
                    lbl_idx = label_to_idx[lbl]
                    break
            targets_split.append(lbl_idx)
        targets_split = np.array(targets_split)
        splits = list(kf.split(np.zeros(len(items)), targets_split))
    else:
        label_to_idx = {name: idx for idx, name in enumerate(task_spec.train_labels)}
        targets_split = np.zeros((len(items), num_labels))
        for i, item in enumerate(items):
            for lbl in item.get("labels", []):
                if lbl in label_to_idx:
                    targets_split[i, label_to_idx[lbl]] = 1.0
        
        if use_ml_strat:
            kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(np.zeros(len(items)), targets_split))
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(kf.split(np.zeros(len(items))))
            
    logging.info("Starting %d-fold cross validation...", n_splits)
    
    oof_probs = np.zeros((len(items), num_labels))
    oof_targets = np.zeros((len(items), num_labels)) if task_spec.task_type == "multilabel" else np.zeros(len(items))
    
    fold_metrics = []
    os.makedirs(task_spec.checkpoint_dir, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        logging.info("--- Fold %d ---", fold)
        
        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]
        
        train_dataset = LoRADataset(train_items, task_spec.train_labels, task_spec.task_type, tokenizer, config["max_seq_length"], config["chunk_overlap"])
        val_dataset = LoRADataset(val_items, task_spec.train_labels, task_spec.task_type, tokenizer, config["max_seq_length"], config["chunk_overlap"])
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
        
        # Load raw backbone model
        logging.info("Loading transformers backbone: %s", config["model_name"])
        raw_backbone = AutoModel.from_pretrained(config["model_name"], trust_remote_code=True)
        
        # Setup LoRA
        lora_target = lora_target_modules_from_set(config["lora_target_set"])
        lora_cfg = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=lora_target,
            lora_dropout=config["lora_dropout"],
            bias="none"
        )
        backbone_lora = get_peft_model(raw_backbone, lora_cfg)
        
        model = LoRAClassifier(
            backbone=backbone_lora,
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_labels=num_labels,
            dropout=config["dropout"]
        ).to(device)
        
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": config["learning_rate_lora"]},
            {"params": model.classifier.parameters(), "lr": config["learning_rate_head"]}
        ], weight_decay=config["weight_decay"])
        
        num_training_steps = len(train_loader) * config["num_epochs"]
        num_warmup_steps = int(num_training_steps * config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        
        if task_spec.task_type == "multilabel":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            
        best_val_loss = float("inf")
        best_metric = -1.0
        
        for epoch in range(config["num_epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            val_res = evaluate(model, val_loader, criterion, device, task_spec.task_type)
            
            curr_metric = val_res["macro_f1"]
            
            if curr_metric > best_metric or (curr_metric == best_metric and val_res["val_loss"] < best_val_loss):
                best_metric = curr_metric
                best_val_loss = val_res["val_loss"]
                # Save checkpoint
                checkpoint_path = os.path.join(task_spec.checkpoint_dir, f"best_model_fold_{fold}.pt")
                # Save state dict
                torch.save(model.state_dict(), checkpoint_path)
                
            if epoch % 10 == 0 or epoch == config["num_epochs"] - 1:
                logging.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | Val Macro F1: {val_res['macro_f1']:.4f}")
                
        # Load best model for OOF predictions
        checkpoint_path = os.path.join(task_spec.checkpoint_dir, f"best_model_fold_{fold}.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        final_res = evaluate(model, val_loader, criterion, device, task_spec.task_type)
        oof_probs[val_idx] = final_res["probs"]
        oof_targets[val_idx] = final_res["targets"]
        fold_metrics.append({
            "fold": fold,
            "val_loss": final_res["val_loss"],
            "macro_f1": final_res["macro_f1"],
            "exact_match_accuracy": final_res["exact_match_accuracy"]
        })
        logging.info("Fold %d Best Val Loss: %.4f | Val Macro F1: %.4f", fold, final_res["val_loss"], final_res["macro_f1"])

    return {
        "oof_probs": oof_probs,
        "oof_targets": oof_targets,
        "fold_metrics": fold_metrics
    }

def optimize_thresholds(
    oof_probs: np.ndarray,
    oof_targets: np.ndarray,
    task_spec: TaskSpec
) -> Dict[str, Any]:
    print("Optimizing classification thresholds...")
    num_labels = len(task_spec.train_labels)
    
    if task_spec.task_type == "multilabel":
        best_thresholds = {}
        for class_idx in range(num_labels):
            class_name = task_spec.train_labels[class_idx]
            best_f1 = -1.0
            best_thresh = 0.5
            
            for thresh in np.arange(0.1, 0.9, 0.02):
                preds = (oof_probs[:, class_idx] >= thresh).astype(float)
                targets = oof_targets[:, class_idx]
                
                prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            best_thresholds[class_name] = float(best_thresh)
        return {"thresholds": best_thresholds, "type": "per_class"}
    else:
        best_thresh = 0.5
        best_f1 = -1.0
        
        for thresh in np.arange(0.1, 0.9, 0.02):
            preds = []
            targets = []
            for i in range(len(oof_probs)):
                max_prob = np.max(oof_probs[i])
                pred_cls = np.argmax(oof_probs[i])
                if max_prob < thresh:
                    pred_label = "abstain"
                else:
                    pred_label = task_spec.train_labels[pred_cls]
                preds.append(pred_label)
                
                target_idx = int(oof_targets[i])
                target_label = task_spec.train_labels[target_idx] if target_idx != -1 else "abstain"
                targets.append(target_label)
                
            prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return {"thresholds": {"abstain_threshold": float(best_thresh)}, "type": "single_abstention"}

def main() -> None:
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

    parser = argparse.ArgumentParser(description="Generic LoRA Fine-Tuning Classifier Trainer")
    parser.add_argument("--task-type", type=str, choices=["multiclass", "multilabel"], required=(cfg.get("task_type") is None), default=cfg.get("task_type"))
    parser.add_argument("--metadata-path", type=str, default="training/document_train_test_split.csv")
    parser.add_argument("--dataset-cache-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints_lora")
    parser.add_argument("--dataset-name", type=str, default=cfg.get("dataset_name", "generic_dataset_lora"))
    parser.add_argument("--model-name", type=str, default=cfg.get("backbone_model_name", "jhu-clsp/mmBERT-base"))
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--abstain-label", type=str, default=cfg.get("abstention_label", "abstain"))
    parser.add_argument("--prompt-name", type=str, default=cfg.get("prompt_name", "generic_prompt"))
    
    # Hyperparameters
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate-lora", type=float, default=2e-4)
    parser.add_argument("--learning-rate-head", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    
    # LoRA config
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-set", type=str, default="attn")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    
    args = parser.parse_args()
    config = vars(args)
    
    set_seed(config["seed"])
    device = choose_device()
    logging.info("Using device: %s", device)
    
    # Load prepared items and dynamic labels
    prepared_items, train_labels = load_or_prepare_dataset_items(config)
    logging.info("Inferred training labels (%d): %s", len(train_labels), train_labels)
    
    task_spec = TaskSpec(
        task_type=config["task_type"],
        dataset_name=config["dataset_name"],
        train_labels=train_labels,
        abstain_label=config["abstain_label"],
        checkpoint_dir=config["checkpoint_dir"]
    )
    
    logging.info("Loading backbone tokenizer: %s", config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        
    # Run CV
    cv_output = run_cross_validation(
        items=prepared_items,
        tokenizer=tokenizer,
        task_spec=task_spec,
        config=config,
        device=device
    )
    
    # Threshold Tuning
    tuning_res = optimize_thresholds(
        oof_probs=cv_output["oof_probs"],
        oof_targets=cv_output["oof_targets"],
        task_spec=task_spec
    )
    
    # Save config, labels, and thresholds metadata
    run_meta = {
        "task_type": task_spec.task_type,
        "dataset_name": task_spec.dataset_name,
        "train_labels": task_spec.train_labels,
        "abstain_label": task_spec.abstain_label,
        "thresholds": tuning_res["thresholds"],
        "threshold_type": tuning_res["type"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "dropout": config["dropout"],
        "model_name": config["model_name"],
        "lora_r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "lora_target_set": config["lora_target_set"]
    }
    
    meta_path = os.path.join(task_spec.checkpoint_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    logging.info("Saved model metadata and thresholds to: %s", meta_path)
    
    # Write summary results
    results_path = os.path.join(task_spec.checkpoint_dir, "cv_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(cv_output["fold_metrics"], f, indent=2)
    logging.info("CV execution completed successfully!")

if __name__ == "__main__":
    main()
