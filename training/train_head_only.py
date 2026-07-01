"""
Generic Head-Only Trainer for pre-computed frozen BERT/transformer embeddings.

Supports both multiclass and multilabel tasks dynamically using command-line arguments.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Global task spec configuration
@dataclass
class TaskSpec:
    task_type: str  # "multiclass" | "multilabel"
    dataset_name: str
    train_labels: List[str]
    abstain_label: str
    checkpoint_dir: str

class FrozenMLPHead(nn.Module):
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

class FrozenEmbeddingDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        embeddings_all: np.ndarray,
        train_labels: List[str],
        task_type: str,
    ) -> None:
        self.items = items
        self.embeddings_all = embeddings_all
        self.train_labels = train_labels
        self.task_type = task_type
        self.label_to_idx = {name: idx for idx, name in enumerate(train_labels)}
        
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        emb_idx = item["embedding_idx"]
        emb = torch.tensor(self.embeddings_all[emb_idx], dtype=torch.float32)
        
        if self.task_type == "multilabel":
            target = torch.zeros(len(self.train_labels), dtype=torch.float32)
            for lbl in item.get("labels", []):
                if lbl in self.label_to_idx:
                    target[self.label_to_idx[lbl]] = 1.0
        else:
            # Multiclass target
            target = torch.tensor(-1, dtype=torch.long)
            for lbl in item.get("labels", []):
                if lbl in self.label_to_idx:
                    target = torch.tensor(self.label_to_idx[lbl], dtype=torch.long)
                    break
        return emb, target

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
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
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
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            
            all_logits.append(logits.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    val_loss = total_loss / len(loader)
    
    if task_type == "multilabel":
        probs = 1 / (1 + np.exp(-all_logits))
        preds = (probs >= 0.5).astype(float)
        
        # Compute macro/micro/weighted metrics
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, preds, average="macro", zero_division=0)
        exact_match = accuracy_score(all_targets, preds)
        
        return {
            "val_loss": val_loss,
            "macro_f1": f1,
            "macro_precision": prec,
            "macro_recall": rec,
            "exact_match_accuracy": exact_match,
            "probs": probs,
            "targets": all_targets
        }
    else:
        probs = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True)
        preds = np.argmax(all_logits, axis=1)
        
        # Filter labeled target indices (ignoring -1 if any)
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
    embeddings_all: np.ndarray,
    task_spec: TaskSpec,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    n_splits = config["n_splits"]
    random_state = config.get("random_state", config.get("seed", 42))
    num_labels = len(task_spec.train_labels)
    
    # Try importing MultilabelStratifiedKFold if multilabel
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
        # build single-label arrays for splits
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
        # Multilabel splits
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
        
        train_dataset = FrozenEmbeddingDataset(train_items, embeddings_all, task_spec.train_labels, task_spec.task_type)
        val_dataset = FrozenEmbeddingDataset(val_items, embeddings_all, task_spec.train_labels, task_spec.task_type)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        model = FrozenMLPHead(
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_labels=num_labels,
            dropout=config["dropout"]
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        
        if task_spec.task_type == "multilabel":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            
        best_val_loss = float("inf")
        best_metric = -1.0
        
        for epoch in range(config["num_epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_res = evaluate(model, val_loader, criterion, device, task_spec.task_type)
            
            # Use macro F1 for selection checkpoint
            curr_metric = val_res["macro_f1"]
            
            if curr_metric > best_metric or (curr_metric == best_metric and val_res["val_loss"] < best_val_loss):
                best_metric = curr_metric
                best_val_loss = val_res["val_loss"]
                # Save checkpoint
                checkpoint_path = os.path.join(task_spec.checkpoint_dir, f"best_model_fold_{fold}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                
            if epoch % 20 == 0 or epoch == config["num_epochs"] - 1:
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
        # Optimize per-class threshold
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
        # Optimize single threshold for abstention
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

    parser = argparse.ArgumentParser(description="Generic Frozen Embedding Classifier Trainer")
    parser.add_argument("--task-type", type=str, choices=["multiclass", "multilabel"], required=(cfg.get("task_type") is None), default=cfg.get("task_type"))
    parser.add_argument("--embeddings-path", type=str, default="data_preparation/train_test_document_embeddings.npy")
    parser.add_argument("--metadata-path", type=str, default="training/document_train_test_split.csv")
    parser.add_argument("--dataset-cache-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--dataset-name", type=str, default=cfg.get("dataset_name", "generic_dataset"))
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--abstain-label", type=str, default=cfg.get("abstention_label", "abstain"))
    parser.add_argument("--prompt-name", type=str, default=cfg.get("prompt_name", "generic_prompt"))
    
    # Hyperparameters
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    
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
    embeddings_path = config["embeddings_path"]
    if not os.path.exists(embeddings_path):
        import subprocess
        import sys
        logging.info("Embeddings file %s not found. Attempting to generate it by running embed_documents.py...", embeddings_path)
        embed_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_preparation", "embed_documents.py")
        if os.path.exists(embed_script):
            try:
                subprocess.run([sys.executable, embed_script], check=True)
                logging.info("Successfully generated embeddings.")
            except subprocess.CalledProcessError as exc:
                logging.error("Failed to run embed_documents.py script: %s", exc)
                raise RuntimeError("Could not generate embeddings. Please check that raw documents exist in data_preparation/training_documents/") from exc
        else:
            raise FileNotFoundError(f"Embeddings file {embeddings_path} does not exist and embed_documents.py script was not found at {embed_script}")

    embeddings_all = np.load(embeddings_path)
    
    # Run CV
    cv_output = run_cross_validation(
        items=prepared_items,
        embeddings_all=embeddings_all,
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
        "dropout": config["dropout"]
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
