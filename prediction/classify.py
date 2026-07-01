"""
Generic Inference/Prediction script.

Loads a trained model checkpoint (head-only or LoRA) and classifies input texts dynamically.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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
        batch_size, num_chunks, seq_len = input_ids.shape
        hidden_size = self.dense.in_features if self.use_intermediate else self.classifier.in_features
        
        ids_flat = input_ids.view(batch_size * num_chunks, seq_len)
        mask_flat = attention_mask.view(batch_size * num_chunks, seq_len)
        
        valid_rows = (mask_flat.sum(dim=1) > 0)
        emb_flat = ids_flat.new_zeros((ids_flat.shape[0], hidden_size), dtype=torch.float32)
        
        if valid_rows.any():
            ids_valid = ids_flat[valid_rows]
            mask_valid = mask_flat[valid_rows]
            
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

def chunk_text(text: str, tokenizer: AutoTokenizer, max_seq_length: int = 512, chunk_overlap: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        tokens = [tokenizer.pad_token_id or 0]
        
    chunks = []
    step = max_seq_length - chunk_overlap
    for i in range(0, len(tokens), step):
        chunks.append(tokens[i : i + max_seq_length])
        if i + max_seq_length >= len(tokens):
            break
            
    chunk_lengths = [len(c) for c in chunks]
    max_chunk_len = max(chunk_lengths)

    chunk_input_ids = []
    chunk_attention_masks = []

    for chunk_tokens in chunks:
        actual_len = len(chunk_tokens)
        pad_id = tokenizer.pad_token_id or 0
        if actual_len < max_chunk_len:
            pad = max_chunk_len - actual_len
            chunk_tokens = chunk_tokens + [pad_id] * pad
            chunk_mask = [1] * actual_len + [0] * pad
        else:
            chunk_mask = [1] * actual_len
        chunk_input_ids.append(chunk_tokens)
        chunk_attention_masks.append(chunk_mask)
        
    # Stack to batch format [1, num_chunks, seq_len]
    return torch.tensor([chunk_input_ids], dtype=torch.long), torch.tensor([chunk_attention_masks], dtype=torch.long)

def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def classify_text(
    text: str,
    meta: Dict[str, Any],
    model_dir: str,
    device: torch.device
) -> Dict[str, Any]:
    # Check if this is a LoRA model or head-only model
    is_lora = "lora_r" in meta
    model_name = meta.get("model_name") or meta.get("mmbert_model_name") or "jhu-clsp/mmBERT-base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        
    input_ids, attention_mask = chunk_text(text, tokenizer, max_seq_length=512, chunk_overlap=0)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # We load fold 0 checkpoint by default for inference
    checkpoint_path = os.path.join(model_dir, "best_model_fold_0.pt")
    if not os.path.exists(checkpoint_path):
        # check if model_fold_0 exists or any model checkpoint exists
        candidates = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint .pt files found in {model_dir}")
        checkpoint_path = os.path.join(model_dir, candidates[0])
        
    logging.info("Loading checkpoint from: %s", checkpoint_path)
    
    if is_lora:
        raw_backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        from peft import LoraConfig, get_peft_model
        lora_target = ["attn.Wqkv", "attn.Wo"] if meta.get("lora_target_set") == "attn" else ["query", "value"]
        lora_cfg = LoraConfig(
            r=meta["lora_r"],
            lora_alpha=meta["lora_alpha"],
            target_modules=lora_target,
            lora_dropout=0.05,
            bias="none"
        )
        backbone_lora = get_peft_model(raw_backbone, lora_cfg)
        model = LoRAClassifier(
            backbone=backbone_lora,
            hidden_size=meta["hidden_size"],
            intermediate_size=meta["intermediate_size"],
            num_labels=len(meta["train_labels"]),
            dropout=0.0
        ).to(device)
    else:
        # head only model needs embedding extracted first
        raw_backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        raw_backbone.eval()
        with torch.no_grad():
            batch_size, num_chunks, seq_len = input_ids.shape
            ids_flat = input_ids.view(batch_size * num_chunks, seq_len)
            mask_flat = attention_mask.view(batch_size * num_chunks, seq_len)
            
            valid_rows = (mask_flat.sum(dim=1) > 0)
            emb_flat = ids_flat.new_zeros((ids_flat.shape[0], meta["hidden_size"]), dtype=torch.float32)
            
            if valid_rows.any():
                ids_valid = ids_flat[valid_rows]
                mask_valid = mask_flat[valid_rows]
                max_len_local = int(mask_valid.sum(dim=1).max().item())
                ids_valid = ids_valid[:, :max_len_local]
                mask_valid = mask_valid[:, :max_len_local]
                
                outputs = raw_backbone(input_ids=ids_valid, attention_mask=mask_valid)
                emb_valid = masked_mean_pool(outputs.last_hidden_state, mask_valid)
                emb_flat[valid_rows] = emb_valid.to(dtype=emb_flat.dtype)
                
            chunk_emb = emb_flat.view(batch_size, num_chunks, meta["hidden_size"])
            chunk_valid = (attention_mask.sum(dim=2) > 0).unsqueeze(-1)
            
            # max-pool chunks to create document vector
            fill_val = torch.finfo(chunk_emb.dtype).min
            masked = chunk_emb.masked_fill(~chunk_valid.expand_as(chunk_emb), fill_val)
            doc_emb = masked.max(dim=1).values
            
        model = FrozenMLPHead(
            hidden_size=meta["hidden_size"],
            intermediate_size=meta["intermediate_size"],
            num_labels=len(meta["train_labels"]),
            dropout=0.0
        ).to(device)
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        if is_lora:
            logits = model(input_ids, attention_mask)
        else:
            logits = model(doc_emb)
            
    task_type = meta["task_type"]
    train_labels = meta["train_labels"]
    abstain_label = meta["abstain_label"]
    thresholds = meta["thresholds"]
    
    predictions = []
    probabilities = {}
    
    if task_type == "multilabel":
        probs = (1 / (1 + torch.exp(-logits))).cpu().numpy()[0]
        for idx, prob in enumerate(probs):
            lbl_name = train_labels[idx]
            probabilities[lbl_name] = float(prob)
            thresh = thresholds.get(lbl_name, 0.5)
            if prob >= thresh:
                predictions.append(lbl_name)
        if not predictions:
            predictions.append(abstain_label)
    else:
        probs = (torch.softmax(logits, dim=1)).cpu().numpy()[0]
        for idx, prob in enumerate(probs):
            lbl_name = train_labels[idx]
            probabilities[lbl_name] = float(prob)
            
        max_idx = np.argmax(probs)
        max_prob = probs[max_idx]
        abstain_thresh = thresholds.get("abstain_threshold", 0.5)
        
        if max_prob >= abstain_thresh:
            predictions.append(train_labels[max_idx])
        else:
            predictions.append(abstain_label)
            
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "task_type": task_type
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Generic Prediction Classification Service CLI")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to checkpoint dir containing best_model_fold_0.pt and model_metadata.json")
    parser.add_argument("--text", type=str, default=None, help="Document text string to classify")
    parser.add_argument("--text-file", type=str, default=None, help="Path to text document file to classify")
    
    args = parser.parse_args()
    
    text = args.text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
            
    if not text:
        raise ValueError("Must specify either --text or --text-file to classify.")
        
    meta_path = os.path.join(args.model_dir, "model_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find model_metadata.json at {meta_path}")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    device = choose_device()
    logging.info("Using device: %s", device)
    
    result = classify_text(text, meta, args.model_dir, device)
    
    print("\n" + "="*40)
    print("Classification Inference Results:")
    print("="*40)
    print(f"Task Type: {result['task_type']}")
    print(f"Predicted Labels: {result['predictions']}")
    print("Class Probabilities:")
    for lbl, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        print(f"  - {lbl}: {prob:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
