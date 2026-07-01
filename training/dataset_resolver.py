"""
Shared utility to resolve datasets for training.

Tries to load from a local cache file first.
If credentials exist:
  - Fetches the dataset from Langfuse.
  - If not found, automatically creates the golden dataset using OpenAI labeling.
If credentials do not exist:
  - Falls back to local training from ground-truth CSV columns.
  - If no ground truth labels are present, stops execution and warns the user.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pymupdf
except ImportError:
    pymupdf = None

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None

try:
    import openai
except ImportError:
    openai = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def has_meaningful_text(s: str, min_alnum: int = 20, min_words: int = 5) -> bool:
    s = s or ""
    alnum = sum(ch.isalnum() for ch in s)
    words = len(re.findall(r"\w+", s))
    return alnum >= min_alnum and words >= min_words

def extract_full_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found at: {file_path}")
        
    suffix = Path(file_path).suffix.lower()
    if suffix != ".pdf":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if pymupdf is None:
        raise ImportError("pymupdf is required to extract text from PDF files. Install with: pip install PyMuPDF")

    full_text = []
    doc = pymupdf.open(file_path)
    for page in doc:
        page_text = page.get_text().strip()
        if not has_meaningful_text(page_text):
            # Try OCR fallback if available
            try:
                page_text = page.get_textpage_ocr(dpi=300, full=True, language="eng+swe+nld+dan+fra").extractText()
            except Exception:
                pass
        full_text.append(unicodedata.normalize("NFC", str(page_text)))
    doc.close()
    return "\n".join(full_text)

def ensure_metadata_split_csv(config: Dict[str, Any]) -> None:
    metadata_path = config["metadata_path"]
    if not os.path.exists(metadata_path):
        logging.info("Metadata split CSV file '%s' not found. Attempting to generate it automatically...", metadata_path)
        
        # 1. Run embed_documents.py if embeddings .npy file is missing
        embeddings_path = config.get("embeddings_path") or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_preparation", "train_test_document_embeddings.npy")
        if not os.path.exists(embeddings_path):
            logging.info("Embeddings file '%s' not found. Running embed_documents.py first...", embeddings_path)
            embed_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_preparation", "embed_documents.py")
            if os.path.exists(embed_script):
                import subprocess
                import sys
                try:
                    subprocess.run([sys.executable, embed_script], check=True)
                    logging.info("Successfully generated embeddings.")
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError("Failed to generate embeddings. Please check that raw documents exist in data_preparation/training_documents/") from exc
            else:
                raise FileNotFoundError(f"embed_documents.py script was not found at {embed_script}")
        
        # 2. Run cluster_data.py
        cluster_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_preparation", "cluster_data.py")
        if os.path.exists(cluster_script):
            logging.info("Running cluster_data.py to generate metadata split CSV...")
            import subprocess
            import sys
            try:
                subprocess.run([sys.executable, cluster_script], check=True)
                logging.info("Successfully generated metadata split CSV file.")
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("Failed to run cluster_data.py script to generate CSV split file.") from exc
        else:
            raise FileNotFoundError(f"cluster_data.py script was not found at {cluster_script}")

def create_train_test_datasets_in_langfuse(
    langfuse_client: Any,
    dataset_name: str,
    prompt_name: str,
    config: Dict[str, Any]
) -> None:
    if openai is None:
        raise ImportError("openai package is required for golden dataset generation. Install with: pip install openai")
        
    ensure_metadata_split_csv(config)
    metadata_path = config["metadata_path"]
    
    train_name = f"{dataset_name}_train"
    test_name = f"{dataset_name}_test"
    logging.info("Creating datasets '%s' and '%s' in Langfuse using OpenAI auto-labeling...", train_name, test_name)
    
    # Try creating the dataset containers
    for name in [train_name, test_name]:
        try:
            langfuse_client.create_dataset(name=name)
        except Exception as exc:
            logging.info("Dataset container creation note for %s: %s", name, exc)

    df = pd.read_csv(metadata_path)
    if "path" not in df.columns:
        raise ValueError(f"Missing required column 'path' in {metadata_path} to locate raw files.")

    # Retrieve prompt configuration from Langfuse
    try:
        prompt = langfuse_client.get_prompt(prompt_name)
    except Exception as exc:
        raise ValueError(f"Could not retrieve prompt '{prompt_name}' from Langfuse. Ensure it is defined in the dashboard.") from exc

    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = None
    if isinstance(getattr(prompt, "config", None), dict):
        model = prompt.config.get("model")
    if not model:
        model = "gpt-4o-mini" # fallback

    for idx, row in df.iterrows():
        file_path = row["path"]
        file_name = row.get("file_name", os.path.basename(file_path))
        logging.info("Processing: %s", file_name)
        
        # Decide which dataset to upload to
        is_gold = False
        if "is_gold_sample" in row:
            val = row["is_gold_sample"]
            if isinstance(val, bool):
                is_gold = val
            elif str(val).lower() in ["true", "1", "yes"]:
                is_gold = True
        else:
            # 80/20 train/test split fallback
            is_gold = (idx % 5 == 0)
            
        target_dataset = test_name if is_gold else train_name
        
        try:
            text_to_classify = extract_full_text(file_path)
            compiled_prompt = prompt.compile(text=text_to_classify)
            
            completion = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": compiled_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            raw_data = json.loads(completion.choices[0].message.content)
            row_metadata = row.to_dict()
            
            langfuse_client.create_dataset_item(
                dataset_name=target_dataset,
                input={"text": text_to_classify},
                expected_output=raw_data,
                metadata=row_metadata
            )
        except Exception as exc:
            logging.error("Failed to process document %s: %s", file_name, exc)

    langfuse_client.flush()
    logging.info("Train and Test datasets generation completed successfully in Langfuse.")

def extract_categories_from_expected_output(expected_output: Any) -> List[str]:
    if not isinstance(expected_output, dict):
        return []
    results = expected_output.get("results", []) or []
    categories: List[str] = []
    for result in results:
        if isinstance(result, dict):
            cat = result.get("category")
            if isinstance(cat, str) and cat:
                categories.append(cat)
    return categories

def resolve_dataset(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    # 1. Load from Cache if it exists
    cache_path = config.get("dataset_cache_path")
    if cache_path and os.path.exists(cache_path):
        logging.info("Loading dataset items from cache: %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        prepared = payload["prepared_items"]
        labels = payload.get("label_mapping", {}).get("train_labels", [])
        if not labels:
            observed = set()
            for row in prepared:
                observed.update(row.get("labels", []))
            labels = sorted(list(observed))
        return prepared, labels

    # 2. Check for Credentials (screening out dummy credentials starting with 'your_')
    lf_public = os.getenv("LANGFUSE_PUBLIC_KEY")
    lf_secret = os.getenv("LANGFUSE_SECRET_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    has_credentials = bool(
        lf_public and not lf_public.startswith("your_") and
        lf_secret and not lf_secret.startswith("your_") and
        openai_key and not openai_key.startswith("your_")
    )
    has_config = bool(config.get("dataset_name") and config.get("prompt_name"))
    
    if has_credentials and has_config and Langfuse is not None:
        # Langfuse Mode
        logging.info("Langfuse credentials found. Fetching from Langfuse...")
        lf_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        langfuse_client = Langfuse(public_key=lf_public, secret_key=lf_secret, host=lf_host)
        
        dataset_name = config["dataset_name"]
        train_name = f"{dataset_name}_train"
        test_name = f"{dataset_name}_test"
        
        # Verify if datasets exist, otherwise create them
        try:
            dataset_train = langfuse_client.get_dataset(train_name)
            dataset_test = langfuse_client.get_dataset(test_name)
        except Exception:
            logging.info("Datasets '%s' and/or '%s' not found in Langfuse.", train_name, test_name)
            prompt_name = config.get("prompt_name", "generic_prompt")
            create_train_test_datasets_in_langfuse(langfuse_client, dataset_name, prompt_name, config)
            dataset_train = langfuse_client.get_dataset(train_name)
            dataset_test = langfuse_client.get_dataset(test_name)
            
        items_train = list(dataset_train.items)
        items_test = list(dataset_test.items)
        logging.info("Loaded %d train items from '%s' and %d test items from '%s'", len(items_train), train_name, len(items_test), test_name)
        
        # Combine them for cross-validation
        items = items_train + items_test
        
        # Extract labels dynamically
        observed_labels = set()
        for item in items:
            cats = extract_categories_from_expected_output(item.expected_output)
            observed_labels.update(cats)
            
        abstain = str(config.get("abstain_label") or config.get("abstention_label") or "abstain").strip().lower()
        train_labels = config.get("labels")
        if not train_labels:
            train_labels = sorted([lbl for lbl in observed_labels if str(lbl).strip().lower() != abstain])
            
        prepared = []
        for i, item in enumerate(items):
            file_name = ""
            if isinstance(item.metadata, dict):
                file_name = item.metadata.get("file_name", "")
            if not file_name:
                file_name = f"doc_{i}"
                
            raw_cats = extract_categories_from_expected_output(item.expected_output)
            row_labels = [c for c in raw_cats if c in train_labels and str(c).strip().lower() != abstain]
            
            text = ""
            if isinstance(item.input, dict):
                text = str(item.input.get("text", "") or "")
            if not text:
                text = str(item.input)
                
            prepared.append({
                "item_idx": i,
                "raw_item_idx": i,
                "file_name": file_name,
                "embedding_idx": i,
                "labels": row_labels,
                "is_abstention": len(row_labels) == 0,
                "text": text
            })
            
        return prepared, train_labels

    else:
        # Local Mode
        logging.info("Langfuse/OpenAI credentials or dataset_name/prompt_name config missing. Falling back to local offline mode...")
        ensure_metadata_split_csv(config)
        metadata_path = config["metadata_path"]
        df = pd.read_csv(metadata_path)
        
        # Check if ground truth labels are present in split CSV
        labels = config.get("labels") or []
        has_gt_column = "label_classes" in df.columns
        
        if not has_gt_column:
            raise ValueError(
                "Execution Stopped:\n"
                "Missing Langfuse/OpenAI credentials in .env, and the local split CSV file "
                "does not contain a 'label_classes' column for ground truth labels.\n"
                "Please set up credentials or manually annotate your split CSV file."
            )
            
        if not labels:
            observed = set()
            for val in df["label_classes"].dropna():
                observed.update([x.strip() for x in str(val).split(",") if x.strip()])
            labels = sorted(list(observed))
            
        if not labels:
            raise ValueError(
                "Execution Stopped:\n"
                "The 'label_classes' column in your split CSV file is empty.\n"
                "Please manually annotate the labels to run local training."
            )

        prepared = []
        for idx, row in df.iterrows():
            row_labels = []
            if pd.notna(row["label_classes"]):
                row_labels = [x.strip() for x in str(row["label_classes"]).split(",") if x.strip()]
            row_labels = [l for l in row_labels if l in labels]
            
            prepared.append({
                "item_idx": idx,
                "raw_item_idx": idx,
                "file_name": row.get("file_name", f"doc_{idx}"),
                "embedding_idx": idx,
                "labels": row_labels,
                "is_abstention": len(row_labels) == 0,
                "text": "Placeholder text since loaded locally without doc chunks."
            })
        return prepared, labels
