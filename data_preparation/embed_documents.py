import os
import torch
import pymupdf
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import re
import gc
import json
import unicodedata

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent

def load_training_config() -> dict:
    config_path = SCRIPT_DIR.parent / "training_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if not line.strip().startswith(("//", "#"))]
            return json.loads("".join(lines))
        except Exception as exc:
            print(f"[CONFIG_ERROR] Failed to read training_config.json: {exc}")
    return {}

cfg = load_training_config()
MODEL_NAME = cfg.get("backbone_model_name", "jhu-clsp/mmBERT-base")
raw_dir_str = cfg.get("raw_documents_dir", "data_preparation/training_documents")
ROOT_FOLDER = SCRIPT_DIR.parent / raw_dir_str
EMBEDDINGS_OUT = SCRIPT_DIR / "train_test_document_embeddings.npy"
METADATA_OUT = SCRIPT_DIR / "document_metadata.json"

# --- FAIL-FAST PRE-FLIGHT CHECKS ---
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm", ".docx"}

if not os.path.exists(ROOT_FOLDER):
    os.makedirs(ROOT_FOLDER, exist_ok=True)

all_files = [f for f in os.listdir(str(ROOT_FOLDER)) if os.path.isfile(os.path.join(str(ROOT_FOLDER), f)) and not f.startswith(".")]

if not all_files:
    raise ValueError(
        f"Error: The document directory '{ROOT_FOLDER}' is empty.\n"
        "Please place your raw documents in this directory before running the training or embedding scripts."
    )

supported_files = [f for f in all_files if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS]

if not supported_files:
    raise ValueError(
        f"Error: No supported document files found in '{ROOT_FOLDER}'.\n"
        f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}\n"
        f"Found files with extensions: {list(set(Path(f).suffix for f in all_files))}"
    )

MAX_SEQ_LENGTH = 8192  # Backbone context window
CHUNK_OVERLAP = 800    # ~10% overlap
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Step 1 & 2: Load Model and Define Chunking
print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("tokenizer loaded")
print("loading model...")
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("model loaded")
print("moving to device:", DEVICE)
model.to(DEVICE)
print("moved to device")
model.eval()
print("eval set")

def get_chunks(text, tokenizer, max_length, overlap):
    """Splits text into overlapping token chunks for the 8k window."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i : i + max_length]
        chunks.append(chunk)
        if i + max_length >= len(tokens): break
    return chunks

# --- DATA PROCESSING LOOP ---
print("Starting text extraction, embedding and max pooling...")
all_document_vectors = []
metadata = []

def has_meaningful_text(s: str, min_alnum: int = 20, min_words: int = 5) -> bool:
    s = s or ""
    alnum = sum(ch.isalnum() for ch in s)
    words = len(re.findall(r"\w+", s))
    return alnum >= min_alnum and words >= min_words

# Iterate through files in training_documents directory
for file_name in os.listdir(str(ROOT_FOLDER)):
    file_path = os.path.join(str(ROOT_FOLDER), file_name)
    if os.path.isdir(file_path) or file_name.startswith("."):
        continue
        
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        print(f"Skipping unsupported file: {file_name}")
        continue

    try:
        # STEP 1: Text Extraction
        full_text = []
        doc = pymupdf.open(file_path)
        for page in doc: # iterate the document pages
            page_text = page.get_text().strip()
            if ((not has_meaningful_text(page_text)) and (suffix == ".pdf")):
                try:
                    page_text = page.get_textpage_ocr(
                        dpi=300, 
                        full=True,
                        language="eng+swe+nld+dan+fra"
                        ).extractText()
                    print(f"---OCR used for page {page.number} in {file_name}")
                except Exception:
                    pass
            
            # Unicode Normalization for cross-lingual consistency
            normalized_text = unicodedata.normalize('NFC', str(page_text))
            full_text.append(normalized_text)
            
        full_text = "\n".join(full_text)
        word_count = len(full_text.split())
        doc.close()

        # STEP 2: Tokenized Chunking
        token_chunks = get_chunks(full_text, tokenizer, MAX_SEQ_LENGTH, CHUNK_OVERLAP)
        
        chunk_embeddings = []
        
        # STEP 3: Embedding on MPS
        with torch.no_grad():
            for chunk in token_chunks:
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    # Prepare input
                    inputs = torch.tensor([chunk]).to(DEVICE)
                    outputs = model(inputs)
                    
                    # We mean-pool the sequence dimension for each chunk initially
                    chunk_vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    chunk_embeddings.append(chunk_vec)

                print(f"---Processed chunk of size {len(chunk)} tokens for document {file_name}")
                
                # Clear local tensors
                del inputs, outputs

        # STEP 4: Max-Pooling (Global Vector)
        doc_vector = np.max(np.vstack(chunk_embeddings), axis=0)
        
        all_document_vectors.append(doc_vector)
        metadata.append({
            'file_name': file_name,
            'word_count': word_count,
            'path': file_path
        })

        # CRITICAL: Clear MPS Cache after every document
        torch.mps.empty_cache()
        gc.collect()
        print(f"Processed: {file_name} ({len(token_chunks)} chunks)")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Convert to Numpy Array
X = np.array(all_document_vectors)

# Save the numerical vectors to a file
np.save(str(EMBEDDINGS_OUT), X)

# Save the Metadata
with METADATA_OUT.open("w", encoding="utf-8") as f:
    json.dump(metadata, f)
print("Embeddings and metadata saved successfully!")
