# Multilingual Classification & HPO Pipeline

An production-ready, automated machine learning pipeline for training, optimizing, and evaluating transformer-based classification models on multilingual text documents. 

This repository highlights a robust ML engineering workflow—spanning unsupervised document clustering, dataset generation, hyperparameter optimization (HPO) grids, Parameter-Efficient Fine-Tuning (PEFT/LoRA), and model serving.

---

## Technical Details

* **Unsupervised Representation Learning & Clustering**: Combines **UMAP** (for non-linear dimensionality reduction) and **HDBSCAN** (for density-based clustering) on document embeddings to automatically discover structure and construct balanced train/test splits.
* **Parameter-Efficient Fine-Tuning (PEFT/LoRA)**: Adapts pre-trained Hugging Face transformer models (such as `mmBERT`) to down-stream tasks by inserting low-rank adapters, significantly reducing memory footprint and training time.
* **Phase-Based Hyperparameter Optimization (HPO)**: Implements a custom sequential search space orchestrator evaluating optimizer parameters (Learning Rate, Weight Decay), regularization (Dropout), classifier architecture layout, and class imbalance techniques (ratio weighting, focal loss, and minor class oversampling).
* **Robust Validation & Guardrails**: Evaluates training runs using multi-fold **Stratified K-Fold** splits, enforcing strict abstention rate guardrails to reject unstable or collapsed configurations.
* **LLM-in-the-Loop Auto-Labeling**: Integrates **OpenAI Chat Completions** and **Langfuse Prompt Management** to automatically annotate raw documents, dynamically resolve active category spaces, and handle label abstentions.
* **Calibration & Serving**: Implements post-training temperature scaling to calibrate confidence output, and exposes a clean inference interface for real-time classifications.

---

## Core Technologies & Tools

* **Deep Learning Framework**: `PyTorch`
* **Transformer Architectures**: Hugging Face `transformers`, `peft` (LoRA)
* **Classical ML & Data Prep**: `scikit-learn`, `pandas`, `numpy`, `scipy`
* **Data Manifold Learning**: `umap-learn`, `hdbscan`
* **Observability & MLOps**: `langfuse`
* **LLM Integrations**: `openai` (Structured JSON outputs)
* **Document Processing**: `PyMuPDF` (Fitz)

---

## Project Structure

```
├── training_config.json         # Central project configuration file
├── requirements.txt             # Python package dependencies
├── .env.example                 # Environment variables template
├── README.md                    # Project overview & usage documentation
├── data_preparation/
│   ├── embed_documents.py       # Computes transformer embedding tensors for raw text
│   └── cluster_data.py          # Auto-generates splits using UMAP & HDBSCAN clustering
├── training/
│   ├── train_and_optimize_hyperparameters_pipeline.py # HPO sweep orchestrator
│   ├── train_head_only.py       # Trainer for MLP classifiers on frozen embeddings
│   ├── train_lora.py            # LoRA fine-tuning trainer
│   ├── test_model.py            # Model evaluation suite
│   └── optimize_abstention_thresholds.py # Tunes confidence thresholds for unknown classes
└── prediction/
    └── classify.py              # Inference service executing classifications
```

---

## Setup Instructions

1. **Clone and Initialize Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Setup Environment Configuration**:
   ```bash
   cp .env.example .env
   ```

---

## Training Execution Paths

### Path A: Local Offline Run (Human Annotations / No Langfuse)
Use this path when your training documents are labeled by human annotators and stored locally.

1. **Configure and Run the Orchestrator**:
   * Place your raw documents under the configured documents directory (default: `data_preparation/training_documents/`).
   * In [training_config.json](training_config.json), leave `"dataset_name"` and `"prompt_name"` empty (`""`).
   * Run the pipeline orchestrator:
     ```bash
     python3 training/train_and_optimize_hyperparameters_pipeline.py --workflow multilabel_frozen
     ```
     *(The pipeline will automatically extract text, compute embeddings, run UMAP + HDBSCAN, and output the local split CSV file if any are missing).*

2. **Annotate Ground Truth Labels**:
   Open the newly generated [training/document_train_test_split.csv](training/document_train_test_split.csv) file and populate the `label_classes` column with your category names (comma-separated for multilabel) for your documents.

3. **Optimize and Train**:
   Rerun the orchestrator command:
   ```bash
   python3 training/train_and_optimize_hyperparameters_pipeline.py --workflow multilabel_frozen
   ```
   The pipeline will read the annotated split CSV, run Stratified K-Fold validation, execute the hyperparameter grid sweeps, tune confidence thresholds, and save the final models.

---

### Path B: LLM-Annotated Run (Langfuse & OpenAI Path)
Use this path to run an automated, LLM-in-the-loop classification setup.

1. **Set Up Credentials**:
   Add your OpenAI API keys and Langfuse Project Keys in your local `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_key
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

2. **Define Langfuse Prompts & Datasets**:
   * Create a classification prompt template in Langfuse (e.g., `classification_prompt`). This prompt defines the classification rules and JSON output schema (see [sample_prompt.txt](sample_prompt.txt) for a template example).
   * Create empty datasets for testing and training in Langfuse.
   * Upload your dataset items under the matching name inside the Langfuse dashboard.

3. **Configure Settings**:
   Configure [training_config.json](training_config.json) to reference your Langfuse resources:
   ```json
   {
     "task_name": "text_classifier",
     "task_type": "multilabel",
     "prompt_name": "classification_prompt",
     "dataset_name": "training_dataset_v1",
     "testset_name": "testset_dataset_v1",
     "abstention_label": "abstain"
   }
   ```

4. **Launch HPO Orchestrator**:
   ```bash
   python3 training/train_and_optimize_hyperparameters_pipeline.py --workflow multilabel_frozen
   ```
   The orchestrator will:
   * Automatically generate local document embeddings and split CSV files if they are missing.
   * Retrieve dataset items and prompt rules from the Langfuse server (or create them automatically via OpenAI auto-labeling if they do not yet exist on the server).
   * Call OpenAI Chat completions on-the-fly to annotate text fragments.
   * Extract unique categories returned by the LLM to dynamically establish the model's target labels.
   * Automatically flag items returning empty label lists as `"is_abstention": True`.
   * Cache results and fine-tune classifiers.

   *Note: A complete classification prompt template detailing the classification categories and JSON schema output format is available in [sample_prompt.txt](sample_prompt.txt).*


