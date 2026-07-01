import pandas as pd
import numpy as np
from pathlib import Path
import umap
import hdbscan
from sklearn.preprocessing import normalize
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_SPLIT_CSV = PROJECT_ROOT / "training" / "document_train_test_split.csv"
EMBEDDINGS_PATH = SCRIPT_DIR / "train_test_document_embeddings.npy"
METADATA_PATH = SCRIPT_DIR / "document_metadata.json"

# --- CLUSTERING & DIVERSITY SAMPLING ---
# Load the data
X = np.load(str(EMBEDDINGS_PATH))
with METADATA_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"Loaded {X.shape[0]} embeddings with {X.shape[1]} dimensions.")

# Fallback for small datasets (avoid UMAP/HDBSCAN solver exceptions)
if X.shape[0] < 15:
    import sys
    print(f"Dataset size too small (N={X.shape[0]}) to perform UMAP/HDBSCAN clustering. Falling back to trivial split...")
    df_final = pd.DataFrame(metadata)
    df_final['cluster'] = 0
    df_final['outlier_score'] = 0.0
    df_final['is_gold_sample'] = False
    
    # Deterministic split: test set contains every 5th item (or at least first item)
    gold_indices = [i for i in range(X.shape[0]) if i % 5 == 0]
    if not gold_indices and X.shape[0] > 0:
        gold_indices = [0]
    
    df_final.loc[gold_indices, 'is_gold_sample'] = True
    df_final.to_csv(str(OUTPUT_SPLIT_CSV), index=False)
    print(f"Success! Gold Set size: {len(gold_indices)}")
    sys.exit(0)

# Step 5: Normalization & UMAP
# Why: Normalization removes length-magnitude bias. 
# UMAP reduces 768 dims to 15 dims to help HDBSCAN find density.
print("Starting normalized UMAP...")
X_norm = normalize(X)
reducer = umap.UMAP(n_components=30, metric='cosine', n_neighbors=5, random_state=42, min_dist=0.0)
X_reduced = reducer.fit_transform(X_norm)

# Step 6: HDBSCAN Clustering
# Why: It finds clusters of varying shapes automatically.
print("Starting clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=5 , metric='euclidean', gen_min_span_tree=True, cluster_selection_method='leaf')
cluster_labels = clusterer.fit_predict(X_reduced)

# --- STEP 7: SCALABLE MEDOID + OUTLIER SAMPLING ---
print("Starting Scalable Medoid + Outlier selection for Gold Set...")
N = X.shape[0]

# Target 20% of the dataset for the test set (gold sample set)
target_test_size = max(5, int(N * 0.20))
target_medoid_count = target_test_size // 2
target_outlier_count = target_test_size - target_medoid_count

gold_set_indices = []

# A. CLUSTER MEDOIDS
unique_labels = set(cluster_labels)
if -1 in unique_labels: 
    unique_labels.remove(-1) # Handle noise (outliers) separately below

C = len(unique_labels)
if C > 0:
    # Determine how many representatives per cluster to sample
    medoids_per_cluster = max(1, target_medoid_count // C)
    print(f"Sampling up to {medoids_per_cluster} medoid(s) per cluster across {C} clusters (Target: {target_medoid_count}).")
    
    for label in unique_labels:
        indices = np.where(cluster_labels == label)[0]
        cluster_points = X_reduced[indices]
        
        # Calculate geometric center of the cluster
        center = np.median(cluster_points, axis=0)
        
        # Calculate Euclidean distances to center
        distances = np.linalg.norm(cluster_points - center, axis=1)
        sorted_cluster_indices = indices[np.argsort(distances)]
        
        num_to_sample = min(medoids_per_cluster, len(sorted_cluster_indices))
        top_indices = sorted_cluster_indices[:num_to_sample]
        gold_set_indices.extend(top_indices.tolist())

# B. GLOBAL OUTLIERS (Anomalies)
outlier_scores = clusterer.outlier_scores_
top_outlier_indices = np.argsort(outlier_scores)[-target_outlier_count:]
print(f"Sampling top {target_outlier_count} global outliers (out of {N} documents).")
gold_set_indices.extend(top_outlier_indices.tolist())

# Merge sets and remove any potential duplicates
gold_set_indices = list(set(gold_set_indices))

# --- UPDATE DATAFRAME ---
df_final = pd.DataFrame(metadata)
df_final['cluster'] = cluster_labels
df_final['outlier_score'] = outlier_scores
df_final['is_gold_sample'] = False
df_final.loc[gold_set_indices, 'is_gold_sample'] = True

# Save to CSV
df_final.to_csv(str(OUTPUT_SPLIT_CSV), index=False)

print(f"Success! Gold Set size: {len(gold_set_indices)} (out of {N} total documents)")
print(f"Breakdown: {len(unique_labels)} clusters sampled + {target_outlier_count} outliers.")
