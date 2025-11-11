# pip install gensim numpy pandas
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------
# 1) Load period embeddings
# -------------------------
m1 = KeyedVectors.load_word2vec_format("period_models/1756-1764_vectors.bin", binary=True)
m2 = KeyedVectors.load_word2vec_format("period_models/1765-1776_vectors.bin", binary=True)
m3 = KeyedVectors.load_word2vec_format("period_models/1777-1783_vectors.bin", binary=True)

# -----------------------------------------
# 2) Shared vocabulary across all 3 periods
# -----------------------------------------
shared_vocab = sorted(set(m1.key_to_index) & set(m2.key_to_index) & set(m3.key_to_index))

# Helper to build a matrix (rows=vocab order, cols=embedding dims) from KeyedVectors
def matrix_from_kv(kv: KeyedVectors, vocab: list[str]) -> np.ndarray:
    # raises KeyError if a word missing; we rely on shared_vocab to avoid that
    return np.vstack([kv[w] for w in vocab]).astype(np.float64)

# ---------------------------------------------------
# 3) Alignment: mean-center + orthogonal Procrustes
#     Align 'target' to 'base' using shared_vocab
# ---------------------------------------------------
def align_vectors(base_kv: KeyedVectors, target_kv: KeyedVectors, vocab: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      Y_aligned: (len(vocab) x dim) target matrix, aligned to base
      R: rotation matrix (dim x dim)
    """
    X = matrix_from_kv(base_kv, vocab)   # base
    Y = matrix_from_kv(target_kv, vocab) # target

    # mean-center (no scaling)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # SVD on Yc^T Xc (classic orthogonal Procrustes)
    M = Yc.T @ Xc
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt

    # Optional: ensure a proper rotation (det=+1), avoid reflections
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    Y_aligned = Yc @ R
    return Y_aligned, R

# Align m2 and m3 to m1 in the shared space
m2_aligned_mat, R2 = align_vectors(m1, m2, shared_vocab)
m3_aligned_mat, R3 = align_vectors(m1, m3, shared_vocab)

# Build index for fast row lookups
idx = {w: i for i, w in enumerate(shared_vocab)}

# -----------------------
# 4) Cosine + comparisons
# -----------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom != 0 else 0.0

def get_vec_period(word: str, period: str) -> np.ndarray:
    """
    Retrieve a word vector for a given period:
     - '1756-1764': from m1 (no rotation)
     - '1765-1776': from m2_aligned_mat
     - '1777-1783': from m3_aligned_mat
    """
    if word not in idx:
        raise KeyError(f"'{word}' not found in shared vocabulary.")
    i = idx[word]
    if period == "1756-1764":
        return m1[word]
    elif period == "1765-1776":
        return m2_aligned_mat[i]
    elif period == "1777-1783":
        return m3_aligned_mat[i]
    else:
        raise ValueError("Unknown period label")

# Tyranny vectors
tyr1 = get_vec_period("tyranny", "1756-1764")
tyr2 = get_vec_period("tyranny", "1765-1776")
tyr3 = get_vec_period("tyranny", "1777-1783")

similarities = {
    "1756-1764 → 1765-1776": cosine(tyr1, tyr2),
    "1765-1776 → 1777-1783": cosine(tyr2, tyr3),
    "1756-1764 → 1777-1783": cosine(tyr1, tyr3),
}
print(similarities)

# Tyranny ↔ Popery within each period
pop1 = get_vec_period("popery", "1756-1764")
pop2 = get_vec_period("popery", "1765-1776")
pop3 = get_vec_period("popery", "1777-1783")

lib1 = get_vec_period("liberty", "1756-1764")
lib2 = get_vec_period("liberty", "1765-1776")
lib3 = get_vec_period("liberty", "1777-1783")

tyr_pop = {
    "1756-1764": cosine(tyr1, pop1),
    "1765-1776": cosine(tyr2, pop2),
    "1777-1783": cosine(tyr3, pop3),
}
print(tyr_pop)

popery_similarities = {
    "1756-1764 → 1765-1776": cosine(pop1, pop2),
    "1765-1776 → 1777-1783": cosine(pop2, pop3),
    "1756-1764 → 1777-1783": cosine(pop1, pop3),
}
print(popery_similarities)
liberty_similarities = {
    "1756-1764 → 1765-1776": cosine(lib1, lib2),
    "1765-1776 → 1777-1783": cosine(lib2, lib3),
    "1756-1764 → 1777-1783": cosine(lib1, lib3),
}
# -----------------------------------------
# 5) Nearest neighbors from a raw matrix M
#    (like your nearest_from_matrix in R)
# -----------------------------------------
def nearest_from_matrix(M: np.ndarray, vocab: list[str], word: str, n: int = 10) -> list[tuple[str, float]]:
    if word not in vocab:
        raise KeyError(f"'{word}' not in vocab")
    wi = vocab.index(word)

    # Normalize matrix rows and query vector for cosine
    row_norms = np.linalg.norm(M, axis=1, keepdims=True)
    M_norm = np.divide(M, np.maximum(row_norms, 1e-12))
    v = M_norm[wi]

    sims = M_norm @ v  # cosine similarities to all rows
    # Drop the word itself and take top-n
    sims[wi] = -np.inf
    top_idx = np.argpartition(-sims, kth=min(n, len(vocab)-1))[:n]
    # Sort those top n
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(vocab[i], float(sims[i])) for i in top_idx]

# Example usage:
# Build matrices for aligned spaces (so we can use nearest_from_matrix)
X1 = matrix_from_kv(m1, shared_vocab)                # base (not centered/rotated)
X2 = m2_aligned_mat                                  # already centered+rotated
X3 = m3_aligned_mat                                  # already centered+rotated

print(nearest_from_matrix(X1, shared_vocab, "tyranny", n=10))
print(nearest_from_matrix(X2, shared_vocab, "tyranny", n=10))
print(nearest_from_matrix(X3, shared_vocab, "tyranny", n=10))

import json

# ---------------------------
# 6) Save results to files
# ---------------------------

output_dir = Path("alignment_results")
output_dir.mkdir(parents=True, exist_ok=True)

# ---- Save cosine similarities ----
similarities_df = pd.DataFrame(
    list(similarities.items()), columns=["Comparison", "Cosine_Similarity"]
)
similarities_df.to_csv(output_dir / "tyranny_similarities.csv", index=False)

# ---- Save Tyranny ↔ Popery associations ----
tyr_pop_df = pd.DataFrame(
    list(tyr_pop.items()), columns=["Period", "Cosine_Similarity"]
)
tyr_pop_df.to_csv(output_dir / "tyranny_popery_association.csv", index=False)

# ---- Save Nearest Neighbors ----
neighbors_data = []
for period_label, M in zip(["1756-1764", "1765-1776", "1777-1783"], [X1, X2, X3]):
    try:
        nn_list = nearest_from_matrix(M, shared_vocab, "tyranny", n=10)
        for word, score in nn_list:
            neighbors_data.append({"Period": period_label, "Neighbor": word, "Cosine_Similarity": score})
    except KeyError:
        print(f"⚠️ 'tyranny' not found in vocabulary for {period_label}")

neighbors_df = pd.DataFrame(neighbors_data)
neighbors_df.to_csv(output_dir / "tyranny_neighbors.csv", index=False)

# ---- Save everything as JSON too ----
results = {
    "tyranny_period_similarities": similarities,
    "tyranny_popery_association": tyr_pop,
    "tyranny_neighbors": neighbors_data,
    "popery_sim": popery_similarities,
    "liberty_sim": liberty_similarities,
}
with open(output_dir / "tyranny_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_dir.resolve()}")

