# visualize_alignment_results.py
# pip install gensim numpy pandas scikit-learn matplotlib

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

PERIODS = ["1756-1764", "1765-1776", "1777-1783"]
MODEL_PATHS = {
    "1756-1764": "period_models/1756-1764_vectors.bin",
    "1765-1776": "period_models/1765-1776_vectors.bin",
    "1777-1783": "period_models/1777-1783_vectors.bin",
}
# Probe categories (tweak to your corpus)
PROBES = {
    "religious":  ["religion","church","priest","papist","papists","catholic","romish","popish","bishop","mass"],
    "institution":["parliament","ministry","constitution","governor","crown","assembly","administration"],
    "moral":      ["virtue","vice","corruption","despotism","licentiousness"],
    "emotion":    ["fear","anger","indignation","zeal","alarm","resentment"]
}

OUTDIR = Path("alignment_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Core math / alignment utils
# ---------------------------
def present_words_in_vocab(words, vocab_set):
    return [w for w in words if w in vocab_set]

def centroid_from_matrix(M: np.ndarray, indices: List[int]) -> np.ndarray | None:
    if not indices: return None
    return M[indices].mean(axis=0)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    return float(a @ b / (da * db)) if da > 0 and db > 0 else 0.0

def matrix_from_kv(kv: KeyedVectors, vocab: List[str]) -> np.ndarray:
    return np.vstack([kv[w] for w in vocab]).astype(np.float64)

def align_to_base(base_kv: KeyedVectors,
                  target_kv: KeyedVectors,
                  shared_vocab: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean-center X (base) and Y (target) on shared vocab, compute R via SVD on Y^T X,
    return Y_aligned = (Y - mean_Y) @ R and the rotation matrix R.
    """
    X = matrix_from_kv(base_kv, shared_vocab)
    Y = matrix_from_kv(target_kv, shared_vocab)

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    M = Yc.T @ Xc
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    # reflection fix
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    Y_aligned = Yc @ R
    return Y_aligned, R

# --------------------
# Loading & alignment
# --------------------

def load_models() -> Tuple[KeyedVectors, KeyedVectors, KeyedVectors]:
    m1 = KeyedVectors.load_word2vec_format(MODEL_PATHS["1756-1764"], binary=True)
    m2 = KeyedVectors.load_word2vec_format(MODEL_PATHS["1765-1776"], binary=True)
    m3 = KeyedVectors.load_word2vec_format(MODEL_PATHS["1777-1783"], binary=True)
    return m1, m2, m3

def build_alignment(m1, m2, m3):
    common = sorted(set(m1.key_to_index) & set(m2.key_to_index) & set(m3.key_to_index))
    if not common:
        raise RuntimeError("No shared vocabulary across the three models.")

    # Align m2 and m3 to m1 on shared vocab
    m2_aligned_mat, _ = align_to_base(m1, m2, common)
    m3_aligned_mat, _ = align_to_base(m1, m3, common)

    # convenience index
    idx = {w: i for i, w in enumerate(common)}

    return common, idx, m2_aligned_mat, m3_aligned_mat

def get_vec(word: str,
            period: str,
            m1, m2_aligned_mat, m3_aligned_mat,
            idx: dict):
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

# ----------------
# Viz helpers
# ----------------

def save_bar(values: dict, title: str, filename: Path):
    labels = list(values.keys())
    scores = [values[k] for k in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(labels)), scores)
    plt.xticks(range(len(labels)), labels, rotation=15, ha="right")
    plt.ylabel("Cosine similarity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def save_line(period_to_score: dict, title: str, filename: Path):
    periods = list(period_to_score.keys())
    scores = [period_to_score[p] for p in periods]
    plt.figure(figsize=(7, 4))
    plt.plot(periods, scores, marker="o")
    plt.xlabel("Period")
    plt.ylabel("Cosine similarity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def pca_2d(points: np.ndarray) -> np.ndarray:
    p = PCA(n_components=2, random_state=0)
    return p.fit_transform(points)

def save_neighborhood_scatter(target: str,
                              neighbors: int,
                              m1, m2_aligned_mat, m3_aligned_mat,
                              vocab: List[str], idx: dict,
                              title: str, filename: Path):
    """
    Build neighborhoods around `target` in each aligned space.
    For m1 we use raw m1 vectors; for m2/m3 we use aligned matrices.
    Scatter in shared PCA with period labels.
    """
    if target not in idx:
        raise KeyError(f"'{target}' not found in shared vocabulary.")

    # Build matrices to cosine against
    M1 = matrix_from_kv(m1, vocab)
    M2 = m2_aligned_mat
    M3 = m3_aligned_mat

    def nearest_from_matrix(M: np.ndarray, w: str, k=15):
        wi = idx[w]
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        Mn = M / np.maximum(norms, 1e-12)
        v = Mn[wi]
        sims = Mn @ v
        sims[wi] = -np.inf
        # top k
        k = min(k, len(vocab) - 1)
        top_idx = np.argpartition(-sims, kth=k)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(vocab[i], float(sims[i])) for i in top_idx]

    k = neighbors
    n1 = nearest_from_matrix(M1, target, k=k)
    n2 = nearest_from_matrix(M2, target, k=k)
    n3 = nearest_from_matrix(M3, target, k=k)

    # Collect points: neighbors + the target itself
    words_1 = [target] + [w for w, _ in n1]
    words_2 = [target] + [w for w, _ in n2]
    words_3 = [target] + [w for w, _ in n3]

    vecs_1 = np.vstack([M1[idx[w]] for w in words_1])
    vecs_2 = np.vstack([M2[idx[w]] for w in words_2])
    vecs_3 = np.vstack([M3[idx[w]] for w in words_3])

    # Stack and project once so all periods share a PCA space
    all_vecs = np.vstack([vecs_1, vecs_2, vecs_3])
    all_2d = pca_2d(all_vecs)

    n1n = len(words_1)
    n2n = len(words_2)
    pts1 = all_2d[0:n1n]
    pts2 = all_2d[n1n:n1n+n2n]
    pts3 = all_2d[n1n+n2n:]

    # Plot
    plt.figure(figsize=(7, 6))
    # neighbors faint, targets bold star
    def draw(points, words, label):
        plt.scatter(points[1:, 0], points[1:, 1], alpha=0.6, label=label)
        plt.scatter(points[0, 0], points[0, 1], marker="*", s=150)

        # annotate a few closest neighbors
        for i in range(1,min(8,points.shape[0])):  # annotate 5 nearest by order
            plt.text(points[i, 0], points[i, 1], words[i], fontsize=8, alpha=0.8)

    draw(pts1, words_1, "1756-1764")
    draw(pts2, words_2, "1765-1776")
    draw(pts3, words_3, "1777-1783")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()
def save_neighborhood_scatter_with_centroids(
    target: str,
    neighbors: int,
    m1, m2_aligned_mat, m3_aligned_mat,
    vocab: List[str], idx: dict,
    probe_sets: dict,
    focus_words: List[str],
    title: str, filename: Path
):
    if target not in idx:
        raise KeyError(f"'{target}' not in shared vocabulary.")

    # Aligned spaces over the same vocab order
    M1 = matrix_from_kv(m1, vocab)
    M2 = m2_aligned_mat
    M3 = m3_aligned_mat
    VSET = set(vocab)

    def nearest_from_matrix(M: np.ndarray, w: str, k=15):
        wi = idx[w]
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
        v = Mn[wi]
        sims = Mn @ v
        sims[wi] = -np.inf
        k = min(k, len(vocab) - 1)
        top = np.argpartition(-sims, kth=k)[:k]
        top = top[np.argsort(-sims[top])]
        return [(vocab[i], float(sims[i])) for i in top]

    k = max(2, neighbors)

    # Target neighborhoods per period
    n1 = nearest_from_matrix(M1, target, k=k)
    n2 = nearest_from_matrix(M2, target, k=k)
    n3 = nearest_from_matrix(M3, target, k=k)

    words_1 = [target] + [w for w, _ in n1]
    words_2 = [target] + [w for w, _ in n2]
    words_3 = [target] + [w for w, _ in n3]

    vecs_1 = np.vstack([M1[idx[w]] for w in words_1])
    vecs_2 = np.vstack([M2[idx[w]] for w in words_2])
    vecs_3 = np.vstack([M3[idx[w]] for w in words_3])

    # Joint PCA so all three panels share the same projection
    ALL = np.vstack([vecs_1, vecs_2, vecs_3])
    p = PCA(n_components=2, random_state=0).fit(ALL)
    ALL_2D = p.transform(ALL)

    n1n = len(words_1)
    n2n = len(words_2)
    pts1 = ALL_2D[:n1n]
    pts2 = ALL_2D[n1n:n1n+n2n]
    pts3 = ALL_2D[n1n+n2n:]

    # Precompute probe indices (only words in shared vocab)
    probe_indices = {
        cat: [idx[w] for w in present_words_in_vocab(ws, VSET)]
        for cat, ws in probe_sets.items()
    }

    # Centroids in vector space, then project via same PCA
    def project_centroids(M):
        out = {}
        for cat, inds in probe_indices.items():
            c = centroid_from_matrix(M, inds)
            out[cat] = p.transform(c.reshape(1, -1))[0] if c is not None else None
        return out

    cent1 = project_centroids(M1)
    cent2 = project_centroids(M2)
    cent3 = project_centroids(M3)

    # Plot
    periods = ["1756-1764", "1765-1776", "1777-1783"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    panels = [
        (axes[0], periods[0], words_1, pts1, cent1),
        (axes[1], periods[1], words_2, pts2, cent2),
        (axes[2], periods[2], words_3, pts3, cent3),
    ]

    for ax, period_label, words, pts, cents in panels:
        # neighbors faint + target star
        ax.scatter(pts[1:,0], pts[1:,1], alpha=0.5)
        ax.scatter(pts[0,0], pts[0,1], marker="*", s=150)
        ax.text(pts[0,0], pts[0,1], target, fontsize=9, weight="bold")

        # annotate a few closest neighbors
        for i in range(1,min(8, pts.shape[0])):
            ax.text(pts[i,0], pts[i,1], words[i], fontsize=8, alpha=0.8)

        # **centroids only** (no probe points)
        for cat, c2 in cents.items():
            if c2 is not None:
                ax.scatter(c2[0], c2[1], marker="P", s=130)  # centroid marker
                ax.text(c2[0]+0.02, c2[1]+0.02, f"{cat} centroid", fontsize=9)

        # dashed lines from focus words to centroids
        for fw in focus_words:
            if fw in words:
                i = words.index(fw)
                x, y = pts[i]
                for cat, c2 in cents.items():
                    if c2 is not None:
                        ax.plot([x, c2[0]], [y, c2[1]], linestyle="--", linewidth=0.7, alpha=0.6)

        ax.set_title(period_label)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(alpha=0.15)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()

def save_similarity_matrix(targets: List[str],
                           period_label: str,
                           get_vec_fn,
                           title: str, filename: Path):
    # Build cosine matrix among targets within a single period
    T = []
    words = []
    for w in targets:
        try:
            T.append(get_vec_fn(w, period_label))
            words.append(w)
        except KeyError:
            pass
    if len(T) < 2:
        return
    T = np.vstack(T)
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    Tn = T / np.maximum(norms, 1e-12)
    S = Tn @ Tn.T

    plt.figure(figsize=(0.6*len(words)+3, 0.6*len(words)+3))
    plt.imshow(S, vmin=-1, vmax=1)
    plt.colorbar(label="cosine")
    plt.xticks(range(len(words)), words, rotation=45, ha="right")
    plt.yticks(range(len(words)), words)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()

# ----------------
# End-to-end run
# ----------------

def main():
    parser = argparse.ArgumentParser(description="Meaningful visualizations of aligned word2vec periods.")
    parser.add_argument("--target", default="tyranny",
                        help="Target word to visualize across all periods.")
    parser.add_argument("--associate", default="popery",
                        help="Associate word to track cosine with the target across periods.")
    parser.add_argument("--neighbors", type=int, default=15,
                        help="Neighborhood size for the scatter plot.")
    parser.add_argument("--matrix-words", nargs="*", default=["tyranny","popery","liberty","parliament","king","corruption"],
                        help="Words to include in the within-period similarity matrix.")
    args = parser.parse_args()

    # Load and align
    for p, path in MODEL_PATHS.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing model: {path}")

    m1, m2, m3 = load_models()
    vocab, idx, m2_aligned, m3_aligned = build_alignment(m1, m2, m3)

    # Quick lambdas for period-vector access
    def vec_period(word: str, period: str) -> np.ndarray:
        return get_vec(word, period, m1, m2_aligned, m3_aligned, idx)

    # 1) Term trajectory across periods (pairwise cosines of target)
    tgt = args.target.lower()
    try:
        v1, v2, v3 = vec_period(tgt, PERIODS[0]), vec_period(tgt, PERIODS[1]), vec_period(tgt, PERIODS[2])
        pairwise = {
            f"{PERIODS[0]} \N{RIGHTWARDS ARROW} {PERIODS[1]}": cosine(v1, v2),
            f"{PERIODS[1]} \N{RIGHTWARDS ARROW} {PERIODS[2]}": cosine(v2, v3),
            f"{PERIODS[0]} \N{RIGHTWARDS ARROW} {PERIODS[2]}": cosine(v1, v3),
        }
        save_bar(pairwise,
                 title=f"Trajectory of '{tgt}' across periods (cosine)",
                 filename=OUTDIR / f"trajectory_{tgt}.png")
    except KeyError:
        print(f"[skip] '{tgt}' not in shared vocab; trajectory plot skipped.")

    # 2) Association over time: target ↔ associate cosine in each period
    assoc = args.associate.lower()
    try:
        a1 = cosine(vec_period(tgt, PERIODS[0]), vec_period(assoc, PERIODS[0]))
        a2 = cosine(vec_period(tgt, PERIODS[1]), vec_period(assoc, PERIODS[1]))
        a3 = cosine(vec_period(tgt, PERIODS[2]), vec_period(assoc, PERIODS[2]))
        series = {PERIODS[0]: a1, PERIODS[1]: a2, PERIODS[2]: a3}
        save_line(series,
                  title=f"Association over time: '{tgt}' ↔ '{assoc}'",
                  filename=OUTDIR / f"assoc_{tgt}_vs_{assoc}.png")
    except KeyError as e:
        print(f"[skip] Association line: {e}")

    # 3) Neighborhood scatter per period around target (shared PCA)
    try:
        save_neighborhood_scatter(
            target=tgt,
            neighbors=max(2, args.neighbors),
            m1=m1, m2_aligned_mat=m2_aligned, m3_aligned_mat=m3_aligned,
            vocab=vocab, idx=idx,
            title=f"Neighborhoods of '{tgt}' across periods (PCA, k={args.neighbors})",
            filename=OUTDIR / f"neighbors_{tgt}_k{args.neighbors}.png",
        )
    except KeyError:
        print(f"[skip] Neighborhood scatter: '{tgt}' not in shared vocab.")
    try: 
         save_neighborhood_scatter_with_centroids(
            target=tgt,
            neighbors=max(2, args.neighbors),
            m1=m1, m2_aligned_mat=m2_aligned, m3_aligned_mat=m3_aligned,
            vocab=vocab, idx=idx,
            probe_sets=PROBES,
            focus_words=["popery","government"],  # customize
            title=f"Neighborhoods of '{tgt}' with Thematic Vocab Centroids",
            filename=OUTDIR / f"neighbors_{tgt}_centroids.png",
        )
    except KeyError:
        print(f"[skip] Neighborhood+centroids: '{tgt}' not in shared vocab.")
    # 4) Similarity matrix within each period for a list of targets
    words = [w.lower() for w in args.matrix_words]
    for period in PERIODS:
        def _get(w, p=period):
            return vec_period(w, p)
        save_similarity_matrix(
            words, period,
            get_vec_fn=_get,
            title=f"Within-period cosine matrix ({period})",
            filename=OUTDIR / f"matrix_{period.replace('-', '')}.png"
        )

    print(f"✅ Done. Plots saved to: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()
