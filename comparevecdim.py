import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

base_dir = "period_models"

# --- your core target words ---
core_words = ["tyranny", "liberty", "oppression", "freedom", "king", "government", "popery"]

# --- probe sets (tweak freely) ---
PROBES = {
    "religious":  ["religion", "church", "priest", "papist", "catholic", "romish", "bishop", "mass"],
    "institution": ["parliament", "ministry", "constitution", "governor", "crown", "assembly"],
    "moral":      ["virtue", "vice", "corruption", "zeal", "licentiousness"],
    "emotion":    ["fear", "anger", "indignation", "hope", "zealotry"]
}

COLORS = {
    "core": "tab:blue",
    "religious": "tab:purple",
    "institution": "tab:green",
    "moral": "tab:red",
    "emotion": "tab:orange",
    "centroid": "black"
}

def load_model(path):
    try:
        return KeyedVectors.load(path, mmap='r')
    except Exception:
        return KeyedVectors.load_word2vec_format(path, binary=True)

def present_words(model, words):
    return [w for w in words if w in model.key_to_index]

def get_matrix(model, words):
    return np.stack([model[w] for w in words])

def centroid(model, words):
    X = get_matrix(model, words)
    return X.mean(axis=0)

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# --- find pairs as before ---
files = sorted([f for f in os.listdir(base_dir) if f.endswith(".bin")])
pairs = {}
for f in files:
    key = f.replace("v1", "").replace("_vectors.bin", "")
    pairs.setdefault(key, []).append(f)

for period, model_files in pairs.items():
    if len(model_files) != 2:
        print(f"Skipping {period}: expected 2 models, found {len(model_files)}")
        continue

    # Identify which is which in your setup
    model_100 = [f for f in model_files if "v1" not in f][0]  # left panel
    model_25  = [f for f in model_files if "v1" in f][0]     # right panel

    M100 = load_model(os.path.join(base_dir, model_100))
    M25  = load_model(os.path.join(base_dir, model_25))

    # Build plotting + centroid data per model
    def build_plot_data(model):
        # words present
        core = present_words(model, core_words)
        probes_present = {k: present_words(model, v) for k, v in PROBES.items()}
        # vectors
        X_core = get_matrix(model, core) if core else np.empty((0, model.vector_size))
        probe_vecs = {k: (get_matrix(model, v) if v else np.empty((0, model.vector_size)))
                      for k, v in probes_present.items()}
        # centroids (only if at least 2 words present)
        cents = {k: (centroid(model, v) if len(v) >= 2 else None) for k, v in probes_present.items()}
        # concatenate all for PCA fit
        mats = [X_core] + [pv for pv in probe_vecs.values() if pv.size]
        ALL = np.vstack(mats) if mats else None
        # PCA to 2D per model (keeps each model’s native geometry)
        pca = PCA(n_components=2, random_state=42)
        ALL_2d = pca.fit_transform(ALL) if ALL is not None else None

        # split back
        idx = 0
        coords = {}
        if X_core.size:
            coords["core"] = (core, ALL_2d[idx: idx + len(core)])
            idx += len(core)
        for k, pv in probe_vecs.items():
            if pv.size:
                coords[k] = (probes_present[k], ALL_2d[idx: idx + len(probes_present[k])])
                idx += len(probes_present[k])

        # centroid 2D coords
        cent_2d = {}
        for k, c in cents.items():
            if c is not None:
                cent_2d[k] = pca.transform(c.reshape(1, -1))[0]
            else:
                cent_2d[k] = None

        return coords, cent_2d, pca

    coords100, cent100, _ = build_plot_data(M100)
    coords25,  cent25,  _ = build_plot_data(M25)

    # --- PLOT ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, title, coords, cents in [
        (axes[0], f"{period} (100-dim)", coords100, cent100),
        (axes[1], f"{period} (25-dim)", coords25, cent25)
    ]:
        # scatter core words
        if "core" in coords:
            words, pts = coords["core"]
            ax.scatter(pts[:,0], pts[:,1], label="core", color=COLORS["core"])
            for i, w in enumerate(words):
                ax.text(pts[i,0]+0.02, pts[i,1]+0.02, w, fontsize=9)

        # scatter probes by category
        for cat in ["religious", "institution", "moral", "emotion"]:
            if cat in coords:
                words, pts = coords[cat]
                ax.scatter(pts[:,0], pts[:,1], s=30, alpha=0.8, label=cat, color=COLORS[cat])
                # optional: light labels for probes
                # for i, w in enumerate(words):
                #     ax.text(pts[i,0]+0.01, pts[i,1]+0.01, w, fontsize=7, alpha=0.8)

        # draw centroids
        for cat in ["religious", "institution", "moral", "emotion"]:
            c = cents.get(cat)
            if c is not None:
                ax.scatter(c[0], c[1], marker="*", s=160, color=COLORS["centroid"])
                ax.text(c[0]+0.02, c[1]+0.02, f"{cat} centroid", fontsize=9, weight="bold")

        # draw lines from popery/government to centroids
        for focus in ["popery", "government"]:
            if "core" in coords:
                words, pts = coords["core"]
                if focus in words:
                    i = words.index(focus)
                    x, y = pts[i]
                    for cat in ["religious", "institution", "moral", "emotion"]:
                        c = cents.get(cat)
                        if c is not None:
                            ax.plot([x, c[0]], [y, c[1]], linestyle="--", linewidth=0.6, alpha=0.6)

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.15)

    plt.suptitle(f"Probe experiment: How do model results compare?  ({period})")
    plt.tight_layout()

    plt.savefig(f"comparemodels/{period}_comparison.png", dpi=300)
