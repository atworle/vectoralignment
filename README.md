# Vector Alignment

This repository contains small utilities for training, aligning and visualizing word embedding models across historical periods. The tools let you train period-specific models, align them into a common space, compare semantic change, and produce diagnostic plots.

## Project purpose

The goal is to analyze how word meanings and associations shift across different time periods by:
- Training word2vec-style embeddings for each period (corpora live in `period_corpus/`).
- Aligning embeddings from different periods into a shared vector space.
- Measuring cosine similarities, neighborhood changes, and plotting results.

## Repository layout

- `period_corpus/` - raw text files per period used to train models (e.g., `1756-1764.txt`).
- `period_models/` - expected location for trained vector models (binary/word2vec format). The visualizer expects these files:
  - `period_models/1756-1764_vectors.bin`
  - `period_models/1765-1776_vectors.bin`
  - `period_models/1777-1783_vectors.bin`
- `alignment_plots/` - directory where visualization output (PNGs) is saved.
- `alignment_results/` - examples and CSV/JSON outputs (existing results included).
- `visualizations.py` - main plotting and analysis script (aligns models, computes cosines, PCA visualizations, centroids, similarity matrices).
- `trainmodel.py` - (utility) expected to produce models in `period_models/` from `period_corpus/` (see usage below).
- `alignvectors.py` and `comparevecdim.py` - helper scripts for model alignment and comparison. Check `--help` for options.
- `requirements.txt` - Python dependencies.

## Requirements

This project uses Python and common data / ML libraries. Install dependencies from `requirements.txt`:

PowerShell
```powershell
python -m pip install -r requirements.txt
```

If you prefer a virtual environment:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt
```

## Using the visualizer (`visualizations.py`)

`visualizations.py` loads three period models, aligns them to the first period, and generates several plots:

- Trajectory bar plot for a target word across periods (cosine similarities)
- Association line plot between a target and an associate over time
- Neighborhood scatterplots (shared PCA) for a target across periods
- Neighborhood scatterplots with thematic centroids (probe terms)
- Within-period similarity matrices for a list of focus words

Basic usage (PowerShell):

```powershell
# Default run (saves plots to alignment_plots/)
python .\visualizations.py

# Specify a different target and associate
python .\visualizations.py --target tyranny --associate popery --neighbors 12

# Provide a custom list for the within-period similarity matrix
python .\visualizations.py --matrix-words tyranny popery liberty parliament king corruption
```

Notes:
- The script expects the models listed at the top of `visualizations.py` (see `MODEL_PATHS`).
- If a required model file is missing it will raise a `FileNotFoundError` — ensure your trained model binaries are present in `period_models/`.
- Use `--help` for script-specific options:
```powershell
python .\visualizations.py --help
```

## Training models (`trainmodel.py`)

`trainmodel.py` is provided to train word embeddings from the text in `period_corpus/` and should write binary models into `period_models/`. Usage and flags vary by script implementation; run:

```powershell
python .\trainmodel.py --help
```

Adjust parameters like vector size, window, min count, and output paths as available in the script.

## Alignment and comparison helpers

- `alignvectors.py`: utility for aligning two or more models. Use `--help` to see available options.
- `comparevecdim.py`: helper to compare vector dimensionality or compare models across vector sizes.

Run each script with `--help` to inspect exact flags and behavior:

```powershell
python .\alignvectors.py --help
python .\comparevecdim.py --help
```

## Output

- Visual outputs are saved to `alignment_plots/` by default (PNG files).
- Derived CSV/JSON results (if produced) are placed under `alignment_results/`.

## Tips & troubleshooting

- If you see `KeyError` messages from `visualizations.py`, the requested words may not be in the shared vocabulary across all models — try picking words present in every model or expand your training corpora.
- If PCA or plotting fails due to memory limits, reduce neighborhood sizes (`--neighbors`) or the number of probe / matrix words.
- Confirm models are in word2vec binary format. `visualizations.py` uses `gensim.models.KeyedVectors.load_word2vec_format(..., binary=True)` by default.

## Development notes

- The central alignment method uses mean-centering and an orthogonal Procrustes-like rotation computed by SVD (`numpy.linalg.svd`) to align target models to the base model.
- Probe sets for thematic centroids are defined in `visualizations.py` (variable `PROBES`) and can be adjusted to your research questions.

## Contact

If you want help adapting the scripts to other corpora, adding new probes, or extending visualizations, open an issue or edit this README with suggested improvements.

---
Generated README to describe repository scripts and usage. Use `--help` on each script for more detailed flags.
