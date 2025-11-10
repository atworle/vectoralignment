# pip install pandas gensim
import re
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# ----------------------------
# 1) Load & prep the dataframe
# ----------------------------
df = pd.read_csv("NHgazette1756-1783.csv", usecols=["sequence", "date", "ocr_eng"]).copy()

# year from first 4 chars of date (e.g., "1768-05-01")
df["year"] = df["date"].astype(str).str.slice(0, 4).astype("int32")

# period buckets
def to_period(y: int) -> str | None:
    if 1756 <= y <= 1764: return "1756-1764"
    if 1765 <= y <= 1776: return "1765-1776"
    if 1777 <= y <= 1783: return "1777-1783"
    return None

df["period"] = df["year"].apply(to_period)

# -----------------------------------------
# 2) Group by period and write raw TXT files
# -----------------------------------------
period_corpus_dir = Path("period_corpus")
period_corpus_dir.mkdir(parents=True, exist_ok=True)

period_corpus = (
    df.dropna(subset=["period"])
      .groupby("period", as_index=False)["ocr_eng"]
      .agg(lambda s: " ".join(map(str, s)))
      .rename(columns={"ocr_eng": "text"})
)

for _, row in period_corpus.iterrows():
    (period_corpus_dir / f"{row['period']}.txt").write_text(row["text"], encoding="utf-8")

# ---------------------------------------------------
# 3) "prep_word2vec" equivalent: clean & write corpus
#     (bundle_ngrams = 1 in your R → just unigrams)
# ---------------------------------------------------
def clean_to_lines(text: str) -> list[str]:
  
    tokens = simple_preprocess(text, deacc=False, min_len=2, max_len=30)
    return [" ".join(tokens)]

# Write one combined cleaned file (like destination="period_corpus/corpus_clean.txt")
combined_clean_path = period_corpus_dir / "corpus_clean.txt"
with combined_clean_path.open("w", encoding="utf-8") as out:
    for pfile in sorted(period_corpus_dir.glob("*.txt")):
        if pfile.name == "corpus_clean.txt":
            continue
        raw = pfile.read_text(encoding="utf-8")
        for line in clean_to_lines(raw):
            out.write(line + "\n")

# ---------------------------------------------
# 4) Train a single model on the combined corpus
# ---------------------------------------------
def sentence_iter_from_textfile(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()

combined_model = Word2Vec(
    sentences=sentence_iter_from_textfile(combined_clean_path),
    vector_size=25,     # vectors = 25
    window=7,           # window = 7
    sg=1,               # negative sampling setup typically goes with skip-gram; R's defaults are skip-gram
    negative=5,         # negative_samples = 5
    epochs=10,          # iter = 10
    workers=2,          # threads = 2
    min_count=5         # typical default; adjust if you want exact parity with your R run
)

# Save in word2vec binary format (close to "vectorsv2.bin")
(combined_model.wv).save_word2vec_format(period_corpus_dir / "vectorsv2.bin", binary=True)

# -----------------------------------------------------------------
# 5) Train one model per period (prep → clean file → train → save)
# -----------------------------------------------------------------
periods = ["1756-1764", "1765-1776", "1777-1783"]
input_dir = period_corpus_dir
output_dir = Path("period_models")
output_dir.mkdir(parents=True, exist_ok=True)

for p in periods:
    input_file = input_dir / f"{p}.txt"
    if not input_file.exists():
        # skip if that bucket didn't produce a file
        continue

    # Clean this period text into its own file (like your second prep_word2vec call)
    raw_text = input_file.read_text(encoding="utf-8")
    clean_lines = clean_to_lines(raw_text)
    clean_file = output_dir / f"{p}.txt"
    clean_file.write_text("\n".join(clean_lines) + "\n", encoding="utf-8")

    # Train the per-period model
    per_period_model = Word2Vec(
        sentences=sentence_iter_from_textfile(clean_file),
        vector_size=25,
        window=7,
        sg=1,
        negative=5,
        epochs=10,
        workers=2,
        min_count=5
    )

    # Save binary vectors (like "{period}_vectors.bin")
    (per_period_model.wv).save_word2vec_format(output_dir / f"{p}_vectors.bin", binary=True)

print("Done. Models saved to:")
print(f"  - {period_corpus_dir/'vectorsv2.bin'}")
print(f"  - {output_dir}/*_vectors.bin")
