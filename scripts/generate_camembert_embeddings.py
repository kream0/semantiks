"""
Generate CamemBERT-based embeddings for the Sémantix French word guessing game.

Uses dangvantuan/sentence-camembert-large to encode ~66k French words into
1024D vectors, then applies PCA to reduce to 512D (or --dimensions).

Outputs:
  - data/camembert_embeddings.tsv  (word TAB comma-separated floats)
  - data/camembert_embeddings.bin  (binary format with header + word index)
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CamemBERT sentence embeddings for Sémantix word list."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write output files (default: data/)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=512,
        help="Target PCA dimensions (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size (default: 256)",
    )
    parser.add_argument(
        "--word-list",
        default="data/french_words.json",
        help="Path to french_words.json (default: data/french_words.json)",
    )
    return parser.parse_args()


def load_word_list(path: str) -> list[str]:
    print(f"Loading word list from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        words = json.load(f)
    print(f"  Loaded {len(words):,} words.")
    return words


def encode_words(
    model: SentenceTransformer,
    words: list[str],
    batch_size: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Encode words in batches, skipping any that raise an exception.

    Returns:
        embeddings  – float32 array, shape (n_encoded, raw_dim)
        good_words  – list of successfully encoded words (same order as rows)
        failed_words – list of words that failed to encode
    """
    good_words: list[str] = []
    failed_words: list[str] = []
    all_embeddings: list[np.ndarray] = []

    total = len(words)
    processed = 0
    start = time.time()

    for batch_start in range(0, total, batch_size):
        batch = words[batch_start : batch_start + batch_size]
        try:
            vecs = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            all_embeddings.append(vecs.astype(np.float32))
            good_words.extend(batch)
        except Exception as exc:
            # Fall back to word-by-word encoding for this batch
            for word in batch:
                try:
                    vec = model.encode(
                        [word],
                        batch_size=1,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                    )
                    all_embeddings.append(vec.astype(np.float32))
                    good_words.append(word)
                except Exception as word_exc:
                    failed_words.append(word)
                    print(
                        f"  [SKIP] '{word}' failed to encode: {word_exc}",
                        file=sys.stderr,
                    )

        processed += len(batch)
        if processed % 1000 == 0 or processed >= total:
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(
                f"  Encoded {processed:,}/{total:,} words "
                f"({processed / total * 100:.1f}%)  "
                f"{rate:.0f} words/s  ETA {eta:.0f}s"
            )

    embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 0), dtype=np.float32)
    return embeddings, good_words, failed_words


def apply_pca(embeddings: np.ndarray, n_components: int) -> tuple[np.ndarray, float]:
    """
    Fit PCA on all embeddings and return the reduced array plus explained variance ratio sum.
    """
    raw_dim = embeddings.shape[1]
    if n_components >= raw_dim:
        print(
            f"  --dimensions ({n_components}) >= raw dim ({raw_dim}); skipping PCA."
        )
        return embeddings.astype(np.float32), 1.0

    print(f"  Fitting PCA: {raw_dim}D → {n_components}D on {embeddings.shape[0]:,} vectors ...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    variance_retained = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA done. Variance retained: {variance_retained * 100:.2f}%")
    return reduced, variance_retained


def save_tsv(path: str, words: list[str], embeddings: np.ndarray) -> int:
    """
    Write TSV: one line per word → "word\tv1,v2,...,vN"
    Returns file size in bytes.
    """
    print(f"  Writing TSV to {path} ...")
    with open(path, "w", encoding="utf-8") as f:
        for word, vec in zip(words, embeddings):
            floats = ",".join(f"{v:.6g}" for v in vec)
            f.write(f"{word}\t{floats}\n")
    size = os.path.getsize(path)
    print(f"  TSV written: {size / 1_048_576:.1f} MB")
    return size


def save_binary(path: str, words: list[str], embeddings: np.ndarray) -> int:
    """
    Write binary file with header:
      - 4 bytes: uint32 vocab_size
      - 4 bytes: uint32 dimensions
      - vocab_size × dimensions × 4 bytes: float32 values (row-major)
      - vocab_size length-prefixed UTF-8 strings:
          uint16 (length in bytes) + raw UTF-8 bytes

    Returns file size in bytes.
    """
    vocab_size, dimensions = embeddings.shape
    print(f"  Writing binary to {path} ...")
    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<II", vocab_size, dimensions))
        # Float32 matrix (row-major, C order)
        f.write(embeddings.astype(np.float32).tobytes(order="C"))
        # Word index: uint16 length-prefix + UTF-8 bytes
        for word in words:
            encoded = word.encode("utf-8")
            f.write(struct.pack("<H", len(encoded)))
            f.write(encoded)
    size = os.path.getsize(path)
    print(f"  Binary written: {size / 1_048_576:.1f} MB")
    return size


def main():
    args = parse_args()

    # Resolve paths relative to the project root (parent of this script's dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    word_list_path = args.word_list
    if not os.path.isabs(word_list_path):
        word_list_path = os.path.join(project_root, word_list_path)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)

    tsv_path = os.path.join(output_dir, "camembert_embeddings.tsv")
    bin_path = os.path.join(output_dir, "camembert_embeddings.bin")

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load word list ──────────────────────────────────────────────────────
    words = load_word_list(word_list_path)

    # ── 2. Load model ─────────────────────────────────────────────────────────
    model_name = "dangvantuan/sentence-camembert-large"
    print(f"\nLoading model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print("  Model loaded.")

    # ── 3. Encode ──────────────────────────────────────────────────────────────
    print(f"\nEncoding {len(words):,} words (batch_size={args.batch_size}) ...")
    embeddings, good_words, failed_words = encode_words(model, words, args.batch_size)
    print(f"  Successfully encoded: {len(good_words):,} words")
    if failed_words:
        print(f"  Failed words ({len(failed_words)}): {failed_words[:20]}")

    if embeddings.size == 0:
        print("ERROR: No words were encoded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ── 4. PCA reduction ───────────────────────────────────────────────────────
    print(f"\nApplying PCA ({args.dimensions}D) ...")
    reduced, variance_retained = apply_pca(embeddings, args.dimensions)

    # ── 5. Save outputs ────────────────────────────────────────────────────────
    print("\nSaving outputs ...")
    tsv_size = save_tsv(tsv_path, good_words, reduced)
    bin_size = save_binary(bin_path, good_words, reduced)

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total words in list:   {len(words):,}")
    print(f"  Successfully encoded:  {len(good_words):,}")
    print(f"  Failed / skipped:      {len(failed_words):,}")
    print(f"  Raw embedding dim:     {embeddings.shape[1]}")
    print(f"  PCA output dim:        {reduced.shape[1]}")
    print(f"  Variance retained:     {variance_retained * 100:.2f}%")
    print(f"  TSV output:            {tsv_path}  ({tsv_size / 1_048_576:.1f} MB)")
    print(f"  Binary output:         {bin_path}  ({bin_size / 1_048_576:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
