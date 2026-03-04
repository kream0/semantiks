"""
Script de précomputation des classements de voisins pour Sémantix.
Calcule les top-1000 mots les plus proches pour chaque mot candidat secret.
"""
import argparse
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone

import numpy as np


# ---------------------------------------------------------------------------
# Embedding loaders
# ---------------------------------------------------------------------------

def load_embeddings_binary(filepath: str):
    """
    Charge les embeddings depuis un fichier binaire.

    Format attendu:
      - uint32: vocab_size
      - uint32: dims
      - vocab_size * dims float32 (row-major)
      - vocab_size strings encodées: uint16 len + bytes UTF-8
    """
    print(f"[1/4] Chargement des embeddings binaires depuis {filepath} ...")
    with open(filepath, "rb") as f:
        vocab_size, dims = struct.unpack("<II", f.read(8))
        print(f"      vocab_size={vocab_size}, dims={dims}")

        total_floats = vocab_size * dims
        raw = f.read(total_floats * 4)
        matrix = np.frombuffer(raw, dtype=np.float32).reshape(vocab_size, dims).copy()

        words = []
        for _ in range(vocab_size):
            (length,) = struct.unpack("<H", f.read(2))
            word = f.read(length).decode("utf-8")
            words.append(word)

    print(f"      {vocab_size} mots chargés depuis le binaire.")
    return words, matrix


def load_embeddings_tsv(filepath: str):
    """
    Charge les embeddings depuis un fichier TSV.

    Format par ligne: word<TAB>v1,v2,...,vN
    """
    print(f"[1/4] Chargement des embeddings TSV depuis {filepath} ...")
    words = []
    vectors = []
    dims = None

    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            tab_idx = line.index("\t")
            word = line[:tab_idx]
            vals = list(map(float, line[tab_idx + 1:].split(",")))

            if dims is None:
                dims = len(vals)
            elif len(vals) != dims:
                print(f"      [AVERTISSEMENT] Ligne {lineno}: dimension incorrecte ({len(vals)} vs {dims}), ignorée.")
                continue

            words.append(word)
            vectors.append(vals)

            if lineno % 10_000 == 0:
                print(f"      {lineno} mots chargés...")

    matrix = np.array(vectors, dtype=np.float32)
    print(f"      {len(words)} mots chargés depuis le TSV.")
    return words, matrix


def load_embeddings(embeddings_dir: str):
    """
    Essaie d'abord le format binaire (camembert_embeddings.bin),
    puis retombe sur le TSV (camembert_embeddings.tsv ou french_embeddings.tsv).
    """
    bin_path = os.path.join(embeddings_dir, "camembert_embeddings.bin")
    tsv_path = os.path.join(embeddings_dir, "camembert_embeddings.tsv")
    fallback_tsv = os.path.join(embeddings_dir, "french_embeddings.tsv")

    if os.path.exists(bin_path):
        return load_embeddings_binary(bin_path)
    elif os.path.exists(tsv_path):
        return load_embeddings_tsv(tsv_path)
    elif os.path.exists(fallback_tsv):
        print(f"      [INFO] camembert_embeddings.tsv introuvable, utilisation de french_embeddings.tsv")
        return load_embeddings_tsv(fallback_tsv)
    else:
        print(
            f"[ERREUR] Aucun fichier d'embeddings trouvé dans {embeddings_dir}.\n"
            f"         Recherché:\n"
            f"           {bin_path}\n"
            f"           {tsv_path}\n"
            f"           {fallback_tsv}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Secret word filtering
# ---------------------------------------------------------------------------

def filter_secret_candidates(words: list[str]) -> list[tuple[int, str]]:
    """
    Retourne une liste de (index_dans_vocab, mot) pour les mots candidats secrets.

    Critères:
      - Longueur >= 4
      - Pas de tiret
      - Pas d'apostrophe
      - Tout en minuscules (w == w.lower())
    """
    candidates = []
    for idx, word in enumerate(words):
        if (
            len(word) >= 4
            and "-" not in word
            and "'" not in word
            and word == word.lower()
        ):
            candidates.append((idx, word))
    return candidates


# ---------------------------------------------------------------------------
# L2 normalisation
# ---------------------------------------------------------------------------

def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalise chaque vecteur ligne à la norme L2 unitaire."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Évite la division par zéro
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


# ---------------------------------------------------------------------------
# Binary output writer
# ---------------------------------------------------------------------------

def write_output_binary(
    output_path: str,
    words: list[str],
    secret_candidates: list[tuple[int, str]],
    all_rankings: list[list[tuple[int, float]]],
    top_n: int,
    dims: int,
) -> None:
    """
    Écrit le fichier binaire de classements précomputés.

    En-tête (17 octets):
      "SEMX"     4 octets  magic ASCII
      version    uint8     valeur 1
      vocab_size uint32    taille totale du vocabulaire
      secret_count uint32  nombre de mots secrets avec classements
      top_n      uint16    valeur 1000
      dims       uint16    valeur 512 (ou réelle)

    Section index des mots (vocab_size entrées):
      word_length uint16
      word_bytes  bytes UTF-8

    Section classements (secret_count entrées):
      secret_word_index uint32
      Pour chacun des top_n voisins:
        neighbor_index uint32
        similarity     float32
    """
    vocab_size = len(words)
    secret_count = len(secret_candidates)

    print(f"\n[4/4] Écriture du fichier binaire dans {output_path} ...")

    with open(output_path, "wb") as f:
        # --- En-tête ---
        f.write(b"SEMX")                                  # magic
        f.write(struct.pack("<B", 1))                      # version uint8
        f.write(struct.pack("<I", vocab_size))             # vocab_size uint32
        f.write(struct.pack("<I", secret_count))           # secret_count uint32
        f.write(struct.pack("<H", top_n))                  # top_n uint16
        f.write(struct.pack("<H", dims))                   # dims uint16

        # --- Index des mots ---
        for word in words:
            encoded = word.encode("utf-8")
            f.write(struct.pack("<H", len(encoded)))
            f.write(encoded)

        # --- Classements ---
        for (secret_idx, _secret_word), neighbors in zip(secret_candidates, all_rankings):
            f.write(struct.pack("<I", secret_idx))
            for neighbor_idx, similarity in neighbors:
                f.write(struct.pack("<I", neighbor_idx))
                f.write(struct.pack("<f", similarity))

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      Fichier créé: {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main precomputation logic
# ---------------------------------------------------------------------------

def precompute_rankings(
    embeddings_dir: str,
    output_dir: str,
    top_n: int,
    batch_size: int,
) -> None:
    total_start = time.time()

    # Step 1: Load embeddings
    words, matrix = load_embeddings(embeddings_dir)
    vocab_size, dims = matrix.shape
    print(f"      Dimensions: {vocab_size} x {dims}")

    # Step 2: L2-normalise all vectors
    print("[2/4] Normalisation L2 de tous les vecteurs...")
    norm_matrix = l2_normalize(matrix)
    del matrix  # free memory
    print("      Normalisation terminée.")

    # Step 3: Filter secret word candidates
    print("[3/4] Filtrage des mots candidats secrets...")
    secret_candidates = filter_secret_candidates(words)
    secret_count = len(secret_candidates)
    print(f"      {secret_count} mots candidats trouvés (sur {vocab_size} total).")

    # Step 4: Batch computation
    print(f"\n[3/4] Calcul des top-{top_n} voisins par lots de {batch_size}...")
    all_rankings: list[list[tuple[int, float]]] = []
    processed = 0

    # Extract only the candidate vectors (subset of norm_matrix)
    candidate_indices = np.array([idx for idx, _ in secret_candidates], dtype=np.int32)

    num_batches = (secret_count + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        batch_start_i = batch_num * batch_size
        batch_end_i = min(batch_start_i + batch_size, secret_count)
        batch_candidates = candidate_indices[batch_start_i:batch_end_i]

        # Shape: (batch, dims)
        batch_vectors = norm_matrix[batch_candidates]

        # Cosine similarity matrix: (batch, vocab_size)
        # Since vectors are L2-normalised, dot product = cosine similarity
        sims = batch_vectors @ norm_matrix.T  # shape: (batch_size, vocab_size)

        actual_batch = len(batch_candidates)

        # Clip top_n in case vocab is smaller than top_n
        k = min(top_n, vocab_size)

        # O(N) partial sort: get top-k indices per row (unsorted)
        top_k_indices = np.argpartition(-sims, k, axis=1)[:, :k]  # shape: (batch, k)

        for i in range(actual_batch):
            row_top_indices = top_k_indices[i]          # shape: (k,)
            row_top_sims = sims[i, row_top_indices]      # shape: (k,)

            # Sort descending by similarity
            sorted_order = np.argsort(-row_top_sims)
            sorted_indices = row_top_indices[sorted_order]
            sorted_sims = row_top_sims[sorted_order]

            neighbors = list(zip(sorted_indices.tolist(), sorted_sims.tolist()))
            all_rankings.append(neighbors)

        processed += actual_batch

        if processed % 100 == 0 or processed == secret_count:
            elapsed = time.time() - total_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (secret_count - processed) / rate if rate > 0 else 0
            print(
                f"      {processed}/{secret_count} mots traités "
                f"({100 * processed / secret_count:.1f}%) "
                f"— {elapsed:.0f}s écoulées, ETA {eta:.0f}s"
            )

    # Step 5: Write binary output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "precomputed_rankings.bin")
    write_output_binary(output_path, words, secret_candidates, all_rankings, top_n, dims)

    # Step 6: Write metadata JSON
    metadata_path = os.path.join(output_dir, "rankings_metadata.json")
    metadata = {
        "vocab_size": vocab_size,
        "secret_count": secret_count,
        "top_n": top_n,
        "dims": dims,
        "generation_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2, ensure_ascii=False)
    print(f"      Métadonnées JSON: {metadata_path}")

    # Summary
    total_elapsed = time.time() - total_start
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("Précomputation terminée!")
    print("=" * 60)
    print(f"  Mots secrets:          {secret_count}")
    print(f"  Classements calculés:  {secret_count * top_n:,}")
    print(f"  Taille fichier:        {output_size_mb:.1f} MB")
    print(f"  Temps total:           {total_elapsed:.1f}s")
    print(f"  Fichier de sortie:     {output_path}")
    print(f"  Métadonnées:           {metadata_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Précompute les top-N voisins pour chaque mot candidat secret de Sémantix."
    )
    parser.add_argument(
        "--embeddings-dir",
        default="data/",
        help="Répertoire contenant les fichiers d'embeddings (défaut: data/)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/",
        help="Répertoire de sortie pour precomputed_rankings.bin (défaut: data/)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1000,
        help="Nombre de voisins à stocker par mot secret (défaut: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Taille des lots pour le calcul matriciel (défaut: 500)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    precompute_rankings(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
        batch_size=args.batch_size,
    )
