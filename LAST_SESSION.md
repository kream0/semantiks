# Last Session

**Session:** 7
**Date:** 2026-03-04
**Focus:** Generate CamemBERT embeddings and precompute rankings — unblock the game

## What Was Accomplished
- Installed Python dependencies in `.venv/` (sentence-transformers, torch CPU, scikit-learn, protobuf, sentencepiece, tiktoken)
- Generated CamemBERT embeddings: 66,579 words, 1024D→512D PCA (94.51% variance retained)
  - `data/camembert_embeddings.tsv` (316.6 MB)
  - `data/camembert_embeddings.bin` (130.7 MB)
- Precomputed top-1000 rankings: 65,649 secret words × 1000 neighbors
  - `data/precomputed_rankings.bin` (501.8 MB)
  - `data/rankings_metadata.json`
- Validated: build passes, server loads CamemBERT + rankings, temperatures well-distributed (-15° to +38°), rank display working

## Files Created
- `.venv/` — Python virtual environment with all deps
- `data/camembert_embeddings.tsv` — CamemBERT 512D embeddings (TSV)
- `data/camembert_embeddings.bin` — CamemBERT 512D embeddings (binary)
- `data/precomputed_rankings.bin` — Top-1000 rankings (SEMX binary)
- `data/rankings_metadata.json` — Rankings metadata

## Current Project Status
- **Build:** Passes (`bun build src/server.ts --outdir dist --target bun`)
- **Game:** Playable with CamemBERT embeddings
- **Lemmatizer:** Works (needs separate `python3 scripts/lemmatizer_service.py`)
- **LLM correction:** Active (MiniMax M2.5 via OpenRouter)

## Next Immediate Action
- Play-test the game in browser to verify UX end-to-end
- Fix any remaining issues (lemmatizer auto-start, LLM reasoning overflow)
- Consider committing the generated data files (large — may need .gitignore or LFS)

## Handoff Notes
- The `.venv/` directory is NOT committed — recreate with: `python3 -m venv .venv && source .venv/bin/activate && pip install sentence-transformers scikit-learn torch --index-url https://download.pytorch.org/whl/cpu && pip install protobuf sentencepiece tiktoken`
- Generated data files (`camembert_embeddings.*`, `precomputed_rankings.bin`) are ~950 MB total — too large for git without LFS
- The server auto-detects CamemBERT vs Fauconnier embeddings based on which files exist in `data/`
- Rankings file uses 65,649 secret words (vs 18,596 in HANDOFF.md) — the filtering logic in `precompute_rankings.py` is more permissive than server.ts

## Infrastructure Status
- **Bun server (port 3000):** Not running (stopped at end of session)
- **Flask lemmatizer (port 3001):** Not running
- **Python venv:** `.venv/` ready to use
