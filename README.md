# Sémantix

Clone of [Cémantix](https://cemantix.certideal.com/) — a French semantic word guessing game. Guess the secret word and receive a "temperature" score based on semantic proximity.

![Bun](https://img.shields.io/badge/Bun-1.3.3+-black?logo=bun)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue?logo=typescript)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)

## How It Works

1. A secret word is chosen daily (deterministic — same word for all players)
2. You guess French words
3. Each guess gets a **temperature** from -100° to 100° based on cosine similarity between CamemBERT sentence embeddings
4. An LLM refines the score with a bounded adjustment (±15°) for more human-like feedback
5. Top 1000 closest words show a proximity rank (1–1000‰)

## Features

- CamemBERT embeddings (512D, 66k+ words) for high-quality French semantic similarity
- LLM temperature correction (MiniMax M2.5 via OpenRouter) — non-blocking, cached in SQLite
- Precomputed top-1000 rankings for instant rank display
- Lemmatization via spaCy (conjugated verbs → infinitive, plurals → singular)
- Autocorrection (Levenshtein distance ≤ 2) and prefix autocompletion (Tab)
- Hint system, give-up, share results, continuous color gradient (cold blue → hot red)
- Word of the day — deterministic daily secret word

## Prerequisites

- [Bun](https://bun.sh/) 1.3.3+
- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key (for LLM correction)

## Setup

### 1. Install dependencies

```bash
# TypeScript dependencies
bun install

# Python dependencies
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key:
# OPENROUTER_API_KEY=sk-or-...
```

### 3. Generate embeddings (one-time, slow)

These scripts generate the large data files needed to play. They are not included in the repo due to size (~950 MB total).

```bash
source .venv/bin/activate

# Generate CamemBERT embeddings (1-3h CPU, ~15min GPU)
python scripts/generate_camembert_embeddings.py

# Precompute top-1000 rankings (2-10 min)
python scripts/precompute_rankings.py
```

This produces:
- `data/camembert_embeddings.tsv` + `.bin` — 66,579 word embeddings (512D)
- `data/precomputed_rankings.bin` — top-1000 neighbors for 65,649 secret words

### 4. Run

```bash
# Terminal 1: Lemmatizer service
source .venv/bin/activate
python scripts/lemmatizer_service.py

# Terminal 2: Game server
bun run src/server.ts
```

Open http://localhost:3000 and start guessing!

On Windows, you can also use `start.bat` to launch both services.

## Architecture

```
src/server.ts                          → Main server: API + embedded frontend + game logic (~1,970 lines)
scripts/
  ├── lemmatizer_service.py            → Flask + spaCy lemmatization service (port 3001)
  ├── generate_camembert_embeddings.py → Offline CamemBERT embedding generation
  ├── precompute_rankings.py           → Offline top-1000 ranking precomputation
  ├── download_fauconnier.py           → Legacy Word2Vec model download
  └── convert-vectors.ts               → Legacy embedding format converter
data/
  ├── french_words.json                → Word list (66,579 lemmas) [tracked in git]
  ├── rankings_metadata.json           → Rankings metadata [tracked in git]
  ├── camembert_embeddings.*           → Generated embeddings [gitignored]
  ├── precomputed_rankings.bin         → Generated rankings [gitignored]
  └── llm_cache.sqlite                 → Runtime LLM cache [gitignored]
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/guess` | Submit a guess (auto-creates session) |
| POST | `/api/hint` | Get a hint about the secret word |
| POST | `/api/give-up` | Reveal the secret word |
| POST | `/api/new-game` | Start a new game |
| POST | `/api/autocomplete` | Get word suggestions |

### Lemmatizer (port 3001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/lemmatize` | Lemmatize a word |
| GET | `/health` | Health check |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Runtime | Bun |
| Backend | TypeScript |
| Frontend | Vanilla HTML/CSS/JS (embedded in server) |
| Embeddings | CamemBERT (sentence-camembert-large, 512D via PCA) |
| LLM | MiniMax M2.5 via OpenRouter |
| Lemmatization | Python Flask + spaCy fr_core_news_sm |
| Cache | SQLite (LLM results) + in-memory Maps |
