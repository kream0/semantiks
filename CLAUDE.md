# Project: Sémantix

French semantic word guessing game (Cémantix clone). Players guess a secret word and receive "temperature" feedback based on Word2Vec cosine similarity, optionally refined by an LLM.

**Monolith architecture:** Bun/TypeScript backend serves embedded HTML/CSS/JS frontend + Python Flask microservice for lemmatization.

```
src/server.ts          → Main server: API, game logic, embedded frontend (1,700+ lines)
scripts/               → Utilities: lemmatizer service, embedding download/conversion
data/                  → Word2Vec embeddings (500D, 66k words), word list JSON
```

## Tech Stack & Versions

- Bun 1.3.3+ (NOT Node.js)
- TypeScript 5 (strict mode, ESNext target)
- Python 3.10+ with Flask 3.0.0 for lemmatizer service
- spaCy 3.7.2 (fr_core_news_sm model)
- Embeddings: **CamemBERT (sentence-camembert-large, 512D)** — primary model
  - Legacy fallback: Word2Vec Fauconnier frWiki 500D CBOW cut10
- LLM: MiniMax M2.5 via OpenRouter API (minimax provider) — categorical judgment with bounded adjustment (±15°)
  - **NOTE:** OpenRouter account has provider routing locked to minimax/deepseek/moonshotai. Use models available on these providers.
  - MiniMax M2.5 is a reasoning model — response is extracted from `reasoning` field when `content` is null.
- Frontend: Vanilla HTML/CSS/JS (embedded in server.ts, NOT a framework)
- Package manager: bun (NOT npm, NOT yarn)
- Python dependencies: sentence-transformers, torch, scikit-learn, numpy

## Architecture Boundaries

- `src/server.ts` is the single server file — API routes, game logic, similarity calculations, and embedded frontend all live here.
- `scripts/lemmatizer_service.py` is a standalone Flask service on port 3001. The main server calls it via HTTP — no direct Python imports from TypeScript.
- `data/` contains mostly read-only assets loaded at startup. **EXCEPTION:** `data/llm_cache.sqlite` is the only runtime-written file (LLM adjustment cache). NEVER write other files in `data/` at runtime.
- `.env` holds `OPENROUTER_API_KEY`. NEVER read, log, or commit this file.
- Embeddings use Float32Array for memory efficiency. NEVER convert to regular arrays.
- Pre-generated assets (`data/camembert_embeddings.bin`, `data/precomputed_rankings.bin`) require offline generation scripts before server start.

## Coding Conventions

- camelCase for variables and functions, UPPERCASE for constants
- Async/await for all I/O (API calls, file reads, lemmatizer calls)
- Map-based caching: `gameSessions`, `llmTemperatureCache`, `similarityCache`
- French word validation regex: `[a-zA-ZàâäéèêëïîôùûüÿœæçÀÂÄÉÈÊËÏÎÔÙÛÜŸŒÆÇ\-']+`
- Temperature scale: -100° to 100° (cosine similarity × 100, rounded to 2 decimals)
- Proximity rank: 1–1000‰ notation for top 1000 closest words
- NEVER use placeholder or stub implementations — provide full, working code
- NEVER add npm dependencies without explicit approval — prefer Bun built-ins

## Workflow Rules

### Validation (run after every change)
- Build: `bun build src/server.ts --outdir dist --target bun`
- Dev server: `bun run --watch src/server.ts`
- Lint: no linter configured yet — type-check with `bun build src/server.ts --outdir dist --target bun`

### Dev Feedback Loop (browser-agent)
Use `agent-browser` (installed globally via `npm install -g agent-browser`) for automated browser testing during development. This is the standard dev loop for UI changes:

1. **Start services:**
   - `python3 scripts/lemmatizer_service.py` (port 3001)
   - `bun run src/server.ts` (port 3000)
2. **Browser-agent commands:**
   - `agent-browser open http://localhost:3000` — open the game page
   - `agent-browser snapshot` — get accessibility tree (text snapshot of page state)
   - `agent-browser screenshot /tmp/test.png` — take visual screenshot
   - `agent-browser fill '#guess-input' 'word'` — type into input
   - `agent-browser press Enter` — submit guess
   - `agent-browser click '#btn-id'` — click a button
   - `agent-browser eval "js expression"` — evaluate JS in page context
3. **Typical test sequence:**
   ```bash
   agent-browser open http://localhost:3000
   agent-browser fill '#guess-input' 'maison' && agent-browser press Enter
   sleep 2 && agent-browser snapshot   # verify guess appeared
   agent-browser screenshot /tmp/test.png  # visual check
   ```
4. **Key element IDs:** `#guess-input`, `#guess-btn`, `#new-game-btn`, `#hint-btn`, `#give-up-btn`, `#message-container`, `#guesses-container`, `#victory-modal`, `#share-btn`
5. **Known limitation:** Headless browser may not trigger `confirm()` dialogs — test those via direct API calls with `curl`.

### Implementation Workflow
- Before modifying a file, read it and understand the existing code first.
- For tasks touching >3 files, start in plan mode and propose a plan.
- Run validation after every change — do NOT batch changes without checking.
- After UI changes, run the browser-agent dev loop to verify visually and functionally.
- Commit frequently with descriptive messages.

## Agent Coordination

### Lead Delegation Rules
- The lead MUST NOT: read source files, run build/test, edit code, debug failures, inspect logs
- The lead ONLY does: plan, spawn agents, coordinate, run deploy/git, update tracking docs
- If you catch yourself about to use Read on a source file or Bash to run a build — STOP. Spawn an agent instead.

### Model Tiering
- Opus: lead only (planning, coordination)
- Sonnet: implementers, reviewers, testers (model: "sonnet")
- Haiku: scouts, single-file fixers, validators, quick searches (model: "haiku")

### Task Description Template
Every agent task prompt MUST include:
- Files owned: [explicit list — agent ONLY touches these]
- Objective: [what to achieve]
- Acceptance criteria: [measurable conditions]
- Patterns: [reference files or snippets to follow]
- Validation: [exact commands to run after changes]

### Completion Summary Format
Append to every agent task prompt:
- TASK COMPLETE: [task subject]
- Status: SUCCESS | FAILED | BLOCKED
- Files modified: [list]
- Changes: [2-3 sentences, no code dumps]
- Validation: PASSED | FAILED ([reason])

### Scout Output Cap
- Report structure and key findings only. Maximum 500 words.
- Do NOT return full file contents.

### Anti-Patterns
- Lead reads source files → spawn Haiku scout instead
- Lead runs build/test/server → spawn agent to run and report
- Lead debugs failures → spawn Sonnet investigator with the error
- Scouts return full file contents → cap at 500 words
- Two agents edit same file → strict file ownership with zero overlap
- Omit model param in Agent calls → always set model explicitly

## Common Mistakes to Avoid

- Do NOT use `node` or `npm` — this project runs on Bun exclusively.
- Do NOT modify `data/` files at runtime — they are read-only assets loaded at startup (except `llm_cache.sqlite`).
- Do NOT convert Float32Array to regular JS arrays — this wastes ~3x memory.
- The frontend is embedded HTML in `src/server.ts`, NOT a separate directory. Do not look for `public/`, `dist/`, or `client/` folders.
- The lemmatizer runs on port 3001, NOT 3000. Port 3000 is the main Bun server.
- LLM temperature correction should ADJUST the cosine similarity score, not replace it. The LLM provides a bounded adjustment (±15°); the base score comes from CamemBERT embeddings.
- Do NOT commit `.env` — it contains the OpenRouter API key.
- Before first server start, run Python generation scripts offline:
  - `scripts/generate_camembert_embeddings.py` — generates CamemBERT embeddings to `data/camembert_embeddings.{tsv,bin}`
  - `scripts/precompute_rankings.py` — generates top-1000 rankings to `data/precomputed_rankings.bin` and metadata to `data/rankings_metadata.json`
  - These scripts may require GPU for reasonable performance.

## References

- Project documentation: see `HANDOFF.md`
- Setup instructions: see `README.md`
- Primary embedding model: sentence-transformers/sentence-camembert-large on HuggingFace
- Legacy embedding model: Word2vec/fauconnier_frWiki_no_phrase_no_postag_500_cbow_cut10 on HuggingFace
- OpenRouter API: https://openrouter.ai/docs
