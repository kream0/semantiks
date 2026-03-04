# TODO

## Quick Resume
**Last session:** 7 (2026-03-04) — Generated CamemBERT embeddings + rankings
**Current state:** Game is playable. All code from Session 6 overhaul is active.
**Start here:** Play-test in browser, then address remaining issues below.

## Priority Tasks

### High
- [ ] Play-test game in browser — verify full UX flow (guess, hint, give-up, share, victory)
- [ ] Investigate secret word count mismatch: server selects 18,596 vs rankings has 65,649
- [ ] Decide on data file strategy: .gitignore the large files or use Git LFS

### Medium
- [ ] Fix MiniMax M2.5 reasoning overflow (`max_tokens=200` too low — content sometimes null)
- [ ] Fix OpenRouter provider routing (account locked to minimax/deepseek/moonshotai)
- [ ] Add `.venv/` and generated data files to `.gitignore`

### Low
- [ ] Consider auto-starting lemmatizer from Bun server (or documenting it better)
- [ ] Clean up legacy data files (`fauconnier_500.vec`, `french_embeddings.tsv`, `cc.fr.300.vec.gz`, `french_vectors_50k.vec`)
- [ ] Initial git commit with all project files

## Completed
- [x] Session 6: Full overhaul — CamemBERT code, LLM categorical scoring, rankings loader, SQLite cache, UX improvements
- [x] Session 7: Generate CamemBERT embeddings (66,579 words, 512D)
- [x] Session 7: Precompute top-1000 rankings (65,649 secret words)
- [x] Session 7: Validate build + server startup + game test
