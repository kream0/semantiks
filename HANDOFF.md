# Sémantix - Clone de Cémantix - Document de passation

## Résumé du projet
Clone du jeu français Cémantix - un jeu de devinette de mots basé sur la similarité sémantique.
Le joueur doit trouver un mot secret en proposant des mots et en recevant une "température" indiquant la proximité sémantique.

## Stack technique
- **Runtime**: Bun.sh (port 3000) + Python Flask (port 3001)
- **Langage**: TypeScript + Python
- **Serveur principal**: Bun.serve()
- **Service de lemmatisation**: Flask + spaCy
- **Frontend**: HTML/CSS/JS inline dans le serveur
- **Embeddings**: CamemBERT (sentence-camembert-large, 512D) — Fauconnier (500D) as legacy fallback
- **LLM**: MiniMax M2.5 via OpenRouter (minimax provider) — categorical judgment with bounded adjustment (±15°)

## Architecture des fichiers
```
semandtics-clone/
├── src/
│   └── server.ts                    # Serveur principal + frontend HTML + LLM
├── scripts/
│   ├── convert-vectors.ts           # Script de conversion des embeddings
│   ├── lemmatizer_service.py        # Service Flask pour lemmatisation spaCy
│   ├── download_fauconnier.py       # Téléchargement modèle Fauconnier depuis HuggingFace (legacy)
│   ├── generate_camembert_embeddings.py   # Génération embeddings CamemBERT (offline, GPU recommandé)
│   └── precompute_rankings.py       # Précomputation des classements top-1000 (offline)
├── data/
│   ├── camembert_embeddings.tsv     # CamemBERT embeddings (512D, 66k mots) - TSV principal
│   ├── camembert_embeddings.bin     # Format binaire optimisé (plus rapide au chargement)
│   ├── precomputed_rankings.bin     # Classements top-1000 pour tous les mots secrets (format SEMX)
│   ├── rankings_metadata.json       # Métadonnées des classements (version, date, count)
│   ├── llm_cache.sqlite             # Cache persistant des ajustements LLM (survit aux redémarrages)
│   ├── french_words.json            # Liste des mots (66,579 lemmes)
│   └── fauconnier_500.vec           # Modèle Word2Vec Fauconnier cut10 (500D) - fallback legacy
├── .env                             # Clé API OpenRouter (OPENROUTER_API_KEY)
├── requirements.txt                 # Dépendances Python
├── start.bat                        # Script de démarrage Windows
└── package.json
```

## Embeddings utilisés (MISE À JOUR SESSION 6 - CamemBERT)
- **Source primaire**: CamemBERT (sentence-camembert-large)
- **Modèle**: Sentence-transformers, fine-tuned pour français
- **Dimensions**: 512
- **Corpus**: Embeddings générés offline, stockés en TSV et format binaire optimisé
- **Mots chargés**: 66,579 lemmes français
- **Mots secrets potentiels**: 18,596
- **Format de stockage**: TSV principal + binaire optimisé (chargement ~2-3x plus rapide)
- **Fallback legacy**: Word2Vec Fauconnier 500D disponible si CamemBERT indisponible

## Services en cours d'exécution
| Service | Port | Description |
|---------|------|-------------|
| Bun (serveur jeu) | 3000 | API REST + frontend + LLM |
| Flask (lemmatisation) | 3001 | spaCy fr_core_news_sm |

## Fonctionnalités implémentées
1. ✅ Interface française colorée avec animations (orbs flottants, confettis)
2. ✅ Calcul de similarité cosinus avec CamemBERT (512D)
3. ✅ Système de température (-100° à 100°)
4. ✅ **Classements top-1000 précompilés** — affichage rapide du rang de proximité
5. ✅ **Mot du jour déterministe** — tous les joueurs reçoivent le même mot par jour
6. ✅ Sessions de jeu multiples
7. ✅ Boutons: Nouvelle Partie, Indice, Abandonner
8. ✅ Layout deux colonnes: jeu à gauche, essais à droite
9. ✅ **Autocomplétion par préfixe** (Tab) - "serpill" → "serpillière"
10. ✅ **Autocorrection automatique** - fautes corrigées à la soumission
11. ✅ **Guesses triées** (les plus proches d'abord)
12. ✅ **Bouton partager** — copie résumé du jeu au presse-papiers
13. ✅ **Gradient de couleurs continu** — évolution lisse de bleu à rouge
14. ✅ **Accessibilité ARIA** — labels descriptifs pour lecteurs d'écran
15. ✅ Cache des similarités pour performance
16. ✅ **Lemmatisation via spaCy** (verbes conjugués → infinitif, etc.)
17. ✅ **Correction LLM de la température** (MiniMax M2.5 via OpenRouter)
   - Évaluation catégorique (5 catégories) → ajustement borné (±15°)
   - Appel LLM **non-bloquant** — le score W2V est renvoyé immédiatement, l'ajustement LLM est caché en arrière-plan
   - Cache persistant en SQLite (survit aux redémarrages)

## API endpoints
- `POST /api/new-game` - Nouvelle partie
- `POST /api/guess` - Deviner un mot (avec lemmatisation + correction LLM)
- `POST /api/hint` - Obtenir un indice
- `POST /api/give-up` - Abandonner (révèle le mot)
- `POST /api/autocomplete` - Suggestions de mots similaires

### Service de lemmatisation (port 3001)
- `POST /lemmatize` - Lemmatise un mot (body: `{word: "mangé"}` → `{lemma: "manger"}`)
- `GET /health` - Vérification du service

## Pour lancer le projet
```bash
# Option 1: Script automatique (Windows)
start.bat

# Option 2: Manuel (2 terminaux)
# Terminal 1:
python scripts/lemmatizer_service.py

# Terminal 2:
bun run src/server.ts
```
Le jeu est accessible sur http://localhost:3000

---

## ✅ Session 7 (04/03/2026) - Data Generation

### Ce qui a été fait
- **Généré les embeddings CamemBERT** — 66,579 mots, 1024D→512D PCA (94.51% variance)
  - `data/camembert_embeddings.tsv` (316.6 MB) + `.bin` (130.7 MB)
- **Précompilé les classements top-1000** — 65,649 mots secrets × 1000 voisins
  - `data/precomputed_rankings.bin` (501.8 MB) + `rankings_metadata.json`
- **Validé le serveur** — build OK, CamemBERT chargé, températures bien distribuées (-15° à +38°), rangs fonctionnels
- **Environnement Python** — `.venv/` avec sentence-transformers, torch CPU, scikit-learn, protobuf

### Dépendances Python ajoutées au venv
```
sentence-transformers==5.2.3
torch==2.10.0+cpu
scikit-learn==1.8.0
protobuf==7.34.0
sentencepiece==0.2.1
tiktoken==0.12.0
```

### Le jeu est maintenant JOUABLE ✅

---

## ✅ Session 6 (04/03/2026) - Full Overhaul

> **Plan détaillé**: [`~/.claude/plans/zippy-foraging-corbato.md`](file:///home/karimel/.claude/plans/zippy-foraging-corbato.md)
> **Transcript**: `~/.claude/projects/.../eb6df293-c8a6-46b5-86d5-b79695ec1540.jsonl`

### Ce qui a été fait (code écrit et fonctionnel)

| Phase | Tâche | Statut |
|-------|-------|--------|
| Phase 0 | Script `generate_camembert_embeddings.py` (CamemBERT 1024D → PCA 512D) | ✅ Écrit |
| Phase 1 | Script `precompute_rankings.py` (top-1000 binaire SEMX) | ✅ Écrit |
| Phase 2 | Prompt LLM catégorique (5 catégories → ±15°) | ✅ Implémenté |
| Phase 3 | Chargement rankings précompilés + fallback | ✅ Implémenté |
| Phase 4 | Cache SQLite persistant + mot du jour déterministe | ✅ Implémenté |
| Phase 5 | UX: tri par temp, partager, gradient, ARIA | ✅ Implémenté |
| Phase 6 | Cleanup docs | ✅ Fait |
| Bonus | Appel LLM non-bloquant (fire-and-forget) | ✅ Implémenté |

### 🚨 CE QUI RESTE — Le jeu est CASSÉ sans ça

**Le serveur tourne encore avec les embeddings Fauconnier Word2Vec (500D, 2014, CBOW, corpus 600M mots).** Les températures sont inutiles — tout se compresse entre -11° et +5°. Le jeu est injouable.

| # | Tâche | Commande | Durée estimée | Bloquant |
|---|-------|----------|---------------|----------|
| 1 | **Générer les embeddings CamemBERT** | `python scripts/generate_camembert_embeddings.py` | 1-3h CPU, ~15min GPU | **OUI** |
| 2 | **Précomputer les rankings top-1000** | `python scripts/precompute_rankings.py` | 2-10 min | **OUI** |
| 3 | Valider les températures CamemBERT | Jouer une partie et vérifier la dispersion | 5 min | Non |
| 4 | Fix OpenRouter provider routing | Dashboard OpenRouter | 2 min | Non |
| 5 | Augmenter `max_tokens` LLM ou ajouter `/no_think` | `src/server.ts` ligne 345 | 1 min | Non |

**Tâche 1 est LE bloqueur.** Sans embeddings CamemBERT, tout le reste du code est inutile — le serveur fallback sur Fauconnier Word2Vec qui produit des scores poubelle.

### Bugs connus
- **Template literal escaping**: `\n` dans le JS embarqué doit être `String.fromCharCode(10)` (fixé)
- **MiniMax M2.5 reasoning overflow**: Avec `max_tokens=200`, le modèle reason trop longtemps et `content` reste `null`. Le fallback extrait la catégorie du champ `reasoning` mais ça échoue parfois (ex: `chimie`, `géographe`)
- **OpenRouter provider routing**: Compte verrouillé sur minimax/deepseek/moonshotai — certains modèles 404

---

## Modifications Session 4 (26/12/2025) - Vocabulaire étendu

### Problème résolu
- Les mots courants comme "serpillère" n'étaient pas dans le dictionnaire
- L'ancien modèle (cut200) n'avait que 31k mots

### Solution
- Passage au modèle Fauconnier `cut10` (mots avec 10+ occurrences au lieu de 200+)
- Vocabulaire doublé: 66,579 mots au lieu de 31,446
- Dimensions réduites à 500 (vs 1000) pour compenser la taille
- **Normalisation des accents** pour recherche tolérante
- **Autocorrection automatique** pour les fautes d'orthographe (distance Levenshtein ≤ 2)
  - Ex: "serpillere" → "serpillière" (correction automatique)
  - Ex: "chatau" → "château" (correction automatique)
- **Autocomplétion par préfixe** - suggestions basées sur le début du mot tapé
  - Ex: "serpill" → suggère "serpillière" en premier
  - Fonctionne avec ou sans accents ("chate" → "château")

### Détails techniques

#### Autocorrection à la soumission
```typescript
// Si mot non trouvé, essayer autocorrection avec Levenshtein distance <= 2
const closestWords = findClosestWords(rawWord, 1);
if (closestWords.length > 0 && levenshteinDistance(rawWord, closest) <= 2) {
  word = closest; // Utiliser le mot corrigé automatiquement
}
```

#### Autocomplétion par préfixe (Tab)
```typescript
// Priorité aux mots commençant par l'input (avec normalisation des accents)
const inputNormalized = normalizeAccents(inputLower);
for (const word of allWords) {
  if (normalizeAccents(word).startsWith(inputNormalized)) {
    scored.push({ word, distance: word.length - input.length, prefixBonus: -10 });
  }
}
```

### Fichiers modifiés
- `scripts/download_fauconnier.py` - Nouveau repo HuggingFace (cut10)
- `scripts/convert-vectors.ts` - Support 500D, limite à 150k mots
- `src/server.ts` - Autocorrection, préfixes, normalisation accents
- `HANDOFF.md` - Documentation mise à jour

---

## Modifications Session 3 (26/12/2025) - Correction LLM

### ✅ IMPLÉMENTÉ: Correction LLM de la température

La température Word2Vec est maintenant corrigée par un LLM pour une UX plus "humaine".

### Configuration LLM (Session 6)
- **API**: OpenRouter (https://openrouter.ai/api/v1/chat/completions)
- **Modèle**: `minimax/minimax-m2.5` (reasoning model)
- **Provider**: minimax (compte verrouillé sur minimax/deepseek/moonshotai)
- **Clé API**: Stockée dans `.env` (variable `OPENROUTER_API_KEY`)

### Flux de traitement d'un guess (Session 6)
```
[Utilisateur entre "sport"]
    → Lemmatisation ("sport")
    → Similarité cosinus embeddings (dot product, <0.1ms)
    → Vérification cache LLM (synchrone, <1ms)
    → Si cache hit: appliquer ajustement → répondre
    → Si cache miss: répondre avec score W2V pur, lancer LLM en arrière-plan
    → LLM catégorise → cache SQLite + mémoire pour prochaine fois
```

### Prompt LLM catégorique (Session 6)
```
Mot secret: "${secretWord}". Mot proposé: "${guessWord}".
La similarité Word2Vec est de ${word2vecTemp}°.
Historique des derniers essais: ${last10guesses}

Choisis UNE catégorie:
- BEAUCOUP_TROP_HAUT (-15°)
- TROP_HAUT (-8°)
- CORRECT (0°)
- TROP_BAS (+8°)
- BEAUCOUP_TROP_BAS (+15°)
```

### Fonctionnalités LLM
- ✅ Appel LLM **non-bloquant** (fire-and-forget, réponse en <10ms)
- ✅ Cache persistant SQLite (survit aux redémarrages)
- ✅ Cache mémoire en premier niveau (Map, <1ms)
- ✅ Historique des 10 derniers essais inclus dans le prompt
- ✅ Fallback sur Word2Vec pur si LLM indisponible
- ✅ Extraction catégorie depuis `reasoning` field (modèles reasoning)
- ✅ Timeout de 10 secondes par requête

### Fichiers modifiés (Session 3)
- `src/server.ts` - Ajout fonction `adjustTemperatureWithLLM()`, intégration dans `/api/guess`
- `.env` (nouveau) - Clé API OpenRouter

### Autres corrections (Session 3)
- ✅ Protection anti-double-submit (flag `isSubmitting`)
- ✅ Vérification des doublons améliorée (mot brut + lemme)
- ✅ `e.preventDefault()` sur touche Entrée

---

## Modifications Session 2 (25/12/2025)

### Changement majeur: Modèle Word2Vec Fauconnier
- Remplacé FastText (300D) par Word2Vec Fauconnier (1000D)
- Même modèle que le vrai Cémantix
- Source: https://huggingface.co/Word2vec/fauconnier_frWiki_no_phrase_no_postag_1000_skip_cut200

### Ajout de la lemmatisation
- Service Flask avec spaCy `fr_core_news_sm`
- Les mots entrés sont convertis en lemmes avant recherche
- Ex: "mangé" → "manger", "chats" → "chat"

### Fichiers créés
- `scripts/lemmatizer_service.py` - Service Flask
- `scripts/download_fauconnier.py` - Téléchargement modèle
- `requirements.txt` - Dépendances Python
- `start.bat` - Script de démarrage Windows

### Fichiers modifiés
- `src/server.ts` - Intégration lemmatisation + support 1000D
- `scripts/convert-vectors.ts` - Conversion modèle Fauconnier

---

## Contexte: Sémantique Word2Vec vs Intuition Humaine

### Le problème (résolu par LLM)
Word2Vec capture les **co-occurrences** dans les textes, pas les relations sémantiques "logiques" humaines.

**Exemple concret** (mot secret: "match"):
| Mot | Word2Vec | LLM corrigé | Attendu |
|-----|----------|-------------|---------|
| équipe | 50.51° | ~50° | Chaud |
| sport | 7.04° | ~45° | Chaud |
| maison | 7.1° | ~-20° | Froid |

### Explication
- Word2Vec ne comprend pas que "match" EST un type de "sport"
- Le LLM corrige cette lacune avec une compréhension sémantique humaine

---

## Dépendances installées

### Python (requirements.txt)
```
flask==3.0.0
flask-cors==4.0.0
spacy==3.7.2 (ou 3.8.11)
gensim==4.3.2
huggingface_hub==0.20.3
```

### Modèle spaCy
```bash
python -m spacy download fr_core_news_sm
```

## Points d'attention
- Le chargement des embeddings prend ~10-15 secondes au démarrage (66k × 500D ou 512D)
- RAM utilisée: ~130-136 MB pour les embeddings Float32Array
- Le service de lemmatisation doit être lancé AVANT le serveur principal
- Les embeddings sont stockés en Float32Array — JAMAIS convertir en Array
- **La clé API OpenRouter doit être configurée dans `.env`**
- **Latence guess: <10ms** (LLM non-bloquant, score W2V immédiat)
- **Latence LLM background: 3-10s** par requête (le cache SQLite évite les appels répétés)
- **CRITIQUE: Générer les embeddings CamemBERT avant de jouer** — sans ça, le jeu utilise Fauconnier Word2Vec qui produit des scores inutiles
