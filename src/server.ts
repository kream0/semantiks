import { serve } from "bun";
import { readFileSync, existsSync } from "fs";
import { Database } from "bun:sqlite";

// Configuration du service de lemmatisation
const LEMMATIZER_URL = "http://localhost:3001";
let lemmatizerAvailable = false;

// Configuration LLM (OpenRouter - Qwen3-8b)
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const LLM_ENABLED = !!OPENROUTER_API_KEY;

// Mapping catégorie → ajustement de température
const CATEGORY_ADJUSTMENTS: Record<string, number> = {
  "BEAUCOUP_TROP_HAUT": -15,
  "TROP_HAUT": -8,
  "CORRECT": 0,
  "TROP_BAS": +8,
  "BEAUCOUP_TROP_BAS": +15,
};

// Cache SQLite persistant pour les ajustements LLM
const cacheDb = new Database("data/llm_cache.sqlite");
cacheDb.run("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, adjustment INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP)");

// Cache in-memory pour les ajustements LLM: Map<"secretWord:guessWord", adjustment>
const llmAdjustmentCache = new Map<string, number>();

// Charger le cache SQLite existant en mémoire au démarrage
const existingCacheRows = cacheDb.query("SELECT key, adjustment FROM llm_cache").all() as { key: string; adjustment: number }[];
for (const row of existingCacheRows) {
  llmAdjustmentCache.set(row.key, row.adjustment);
}
console.log(`[Cache] Chargé ${llmAdjustmentCache.size} entrées LLM depuis SQLite`);

// Persister un ajustement LLM dans le cache SQLite + mémoire
function cacheLLMAdjustment(key: string, adjustment: number): void {
  llmAdjustmentCache.set(key, adjustment);
  cacheDb.run("INSERT OR REPLACE INTO llm_cache (key, adjustment) VALUES (?, ?)", [key, adjustment]);
}

// Fonction pour lemmatiser un mot via le service spaCy
async function lemmatize(word: string): Promise<string> {
  if (!lemmatizerAvailable) return word;

  try {
    const response = await fetch(`${LEMMATIZER_URL}/lemmatize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ word })
    });

    if (!response.ok) return word;

    const data = await response.json() as { lemma: string };
    return data.lemma || word;
  } catch {
    return word; // Fallback si service indisponible
  }
}

// Vérifier si le service de lemmatisation est disponible
async function checkLemmatizerService(): Promise<boolean> {
  try {
    const response = await fetch(`${LEMMATIZER_URL}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(2000)
    });
    if (response.ok) {
      console.log("[OK] Service de lemmatisation disponible sur " + LEMMATIZER_URL);
      return true;
    }
    return false;
  } catch {
    console.log("[WARN] Service de lemmatisation non disponible - fonctionnement sans lemmatisation");
    return false;
  }
}

// Detect which embeddings file to use: prefer CamemBERT if available, fallback to Word2Vec
const CAMEMBERT_PATH = "data/camembert_embeddings.tsv";
const WORD2VEC_PATH = "data/french_embeddings.tsv";
const embeddingsPath = existsSync(CAMEMBERT_PATH) ? CAMEMBERT_PATH : WORD2VEC_PATH;

console.log(`Loading French word embeddings from: ${embeddingsPath}`);

// Load word embeddings from file (TSV format: word\tv1,v2,...,vN)
const embeddingsFile = readFileSync(embeddingsPath, "utf-8");
const embeddingLines = embeddingsFile.trim().split("\n");

const wordVectors: Map<string, Float32Array> = new Map();
const allWords: string[] = [];

// Normaliser les accents pour la recherche tolérante
function normalizeAccents(str: string): string {
  return str.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
}

// Map de mots normalisés → mot original pour recherche tolérante
const normalizedToOriginal: Map<string, string> = new Map();

for (const line of embeddingLines) {
  const [word, vectorStr] = line.split("\t");
  if (word && vectorStr) {
    const vector = new Float32Array(vectorStr.split(",").map(Number));
    wordVectors.set(word, vector);
    allWords.push(word);
    // Ajouter la version normalisée pour la recherche tolérante
    const normalized = normalizeAccents(word);
    if (!normalizedToOriginal.has(normalized)) {
      normalizedToOriginal.set(normalized, word);
    }
  }
}

// Déterminer la dimension des vecteurs depuis la première ligne (supporte 300D, 500D, 512D, etc.)
const sampleVector = wordVectors.values().next().value;
const vectorDimension = sampleVector ? sampleVector.length : 300;

const embeddingModelName = embeddingsPath === CAMEMBERT_PATH ? "CamemBERT" : "Fauconnier Word2Vec";
console.log(`Loaded ${wordVectors.size} word embeddings (${vectorDimension} dimensions) — Model: ${embeddingModelName}`);

// Vérifier le service de lemmatisation au démarrage
checkLemmatizerService().then(available => {
  lemmatizerAvailable = available;
});

// Precomputed rankings: Map<secretWord, { indices: Uint32Array, scores: Float32Array }>
// Loaded from data/precomputed_rankings.bin at startup for O(1) proximity rank lookups.
let precomputedRankings: Map<string, { indices: Uint32Array; scores: Float32Array }> = new Map();
let wordToIndex: Map<string, number> = new Map();

async function loadPrecomputedRankings(): Promise<void> {
  const rankingsPath = "data/precomputed_rankings.bin";
  const file = Bun.file(rankingsPath);
  if (!(await file.exists())) {
    console.log("[Rankings] No precomputed rankings found — will use on-the-fly computation");
    return;
  }

  const buffer = await file.arrayBuffer();
  const view = new DataView(buffer);
  let offset = 0;

  // Parse header (17 bytes)
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
  );
  if (magic !== "SEMX") throw new Error(`Invalid rankings file magic: expected SEMX, got ${magic}`);
  offset = 4;
  const _version = view.getUint8(offset); offset += 1;
  const vocabSize = view.getUint32(offset, true); offset += 4;
  const secretCount = view.getUint32(offset, true); offset += 4;
  const topN = view.getUint16(offset, true); offset += 2;
  const _dims = view.getUint16(offset, true); offset += 2;

  // Parse word index
  const indexToWord: string[] = [];
  const decoder = new TextDecoder();
  for (let i = 0; i < vocabSize; i++) {
    const wordLen = view.getUint16(offset, true); offset += 2;
    const wordBytes = new Uint8Array(buffer, offset, wordLen);
    const word = decoder.decode(wordBytes);
    indexToWord.push(word);
    wordToIndex.set(word, i);
    offset += wordLen;
  }

  // Parse rankings: vocab may exceed 65535, so use Uint32Array for neighbor indices
  for (let i = 0; i < secretCount; i++) {
    const secretIdx = view.getUint32(offset, true); offset += 4;
    const indices = new Uint32Array(topN);
    const scores = new Float32Array(topN);
    for (let j = 0; j < topN; j++) {
      indices[j] = view.getUint32(offset, true); offset += 4;
      scores[j] = view.getFloat32(offset, true); offset += 4;
    }
    precomputedRankings.set(indexToWord[secretIdx], { indices, scores });
  }

  console.log(`[Rankings] Loaded precomputed rankings: ${secretCount} secret words, top ${topN} each`);
}

// Load precomputed rankings at startup (non-blocking — falls back gracefully)
loadPrecomputedRankings().catch(err => {
  console.error("[Rankings] Failed to load precomputed rankings:", err);
});

// Filter for good secret words (common nouns/adjectives, 4-12 letters)
const secretWordCandidates = allWords.filter(word => {
  if (word.length < 4 || word.length > 12) return false;
  const skipWords = new Set(["dans", "pour", "avec", "sans", "sous", "chez", "vers", "mais", "donc", "comme", "quand", "cette", "sont", "être", "avoir", "fait", "faire", "peut", "tout", "tous", "plus", "moins", "très", "bien", "aussi"]);
  if (skipWords.has(word)) return false;
  const index = allWords.indexOf(word);
  return index < 20000 && index > 100;
});

console.log(`Selected ${secretWordCandidates.length} potential secret words`);

// Levenshtein distance for autocorrection
function levenshteinDistance(a: string, b: string): number {
  const matrix: number[][] = [];

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }

  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }

  return matrix[b.length][a.length];
}

// Find closest words by spelling (for autocorrection)
function findClosestWords(input: string, limit: number = 5): string[] {
  if (!input || input.length < 2) return [];

  const inputLower = input.toLowerCase();
  const inputNormalized = normalizeAccents(inputLower);
  const scored: Array<{ word: string; distance: number; prefixBonus: number }> = [];

  for (const word of allWords) {
    const wordNormalized = normalizeAccents(word);

    // Inclure les mots qui commencent par l'input (préfixe) - priorité haute
    if (wordNormalized.startsWith(inputNormalized) || word.startsWith(inputLower)) {
      const distance = word.length - inputLower.length; // Distance = caractères manquants
      scored.push({ word, distance, prefixBonus: -10 }); // Bonus fort pour les préfixes
      continue;
    }

    // Skip if length difference is too big (pour Levenshtein)
    if (Math.abs(word.length - inputLower.length) > 4) continue;

    const distance = levenshteinDistance(inputLower, word);

    // Only consider words within reasonable distance
    if (distance <= Math.max(3, Math.floor(inputLower.length / 2))) {
      const startsWithBonus = word.startsWith(inputLower.slice(0, 2)) ? -2 : 0;
      scored.push({ word, distance, prefixBonus: startsWithBonus });
    }
  }

  // Sort by prefix bonus first, then by distance
  scored.sort((a, b) => {
    const bonusDiff = a.prefixBonus - b.prefixBonus;
    if (bonusDiff !== 0) return bonusDiff;
    return a.distance - b.distance;
  });

  return scored.slice(0, limit).map(s => s.word);
}

// Calculate cosine similarity between two vectors
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Calculate semantic similarity between two words (returns temperature -100 to 100)
function calculateSimilarity(word1: string, word2: string): number {
  if (word1 === word2) return 100;

  const vec1 = wordVectors.get(word1);
  const vec2 = wordVectors.get(word2);

  if (!vec1 || !vec2) return -100;

  const similarity = cosineSimilarity(vec1, vec2);
  return Math.round(similarity * 100 * 100) / 100;
}

// Obtenir l'ajustement LLM catégoriel pour affiner la température Word2Vec
async function getLLMAdjustment(
  secretWord: string,
  guessWord: string,
  word2vecTemp: number,
  previousGuesses: Array<{ word: string; temperature: number }>
): Promise<number> {
  const cacheKey = `${secretWord}:${guessWord}`;

  // Vérifier le cache in-memory (chargé depuis SQLite au démarrage)
  if (llmAdjustmentCache.has(cacheKey)) {
    return llmAdjustmentCache.get(cacheKey)!;
  }

  // Construire l'historique des 10 derniers essais
  const recentGuesses = previousGuesses.slice(-10);
  const last10guesses = recentGuesses.length > 0
    ? recentGuesses.map(g => `- "${g.word}": ${g.temperature}°`).join("\n")
    : "(aucun essai précédent)";

  const prompt = `Mot secret: "${secretWord}". Mot proposé: "${guessWord}".
La similarité Word2Vec est de ${word2vecTemp}°.

Historique des derniers essais:
${last10guesses}

La similarité Word2Vec est-elle correcte pour ce couple de mots?
Choisis UNE catégorie:
- BEAUCOUP_TROP_HAUT (le lien réel est bien plus faible)
- TROP_HAUT (le lien réel est un peu plus faible)
- CORRECT (la similarité est juste)
- TROP_BAS (le lien réel est un peu plus fort)
- BEAUCOUP_TROP_BAS (le lien réel est bien plus fort)

Réponds avec UN SEUL MOT: le nom de la catégorie.`;

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "Semantix Game"
    },
    body: JSON.stringify({
      model: "minimax/minimax-m2.5",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 400,
      temperature: 0
    }),
    signal: AbortSignal.timeout(10000) // Timeout 10s
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`[LLM] API error ${response.status}:`, errorText);
    throw new Error(`LLM API error: ${response.status}`);
  }

  const data = await response.json() as {
    choices?: Array<{ message?: { content?: string; reasoning?: string } }>;
    error?: { message?: string };
  };

  if (data.error) {
    console.error("[LLM] API returned error:", data.error);
    throw new Error(`LLM error: ${data.error.message}`);
  }

  // Some reasoning models (MiniMax M2.5) put the answer in content but reason in reasoning field
  // Extract content first, fallback to searching reasoning field for category keywords
  const msg = data.choices?.[0]?.message;
  let content = msg?.content?.trim();
  if (!content && msg?.reasoning) {
    // Search reasoning text for a category keyword
    const reasoningUpper = msg.reasoning.toUpperCase();
    const foundCat = Object.keys(CATEGORY_ADJUSTMENTS).find(cat => reasoningUpper.includes(cat));
    if (foundCat) {
      content = foundCat;
      console.log(`[LLM] Extracted category from reasoning field for ${guessWord}: ${foundCat}`);
    }
  }
  if (!content) {
    console.log(`[LLM] No content in response for ${guessWord}:`, JSON.stringify(data).substring(0, 500));
  }
  console.log(`[LLM] Raw response for ${guessWord}:`, content);

  // Parser la catégorie depuis la réponse
  const responseText = (content ?? "").trim().toUpperCase();
  const category = Object.keys(CATEGORY_ADJUSTMENTS).find(cat => responseText.includes(cat));
  const adjustment = category !== undefined ? CATEGORY_ADJUSTMENTS[category] : 0; // fallback: pas d'ajustement

  if (!category) {
    console.warn(`[LLM] Catégorie non reconnue dans la réponse: "${content}" — ajustement 0 appliqué`);
  } else {
    console.log(`[LLM] Catégorie: ${category}, Ajustement: ${adjustment > 0 ? "+" : ""}${adjustment}°`);
  }

  // Persister dans le cache SQLite + mémoire
  cacheLLMAdjustment(cacheKey, adjustment);
  return adjustment;
}

// Pre-compute similarities for secret word (cached per session)
interface SimilarityCache {
  secretWord: string;
  rankings: Array<{ word: string; similarity: number }>;
}

const similarityCaches = new Map<string, SimilarityCache>();

function getOrCreateSimilarityCache(secretWord: string): SimilarityCache {
  let cache = similarityCaches.get(secretWord);
  if (cache) return cache;

  console.log(`Computing similarity rankings for: ${secretWord}`);
  const startTime = Date.now();

  const secretVec = wordVectors.get(secretWord);
  if (!secretVec) throw new Error(`Secret word not found: ${secretWord}`);

  const rankings: Array<{ word: string; similarity: number }> = [];

  for (const word of allWords) {
    if (word !== secretWord) {
      const vec = wordVectors.get(word);
      if (vec) {
        const similarity = cosineSimilarity(secretVec, vec);
        rankings.push({ word, similarity });
      }
    }
  }

  rankings.sort((a, b) => b.similarity - a.similarity);

  cache = { secretWord, rankings };
  similarityCaches.set(secretWord, cache);

  console.log(`Computed rankings in ${Date.now() - startTime}ms`);
  return cache;
}

// Fallback: compute proximity rank on-the-fly using lazy similarity cache
function getProximityRankFallback(secretWord: string, guessWord: string): number | null {
  const cache = getOrCreateSimilarityCache(secretWord);
  const rank = cache.rankings.findIndex(r => r.word === guessWord);

  if (rank !== -1 && rank < 1000) {
    return 1000 - rank;
  }

  return null;
}

// Get rank among top 1000 closest words — uses precomputed rankings if available
function getProximityRank(secretWord: string, guessWord: string): number | null {
  const rankings = precomputedRankings.get(secretWord);
  if (!rankings) {
    // Fallback to on-the-fly computation if rankings not precomputed
    return getProximityRankFallback(secretWord, guessWord);
  }

  const guessIdx = wordToIndex.get(guessWord);
  if (guessIdx === undefined) return null;

  // Search for the guess word in the top-1000 indices
  for (let i = 0; i < rankings.indices.length; i++) {
    if (rankings.indices[i] === guessIdx) {
      return 1000 - i; // Rank 1000 = closest, 1 = 1000th closest
    }
  }

  return null; // Not in top 1000
}

// Game sessions storage
interface GameSession {
  secretWord: string;
  guesses: Array<{ word: string; temperature: number; rank: number | null; timestamp: number }>;
  startTime: number;
  solved: boolean;
}

const gameSessions = new Map<string, GameSession>();

// Sélection déterministe du mot du jour (même mot pour tous les joueurs le même jour)
function getDailySecretWord(candidates: string[]): string {
  const daysSinceEpoch = Math.floor(Date.now() / 86400000);
  const index = ((daysSinceEpoch * 2654435761) >>> 0) % candidates.length;
  return candidates[index];
}

// Select the daily secret word (deterministic based on current day)
function selectSecretWord(): string {
  return getDailySecretWord(secretWordCandidates);
}

// HTML Template with side panel layout
const htmlTemplate = `
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sémantix - Trouvez le mot secret !</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    :root {
      --primary: #6366f1;
      --primary-dark: #4f46e5;
      --secondary: #ec4899;
      --accent: #14b8a6;
      --success: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
      --bg-dark: #0f172a;
      --bg-card: #1e293b;
      --bg-input: #334155;
      --text-light: #f1f5f9;
      --text-muted: #94a3b8;
    }

    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a2e 50%, #16213e 100%);
      min-height: 100vh;
      color: var(--text-light);
      overflow-x: hidden;
    }

    .background-animation {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 0;
      overflow: hidden;
    }

    .floating-orb {
      position: absolute;
      border-radius: 50%;
      filter: blur(80px);
      opacity: 0.3;
      animation: float 20s infinite ease-in-out;
    }

    .orb-1 {
      width: 400px;
      height: 400px;
      background: var(--primary);
      top: -100px;
      left: -100px;
    }

    .orb-2 {
      width: 300px;
      height: 300px;
      background: var(--secondary);
      top: 50%;
      right: -100px;
      animation-delay: -5s;
    }

    .orb-3 {
      width: 350px;
      height: 350px;
      background: var(--accent);
      bottom: -100px;
      left: 30%;
      animation-delay: -10s;
    }

    @keyframes float {
      0%, 100% { transform: translate(0, 0) scale(1); }
      25% { transform: translate(50px, 50px) scale(1.1); }
      50% { transform: translate(0, 100px) scale(0.9); }
      75% { transform: translate(-50px, 50px) scale(1.05); }
    }

    .main-layout {
      position: relative;
      z-index: 1;
      display: flex;
      justify-content: center;
      min-height: 100vh;
      gap: 20px;
      padding: 20px;
    }

    .game-panel {
      width: 100%;
      max-width: 600px;
    }

    .guesses-panel {
      width: 350px;
      flex-shrink: 0;
    }

    header {
      text-align: center;
      padding: 20px 0 30px;
    }

    h1 {
      font-size: 3rem;
      font-weight: 800;
      background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 8px;
      letter-spacing: -2px;
    }

    .subtitle {
      color: var(--text-muted);
      font-size: 1rem;
    }

    .game-card {
      background: rgba(30, 41, 59, 0.8);
      backdrop-filter: blur(20px);
      border-radius: 24px;
      padding: 25px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }

    .input-wrapper {
      position: relative;
      margin-bottom: 15px;
    }

    #guess-input {
      width: 100%;
      padding: 18px 24px;
      font-size: 1.2rem;
      background: var(--bg-input);
      border: 2px solid transparent;
      border-radius: 16px;
      color: var(--text-light);
      outline: none;
      transition: all 0.3s ease;
    }

    #guess-input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
    }

    #guess-input::placeholder {
      color: var(--text-muted);
    }

    .autocomplete-hint {
      position: absolute;
      right: 15px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-muted);
      font-size: 0.85rem;
      pointer-events: none;
      display: none;
    }

    .autocomplete-hint.visible {
      display: block;
    }

    .autocomplete-hint kbd {
      background: var(--bg-dark);
      padding: 2px 8px;
      border-radius: 4px;
      font-family: inherit;
      border: 1px solid var(--text-muted);
    }

    .suggestion-box {
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      background: var(--bg-card);
      border-radius: 12px;
      margin-top: 8px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      overflow: hidden;
      display: none;
      z-index: 10;
    }

    .suggestion-box.visible {
      display: block;
    }

    .suggestion-item {
      padding: 12px 20px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: background 0.2s;
    }

    .suggestion-item:hover, .suggestion-item.selected {
      background: var(--primary);
    }

    .suggestion-word {
      font-weight: 600;
    }

    .suggestion-distance {
      font-size: 0.8rem;
      color: var(--text-muted);
      background: rgba(0,0,0,0.3);
      padding: 2px 8px;
      border-radius: 10px;
    }

    .button-row {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
    }

    .btn {
      padding: 14px 24px;
      font-size: 1rem;
      font-weight: 600;
      border: none;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.3s ease;
      text-transform: uppercase;
      letter-spacing: 1px;
      flex: 1;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
      color: white;
      box-shadow: 0 10px 40px rgba(99, 102, 241, 0.4);
    }

    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 15px 50px rgba(99, 102, 241, 0.6);
    }

    .btn-secondary {
      background: var(--bg-input);
      color: var(--text-light);
    }

    .btn-secondary:hover {
      background: rgba(51, 65, 85, 0.8);
    }

    .stats-bar {
      display: flex;
      justify-content: space-around;
      margin-bottom: 20px;
      padding: 15px;
      background: rgba(15, 23, 42, 0.5);
      border-radius: 16px;
    }

    .stat {
      text-align: center;
    }

    .stat-value {
      font-size: 1.8rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent) 0%, var(--success) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .stat-label {
      color: var(--text-muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .legend {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      padding: 15px;
      background: rgba(15, 23, 42, 0.5);
      border-radius: 16px;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .legend-color {
      width: 18px;
      height: 18px;
      border-radius: 6px;
    }

    .legend-text {
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    .message {
      text-align: center;
      padding: 15px;
      border-radius: 12px;
      margin-bottom: 15px;
      font-size: 1rem;
    }

    .message.error {
      background: rgba(239, 68, 68, 0.2);
      color: #fca5a5;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .message.success {
      background: rgba(34, 197, 94, 0.2);
      color: #86efac;
      border: 1px solid rgba(34, 197, 94, 0.3);
    }

    /* Guesses Panel */
    .guesses-card {
      background: rgba(30, 41, 59, 0.8);
      backdrop-filter: blur(20px);
      border-radius: 24px;
      padding: 20px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      height: calc(100vh - 40px);
      display: flex;
      flex-direction: column;
    }

    .guesses-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 15px;
      padding-bottom: 15px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    .guesses-title {
      font-size: 1.2rem;
      font-weight: 700;
      color: var(--text-light);
    }

    .guesses-count {
      background: var(--primary);
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 600;
    }

    .guesses-container {
      flex: 1;
      overflow-y: auto;
      padding-right: 5px;
    }

    .guesses-container::-webkit-scrollbar {
      width: 6px;
    }

    .guesses-container::-webkit-scrollbar-track {
      background: var(--bg-input);
      border-radius: 3px;
    }

    .guesses-container::-webkit-scrollbar-thumb {
      background: var(--primary);
      border-radius: 3px;
    }

    .guess-row {
      display: grid;
      grid-template-columns: 1fr 70px 50px;
      gap: 10px;
      padding: 12px 15px;
      margin-bottom: 8px;
      background: rgba(15, 23, 42, 0.6);
      border-radius: 10px;
      align-items: center;
      animation: slideIn 0.3s ease;
      border-left: 3px solid transparent;
    }

    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateX(-10px);
      }
      to {
        opacity: 1;
        transform: translateX(0);
      }
    }

    .guess-row.hot {
      border-left-color: #ef4444;
      background: linear-gradient(90deg, rgba(239, 68, 68, 0.2) 0%, rgba(15, 23, 42, 0.6) 100%);
    }

    .guess-row.warm {
      border-left-color: #f59e0b;
      background: linear-gradient(90deg, rgba(245, 158, 11, 0.2) 0%, rgba(15, 23, 42, 0.6) 100%);
    }

    .guess-row.cool {
      border-left-color: #3b82f6;
      background: linear-gradient(90deg, rgba(59, 130, 246, 0.2) 0%, rgba(15, 23, 42, 0.6) 100%);
    }

    .guess-row.cold {
      border-left-color: #6366f1;
      background: linear-gradient(90deg, rgba(99, 102, 241, 0.1) 0%, rgba(15, 23, 42, 0.6) 100%);
    }

    .guess-row.found {
      border-left-color: var(--success);
      background: linear-gradient(90deg, rgba(34, 197, 94, 0.3) 0%, rgba(15, 23, 42, 0.6) 100%);
      animation: pulse 1s ease infinite;
    }

    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 15px rgba(34, 197, 94, 0.3); }
      50% { box-shadow: 0 0 25px rgba(34, 197, 94, 0.6); }
    }

    .guess-word {
      font-weight: 600;
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .guess-temp {
      font-weight: 700;
      font-size: 0.9rem;
      text-align: right;
    }

    .guess-rank {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-align: right;
    }

    .guess-rank.has-rank {
      color: var(--secondary);
      font-weight: 600;
    }

    .victory-modal {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      z-index: 100;
      align-items: center;
      justify-content: center;
      backdrop-filter: blur(10px);
    }

    .victory-modal.active {
      display: flex;
    }

    .victory-content {
      background: var(--bg-card);
      padding: 50px;
      border-radius: 32px;
      text-align: center;
      max-width: 500px;
      animation: bounceIn 0.5s ease;
      border: 2px solid var(--success);
      box-shadow: 0 0 100px rgba(34, 197, 94, 0.5);
    }

    @keyframes bounceIn {
      0% { transform: scale(0.5); opacity: 0; }
      50% { transform: scale(1.05); }
      100% { transform: scale(1); opacity: 1; }
    }

    .victory-emoji {
      font-size: 5rem;
      margin-bottom: 20px;
    }

    .victory-title {
      font-size: 2.5rem;
      font-weight: 800;
      background: linear-gradient(135deg, var(--success) 0%, var(--accent) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 15px;
    }

    .victory-word {
      font-size: 2rem;
      font-weight: 700;
      color: var(--text-light);
      margin-bottom: 10px;
      text-transform: uppercase;
      letter-spacing: 3px;
    }

    .victory-stats {
      color: var(--text-muted);
      margin-bottom: 30px;
      font-size: 1.1rem;
    }

    .info-section {
      margin-top: 20px;
    }

    .info-toggle {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      cursor: pointer;
      color: var(--text-muted);
      padding: 10px;
      transition: color 0.3s ease;
      font-size: 0.9rem;
    }

    .info-toggle:hover {
      color: var(--text-light);
    }

    .info-content {
      display: none;
      padding: 20px;
      background: rgba(15, 23, 42, 0.5);
      border-radius: 16px;
      margin-top: 10px;
      line-height: 1.6;
      color: var(--text-muted);
      font-size: 0.9rem;
    }

    .info-content.active {
      display: block;
    }

    .info-content h3 {
      color: var(--text-light);
      margin-bottom: 10px;
      font-size: 1.1rem;
    }

    .info-content p {
      margin-bottom: 12px;
    }

    .confetti {
      position: fixed;
      top: -20px;
      z-index: 99;
      animation: confettiFall 3s linear forwards;
    }

    @keyframes confettiFall {
      to {
        transform: translateY(100vh) rotate(720deg);
        opacity: 0;
      }
    }

    .word-count {
      text-align: center;
      color: var(--text-muted);
      font-size: 0.75rem;
      margin-top: 15px;
    }

    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 200px;
      color: var(--text-muted);
      text-align: center;
    }

    .empty-state-icon {
      font-size: 3rem;
      margin-bottom: 15px;
      opacity: 0.5;
    }

    @media (max-width: 1024px) {
      .main-layout {
        flex-direction: column;
      }

      .game-panel {
        max-width: 100%;
      }

      .guesses-panel {
        width: 100%;
      }

      .guesses-card {
        height: auto;
        max-height: 400px;
      }
    }

    @media (max-width: 600px) {
      h1 {
        font-size: 2rem;
      }

      .button-row {
        flex-wrap: wrap;
      }

      .btn {
        padding: 12px 16px;
        font-size: 0.85rem;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      * { animation: none !important; transition: none !important; }
    }

    .btn-share {
      background: linear-gradient(135deg, #22c55e 0%, #14b8a6 100%);
      color: white;
      box-shadow: 0 10px 40px rgba(34, 197, 94, 0.4);
      margin-top: 10px;
    }

    .btn-share:hover {
      transform: translateY(-2px);
      box-shadow: 0 15px 50px rgba(34, 197, 94, 0.6);
    }

    .share-feedback {
      font-size: 0.9rem;
      color: var(--success);
      margin-top: 8px;
      min-height: 1.2em;
    }
  </style>
</head>
<body>
  <div class="background-animation">
    <div class="floating-orb orb-1"></div>
    <div class="floating-orb orb-2"></div>
    <div class="floating-orb orb-3"></div>
  </div>

  <div class="main-layout">
    <div class="game-panel">
      <header>
        <h1>Sémantix</h1>
        <p class="subtitle">Trouvez le mot secret grâce à la proximité sémantique !</p>
      </header>

      <div class="game-card">
        <div class="stats-bar">
          <div class="stat">
            <div class="stat-value" id="guess-count">0</div>
            <div class="stat-label">Essais</div>
          </div>
          <div class="stat">
            <div class="stat-value" id="best-temp">--</div>
            <div class="stat-label">Meilleure temp.</div>
          </div>
          <div class="stat">
            <div class="stat-value" id="best-rank">--</div>
            <div class="stat-label">Meilleur rang</div>
          </div>
        </div>

        <div class="input-wrapper">
          <input type="text" id="guess-input" placeholder="Entrez un mot français..." autocomplete="off" autofocus>
          <div class="autocomplete-hint" id="autocomplete-hint">
            <kbd>Tab</kbd> pour corriger
          </div>
          <div class="suggestion-box" id="suggestion-box"></div>
        </div>

        <div class="button-row">
          <button class="btn btn-primary" id="guess-btn">Deviner</button>
        </div>

        <div class="button-row">
          <button class="btn btn-secondary" id="new-game-btn">Nouvelle Partie</button>
          <button class="btn btn-secondary" id="hint-btn">Indice</button>
          <button class="btn btn-secondary" id="give-up-btn">Abandonner</button>
        </div>

        <div id="message-container" role="status" aria-live="polite"></div>

        <div class="legend">
          <div class="legend-item">
            <div class="legend-color" style="background: linear-gradient(135deg, #ef4444, #dc2626);"></div>
            <span class="legend-text">Brûlant (> 40°)</span>
          </div>
          <div class="legend-item">
            <div class="legend-color" style="background: linear-gradient(135deg, #f59e0b, #d97706);"></div>
            <span class="legend-text">Chaud (20° - 40°)</span>
          </div>
          <div class="legend-item">
            <div class="legend-color" style="background: linear-gradient(135deg, #3b82f6, #2563eb);"></div>
            <span class="legend-text">Tiède (0° - 20°)</span>
          </div>
          <div class="legend-item">
            <div class="legend-color" style="background: linear-gradient(135deg, #6366f1, #4f46e5);"></div>
            <span class="legend-text">Froid (< 0°)</span>
          </div>
        </div>

        <div class="word-count">Dictionnaire: 66 000+ mots français (${embeddingModelName})</div>

        <div class="info-section">
          <div class="info-toggle" id="info-toggle">
            <span>Comment jouer ?</span>
            <span id="info-arrow">▼</span>
          </div>
          <div class="info-content" id="info-content">
            <h3>Le but du jeu</h3>
            <p>
              Trouvez le mot secret en vous approchant <strong>contextuellement</strong>.
              Chaque mot reçoit une température indiquant sa proximité sémantique.
            </p>

            <h3>Autocorrection</h3>
            <p>
              Si votre mot n'existe pas, appuyez sur <strong>Tab</strong> pour voir des suggestions
              de mots similaires. Utilisez les flèches et Entrée pour sélectionner.
            </p>

            <h3>Indices de progression</h3>
            <p>
              Si votre mot est dans les 1000 plus proches, un rang de 1 à 1000‰ apparaît.
            </p>
          </div>
        </div>
      </div>
    </div>

    <div class="guesses-panel">
      <div class="guesses-card">
        <div class="guesses-header">
          <span class="guesses-title">Vos essais</span>
          <span class="guesses-count" id="guesses-count-badge">0</span>
        </div>
        <div class="guesses-container" id="guesses-container" aria-live="polite" aria-label="Liste de vos essais">
          <div class="empty-state">
            <div class="empty-state-icon">🎯</div>
            <p>Entrez un mot pour commencer !</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="victory-modal" id="victory-modal">
    <div class="victory-content">
      <div class="victory-emoji">🎉</div>
      <div class="victory-title">Félicitations !</div>
      <div class="victory-word" id="victory-word"></div>
      <div class="victory-stats" id="victory-stats"></div>
      <button class="btn btn-primary" id="play-again-btn">Rejouer</button>
      <button class="btn btn-share" id="share-btn">Partager</button>
      <div class="share-feedback" id="share-feedback" role="status" aria-live="polite"></div>
    </div>
  </div>

  <script>
    let sessionId = localStorage.getItem('sessionId') || crypto.randomUUID();
    localStorage.setItem('sessionId', sessionId);

    const guessInput = document.getElementById('guess-input');
    const guessBtn = document.getElementById('guess-btn');
    const newGameBtn = document.getElementById('new-game-btn');
    const hintBtn = document.getElementById('hint-btn');
    const giveUpBtn = document.getElementById('give-up-btn');
    const guessesContainer = document.getElementById('guesses-container');
    const messageContainer = document.getElementById('message-container');
    const guessCount = document.getElementById('guess-count');
    const guessesCountBadge = document.getElementById('guesses-count-badge');
    const bestTemp = document.getElementById('best-temp');
    const bestRank = document.getElementById('best-rank');
    const victoryModal = document.getElementById('victory-modal');
    const victoryWord = document.getElementById('victory-word');
    const victoryStats = document.getElementById('victory-stats');
    const playAgainBtn = document.getElementById('play-again-btn');
    const infoToggle = document.getElementById('info-toggle');
    const infoContent = document.getElementById('info-content');
    const infoArrow = document.getElementById('info-arrow');
    const autocompleteHint = document.getElementById('autocomplete-hint');
    const suggestionBox = document.getElementById('suggestion-box');

    let guesses = []; // Ordered by time of guess (for share)
    let bestTempValue = -Infinity;
    let bestRankValue = 0;
    let suggestions = [];
    let selectedSuggestionIndex = -1;
    let isSubmitting = false;

    // Share button elements
    const shareBtn = document.getElementById('share-btn');
    const shareFeedback = document.getElementById('share-feedback');

    // Epoch for day number calculation (Jan 1 2024)
    const SHARE_EPOCH = new Date('2024-01-01T00:00:00Z').getTime();

    function getShareDayNumber() {
      return Math.floor((Date.now() - SHARE_EPOCH) / 86400000);
    }

    function getShareSquare(temp) {
      if (temp >= 100) return '🔥';
      if (temp >= 80) return '🟥';
      if (temp >= 50) return '🟧';
      if (temp >= 20) return '🟨';
      if (temp >= -10) return '⬜';
      if (temp >= -50) return '🟪';
      return '🟦';
    }

    function buildShareText(attempts) {
      const dayNum = getShareDayNumber();
      const squares = guesses.map(g => getShareSquare(g.temperature)).join('');
      return 'Sémantix #' + dayNum + ' - Trouvé en ' + attempts + ' essai' + (attempts > 1 ? 's' : '') + '!' + String.fromCharCode(10) + squares;
    }

    async function shareResult(attempts) {
      const text = buildShareText(attempts);
      try {
        await navigator.clipboard.writeText(text);
        shareFeedback.textContent = 'Copié !';
        setTimeout(() => { shareFeedback.textContent = ''; }, 3000);
      } catch (err) {
        shareFeedback.textContent = 'Impossible de copier — copiez manuellement: ' + text;
      }
    }

    function showMessage(text, type = 'error') {
      messageContainer.innerHTML = '<div class="message ' + type + '">' + text + '</div>';
      setTimeout(() => messageContainer.innerHTML = '', 4000);
    }

    function getTemperatureClass(temp) {
      if (temp >= 100) return 'found';
      if (temp >= 40) return 'hot';
      if (temp >= 20) return 'warm';
      if (temp >= 0) return 'cool';
      return 'cold';
    }

    function getTemperatureColor(temp) {
      if (temp >= 100) return 'rgb(255, 60, 0)'; // Mot trouvé
      if (temp <= -20) {
        // Blue shades: -100 to -20
        const t = (temp + 100) / 80; // 0 to 1
        return 'rgb(' + Math.round(50 + t * 100) + ', ' + Math.round(50 + t * 150) + ', ' + Math.round(200 + t * 55) + ')';
      } else if (temp <= 20) {
        // Blue to white: -20 to 20
        const t = (temp + 20) / 40; // 0 to 1
        return 'rgb(' + Math.round(150 + t * 105) + ', ' + Math.round(200 + t * 55) + ', 255)';
      } else if (temp <= 60) {
        // White to orange: 20 to 60
        const t = (temp - 20) / 40; // 0 to 1
        return 'rgb(255, ' + Math.round(255 - t * 120) + ', ' + Math.round(255 - t * 200) + ')';
      } else {
        // Orange to red: 60 to 100
        const t = (temp - 60) / 40; // 0 to 1
        return 'rgb(' + Math.round(255 - t * 55) + ', ' + Math.round(135 - t * 105) + ', ' + Math.round(55 - t * 55) + ')';
      }
    }

    function updateStats() {
      guessCount.textContent = guesses.length;
      guessesCountBadge.textContent = guesses.length;
      bestTemp.textContent = bestTempValue > -Infinity ? bestTempValue.toFixed(1) + '°' : '--';
      bestRank.textContent = bestRankValue > 0 ? bestRankValue + '‰' : '--';
    }

    function renderGuesses() {
      if (guesses.length === 0) {
        guessesContainer.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🎯</div><p>Entrez un mot pour commencer !</p></div>';
        return;
      }

      const sortedGuesses = [...guesses].sort((a, b) => b.temperature - a.temperature);

      guessesContainer.innerHTML = sortedGuesses.map((guess) => {
        const tempClass = getTemperatureClass(guess.temperature);
        const tempColor = getTemperatureColor(guess.temperature);
        const rankText = guess.rank ? guess.rank + '‰' : '--';
        const ariaLabel = "Votre mot '" + guess.word + "' a une température de " + guess.temperature.toFixed(1) + " degrés" + (guess.rank ? ', rang ' + guess.rank + ' sur 1000' : '');

        return '<div class="guess-row ' + tempClass + '" aria-label="' + ariaLabel + '">' +
          '<div class="guess-word">' + guess.word + '</div>' +
          '<div class="guess-temp" style="color: ' + tempColor + '">' + guess.temperature.toFixed(1) + '°</div>' +
          '<div class="guess-rank ' + (guess.rank ? 'has-rank' : '') + '">' +
            rankText +
          '</div>' +
        '</div>';
      }).join('');
    }

    function createConfetti() {
      const colors = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#6366f1', '#ec4899'];
      for (let i = 0; i < 50; i++) {
        setTimeout(() => {
          const confetti = document.createElement('div');
          confetti.className = 'confetti';
          confetti.style.left = Math.random() * 100 + 'vw';
          confetti.style.background = colors[Math.floor(Math.random() * colors.length)];
          confetti.style.width = (Math.random() * 10 + 5) + 'px';
          confetti.style.height = (Math.random() * 10 + 5) + 'px';
          confetti.style.borderRadius = Math.random() > 0.5 ? '50%' : '0';
          document.body.appendChild(confetti);
          setTimeout(() => confetti.remove(), 3000);
        }, i * 30);
      }
    }

    function showVictory(word, attempts) {
      victoryWord.textContent = word;
      victoryStats.textContent = 'Trouvé en ' + attempts + ' essai' + (attempts > 1 ? 's' : '') + ' !';
      victoryModal.classList.add('active');
      createConfetti();
      // Wire share button to this game's attempt count
      shareBtn.onclick = () => shareResult(attempts);
      shareFeedback.textContent = '';
    }

    function hideSuggestions() {
      suggestionBox.classList.remove('visible');
      autocompleteHint.classList.remove('visible');
      suggestions = [];
      selectedSuggestionIndex = -1;
    }

    function showSuggestions(words) {
      suggestions = words;
      selectedSuggestionIndex = 0;

      suggestionBox.innerHTML = words.map((word, index) =>
        '<div class="suggestion-item ' + (index === 0 ? 'selected' : '') + '" data-index="' + index + '">' +
          '<span class="suggestion-word">' + word + '</span>' +
        '</div>'
      ).join('');

      suggestionBox.classList.add('visible');
    }

    function updateSuggestionSelection() {
      const items = suggestionBox.querySelectorAll('.suggestion-item');
      items.forEach((item, index) => {
        item.classList.toggle('selected', index === selectedSuggestionIndex);
      });
    }

    function selectSuggestion(index) {
      if (suggestions[index]) {
        guessInput.value = suggestions[index];
        hideSuggestions();
        guessInput.focus();
      }
    }

    async function fetchSuggestions(word) {
      try {
        const response = await fetch('/api/autocomplete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ word })
        });
        const data = await response.json();
        if (data.suggestions && data.suggestions.length > 0) {
          showSuggestions(data.suggestions);
        }
      } catch (error) {
        console.error('Autocomplete error:', error);
      }
    }

    async function makeGuess() {
      const word = guessInput.value.trim().toLowerCase();
      if (!word) return;

      // Protection anti-double-submit
      if (isSubmitting) return;
      isSubmitting = true;

      hideSuggestions();

      // Vérification des doublons (mot brut ET lemmes déjà soumis)
      if (guesses.some(g => g.word === word || g.word === word.toLowerCase())) {
        showMessage('Vous avez déjà essayé ce mot !');
        isSubmitting = false;
        return;
      }

      try {
        const response = await fetch('/api/guess', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId, word })
        });

        const data = await response.json();

        if (data.error) {
          // Show autocomplete hint if word not found
          if (data.error.includes('dictionnaire')) {
            autocompleteHint.classList.add('visible');
          }
          showMessage(data.error);
          isSubmitting = false;
          return;
        }

        // Vérifier si le lemme retourné existe déjà
        if (guesses.some(g => g.word === data.word)) {
          showMessage('Vous avez déjà essayé ce mot (lemme: ' + data.word + ') !');
          isSubmitting = false;
          return;
        }

        guesses.push({
          word: data.word,
          temperature: data.temperature,
          rank: data.rank
        });

        if (data.temperature > bestTempValue) {
          bestTempValue = data.temperature;
        }

        if (data.rank && data.rank > bestRankValue) {
          bestRankValue = data.rank;
        }

        updateStats();
        renderGuesses();

        guessInput.value = '';
        guessInput.focus();

        if (data.solved) {
          showVictory(data.word, guesses.length);
        }

      } catch (error) {
        showMessage('Erreur de connexion au serveur');
      } finally {
        isSubmitting = false;
      }
    }

    async function startNewGame() {
      try {
        const response = await fetch('/api/new-game', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId })
        });

        const data = await response.json();

        if (data.success) {
          guesses = [];
          bestTempValue = -Infinity;
          bestRankValue = 0;
          updateStats();
          renderGuesses();
          victoryModal.classList.remove('active');
          hideSuggestions();
          guessInput.value = '';
          guessInput.focus();
          if (shareFeedback) shareFeedback.textContent = '';
          showMessage('Nouvelle partie ! Trouvez le mot secret.', 'success');
        }
      } catch (error) {
        showMessage('Erreur de connexion au serveur');
      }
    }

    async function getHint() {
      try {
        const response = await fetch('/api/hint', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId })
        });

        const data = await response.json();

        if (data.hint) {
          showMessage('Indice: ' + data.hint, 'success');
        } else if (data.error) {
          showMessage(data.error);
        }
      } catch (error) {
        showMessage('Erreur de connexion au serveur');
      }
    }

    async function giveUp() {
      try {
        const response = await fetch('/api/give-up', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId })
        });

        const data = await response.json();

        if (data.secretWord) {
          showMessage('Le mot secret était: ' + data.secretWord.toUpperCase(), 'error');
        } else if (data.error) {
          showMessage(data.error);
        }
      } catch (error) {
        showMessage('Erreur de connexion au serveur');
      }
    }

    // Event listeners
    guessBtn.addEventListener('click', makeGuess);

    guessInput.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const word = guessInput.value.trim();
        if (word.length >= 2) {
          fetchSuggestions(word);
        }
      } else if (e.key === 'ArrowDown' && suggestions.length > 0) {
        e.preventDefault();
        selectedSuggestionIndex = Math.min(selectedSuggestionIndex + 1, suggestions.length - 1);
        updateSuggestionSelection();
      } else if (e.key === 'ArrowUp' && suggestions.length > 0) {
        e.preventDefault();
        selectedSuggestionIndex = Math.max(selectedSuggestionIndex - 1, 0);
        updateSuggestionSelection();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (suggestions.length > 0 && selectedSuggestionIndex >= 0) {
          selectSuggestion(selectedSuggestionIndex);
        } else {
          makeGuess();
        }
      } else if (e.key === 'Escape') {
        hideSuggestions();
      }
    });

    guessInput.addEventListener('input', () => {
      hideSuggestions();
    });

    suggestionBox.addEventListener('click', (e) => {
      const item = e.target.closest('.suggestion-item');
      if (item) {
        const index = parseInt(item.dataset.index);
        selectSuggestion(index);
      }
    });

    newGameBtn.addEventListener('click', startNewGame);
    hintBtn.addEventListener('click', getHint);
    giveUpBtn.addEventListener('click', giveUp);
    playAgainBtn.addEventListener('click', startNewGame);

    infoToggle.addEventListener('click', () => {
      infoContent.classList.toggle('active');
      infoArrow.textContent = infoContent.classList.contains('active') ? '▲' : '▼';
    });

    // Close suggestions when clicking outside
    document.addEventListener('click', (e) => {
      if (!guessInput.contains(e.target) && !suggestionBox.contains(e.target)) {
        hideSuggestions();
      }
    });

    // Initialize game on load
    startNewGame();
  </script>
</body>
</html>
`;

// Main server
const server = serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);

    // Serve static HTML
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(htmlTemplate, {
        headers: { "Content-Type": "text/html; charset=utf-8" }
      });
    }

    // API: Autocomplete/correction
    if (url.pathname === "/api/autocomplete" && req.method === "POST") {
      const body = await req.json();
      const word = body.word?.toLowerCase().trim();

      if (!word || word.length < 2) {
        return Response.json({ suggestions: [] });
      }

      const suggestions = findClosestWords(word, 5);
      return Response.json({ suggestions });
    }

    // API: New game
    if (url.pathname === "/api/new-game" && req.method === "POST") {
      const body = await req.json();
      const sessionId = body.sessionId;

      const secretWord = selectSecretWord();
      // Only warm up on-the-fly similarity cache if precomputed rankings are not available
      if (!precomputedRankings.has(secretWord)) {
        getOrCreateSimilarityCache(secretWord);
      }

      const session: GameSession = {
        secretWord,
        guesses: [],
        startTime: Date.now(),
        solved: false
      };

      gameSessions.set(sessionId, session);

      return Response.json({ success: true });
    }

    // API: Make a guess
    if (url.pathname === "/api/guess" && req.method === "POST") {
      const body = await req.json();
      const sessionId = body.sessionId;
      const rawWord = body.word?.toLowerCase().trim();

      let session = gameSessions.get(sessionId);

      if (!session) {
        const secretWord = selectSecretWord();
        // Only warm up on-the-fly similarity cache if precomputed rankings are not available
        if (!precomputedRankings.has(secretWord)) {
          getOrCreateSimilarityCache(secretWord);
        }

        session = {
          secretWord,
          guesses: [],
          startTime: Date.now(),
          solved: false
        };
        gameSessions.set(sessionId, session);
      }

      if (session.solved) {
        return Response.json({ error: "Partie terminée ! Commencez une nouvelle partie." });
      }

      if (!rawWord) {
        return Response.json({ error: "Veuillez entrer un mot." });
      }

      // Lemmatiser le mot pour correspondre au modèle Fauconnier
      const lemma = await lemmatize(rawWord);

      // Chercher d'abord le lemme, puis le mot brut, puis avec normalisation d'accents
      let word = lemma;
      let wasAutoCorrected = false;
      if (!wordVectors.has(lemma)) {
        if (wordVectors.has(rawWord)) {
          word = rawWord;
        } else {
          // Essayer avec la version normalisée (sans accents)
          const normalizedInput = normalizeAccents(rawWord);
          const originalWord = normalizedToOriginal.get(normalizedInput);
          if (originalWord && wordVectors.has(originalWord)) {
            word = originalWord;
          } else {
            // Essayer l'autocorrection pour les fautes d'orthographe proches (distance <= 2)
            const closestWords = findClosestWords(rawWord, 1);
            if (closestWords.length > 0) {
              const closest = closestWords[0];
              const distance = levenshteinDistance(rawWord.toLowerCase(), closest);
              if (distance <= 2) {
                word = closest;
                wasAutoCorrected = true;
              } else {
                return Response.json({
                  error: `Le mot "${rawWord}" n'est pas dans le dictionnaire. Appuyez sur Tab pour des suggestions.`,
                  lemma: lemma !== rawWord ? lemma : undefined
                });
              }
            } else {
              return Response.json({
                error: `Le mot "${rawWord}" n'est pas dans le dictionnaire. Appuyez sur Tab pour des suggestions.`,
                lemma: lemma !== rawWord ? lemma : undefined
              });
            }
          }
        }
      }

      const word2vecTemp = calculateSimilarity(session.secretWord, word);
      let temperature = word2vecTemp;

      // Ajustement LLM : vérification synchrone du cache d'abord (< 1ms)
      const llmCacheKey = `${session.secretWord}:${word}`;
      const cachedAdjustment = llmAdjustmentCache.get(llmCacheKey);

      if (cachedAdjustment !== undefined) {
        // Cache hit — appliquer l'ajustement immédiatement
        temperature = Math.max(-100, Math.min(100, word2vecTemp + cachedAdjustment));
        console.log(`[LLM] Cache hit ${word}: ${temperature}° (W2V: ${word2vecTemp}°, ajustement: ${cachedAdjustment > 0 ? "+" : ""}${cachedAdjustment}°)`);
      } else if (LLM_ENABLED && word !== session.secretWord) {
        // Cache miss — lancer l'appel LLM en arrière-plan (non-bloquant)
        // Préparer l'historique des essais précédents pour le contexte LLM
        const previousGuesses = session.guesses.map(g => ({
          word: g.word,
          temperature: g.temperature
        }));

        getLLMAdjustment(session.secretWord, word, word2vecTemp, previousGuesses)
          .then(adj => {
            console.log(`[LLM] Background adjustment cached for ${word}: ${adj > 0 ? "+" : ""}${adj}°`);
          })
          .catch(err => {
            console.warn(`[LLM] Background call failed for ${word}:`, err.message);
          });
        // Retourner le score Word2Vec pur immédiatement (l'ajustement LLM sera appliqué à la prochaine occurrence)
      }

      const rank = getProximityRank(session.secretWord, word);
      const solved = word === session.secretWord;

      session.guesses.push({
        word,
        temperature,
        rank,
        timestamp: Date.now()
      });

      if (solved) {
        session.solved = true;
      }

      return Response.json({
        word,
        temperature,
        rank,
        solved,
        guessNumber: session.guesses.length
      });
    }

    // API: Get hint
    if (url.pathname === "/api/hint" && req.method === "POST") {
      const body = await req.json();
      const sessionId = body.sessionId;

      const session = gameSessions.get(sessionId);

      if (!session) {
        return Response.json({ error: "Aucune partie en cours." });
      }

      if (session.solved) {
        return Response.json({ error: "Partie déjà terminée !" });
      }

      const secretWord = session.secretWord;
      const cache = getOrCreateSimilarityCache(secretWord);

      const closeWords = cache.rankings.slice(0, 50);
      const hintWord = closeWords[Math.floor(Math.random() * closeWords.length)];

      const hints = [
        `Le mot contient ${secretWord.length} lettres.`,
        `Le mot commence par "${secretWord[0].toUpperCase()}".`,
        `Le mot finit par "${secretWord[secretWord.length - 1]}".`,
        `Un mot proche: "${hintWord.word}" (${(hintWord.similarity * 100).toFixed(1)}°)`,
      ];

      const hint = hints[Math.floor(Math.random() * hints.length)];

      return Response.json({ hint });
    }

    // API: Give up
    if (url.pathname === "/api/give-up" && req.method === "POST") {
      const body = await req.json();
      const sessionId = body.sessionId;

      const session = gameSessions.get(sessionId);

      if (!session) {
        return Response.json({ error: "Aucune partie en cours." });
      }

      session.solved = true;
      return Response.json({ secretWord: session.secretWord });
    }

    // 404 for other routes
    return new Response("Not Found", { status: 404 });
  }
});

const modelType = embeddingModelName;
const lemmaStatus = lemmatizerAvailable ? "Actif (spaCy)" : "Inactif";
const llmStatus = LLM_ENABLED ? "Actif (MiniMax M2.5)" : "Inactif (pas de clé)";

console.log(`
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🎮  SÉMANTIX - Le jeu de mots sémantique                  ║
║                                                              ║
║   Serveur démarré sur: http://localhost:${server.port}               ║
║                                                              ║
║   Modèle: ${modelType.padEnd(20)} (${vectorDimension} dims)      ║
║   Dictionnaire: ${String(wordVectors.size).padEnd(6)} mots                          ║
║   Lemmatisation: ${lemmaStatus.padEnd(20)}                   ║
║   Correction LLM: ${llmStatus.padEnd(20)}                   ║
║                                                              ║
║   Ouvrez votre navigateur et commencez à jouer !            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
`);
