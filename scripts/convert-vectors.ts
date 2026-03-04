import { readFileSync, writeFileSync, existsSync } from "fs";

// Configuration - Modèle Fauconnier avec 500 dimensions (cut10 pour vocabulaire étendu)
const VECTOR_DIMENSIONS = 500;  // Modèle cut10 utilise 500 dimensions
const INPUT_FILE = "data/fauconnier_500.vec";  // Fichier téléchargé depuis HuggingFace (cut10, 500D)
const OUTPUT_TSV = "data/french_embeddings.tsv";
const OUTPUT_JSON = "data/french_words.json";
const MAX_WORDS = 150000;  // Limiter aux 150k mots les plus fréquents (cut50 a ~100k mots)

console.log("="
.repeat(60));
console.log("Conversion des embeddings Word2Vec Fauconnier");
console.log("="
.repeat(60));

// Vérifier que le fichier source existe
if (!existsSync(INPUT_FILE)) {
  console.error(`\nERREUR: Fichier ${INPUT_FILE} non trouvé!`);
  console.error("Exécutez d'abord: python scripts/download_fauconnier.py");
  process.exit(1);
}

console.log(`\nLecture de ${INPUT_FILE}...`);

// Read the vector file
const vectorFile = readFileSync(INPUT_FILE, "utf-8");
const lines = vectorFile.trim().split("\n");

// La première ligne contient souvent les métadonnées (nombre de mots, dimensions)
let startIndex = 0;
const firstLineParts = lines[0].split(" ");
if (firstLineParts.length === 2 && !isNaN(Number(firstLineParts[0]))) {
  console.log(`Métadonnées: ${firstLineParts[0]} mots, ${firstLineParts[1]} dimensions`);
  startIndex = 1;
}

// French word pattern: only letters (including accented), no numbers, no special chars
// Le modèle Fauconnier utilise des lemmes, donc tous les mots sont en forme canonique
const frenchWordPattern = /^[a-zA-ZàâäéèêëïîôùûüÿœæçÀÂÄÉÈÊËÏÎÔÙÛÜŸŒÆÇ\-']+$/;

const vectors: Record<string, number[]> = {};
const validWords: string[] = [];

let processed = 0;
let accepted = 0;
let skippedDimension = 0;
let skippedPattern = 0;

console.log("\nTraitement des vecteurs...");

for (let i = startIndex; i < lines.length && accepted < MAX_WORDS; i++) {
  processed++;
  const parts = lines[i].split(" ");
  const word = parts[0].toLowerCase();

  // Filter criteria:
  // 1. Must match French word pattern (letters only, avec tirets et apostrophes pour lemmes composés)
  // 2. Must be at least 2 characters
  // 3. Must be at most 25 characters (lemmes peuvent être plus longs)
  // 4. No duplicates
  if (!frenchWordPattern.test(word)) {
    skippedPattern++;
    continue;
  }

  if (word.length < 2 || word.length > 25 || vectors[word]) {
    continue;
  }

  const vector = parts.slice(1).map(Number);

  // Accepter les dimensions 500 (cut10), 1000 (cut200) ou 300 (FastText) pour rétrocompatibilité
  if ((vector.length === VECTOR_DIMENSIONS || vector.length === 1000 || vector.length === 300) && !vector.some(isNaN)) {
    vectors[word] = vector;
    validWords.push(word);
    accepted++;
  } else {
    skippedDimension++;
  }

  if (processed % 50000 === 0) {
    console.log(`  Traité: ${processed} lignes, accepté: ${accepted} mots`);
  }
}

console.log(`\n${"=".repeat(40)}`);
console.log(`Total traité: ${processed}`);
console.log(`Total accepté: ${accepted}`);
console.log(`Ignorés (pattern): ${skippedPattern}`);
console.log(`Ignorés (dimensions): ${skippedDimension}`);
console.log(`${"=".repeat(40)}`);

console.log(`\nExemples de mots:`, validWords.slice(0, 30).join(", "));

// Déterminer la dimension réelle des vecteurs
const actualDimension = validWords.length > 0 ? vectors[validWords[0]].length : VECTOR_DIMENSIONS;
console.log(`\nDimension des vecteurs: ${actualDimension}`);

// Save as TSV format for loading by server
// Format: word\tvector values separated by comma
console.log(`\nÉcriture de ${OUTPUT_TSV}...`);
const outputLines: string[] = [];
for (const word of validWords) {
  // Reduce precision to save space (4 decimal places)
  const reducedVector = vectors[word].map(v => v.toFixed(4));
  outputLines.push(`${word}\t${reducedVector.join(",")}`);
}

writeFileSync(OUTPUT_TSV, outputLines.join("\n"));
const tsvSize = (Buffer.byteLength(outputLines.join("\n")) / (1024 * 1024)).toFixed(1);
console.log(`[OK] ${OUTPUT_TSV} créé (${tsvSize} MB)`);

// Also save just the word list for quick lookup
writeFileSync(OUTPUT_JSON, JSON.stringify(validWords));
const jsonSize = (Buffer.byteLength(JSON.stringify(validWords)) / 1024).toFixed(1);
console.log(`[OK] ${OUTPUT_JSON} créé (${jsonSize} KB)`);

console.log(`\n${"=".repeat(60)}`);
console.log("Conversion terminée avec succès!");
console.log(`${"=".repeat(60)}`);
