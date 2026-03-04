"""
Service de lemmatisation pour Sémantix
Utilise spaCy pour convertir les mots français en leurs lemmes
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy

app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin depuis le serveur Bun

# Charger le modèle français de spaCy
print("Chargement du modèle spaCy fr_core_news_sm...")
try:
    nlp = spacy.load("fr_core_news_sm")
    print("Modèle chargé avec succès!")
except OSError:
    print("ERREUR: Modèle spaCy non trouvé!")
    print("Installez-le avec: python -m spacy download fr_core_news_sm")
    exit(1)


@app.route("/lemmatize", methods=["POST"])
def lemmatize():
    """Lemmatise un mot français"""
    data = request.get_json()
    word = data.get("word", "").lower().strip()

    if not word:
        return jsonify({"lemma": "", "error": "Mot vide"})

    # Traiter le mot avec spaCy
    doc = nlp(word)

    if len(doc) == 0:
        return jsonify({"lemma": word})

    # Prendre le lemme du premier token
    lemma = doc[0].lemma_.lower()

    return jsonify({
        "lemma": lemma,
        "original": word,
        "pos": doc[0].pos_  # Part of speech (VERB, NOUN, etc.)
    })


@app.route("/lemmatize_batch", methods=["POST"])
def lemmatize_batch():
    """Lemmatise plusieurs mots en une seule requête"""
    data = request.get_json()
    words = data.get("words", [])

    results = []
    for word in words:
        word = word.lower().strip()
        if word:
            doc = nlp(word)
            lemma = doc[0].lemma_.lower() if len(doc) > 0 else word
            results.append({"original": word, "lemma": lemma})

    return jsonify({"results": results})


@app.route("/health", methods=["GET"])
def health():
    """Endpoint de vérification de santé"""
    return jsonify({"status": "ok", "model": "fr_core_news_sm"})


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Service de Lemmatisation Sémantix")
    print("="*50)
    print(f"Démarrage sur http://localhost:3001")
    print("="*50 + "\n")

    app.run(host="0.0.0.0", port=3001, debug=False)
