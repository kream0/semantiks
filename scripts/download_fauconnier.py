"""
Script de téléchargement et conversion du modèle Word2Vec de Fauconnier
Ce modèle est celui utilisé par le vrai Cémantix
"""
import os
import sys

print("="*60)
print("Téléchargement du modèle Word2Vec de Jean-Philippe Fauconnier")
print("="*60)

# Vérifier que gensim est installé
try:
    from gensim.models import KeyedVectors
    print("[OK] gensim installé")
except ImportError:
    print("[ERREUR] gensim non installé!")
    print("Exécutez: pip install gensim")
    sys.exit(1)

# Vérifier que huggingface_hub est installé
try:
    from huggingface_hub import hf_hub_download
    print("[OK] huggingface_hub installé")
except ImportError:
    print("[ERREUR] huggingface_hub non installé!")
    print("Exécutez: pip install huggingface_hub")
    sys.exit(1)

# Créer le dossier data s'il n'existe pas
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_dir, exist_ok=True)

output_file = os.path.join(data_dir, "fauconnier_500.vec")

# Vérifier si le fichier existe déjà
if os.path.exists(output_file):
    print(f"\n[INFO] Le fichier {output_file} existe déjà.")
    response = input("Voulez-vous le re-télécharger? (o/N): ")
    if response.lower() != 'o':
        print("Téléchargement annulé.")
        sys.exit(0)

print("\n[1/3] Téléchargement depuis Hugging Face...")
print("      Repo: Word2vec/fauconnier_frWiki_no_phrase_no_postag_500_cbow_cut10")
print("      (Version cut10 avec ~300k mots - vocabulaire plus riche)")
print("      Cela peut prendre plusieurs minutes...")

try:
    model_path = hf_hub_download(
        repo_id="Word2vec/fauconnier_frWiki_no_phrase_no_postag_500_cbow_cut10",
        filename="frWiki_no_phrase_no_postag_500_cbow_cut10.bin",
        cache_dir=os.path.join(data_dir, ".cache")
    )
    print(f"[OK] Téléchargé: {model_path}")
except Exception as e:
    print(f"[ERREUR] Échec du téléchargement: {e}")
    sys.exit(1)

print("\n[2/3] Chargement du modèle binaire Word2Vec...")
try:
    model = KeyedVectors.load_word2vec_format(
        model_path,
        binary=True,
        unicode_errors='ignore'
    )
    print(f"[OK] Modèle chargé: {len(model)} mots, {model.vector_size} dimensions")
except Exception as e:
    print(f"[ERREUR] Échec du chargement: {e}")
    sys.exit(1)

print("\n[3/3] Conversion en format texte...")
try:
    model.save_word2vec_format(output_file, binary=False)

    # Afficher la taille du fichier
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[OK] Fichier créé: {output_file}")
    print(f"     Taille: {file_size:.1f} MB")
except Exception as e:
    print(f"[ERREUR] Échec de la conversion: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Téléchargement terminé avec succès!")
print("="*60)
print("\nProchaines étapes:")
print("1. Exécutez le script de conversion TypeScript:")
print("   bun run scripts/convert-vectors.ts")
print("2. Démarrez le service de lemmatisation:")
print("   python scripts/lemmatizer_service.py")
print("3. Démarrez le serveur principal:")
print("   bun run src/server.ts")
