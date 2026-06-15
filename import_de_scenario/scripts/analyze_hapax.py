import json
from collections import Counter
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def load_json(input_file):
    """Charge un fichier JSON"""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON: {e}")
        return {}

def tokenize(text):
    """Tokenise un texte en mots (minuscules, sans ponctuation)"""
    text = text.lower()
    tokens = re.findall(r'\b[a-zàâäéèêëîïôùûüçœæ]+\b', text)
    return tokens

def build_dict(data):
    """Construit le dictionnaire de fréquences à partir des répliques"""
    all_words = []
    for replique in data:
        # Adapter la clé selon la structure du JSON
        text = replique.get("line")
        if text:
            all_words.extend(tokenize(text))
    return dict(Counter(all_words))

###################################
#Chargement des 2 fichiers

file_gold = "../transcription_gold.json"
corpus = load_json(file_gold)
corpus_gold = corpus["repliques"]

freq_words_gold = build_dict(corpus_gold)

# Garder les mots uniques (hapax)
hapax_gold = {word: count for word, count in freq_words_gold.items() if count == 1}

with open("../hapax_gold_list.txt", "w") as file :
    for k, v in sorted(hapax_gold.items()) :
        file.write(f"{k}\n")

#--------------------------------------
# Même chose pour le fichier de transcription

file_transcript = "../data/clean/transcription_USGS_clean.json"
corpus = load_json(file_transcript)
corpus_transcript = corpus["repliques"].values()

freq_words_transcript = build_dict(corpus_transcript)

# Garder les mots uniques (hapax)
hapax_transcript = {word: count for word, count in freq_words_transcript.items() if count == 1}

with open("../hapax_transcript_list.txt", "w") as file :
    for k, v in sorted(hapax_transcript.items()) :
        file.write(f"{k}\n")

######################################
# Comparaison des 2 dicos --> mots en commun / différences

dico_gold = "../hapax_gold_list.txt"
dico_transcript = "../hapax_transcript_list.txt"

with open(dico_gold, "r") as file_gold:
    gold_hapax_dico = set(word.strip() for word in file_gold.readlines())

with open(dico_transcript, "r") as file_transcript:
    transcript_hapax_dico = set(word.strip() for word in file_transcript.readlines())

common_words = sorted(list(gold_hapax_dico & transcript_hapax_dico)) #rangé par ordre alphabétique
# print(common_words)
# print(len(common_words))  #260 avec doublons

########################################
##### ALIGNEMENT
table_transcript = []
table_gold = []
for word in common_words:
    matches_gold = [r for r in corpus_gold if word in tokenize(r.get("line", ""))]
    matches_transcript = [r for r in corpus_transcript if word in tokenize(r.get("line", ""))]

    if matches_transcript :
        for repl_transcript in matches_transcript:
                table_transcript.append({
                    "word": word,
                    "sent_transcript": repl_transcript.get("line", ""),
                })

    if matches_gold:
        for repl_gold in matches_gold:
            table_gold.append({
                    "word": word,
                    "sent_gold": repl_gold.get("line", ""),
                    "speaker_gold": repl_gold.get("speaker", {})
                })


df = pd.DataFrame(table_transcript, columns=["word", "sent_transcript"])
df.to_csv("common_words_transcript.csv", index=False, encoding="utf-8")
df = pd.DataFrame(table_gold, columns=["word", "sent_gold", "speaker_gold"])
df.to_csv("common_words_gold.csv", index=False, encoding="utf-8")

df_transcript = pd.read_csv("common_words_transcript.csv", encoding="utf-8")
df_gold = pd.read_csv("common_words_gold.csv", encoding="utf-8")
df_merged = pd.merge(df_gold, df_transcript, on="word", how="inner")
df_merged = df_merged[["word", "sent_gold", "speaker_gold", "sent_transcript"]]

df_merged.to_csv("common_words_merged.csv", index=False, encoding="utf-8")

model = SentenceTransformer("../model/", trust_remote_code=True)

def similarity_score(a, b):
    if a == b:
        return 1.0
    embeddings_a = model.encode([a])
    embeddings_b = model.encode([b])
    score = model.similarity(embeddings_a, embeddings_b)
    return float(score[0][0])

df_merged["similarity"] = df_merged.apply(
    lambda row: similarity_score(row["sent_transcript"], row["sent_gold"]), axis=1
)

# Suppression des phrases ayant moins de 0.4 similarité
df_merged = df_merged[df_merged["similarity"] >= 0.4]
initial_count = len(df_merged)
print(f"Nombre de lignes restantes : {initial_count}")

##### Suppression des phrases doublons
df_merged = df_merged.drop_duplicates(subset=["sent_transcript"], keep="first")
print(f"Nombre de lignes après suppression des doublons : {len(df_merged)}")
print(f"Doublons supprimés : {initial_count - len(df_merged)}")

df_merged.to_csv("common_words_merged.csv", index=False, encoding="utf-8")  #Sauvegarde

########################################
##### On retrouve les phrases du CSV dans le fichier transcription pour appliquer la correction
# Chargement du JSON
input_file = "../transcription_daia_corrige.json"
with open(input_file, "r", encoding="utf-8") as f:
    transcription_data = json.load(f)
transcription_data_rep = transcription_data.get("repliques", {})

# Chargement du CSV
alignements = pd.read_csv("common_words_merged.csv", encoding="utf-8")
sent_transcript = alignements["sent_transcript"].tolist()
speaker_gold = alignements["speaker_gold"].tolist()

# Condition de CORRECTION :
modifications = []
for key in transcription_data_rep:
    spk = transcription_data_rep[key]
    line = spk.get("line", "")
    if line in sent_transcript:
    # Si "line" est = à sent_transcript, alors on applique la correction du speaker
        idx = sent_transcript.index(line)
        nouveau_speaker = speaker_gold[idx]
        ancien_speaker = spk.get("speaker_id", "")
        if ancien_speaker == nouveau_speaker:  # Pas de modification si identiques
            continue
        modifications.append({
            "key": key,
            "ancien_speaker": ancien_speaker,
            "nouveau_speaker": nouveau_speaker,
            "line": line
        })
        if isinstance(spk, dict):
            transcription_data_rep[key]["speaker_id"] = nouveau_speaker

# Sauvegarde JSON
transcription_data["repliques"] = transcription_data_rep
with open("../transcription_daia_corrige2.json", "w", encoding="utf-8") as f:
    json.dump(transcription_data, f, ensure_ascii=False, indent=2)


#####################################
##### SAUVEGARDE DES MODIFICATIONS
with open("../results_occurences.txt", "w", encoding="utf-8") as f:
    f.write(f"{'='*60}\n")
    f.write(f"CORRECTIONS DES SPEAKERS\n")
    f.write(f"Nombre de modifications : {len(modifications)}\n")
    f.write(f"{'='*60}\n\n")
    for mod in modifications:
        f.write(f"[Réplique {mod['key']}]\n")
        f.write(f"  Speaker : {mod['ancien_speaker']} --> {mod['nouveau_speaker']}\n")
        f.write(f"   Line   : {mod['line']}\n")
        f.write(f"{'-'*40}\n")

print(f"{len(modifications)} modifications enregistrées (dans results_occurences.txt)")
