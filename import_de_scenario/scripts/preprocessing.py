import json
import os
import re

def load_json(input_file):
    """Charge un fichier JSON"""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON: {e}")
        return {}

def normaliser_noms(text):
    replacements = [
        (r"\bPacotte\b", "Pacott"),
        (r"\bAssous\b", "Assoult"),
        (r"\bGobert\b", "Gaubert"),
        (r"\b(Gallant|Galon|Galland|Galand)\b", "Gallant"),
        (r"\bLefranc\b", "LeFranc"),
        (r"\bbrigadier\b", "Brigadier")
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    return text


def normaliser_phrases(replique):
    if not isinstance(replique, str):
        replique = str(replique) if repliques is not None else ""
    replique_clean = re.sub(r"\s+", " ", replique)
    return replique_clean

def words_concatenation(dictionary: dict):
    repliques_finales = {}
    final_id = 0
    current_words = []
    current_speaker = None

    for key in dictionary.keys():
        element = dictionary[key]
        content = element["content"]
        speaker_id = element["speaker_id"]
        if current_speaker is None:
            current_speaker = speaker_id
        current_words.append(content)
        if re.search(r'[.!?]$', content.strip()):  #split en phrases
            phrase = " ".join(current_words).strip()
            phrase = normaliser_phrases(phrase)
            final_id += 1
            repliques_finales[final_id] = {
                "content": phrase,
                "speaker_id": current_speaker,
            }
            current_words = []
            current_speaker = None

    return repliques_finales


if __name__ == "__main__" :
    ##### CHARGEMENT DU FICHIER JSON
    transcription = "../data/raw/USGS_1804_daia.json"  # à modifier le chemin si besoin
    corpus_transcript = load_json(transcription)

    ##### CONVERSION EN DICT : Dict qui va contenir tous les mots : id + valeur
    words_dict = {
        i: dict(element)
        for i, element in enumerate(corpus_transcript.get("words", []))
    }

    ## FORMATION des répliques
    repliques = words_concatenation(words_dict)

    ## IMPLEMENTATION du champ "repliques"
    corpus_transcript["repliques"] = repliques


    ## Sauvegarde de la sortie
    input_path = transcription
    filename = os.path.basename(input_path)
    output_path = "../data/clean/"

    output_path = "../data/clean/transcription_USGS_clean.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus_transcript, f, indent=4, ensure_ascii=False)
    print(f"Fichier sauvegardé : {output_path}")
