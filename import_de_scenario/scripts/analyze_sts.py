from sentence_transformers import SentenceTransformer, util
from collections import Counter
import torch
import json
import os
import re

### CONFIGURATION
SCRIPT_DIR         = "/home/rachou/Documents/stage/USGS 1804"
MODEL_SAVE_DIR_STD = os.path.join(SCRIPT_DIR, "model")
BASE_MODEL         = "sentence-transformers/static-similarity-mrl-multilingual-v1"
SCORE_THRESHOLD    = 0.65

def moy_phrases_par_scene(repliques):
    "Fonction qui calcule la moyenne du nb de répliques par scène."

    # Gère les deux cas : liste ou dict
    items = repliques if isinstance(repliques, list) else repliques.values()

    scenes = Counter(rep["scene_index"] for rep in items)

    if not scenes:
        return 0.0

    return sum(scenes.values()) / len(scenes)

def load_json(filename):
    filepath = os.path.join(SCRIPT_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Fichier introuvable : {filepath}")
        return None

def normaliser_phrase(sent):
    """Nettoie et normalise une phrase"""
    sent = re.sub(r'  ', ' ', sent) # Remplacer tous les types d'espaces par un espace classique
    sent = re.sub(r'\s+', ' ', sent) # Supprimer les espaces multiples
    sent = re.sub(r'([.!?,;:])([^\s])', r'\1 \2', sent) # Espace après la ponctuation si manquant
    return sent

# SAUVEGARDE DU MODÈLE DE BASE (sans entraînement)
def save_model(base_model_name, save_dir):
    print(f"\n{'='*60}")
    print("SAUVEGARDE du modèle de base (sans entraînement)")
    print(f"{'='*60}")
    model = SentenceTransformer(base_model_name, trust_remote_code=True)
    model.save(save_dir)
    print(f"Modèle de base sauvegardé dans : {save_dir}")
    return model


### ALIGNEMENT

def run_alignment(model, gold_embeddings, gold_lines, gold_speakers,transcript_keys, transcript_lines, transcript_items,n_gold, n_transcript):
    results            = []
    corrections        = []
    transcript_corrige = json.loads(json.dumps(corpus_transcript))

    true_positives  = 0
    total_aligned   = 0
    unaligned_count = 0

    for i, t_line in enumerate(transcript_lines):
        key    = transcript_keys[i]
        center = int(i * n_gold / n_transcript)
        start  = max(0, center - WINDOW_SIZE)
        end    = min(n_gold, center + WINDOW_SIZE + 1)

        t_emb             = model.encode(t_line, convert_to_tensor=True)
        window_embeddings = gold_embeddings[start:end]
        scores            = util.cos_sim(t_emb, window_embeddings)[0]

        best_local_idx   = torch.argmax(scores).item()
        best_score       = scores[best_local_idx].item()
        best_gold_idx    = start + best_local_idx
        best_gold_line   = gold_lines[best_gold_idx]
        best_speaker     = gold_speakers[best_gold_idx]
        original_speaker = transcript_items[key]["speaker_id"]

        if best_score >= SCORE_THRESHOLD:
            total_aligned += 1
            transcript_corrige["repliques"][key]["speaker_id"] = best_speaker
            if str(original_speaker) != str(best_speaker):
                corrections.append({
                    "phrase"          : t_line,
                    "ancien_speaker"  : original_speaker,
                    "nouveau_speaker" : best_speaker,
                    "gold_phrase"     : best_gold_line,
                    "score"           : best_score
                })
            if t_line.strip().lower() == best_gold_line.strip().lower():
                true_positives += 1
        else:
            unaligned_count += 1

        results.append({
            "index"            : i + 1,
            "transcript"       : t_line,
            "gold"             : best_gold_line,
            "score"            : best_score,
            "speaker_gold"     : best_speaker,
            "speaker_original" : original_speaker,
            "aligne"           : best_score >= SCORE_THRESHOLD
        })

    precision = true_positives / total_aligned if total_aligned > 0 else 0
    recall    = true_positives / n_transcript  if n_transcript  > 0 else 0
    f_score   = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return results, corrections, transcript_corrige, {
        "total_aligned"   : total_aligned,
        "unaligned_count" : unaligned_count,
        "corrections"     : len(corrections),
        "true_positives"  : true_positives,
        "precision"       : precision,
        "recall"          : recall,
        "f_score"         : f_score
    }


### SAUVEGARDE DES RÉSULTATS

def save_results(model_name, results, corrections, transcript_corrige, metrics, n_gold, n_transcript):
    safe_name   = model_name.replace("/", "_").replace(" ", "_")
    output_txt  = os.path.join(SCRIPT_DIR, f"resultats_{safe_name}.txt")
    output_json = os.path.join(SCRIPT_DIR, f"transcription_daia_corrige.json")

    with open(output_txt, "w", encoding="utf-8") as out:
        out.write(f"=== RÉSULTATS — {model_name} ===\n")
        out.write(f"Fenêtre : ±{WINDOW_SIZE} | Seuil : {SCORE_THRESHOLD}\n")
        out.write(f"Gold : {n_gold} phrases | Transcript : {n_transcript} phrases\n")
        #out.write(f"Onomatopées ignorées : {len(onomatopees_ignorees)}\n\n")

        out.write("=== ALIGNEMENTS ===\n\n")
        for r in results:
            status = "✓" if r["aligne"] else "✗ (sous seuil)"
            out.write(f"Paire {r['index']} [{status}]\n")
            out.write(f"  Transcript ({r['speaker_original']}) : {r['transcript']}\n")
            out.write(f"  Gold       ({r['speaker_gold']})     : {r['gold']}\n")
            out.write(f"  Score : {r['score']:.3f}\n\n")

        out.write("\n=== CORRECTIONS DE SPEAKERS ===\n\n")
        if corrections:
            for c in corrections:
                out.write(f"Phrase     : {c['phrase']}\n")
                out.write(f"  {c['ancien_speaker']} -> {c['nouveau_speaker']} (score: {c['score']:.3f})\n")
                out.write(f"  Gold ref : {c['gold_phrase']}\n\n")
        else:
            out.write("Aucune correction effectuée.\n\n")

        out.write("\n=== ÉVALUATIONS ===\n")
        out.write(f"Alignées (≥ seuil)   : {metrics['total_aligned']}\n")
        out.write(f"Non alignées         : {metrics['unaligned_count']}\n")
        out.write(f"Speakers corrigés    : {metrics['corrections']}\n")
        out.write(f"Precision            : {metrics['precision']:.3f}\n")
        out.write(f"Recall               : {metrics['recall']:.3f}\n")
        out.write(f"F-score              : {metrics['f_score']:.3f}\n")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcript_corrige, f, ensure_ascii=False, indent=2)

    print(f"  → {output_txt}")
    print(f"  → {output_json}")


### MAIN

if __name__ == "__main__" :
    corpus_gold       = load_json("transcription_gold.json") #fichier de référence
    corpus_transcript = load_json("data/clean/transcription_USGS_clean.json")

    if corpus_gold is None or corpus_transcript is None:
        raise SystemExit("Impossible de continuer sans les fichiers JSON.")

    gold_repliques = corpus_gold["repliques"]
    gold_lines     = [r["line"] for r in gold_repliques]
    gold_speakers  = [r["speaker"] for r in gold_repliques]

    transcript_items = corpus_transcript["repliques"]
    transcript_keys  = list(transcript_items.keys())
    transcript_lines = [transcript_items[k]["line"] for k in transcript_keys]

    moyenne = moy_phrases_par_scene(gold_repliques)
    WINDOW_SIZE = int(round(moyenne))
    print(f"Moyenne : {moyenne:.2f} répliques/scène  →  fenêtre : ±{WINDOW_SIZE}")

    n_gold       = len(gold_lines)
    n_transcript = len(transcript_lines)
    print(f"{n_gold} répliques gold | {n_transcript} répliques transcript (après filtrage)")
    #print(f"{len(onomatopees_ignorees)} onomatopées ignorées\n")

    # Chargement/sauvegarde du modèle de base
    if os.path.exists(MODEL_SAVE_DIR_STD) and os.listdir(MODEL_SAVE_DIR_STD):
        print(f"Modèle de base déjà sauvegardé → chargement depuis : {MODEL_SAVE_DIR_STD}")
        model = SentenceTransformer(MODEL_SAVE_DIR_STD, trust_remote_code=True)
    else:
        model = save_model(BASE_MODEL, MODEL_SAVE_DIR_STD)

    # Encodage du gold + alignement
    print("\nEncodage du corpus gold...")
    gold_embeddings = model.encode(gold_lines, convert_to_tensor=True)

    print("Alignement en cours...")
    results, corrections, transcript_corrige, metrics = run_alignment(
        model, gold_embeddings, gold_lines, gold_speakers,
        transcript_keys, transcript_lines, transcript_items, n_gold, n_transcript
    )
    #Ajout des champs du fichier de référence
    transcript_corrige["scenes"]   = corpus_gold.get("scenes", [])
    transcript_corrige["speakers"] = corpus_gold.get("speakers", [])

    print("Sauvegarde des fichiers...")
    save_results("base", results, corrections, transcript_corrige, metrics, n_gold, n_transcript)
