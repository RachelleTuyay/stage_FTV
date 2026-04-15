import re
import json
import stanza
import os

# ── Chargement Stanza ─────────────────────────
stanza_dir = os.path.join(os.path.expanduser('~'), 'stanza_resources', 'fr')
if not os.path.exists(stanza_dir):
    stanza.download('fr', processors='tokenize,ner')
nlp = stanza.Pipeline('fr', processors='tokenize,ner', tokenize_no_ssplit=True)


# ── Fonctions utilitaires ─────────────────────

def detecter_locuteurs_multiples(line):
    """Détecte 2 locuteurs séparés par &. Retourne (loc1, loc2) ou None."""
    match = re.match(r'^([A-ZÉÈÀÇ]+)\s&\s([A-ZÉÈÀÇ]+)', line.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None


def extraire_locuteurs(line, current_speaker):
    """Détecte un locuteur au début de la ligne."""
    match_debut = re.match(r'^([A-ZÉÈÀÇ0-9\s\(\)-]{2,})\s+(.*)', line)
    if match_debut:
        return match_debut.group(1).strip(), match_debut.group(2).strip()
    if re.match(r'^[A-ZÉÈÀÇ0-9\s\(\)-]{2,}$', line.strip()):
        return line.strip(), ""
    return current_speaker, line


def fusionner_replique_multiligne(lines, index):
    """Concatène toutes les lignes d'une réplique en une seule."""
    replique = lines[index].strip()
    i = index + 1
    while i < len(lines):
        ligne_suivante = lines[i].strip()
        if (
            not ligne_suivante
            or re.match(r'^[A-Z0-9ÉÈÀÇ\s\(\)]+$', ligne_suivante)
            or re.search(r'\b(INT|EXT)\b', ligne_suivante)
        ):
            break
        replique += " " + ligne_suivante
        i += 1
    return replique, i


def detect_locuteur_stanza(line, current_speaker):
    """Utilise Stanza pour détecter un locuteur si la regex échoue."""
    if current_speaker:
        return current_speaker, line
    doc = nlp(line)
    for sent in doc.sentences:
        for ent in sent.ents:
            if ent.type == "PER":
                locuteur = ent.text
                replique = line.replace(locuteur, "").strip(" :–—")
                return locuteur, replique
    return current_speaker, line


# ── Fonctions principales ─────────────────────

def normaliser_ligne(line):
    """Applique les normalisations communes à une ligne."""
    line = re.sub(r"\s*[-|–]\s*", " - ", line, count=1)
    line = re.sub(r"^(\d+\s?[A-C]?)\s+-?(INT|EXT)\b", r"\1 - \2", line)
    line = re.sub(r"^(\d+)$", "", line)
    line = re.sub(r"\([^)]*\)", "", line)
    line = re.sub(r"^(\d+)\s([A-C])", r"\1\2", line)
    line = re.sub(r"^EXT\s?/\s?INT\b.", "EXT/INT.", line)
    return line.strip()


def est_ligne_inutile(line):
    """Retourne True si la ligne doit être ignorée."""
    if re.match(r"^(\d+)\.?\s*-?\s*(SUP?PRIM[ÉE]E?|ANNUL[ÉE]E?)$", line, re.IGNORECASE):
        return True
    if re.match(r"^\s*JOUR\s+\d+\.?\s*$", line, re.IGNORECASE):
        return True
    if re.match(r"^(TEASER|FIN|GENERIQUE|GÉNÉRIQUE|FIN GENERIQUE|FIN GÉNÉRIQUE)\s*$", line):
        return True
    if line.startswith("«"):
        return True
    if re.match(r"SÉQUENCE EN ALTERNANCE AVEC LA SUIVANTE", line):
        return True
    return False


def decouper_en_scenes(content):
    """Retourne une liste de blocs, chacun commençant par un intitulé de scène."""
    scenes = []
    bloc_courant = []

    for line in content:
        line = normaliser_ligne(line)
        if not line or est_ligne_inutile(line):
            if bloc_courant:
                bloc_courant.append("")
            continue
        if re.search(r'\b(EXT|INT)\b', line):
            if bloc_courant:
                scenes.append(bloc_courant)
            bloc_courant = [line]
        else:
            bloc_courant.append(line)

    if bloc_courant:
        scenes.append(bloc_courant)

    return scenes


def traiter_scene(bloc):
    """Traite un bloc de scène et retourne (intitulé, répliques)."""
    intitule = bloc[0]
    repliques = []
    in_narrative = True
    current_speaker = None
    lines = bloc[1:]
    i = 0

    while i < len(lines):
        line = lines[i]
        i += 1

        if not line:
            current_speaker = None
            in_narrative = True
            continue

        # ── Filtrage descriptions narratives ──────
        if in_narrative:
            if re.match(r'^[A-ZÉÈÀÇ0-9][A-ZÉÈÀÇ0-9\s\(\)\-\./&]*$', line):
                in_narrative = False
                # laisse passer au bloc locuteur
            else:
                continue

        # ── Locuteur ──────────────────────────────
        if re.match(r'^[A-ZÉÈÀÇ0-9][A-ZÉÈÀÇ0-9\s\(\)\-\./&]*$', line):
            double = detecter_locuteurs_multiples(line)
            if double:
                current_speaker = double
            else:
                current_speaker = line.strip()
            continue

        # ── Réplique ──────────────────────────────
        if current_speaker and line:
            line, i = fusionner_replique_multiligne(lines, i - 1)
            line = re.sub(r'\([^)]*\)', '', line).strip()

            current_speaker, line = detect_locuteur_stanza(line, current_speaker)

            if line.strip():
                phrases = re.split(r'(?<=[.!?])\s*', line.strip())
                speakers = list(current_speaker) if isinstance(current_speaker, tuple) else [current_speaker]
                for speaker in speakers:
                    for phrase in phrases:
                        if phrase and phrase != ".":
                            repliques.append({"speaker": speaker, "line": phrase.strip()})

            current_speaker = None
            in_narrative = True

    return intitule, repliques


def simplify_script_to_json(input_file_path, output_file_path):
    """Simplifie un scénario en JSON avec 'scenes' et 'repliques'."""
    with open(input_file_path, "r", encoding="utf-8") as f:
        content = f.readlines()

    blocs = decouper_en_scenes(content)
    json_output = {"scenes": [], "repliques": []}

    for scene_index, bloc in enumerate(blocs):
        intitule, repliques = traiter_scene(bloc)

        if repliques:
            json_output["scenes"].append({
                "scene_number": len(json_output["scenes"]) + 1,
                "old_scene_number": intitule
            })
            for r in repliques:
                json_output["repliques"].append({
                    "scene_index": len(json_output["scenes"]) - 1,
                    "speaker": r["speaker"],
                    "line": r["line"],
                })

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    print(f"JSON créé : {output_file_path}")


if __name__ == "__main__":
    simplify_script_to_json("../ocr_output_AH35.txt", "../transcription_gold.json")
