import re
import json

#   En-tête de scène  →  "1804/01 – EXT JOUR. ABORDS APPART LUCAS :" / "# 1804/06 – INT JOUR. LYCÉE Couloir : vignette B3" / "**1804/08 – INT JOUR. LYCÉE Toilettes :** studio B4"
#   Locuteur          →  "1. EMMA" / "## 25. LUCAS" /  "# 94. NOURA" / "**CHARLOTTE – NOURA**"
#   Réplique          →  ligne immédiatement après le locuteur
#
# Seules les scènes "1804/*" sont retenues.
# Tout le reste (didascalies, pieds de page, lignes vides) est ignoré.


# En tetes de scènes
_SCENE_RE = re.compile(r'^(\d{1,4}/\d{1,2})\s*[\u002D\u2012\u2013\u2014]\s*(.+)$')

# Pied de page : "Un si grand soleil - EP 1804 - ..."
_FOOTER_RE = re.compile(r'un si grand soleil', re.IGNORECASE)

# Caractères autorisés dans un nom de personnage
_NAME_CHARS = r'[A-ZÉÈÊËÀÂÙÛÎÏÔÇŒÆ][A-ZÉÈÊËÀÂÙÛÎÏÔÇŒÆ0-9 \'\-]*'

# Locuteur
_SPEAKER_NUMBERED_RE = re.compile(r'^(?:[#]+\s*)?\d+\.\s+(' + _NAME_CHARS + r')$')

# Locuteur en gras sans numéro : "**CHARLOTTE – NOURA**" ou "**NOURA**"
_SPEAKER_BOLD_RE = re.compile(
    r'^\*\*(' + _NAME_CHARS + r'(?:\s*[–\-]\s*' + _NAME_CHARS + r')*)\*\*$'
)

# Séparation en phrases : coupe après . ? ! suivi d'un espace + majuscule
_SENTENCE_RE = re.compile(r'(?<=[.?!])\s+(?=[A-ZÉÈÊËÀÂÙÛÎÏÔÇŒÆ"\'])')


def strip_markdown(line: str) -> str:
    """Retire les préfixes #/## et les marqueurs ** d'une ligne."""
    line = line.strip().strip('\ufeff\u00a0\u200b')
    line = re.sub(r'^#+\s*', '', line)
    line = re.sub(r'\*\*', '', line)
    return line.strip()


def clean_line(line: str) -> str:
    """Nettoie une réplique : retire les indications de jeu *(...)* et (...)."""
    line = re.sub(r'\*\([^)]*\)\*', '', line)   # *(blessée)*
    line = re.sub(r'\([^)]*\)', '', line)         # (penaud)
    return line.strip()


def split_sentences(text: str) -> list[str]:
    """
    Découpe une réplique en phrases individuelles.
    Ex. : "Je sais pas. Il me déprime." → ["Je sais pas.", "Il me déprime."]
    """
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def parse_scene_header(line: str):
    """
    Retourne (scene_number, scene_desc) si la ligne est un en-tête de scène
    dont le numéro commence par "1804/", sinon (None, None).
    """
    cleaned = strip_markdown(line)
    m = _SCENE_RE.match(cleaned)
    if not m:
        return None, None
    scene_number = m.group(1).strip()
    if not scene_number.startswith("1804/"):
        return None, None
    scene_desc = m.group(2).strip()
    return scene_number, scene_desc


def parse_speaker(line: str):
    """
    Retourne le(s) nom(s) de locuteur si la ligne est une ligne de locuteur,
    sinon None.
    """
    stripped = line.strip()

    # Locuteur numéroté (avec ou sans #/##)
    m = _SPEAKER_NUMBERED_RE.match(stripped)
    if m:
        return m.group(1).strip()

    # Locuteur en gras sans numéro : **NOM** ou **NOM – NOM**
    m = _SPEAKER_BOLD_RE.match(stripped)
    if m:
        raw = m.group(1)
        parts = [p.strip() for p in re.split(r'\s*[–\-]\s*', raw) if p.strip()]
        return parts if len(parts) > 1 else parts[0]

    return None


def simplify_script_to_json(input_file_path: str, output_file_path: str):
    with open(input_file_path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    json_output     = {"scenes": [], "repliques": []}
    current_scene   = None
    current_speaker = None
    all_speakers    = set()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            current_speaker = None
            continue

        # ── Pied de page ─────────────────────────────────────────────────────
        if _FOOTER_RE.search(stripped):
            continue

        # ── Titre narratif en gras (ex. "**Noura se confie à Charlotte**") ──
        # Contient des minuscules → c'est une didascalie, pas un locuteur
        if re.match(r'^\*\*[^*]+\*\*$', stripped) and re.search(r'[a-zéèêëàâùûîïôçœæ]', stripped):
            current_speaker = None
            continue

        # ── En-tête de scène ─────────────────────────────────────────────────
        scene_number, scene_desc = parse_scene_header(stripped)
        if scene_number:
            json_output["scenes"].append({
                "scene_number": scene_number,
                "scene_desc":   scene_desc,
            })
            current_scene   = scene_number
            current_speaker = None
            continue

        # ── Locuteur ─────────────────────────────────────────────────────────
        speaker = parse_speaker(stripped)
        if speaker and current_scene:
            current_speaker = speaker
            continue

        # ── Réplique (ligne suivant immédiatement un locuteur) ───────────────
        if current_speaker and current_scene:
            replique = clean_line(stripped)
            if replique:
                speakers = current_speaker if isinstance(current_speaker, list) else [current_speaker]
                phrases  = split_sentences(replique)
                for sp in speakers:
                    all_speakers.add(sp)
                    for phrase in phrases:
                        json_output["repliques"].append({
                            "scene_index": current_scene,
                            "speaker":     sp,
                            "line":        phrase,
                        })
            current_speaker = None
            continue

        # ── Tout le reste (didascalies, notes de tournage…) est ignoré ───────

    # Finalisation
    json_output["speakers"] = sorted(all_speakers)

    valid_scenes = {r["scene_index"] for r in json_output["repliques"]}
    json_output["scenes"] = [
        s for s in json_output["scenes"]
        if s["scene_number"] in valid_scenes
    ]

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    simplify_script_to_json("../ocr_output_USGS.txt", "../transcription_gold.json")
