import streamlit as st
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json
import re
import os
import sys
import subprocess
import time
import torch
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR  = os.path.join(SCRIPT_DIR, "model") #static-similarity-mrl-multilingual-v1
SCORE_THRESHOLD = 0.65

@dataclass
class Corpus :

    @classmethod
    def format_duration(cls, elapsed: float) -> str:
        if elapsed < 60:
            return f"{elapsed:.2f}s"
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.2f}s"

    @classmethod
    def load_json_file(cls, uploaded_file) -> dict:
        """Charge un corpus depuis un st.uploaded_file."""
        try:
            uploaded_file.seek(0)
            return json.load(uploaded_file)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier JSON : {e}")
            return {}

    @classmethod
    def to_download_json(cls, data: dict, filename: str):
        """Bouton de téléchargement d'un dict → JSON."""
        output = json.dumps(data, ensure_ascii=False, indent=2)
        st.download_button(
            label=f"📥 Télécharger {filename}",
            data=output,
            file_name=filename,
            mime="application/json",
        )

@dataclass
class STSAnalyzer :

    @classmethod
    def tokenize(cls, text: str):
        text = text.lower()
        return re.findall(r'\b[a-zàâäéèêëîïôùûüçœæ]+\b', text)

    @classmethod
    def normaliser_phrase(cls, sent: str) -> str:
        sent = re.sub(r'  ', ' ', sent)
        sent = re.sub(r'\s+', ' ', sent)
        sent = re.sub(r'([.!?,;:])([^\s])', r'\1 \2', sent)
        return sent

    @classmethod
    def moy_phrases_par_scene(cls, repliques: list) -> float:
        items = repliques if isinstance(repliques, list) else repliques.values()
        scenes = Counter(rep["scene_index"] for rep in items)
        return sum(scenes.values()) / len(scenes) if scenes else 0.0

    @classmethod
    def run_alignment(cls, model, gold_embeddings, gold_lines, gold_speakers, gold_repliques, transcript_keys, transcript_lines, transcript_items, n_gold, n_transcript, window_size, corpus_transcript):
        results, corrections = [], []
        # Deep-copy du corpus complet (comme comparaison_sent_to_sent), pas seulement les répliques
        transcript_corrige = json.loads(json.dumps(corpus_transcript))
        true_positives = total_aligned = unaligned_count = 0

        # ── OPTIMISATION : encodage batch de tout le transcript en un seul appel ──
        # (avant : model.encode() était appelé une fois par ligne dans la boucle, ce qui
        #  multipliait les appels au modèle par le nombre de répliques du transcript)
        transcript_embeddings = model.encode(transcript_lines, convert_to_tensor=True)

        for i, t_line in enumerate(transcript_lines):
            key    = transcript_keys[i]
            center = int(i * n_gold / n_transcript)
            start  = max(0, center - window_size)
            end    = min(n_gold, center + window_size + 1)

            t_emb             = transcript_embeddings[i]  # récupéré depuis le batch, plus de ré-encodage
            window_embeddings = gold_embeddings[start:end]
            scores            = util.cos_sim(t_emb, window_embeddings)[0]

            best_local_idx   = torch.argmax(scores).item()
            best_score       = scores[best_local_idx].item()
            best_gold_idx    = start + best_local_idx
            best_gold_line   = gold_lines[best_gold_idx]
            best_speaker     = gold_speakers[best_gold_idx]

            # Détecte le bon champ : speaker_id (comparaison_sent_to_sent) ou speaker (App)
            rep = transcript_items[key]
            if "speaker_id" in rep:
                speaker_field    = "speaker_id"
                original_speaker = rep["speaker_id"]
            else:
                speaker_field    = "speaker"
                original_speaker = rep.get("speaker", "")

            if best_score >= SCORE_THRESHOLD:
                total_aligned += 1
                if str(original_speaker) != str(best_speaker):
                    transcript_corrige["repliques"][key][speaker_field] = best_speaker
                    # Ajout du scene_index du gold uniquement sur les répliques corrigées
                    gold_scene_index = gold_repliques[best_gold_idx].get("scene_index")
                    if gold_scene_index is not None:
                        transcript_corrige["repliques"][key]["scene_index"] = gold_scene_index
                    corrections.append({
                        "phrase"         : t_line,
                        "ancien_speaker" : original_speaker,
                        "nouveau_speaker": best_speaker,
                        "gold_phrase"    : best_gold_line,
                        "score"          : best_score,
                        "scene_index"    : gold_scene_index,
                    })
                if t_line.strip().lower() == best_gold_line.strip().lower():
                    pass  # conservé pour ne pas modifier la structure de la boucle
            else:
                unaligned_count += 1

            results.append({
                "index"            : i + 1,
                "transcript"       : t_line,
                "gold"             : best_gold_line,
                "score"            : round(best_score, 3),
                "speaker_gold"     : best_speaker,
                "speaker_original" : original_speaker,
                "aligne"           : best_score >= SCORE_THRESHOLD,
            })

        return results, corrections, transcript_corrige, {
            "total_aligned"  : total_aligned,
            "unaligned_count": unaligned_count,
            "corrections"    : len(corrections),
        }

    @classmethod
    def run_analyse_sts(cls, file_gold, file_transcript, return_corpus=False):
        """Lance l'analyse STS complète et affiche les résultats dans Streamlit."""
        corpus_gold       = Corpus.load_json_file(file_gold)
        corpus_transcript = Corpus.load_json_file(file_transcript)

        if not corpus_gold or not corpus_transcript:
            st.error("Impossible de charger les fichiers JSON.")
            return

        gold_repliques   = corpus_gold.get("repliques", [])
        gold_lines       = [r["line"] for r in gold_repliques]
        gold_speakers_map = {
            v["name"]: v["id"]
            for v in SpeakerEnricher.enrich_speakers(corpus_gold["speakers"])
        }

        gold_speakers = [
            gold_speakers_map.get(r["speaker"], r["speaker"])
            for r in gold_repliques
        ]

        transcript_items = corpus_transcript.get("repliques", {})
        # Supporte liste ou dict
        if isinstance(transcript_items, list):
            transcript_keys  = [str(i) for i in range(len(transcript_items))]
            transcript_lines = [r["line"] for r in transcript_items]
            transcript_items = {str(i): transcript_items[i] for i in range(len(transcript_items))}
        else:
            transcript_keys  = list(transcript_items.keys())
            transcript_lines = [transcript_items[k]["line"] for k in transcript_keys]

        moyenne     = cls.moy_phrases_par_scene(gold_repliques)
        window_size = int(round(moyenne))

        n_gold       = len(gold_lines)
        n_transcript = len(transcript_lines)

        st.info(f"Gold : {n_gold} répliques | Transcript : {n_transcript} répliques | Fenêtre : ±{window_size}")

        model = ModelLoader.load_model(MODEL_SAVE_DIR)

        with st.spinner("Encodage du corpus gold…"):
            gold_embeddings = model.encode(gold_lines, convert_to_tensor=True)

        with st.spinner("Alignement en cours…"):
            results, corrections, transcript_corrige, metrics = cls.run_alignment(
                model, gold_embeddings, gold_lines, gold_speakers, gold_repliques,
                transcript_keys, transcript_lines, transcript_items,
                n_gold, n_transcript, window_size, corpus_transcript,
            )

        # Ajout des champs du gold manquants (scenes) — comme comparaison_sent_to_sent
        if "scenes" not in transcript_corrige and "scenes" in corpus_gold:
            transcript_corrige["scenes"] = corpus_gold["scenes"]
        # Renommage scene_number → scene_index dans scenes
        if "scenes" in transcript_corrige:
            for scene in (transcript_corrige["scenes"] if isinstance(transcript_corrige["scenes"], list)
                        else transcript_corrige["scenes"].values()):
                if "scene_number" in scene:
                    scene["scene_index"] = scene.pop("scene_number")
        # speakers : toujours ecrase par celui du gold (reference authoritative) + enrichissement
        if "speakers" in corpus_gold:
            transcript_corrige["speakers"] = SpeakerEnricher.enrich_speakers(corpus_gold["speakers"])

        # ── Affichage des métriques ──
        st.markdown("#### Quelques chiffres...")
        col1, col2, col3 = st.columns(3)
        col1.metric("Alignées",     metrics["total_aligned"])
        col2.metric("Non alignées", metrics["unaligned_count"])
        col3.metric("Corrections",  metrics["corrections"])

        # ── Tableau des alignements ──
        with st.expander("📋 Détail des alignements"):
            df_results = pd.DataFrame(results)
            df_results["aligne"] = df_results["aligne"].map({True: "✓", False: "✗"})
            st.dataframe(df_results, use_container_width=True)

        # ── Corrections de speakers ──
        with st.expander(f"🔧 Corrections de speakers ({len(corrections)})"):
            if corrections:
                st.dataframe(pd.DataFrame(corrections), use_container_width=True)
            else:
                st.info("Aucune correction effectuée.")

        # ── Téléchargement ──
        if not return_corpus:
            Corpus.to_download_json(transcript_corrige, "transcription_corrigee_sts.json")

        # Attribution du scene_index et du repl_id aux mots
        transcript_corrige = SpeakerEnricher.assign_scene_index_to_words(transcript_corrige)

        if return_corpus:
            return transcript_corrige
    return

@dataclass
class ModelLoader :
    @classmethod
    @st.cache_resource(show_spinner="Chargement du modèle SentenceTransformer...")
    def load_model(cls, model_dir: str) -> SentenceTransformer:
        if os.path.exists(model_dir) and os.listdir(model_dir):
            return SentenceTransformer(model_dir, trust_remote_code=True)
        st.warning("Modèle introuvable, téléchargement depuis HuggingFace....")
        model = SentenceTransformer(
            "sentence-transformers/static-similarity-mrl-multilingual-v1",
            trust_remote_code=True,
        )
        model.save(model_dir)
        return model

    @classmethod
    @st.cache_resource(show_spinner="Chargement du pipeline Stanza...")
    def load_stanza_pipeline(cls):
        """
        OPTIMISATION : pipeline Stanza séparé dans son propre cache_resource.
        Avant, stanza.Pipeline(...) était instancié à l'intérieur de _annotate_gold_ner,
        qui est en cache_data : si l'utilisateur change de fichier gold, le cache
        (basé sur le contenu JSON) rate ET le pipeline (chargement lourd, plusieurs
        secondes) est reconstruit inutilement. Ici, le pipeline est chargé une seule
        fois pour toute la session, quel que soit le gold utilisé.
        """
        import stanza
        return stanza.Pipeline(
            lang='fr',
            processors='tokenize,mwt,pos,ner',
            verbose=False,
        )

@dataclass
class HapaxAnalyzer :
    @classmethod
    def build_dict(cls, repliques) -> dict:
        all_words = []
        items = repliques if isinstance(repliques, list) else repliques.values()
        for r in items:
            text = r.get("line", "")
            if text:
                all_words.extend(STSAnalyzer.tokenize(text))
        return dict(Counter(all_words))

    @classmethod
    def similarity_scores_hapax_batch(cls, model, phrases_a: list, phrases_b: list) -> list:
        """
        Calcule les scores de similarité pour des paires (a[i], b[i]) en batch.
        OPTIMISATION : remplace l'ancien similarity_score_hapax() appelé ligne par ligne
        via df.apply(), qui ré-encodait chaque phrase individuellement (2 x N appels
        model.encode() pour N paires). Ici : 2 appels encode() au total, puis comparaison
        paire-à-paire (rapide, en mémoire).
        """
        emb_a = model.encode(phrases_a, convert_to_tensor=True)
        emb_b = model.encode(phrases_b, convert_to_tensor=True)
        # Similarité "diagonale" : on ne veut que sim(a[i], b[i]), pas la matrice complète
        scores = util.pairwise_cos_sim(emb_a, emb_b) if hasattr(util, "pairwise_cos_sim") \
            else torch.nn.functional.cosine_similarity(emb_a, emb_b)
        return [float(s) for s in scores]

    @classmethod
    def run_analyse_hapax(cls, file_gold, file_transcript, return_corpus=False):
        """Lance l'analyse Hapax complète et affiche les résultats dans Streamlit."""
        corpus_gold       = Corpus.load_json_file(file_gold)
        corpus_transcript = Corpus.load_json_file(file_transcript)

        if not corpus_gold or not corpus_transcript:
            st.error("Impossible de charger les fichiers JSON.")
            return

        gold_repliques       = corpus_gold.get("repliques", [])
        transcript_repliques = corpus_transcript.get("repliques", {})

        gold_speakers_map = {
            v["name"]: v["id"]
            for v in SpeakerEnricher.enrich_speakers(corpus_gold["speakers"])
        }
        # Normalisation en liste
        transcript_list = (
            transcript_repliques
            if isinstance(transcript_repliques, list)
            else list(transcript_repliques.values())
        )

        # ── Construction des hapax ──
        freq_gold       = cls.build_dict(gold_repliques)
        freq_transcript = cls.build_dict(transcript_list)

        hapax_gold       = {w for w, c in freq_gold.items()       if c == 1}
        hapax_transcript = {w for w, c in freq_transcript.items() if c == 1}
        common_words     = sorted(hapax_gold & hapax_transcript)
        common_words_set = set(common_words)

        st.info(f"Hapax gold : {len(hapax_gold)} | Hapax transcript : {len(hapax_transcript)} | Mots communs : {len(common_words)}")

        # ── Construction des tables ──
        # OPTIMISATION : tokenize() est maintenant appelé UNE FOIS par réplique (mis en cache
        # dans gold_tokens_cache / transcript_tokens_cache), au lieu d'être ré-exécuté pour
        # chaque mot commun x chaque réplique (complexité O(mots x répliques) -> O(répliques)).
        gold_tokens_cache       = [(r, set(STSAnalyzer.tokenize(r.get("line", "")))) for r in gold_repliques]
        transcript_tokens_cache = [(r, set(STSAnalyzer.tokenize(r.get("line", ""))))  for r in transcript_list]

        table_gold, table_transcript = [], []

        for r, tokens in gold_tokens_cache:
            for word in common_words_set & tokens:
                table_gold.append({
                    "word"        : word,
                    "sent_gold"   : r.get("line", ""),
                    "speaker_gold": gold_speakers_map.get(
                        r.get("speaker", ""),
                        r.get("speaker", "")
                    ),
                })

        for r, tokens in transcript_tokens_cache:
            for word in common_words_set & tokens:
                table_transcript.append({
                    "word"           : word,
                    "sent_transcript": r.get("line", ""),
                })

        df_gold       = pd.DataFrame(table_gold)
        df_transcript = pd.DataFrame(table_transcript)

        if df_gold.empty or df_transcript.empty:
            st.warning("Aucun hapax commun trouvé.")
            return

        df_merged = pd.merge(df_gold, df_transcript, on="word", how="inner")
        df_merged = df_merged[["word", "sent_gold", "speaker_gold", "sent_transcript"]]

        # ── Scores de similarité ──
        # OPTIMISATION : batch encoding (2 appels model.encode() au total) au lieu d'un
        # appel .apply() ligne par ligne qui ré-encodait chaque phrase individuellement.
        model = ModelLoader.load_model(MODEL_SAVE_DIR)
        with st.spinner("Calcul des scores de similarité…"):
            df_merged["similarity"] = cls.similarity_scores_hapax_batch(
                model,
                df_merged["sent_transcript"].tolist(),
                df_merged["sent_gold"].tolist(),
            )

        df_merged = df_merged[df_merged["similarity"] >= 0.4]
        df_merged = df_merged.drop_duplicates(subset=["sent_transcript"], keep="first")

        st.info(f"Paires retenues (similarité ≥ 0.4) : {len(df_merged)}")

        # ── Application des corrections ──
        sent_list    = df_merged["sent_transcript"].tolist()
        speaker_list = df_merged["speaker_gold"].tolist()
        gold_sent_list = df_merged["sent_gold"].tolist()

        # Lookup gold line → scene_index
        gold_scene_lookup = {r.get("line", ""): r.get("scene_index") for r in gold_repliques}

        transcript_corrige = json.loads(json.dumps(corpus_transcript))
        items = transcript_corrige.get("repliques", {})
        modifications = []

        # Détecte le bon champ speaker (speaker_id ou speaker)
        sample = next(iter(items if isinstance(items, list) else items.values()), {})
        speaker_field = "speaker_id" if "speaker_id" in sample else "speaker"

        iterate_over = items if isinstance(items, list) else items.values()
        for r in iterate_over:
            line = r.get("line", "")
            if line in sent_list:
                idx              = sent_list.index(line)
                nouveau_speaker  = speaker_list[idx]
                ancien_speaker   = r.get(speaker_field, r.get("speaker", r.get("speaker_id", "")))
                if str(ancien_speaker) != str(nouveau_speaker):
                    # Récupère le scene_index depuis la ligne gold correspondante
                    gold_line        = gold_sent_list[idx]
                    gold_scene_index = gold_scene_lookup.get(gold_line)
                    modifications.append({
                        "ancien_speaker" : ancien_speaker,
                        "nouveau_speaker": nouveau_speaker,
                        "line"           : line,
                        "scene_index"    : gold_scene_index,
                    })
                    r[speaker_field] = nouveau_speaker
                    if gold_scene_index is not None:
                        r["scene_index"] = gold_scene_index

        # Preservation de tous les champs du corpus original sauf repliques
        for key, value in corpus_transcript.items():
            if key != "repliques" and key not in transcript_corrige:
                transcript_corrige[key] = value
        # Renommage scene_number → scene_index dans scenes
        if "scenes" in transcript_corrige:
            for scene in (transcript_corrige["scenes"] if isinstance(transcript_corrige["scenes"], list)
                        else transcript_corrige["scenes"].values()):
                if "scene_number" in scene:
                    scene["scene_index"] = scene.pop("scene_number")
        # speakers : toujours ecrase par celui du gold (reference authoritative) + enrichissement
        if "speakers" in corpus_gold:
            transcript_corrige["speakers"] = SpeakerEnricher.enrich_speakers(corpus_gold["speakers"])

        # ── Affichage ──
        st.metric("Corrections effectuées", len(modifications))

        with st.expander("📋 Hapax communs & similarités"):
            st.dataframe(df_merged.reset_index(drop=True), use_container_width=True)

        with st.expander(f"🔧 Corrections de speakers ({len(modifications)})"):
            if modifications:
                st.dataframe(pd.DataFrame(modifications), use_container_width=True)
            else:
                st.info("Aucune correction effectuée.")

        if not return_corpus:
            Corpus.to_download_json(transcript_corrige, "transcription_corrigee_hapax.json")

        # Attribution du scene_index et du repl_id aux mots
        transcript_corrige = SpeakerEnricher.assign_scene_index_to_words(transcript_corrige)

        if return_corpus:
            return transcript_corrige

@dataclass
class NEROption :

    @classmethod
    def highlight_ner_html(cls, text: str, entities: list) -> str:
        """Retourne du HTML avec les entités surlignées par couleur selon leur type."""
        COLORS = {
            "PER" : "#FFD700",  # or
            "ORG" : "#90EE90",  # vert clair
            "LOC" : "#87CEEB",  # bleu ciel
            "MISC": "#FFB6C1",  # rose
        }
        DEFAULT_COLOR = "#E0E0E0"

        result   = ""
        last_end = 0
        for start, end, etype, etext in sorted(entities, key=lambda x: x[0]):
            result += text[last_end:start]
            color   = COLORS.get(etype, DEFAULT_COLOR)
            result += (
                f'<mark style="background-color:{color};border-radius:4px;'
                f'padding:1px 4px;" title="{etype}">{etext} <sup style="font-size:0.65em">{etype}</sup></mark>'
            )
            last_end = end
        result += text[last_end:]
        return result

    @classmethod
    @st.cache_data(show_spinner=False)
    def _annotate_gold_ner(cls, gold_json_str: str) -> tuple:
        """
        Annote toutes les répliques du gold en une passe batch Stanza.
        Mis en cache : ne se ré-exécute que si le contenu du gold change.
        Retourne (html_blocks, total_ents).
        """
        gold_repliques = json.loads(gold_json_str).get("repliques", [])

        # ── Pipeline de meilleure qualité : mwt+pos affinent la reconnaissance NER ──
        nlp = ModelLoader.load_stanza_pipeline()

        # ── Batch : on envoie tous les textes non-vides d'un seul coup ──
        texts    = [r.get("line", "") for r in gold_repliques]
        speakers = [r.get("speaker", r.get("speaker_id", "?")) for r in gold_repliques]

        non_empty_texts = [t for t in texts if t]
        docs = nlp.bulk_process(non_empty_texts)

        # Reconstruction : on réaligne docs (seulement textes non-vides) avec la liste complète
        doc_iter    = iter(docs)
        html_blocks = []
        total_ents  = 0

        for text, speaker in zip(texts, speakers):
            if not text:
                continue
            doc  = next(doc_iter)
            ents = []
            for sent in doc.sentences:
                for ent in sent.ents:
                    ents.append((ent.start_char, ent.end_char, ent.type, ent.text))

            if ents:
                total_ents += len(ents)
                annotated   = cls.highlight_ner_html(text, ents)
                html_blocks.append(
                    f'<div style="margin-bottom:6px"><strong>{speaker}</strong> — {annotated}</div>'
                )
            else:
                html_blocks.append(
                    f'<div style="margin-bottom:6px;color:#888"><strong>{speaker}</strong> — {text}</div>'
                )
        return html_blocks, total_ents

    @classmethod
    def run_option_ner(cls, file_gold, file_transcript, corpus_in=None, return_corpus=False):
        """Extrait et surligne les NER détectés dans le fichier de référence (gold)."""
        try:
            import stanza
        except ImportError:
            st.error("Le module `stanza` n'est pas installé. Lancez : `pip install stanza`")
            return corpus_in if return_corpus else None

        corpus_gold = Corpus.load_json_file(file_gold)

        # corpus_in / corpus_transcript conservés pour le return_corpus, non utilisés pour l'annotation
        if corpus_in is not None:
            corpus_transcript = json.loads(json.dumps(corpus_in))
        else:
            corpus_transcript = Corpus.load_json_file(file_transcript) if file_transcript else {}

        if not corpus_gold:
            st.error("Impossible de charger le fichier gold.")
            return corpus_in if return_corpus else None

        gold_repliques = corpus_gold.get("repliques", [])

        st.markdown("#### Entités NER surlignées dans le fichier de référence")
        st.markdown("PS : cette option est en cours de développement !")
        st.markdown("**Légende :** 🟡 PER &nbsp; 🟢 ORG &nbsp; 🔵 LOC &nbsp; 🟣 MISC")

        # La clé de cache est le contenu JSON du gold — st.cache_data la hache automatiquement
        gold_json_str = json.dumps(corpus_gold, ensure_ascii=False, sort_keys=True)

        with st.spinner("Annotation NER en cours..."):
            html_blocks, total_ents = cls._annotate_gold_ner(gold_json_str)

        # Affichage en une seule passe
        if html_blocks:
            st.markdown("\n".join(html_blocks), unsafe_allow_html=True)
            st.caption(f"{total_ents} entité(s) détectée(s) sur {len(gold_repliques)} réplique(s).")
        else:
            st.info("Aucune réplique trouvée dans le gold.")

        if return_corpus:
            return corpus_transcript

@dataclass
class TiretOption :
    @classmethod
    def run_option_tirets(cls, file_transcript, corpus_in=None, return_corpus=False):
        corpus = json.loads(json.dumps(corpus_in)) if corpus_in is not None else Corpus.load_json_file(file_transcript)
        if not corpus:
            return corpus_in if return_corpus else None

        repliques = corpus.get("repliques", [])
        if not repliques:
            st.warning("Aucune réplique trouvée.")
            return corpus_in if return_corpus else None

        items = repliques if isinstance(repliques, list) else list(repliques.values())

        previous_speaker = None
        count = 0
        for r in items:
            speaker = r.get("speaker", r.get("speaker_id", ""))
            line    = r.get("line", "")
            if speaker != previous_speaker:
                if not line.startswith("–") and not line.startswith("-"):
                    r["line"] = "– " + line
                    count += 1
            previous_speaker = speaker

        if isinstance(repliques, list):
            corpus["repliques"] = items
        else:
            keys = list(corpus["repliques"].keys())
            corpus["repliques"] = {keys[i]: items[i] for i in range(len(keys))}

        with st.expander("Notes – Tirets", expanded=True):
            st.markdown(f"✓ Tirets ajoutés !")
            if not return_corpus:
                Corpus.to_download_json(corpus, "transcription_tirets.json")

        if return_corpus:
            return corpus

@dataclass
class PunctOption :

    @classmethod
    def run_option_ponctuation(cls, file_transcript, corpus_in=None, return_corpus=False):
        corpus = json.loads(json.dumps(corpus_in)) if corpus_in is not None else Corpus.load_json_file(file_transcript)
        if not corpus:
            return corpus_in if return_corpus else None

        repliques = corpus.get("repliques", [])
        items     = repliques if isinstance(repliques, list) else list(repliques.values())

        for r in items:
            line = r.get("line", "")
            line = re.sub(r'\s+([?.!,;:])', r'\1', line)
            line = re.sub(r'([?.!,;:])([^\s])', r'\1 \2', line)
            line = re.sub(r'\s+', ' ', line).strip()
            r["line"] = line

        if isinstance(repliques, list):
            corpus["repliques"] = items
        else:
            keys = list(corpus["repliques"].keys())
            corpus["repliques"] = {keys[i]: items[i] for i in range(len(keys))}

        with st.expander("Notes – Ponctuation", expanded=True):
            st.markdown("✓ Ponctuation normalisée.")
            if not return_corpus:
                Corpus.to_download_json(corpus, "transcription_ponctuation.json")

        if return_corpus:
            return corpus

@dataclass
class SpeakerEnricher :

    @classmethod
    def enrich_speakers(cls, speakers_raw) -> list:
        """
        Enrichit le champ speakers avec les métadonnées name et id.
        Accepte une liste ou un dict.
        Retourne toujours une liste : [{"name": ..., "id": ...}, ...]
        """
        if isinstance(speakers_raw, list):
            pairs = list(enumerate(speakers_raw, start=1))
        else:
            pairs = list(speakers_raw.items())
        result = []
        for seq, (k, v) in enumerate(pairs, start=1):
            # v peut être un str (juste le nom) ou déjà un dict
            name = v if isinstance(v, str) else v.get("name", str(v))
            result.append({
                "name": name,
                "id"  : f"S{seq}",
            })
        return result

    @classmethod
    def assign_scene_index_to_words(cls, transcript_corrige: dict) -> dict:
        """
        Attribue le scene_index à chaque mot du champ 'words' en se basant
        directement sur le champ 'repl_id' du mot, qui référence l'index/clé
        de la réplique correspondante dans 'repliques'.
        """
        words     = transcript_corrige.get("words", {})
        repliques = transcript_corrige.get("repliques", {})

        if not words or not repliques:
            return transcript_corrige

        # Normalise repliques en dict indexé par clé → réplique
        if isinstance(repliques, list):
            repliques_by_id = {i: r for i, r in enumerate(repliques)}
        else:
            repliques_by_id = repliques

        # Construit un index : repl_id (int) → scene_index (peut être None)
        scene_index_by_repl_id = {}
        speaker_id_by_repl_id = {}

        for rid, r in repliques_by_id.items():
            try:
                key = int(rid)
            except (ValueError, TypeError):
                key = rid

            scene_index_by_repl_id[key] = r.get("scene_index", None)
            speaker_id_by_repl_id[key] = r.get(
                "speaker_id",
                r.get("speaker", None)
            )
        # Normalise words en liste ordonnée de (key, dict)
        if isinstance(words, list):
            words_pairs = list(enumerate(words))
        else:
            words_pairs = [(k, words[k]) for k in sorted(words.keys(), key=lambda x: int(x))]

        # Attribution directe via repl_id
        for key, word in words_pairs:
            repl_id = word.get("repl_id", None)
            if repl_id is None:
                continue

            scene_index = scene_index_by_repl_id.get(repl_id, None)
            speaker_id = speaker_id_by_repl_id.get(repl_id, None)

            if scene_index is not None:
                word["scene_index"] = scene_index

            if speaker_id is not None:
                word["speaker_id"] = speaker_id

        return transcript_corrige


##### APP STREAMLIT (main)

apptitle = 'Import de scénario'
st.set_page_config(page_title=apptitle, page_icon=":bookmark_tabs:", layout="wide")
st.title('Import de scénario')
st.markdown("---")

##### Import fichiers JSON

st.subheader("Import des fichiers JSON")
col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded_transcript = st.file_uploader("📄 Fichier transcript (à corriger)", type=["json"], key="transcript")
with col_up2:
    uploaded_gold = st.file_uploader("📄 Fichier gold (référence)", type=["json"], key="gold")

if uploaded_transcript:
    uploaded_transcript.seek(0)
    corpus_preview = json.load(uploaded_transcript)
    n = len(corpus_preview.get("repliques", []))
    st.success(f"Transcript chargé : {n} réplique(s).")

if uploaded_gold:
    uploaded_gold.seek(0)
    corpus_preview = json.load(uploaded_gold)
    n = len(corpus_preview.get("repliques", []))
    st.success(f"Gold chargé : {n} réplique(s).")

### Sidebar : Analyse

st.sidebar.markdown("## Analyse à lancer :")
select_analyse = st.sidebar.selectbox(
    'Analyse :',
    ['Analyse STS', 'Analyse Hapax', 'Both'],
    key='analyse',
)

st.markdown("""
* **Analyse STS** : comparaison phrase par phrase → correction des speakers et ajout des id de scènes.
* **Analyse Hapax** : analyse par mot unique → correction des speakers.
* **Both** : STS + Hapax.
""")

### Sidebar : Options

st.sidebar.markdown("## Options de post-traitement :")
opt_ner         = st.sidebar.checkbox("Extraction des NER", key="opt_ner")
opt_tirets      = st.sidebar.checkbox("Ajout des tirets", key="opt_tirets")
opt_ponctuation = st.sidebar.checkbox("Correction de la ponctuation", key="opt_ponctuation")

st.markdown("""---
* **NER** : extraction des entités nommées du gold → surlignage dans le transcript. (en cours de développement)
* **Tirets** : tiret ajouté à chaque changement de speaker.
* **Ponctuation** : normalisation typographique française. (option à venir)
""")



### Boutons sidebar

st.sidebar.markdown("---")

# ── Bouton Lancer ──
lancer = st.sidebar.button("▶ Lancer l'analyse", type="primary", use_container_width=True)

if lancer:
    # Vérification des fichiers nécessaires
    needs_gold = select_analyse in ['Analyse STS', 'Analyse Hapax', 'Both'] or opt_ner
    if uploaded_transcript is None:
        st.warning("Veuillez charger le fichier transcript.")
    elif needs_gold and uploaded_gold is None:
        st.warning("Veuillez charger le fichier gold (nécessaire pour cette analyse).")
    else:
        # ── Analyses ── (on accumule le corpus corrigé)
        st.markdown("### Résultats de l'analyse")

        corpus_final = None  # contiendra le corpus cumulativement corrigé

        if select_analyse == 'Analyse STS':
            with st.expander("Analyse STS", expanded=True):
                corpus_final = STSAnalyzer.run_analyse_sts(uploaded_gold, uploaded_transcript, return_corpus=True)
            if corpus_final:
                repliques_finales_sts = corpus_final.get("repliques", {})
                items_finaux_sts = repliques_finales_sts if isinstance(repliques_finales_sts, list) else list(repliques_finales_sts.values())
                uploaded_transcript.seek(0)
                corpus_original_sts = json.load(uploaded_transcript)
                repliques_originales_sts = corpus_original_sts.get("repliques", {})
                items_originaux_sts = repliques_originales_sts if isinstance(repliques_originales_sts, list) else list(repliques_originales_sts.values())
                speaker_field_sts = "speaker_id" if items_originaux_sts and "speaker_id" in items_originaux_sts[0] else "speaker"
                total_corrections_sts = sum(
                    1 for orig, final in zip(items_originaux_sts, items_finaux_sts)
                    if str(orig.get(speaker_field_sts, "")) != str(final.get(speaker_field_sts, ""))
                )
                n_repliques_sts = len(items_originaux_sts)
                pct_corrections_sts = round(100 * total_corrections_sts / n_repliques_sts, 1) if n_repliques_sts else 0.0
                st.markdown("---")
                st.markdown("#### Récapitulatif STS")
                rs1, rs2, rs3 = st.columns(3)
                rs1.metric("Total répliques",   n_repliques_sts)
                rs2.metric("Corrections",        total_corrections_sts)
                rs3.metric("Taux de correction", f"{pct_corrections_sts} %")
                Corpus.to_download_json(corpus_final, "transcription_corrigee_sts.json")

        elif select_analyse == 'Analyse Hapax':
            with st.expander("Analyse Hapax", expanded=True):
                corpus_final = HapaxAnalyzer.run_analyse_hapax(uploaded_gold, uploaded_transcript, return_corpus=True)
            if corpus_final:
                repliques_finales_hpx = corpus_final.get("repliques", {})
                items_finaux_hpx = repliques_finales_hpx if isinstance(repliques_finales_hpx, list) else list(repliques_finales_hpx.values())
                uploaded_transcript.seek(0)
                corpus_original_hpx = json.load(uploaded_transcript)
                repliques_originales_hpx = corpus_original_hpx.get("repliques", {})
                items_originaux_hpx = repliques_originales_hpx if isinstance(repliques_originales_hpx, list) else list(repliques_originales_hpx.values())
                speaker_field_hpx = "speaker_id" if items_originaux_hpx and "speaker_id" in items_originaux_hpx[0] else "speaker"
                total_corrections_hpx = sum(
                    1 for orig, final in zip(items_originaux_hpx, items_finaux_hpx)
                    if str(orig.get(speaker_field_hpx, "")) != str(final.get(speaker_field_hpx, ""))
                )
                n_repliques_hpx = len(items_originaux_hpx)
                pct_corrections_hpx = round(100 * total_corrections_hpx / n_repliques_hpx, 1) if n_repliques_hpx else 0.0
                st.markdown("---")
                st.markdown("#### Récapitulatif Hapax")
                rh1, rh2, rh3 = st.columns(3)
                rh1.metric("Total répliques",   n_repliques_hpx)
                rh2.metric("Corrections",        total_corrections_hpx)
                rh3.metric("Taux de correction", f"{pct_corrections_hpx} %")
                Corpus.to_download_json(corpus_final, "transcription_corrigee_hapax.json")

        elif select_analyse == 'Both':
            with st.expander("Analyse STS", expanded=True):
                corpus_final = STSAnalyzer.run_analyse_sts(uploaded_gold, uploaded_transcript, return_corpus=True)

            if corpus_final:
                import io
                transcript_after_sts = io.BytesIO(
                    json.dumps(corpus_final, ensure_ascii=False, indent=2).encode("utf-8")
                )
                transcript_after_sts.name = "transcript_after_sts-hapax.json"

                with st.expander("Analyse Hapax", expanded=True):
                    corpus_final = HapaxAnalyzer.run_analyse_hapax(uploaded_gold, transcript_after_sts, return_corpus=True)

                # ── Récap total des corrections STS + Hapax ──
                if corpus_final:
                    repliques_finales = corpus_final.get("repliques", {})
                    items_finaux = repliques_finales if isinstance(repliques_finales, list) else list(repliques_finales.values())
                    uploaded_transcript.seek(0)
                    corpus_original_both = json.load(uploaded_transcript)
                    repliques_originales = corpus_original_both.get("repliques", {})
                    items_originaux = repliques_originales if isinstance(repliques_originales, list) else list(repliques_originales.values())
                    speaker_field_both = "speaker_id" if items_originaux and "speaker_id" in items_originaux[0] else "speaker"
                    total_corrections_both = sum(
                        1 for orig, final in zip(items_originaux, items_finaux)
                        if str(orig.get(speaker_field_both, "")) != str(final.get(speaker_field_both, ""))
                    )
                    n_repliques_both    = len(items_originaux)
                    pct_corrections_both = round(100 * total_corrections_both / n_repliques_both, 1) if n_repliques_both else 0.0
                    st.markdown("---")
                    st.markdown("#### Récapitulatif Both (STS + Hapax cumulés)")
                    rb1, rb2, rb3 = st.columns(3)
                    rb1.metric("Total répliques",   n_repliques_both)
                    rb2.metric("Corrections",        total_corrections_both)
                    rb3.metric("Taux de correction", f"{pct_corrections_both} %")
            else:
                st.warning("L'analyse STS n'a pas retourné de corpus, Hapax ignoré.")

        # ── Options (appliquées sur le corpus déjà corrigé) ──
        if any([opt_ner, opt_tirets, opt_ponctuation]):
            st.markdown("### Résultats des options")

            if opt_ner:
                with st.expander("Extraction NER", expanded=True):
                    corpus_final = NEROption.run_option_ner(
                        uploaded_gold, uploaded_transcript,
                        corpus_in=corpus_final,
                        return_corpus=True,
                    )

            if opt_tirets:
                with st.expander("Tirets", expanded=True):
                    corpus_final = TiretOption.run_option_tirets(
                        uploaded_transcript,
                        corpus_in=corpus_final,
                        return_corpus=True,
                    )

            if opt_ponctuation:
                with st.expander("Ponctuation", expanded=True):
                    corpus_final = PunctOption.run_option_ponctuation(
                        uploaded_transcript,
                        corpus_in=corpus_final,
                        return_corpus=True,
                    )

        # ── Téléchargement du fichier final cumulé ──
        if corpus_final:
            st.markdown("---")
            st.markdown("### 📦 Fichier final")

            # Récapitulatif des paramètres appliqués
            params_appliques = [select_analyse]
            if opt_ner:         params_appliques.append("NER")
            if opt_tirets:      params_appliques.append("Tirets")
            if opt_ponctuation: params_appliques.append("Ponctuation")
            st.info(f"Paramètres appliqués : **{' + '.join(params_appliques)}**")

            # Vérification que tous les champs originaux sont présents
            if uploaded_transcript:
                uploaded_transcript.seek(0)
                corpus_original = json.load(uploaded_transcript)
                for key, value in corpus_original.items():
                    if key not in corpus_final:
                        corpus_final[key] = value

            output_final = json.dumps(corpus_final, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Télécharger la version corrigée",
                data=output_final,
                file_name="transcription_finale.json",
                mime="application/json",
                type="primary",
            )
        else:
            st.info("Aucune option sélectionnée.")

##### INFOS
st.markdown("---")
st.subheader("À propos de l'UI :")
st.markdown("""
* [Voir le code](https://github.com/RachelleTuyay/stage_FTV/tree/main/import_de_scenario)
* Auteur : ME :)
""")
