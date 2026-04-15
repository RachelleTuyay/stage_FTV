import os
from mistralai.client import Mistral

# Récupération de la clé API Mistral
api_key = "dLpPAVlqYV5sY2PPVuiTjhod6v0lUY5x"
client = Mistral(api_key=api_key)

file_path = "../data/raw/GS_EP_1804_CDVDEF.pdf"
txt_output_path = "../ocr_output_USGS.txt"

# Étape 1 : Upload du fichier
with open(file_path, "rb") as f:
    uploaded_file = client.files.upload(
        file={
            "file_name": os.path.basename(file_path),
            "content": f,
        },
        purpose="ocr"
    )

print(f"Fichier uploadé, ID : {uploaded_file.id}")

# Étape 2 : Obtenir une URL signée pour le fichier
signed_url = client.files.get_signed_url(file_id=uploaded_file.id)

# Étape 3 : Lancer l'OCR avec le modèle
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": signed_url.url,
    }
)


with open(txt_output_path, "w", encoding="utf-8") as f:
    for page in ocr_response.pages:
        text = page.markdown  # texte extrait de chaque page
        text = text.replace("-\n", "")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if text:
            f.write(text + "\n\n")

print(f"Conversion terminée ! Le texte est dans {txt_output_path}")

# Optionnel : supprimer le fichier uploadé après usage
client.files.delete(file_id=uploaded_file.id)
