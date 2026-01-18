import os
import tempfile
from ingestion.pdf_to_text import extract_pdf
from ingestion.img_to_text import extract_image
from ingestion.doc_to_text import extract_docx

def ingest_files(files):
    documents = []

    for file in files:
        filename = file.filename.lower()

        # Save uploaded file to temp location
        suffix = os.path.splitext(filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        # Route by file type
        if filename.endswith(".pdf"):
            documents.extend(extract_pdf(temp_path))

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            documents.extend(extract_image(temp_path))

        elif filename.endswith(".docx"):
            documents.extend(extract_docx(temp_path))

    return documents
