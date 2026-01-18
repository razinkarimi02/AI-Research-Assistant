import uuid
import fitz  # PyMuPDF

def extract_pdf(pdf_path: str) -> list[dict]:
    documents = []

    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")

        if not text.strip():
            continue

        documents.append({
            "id": str(uuid.uuid4()),
            "source": "pdf",
            "filename": pdf_path,
            "page": page_num,
            "text": text,
            "metadata": {
                "page": page_num,
                "source": "pdf"
            }
        })

    return documents
