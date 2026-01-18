import uuid
from docx import Document as DocxDocument
from schemas.document import Document

def extract_docx(docx_path: str) -> list[Document]:
    documents: list[Document] = []

    doc = DocxDocument(docx_path)

    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)

    text = "\n".join(full_text)

    if text.strip():
        documents.append({
            "id": str(uuid.uuid4()),
            "source": "docx",
            "filename": docx_path,
            "text": text,
            "page": None,
            "metadata": {}
        })

    return documents
