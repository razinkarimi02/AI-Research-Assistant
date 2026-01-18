import uuid
from PIL import Image
import pytesseract
from schemas.document import Document

def extract_image(image_path: str) -> list[Document]:
    documents: list[Document] = []

    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    if text.strip():
        documents.append({
            "id": str(uuid.uuid4()),
            "source": "image",
            "filename": image_path,
            "text": text,
            "page": None,
            "metadata": {
                "ocr_engine": "tesseract"
            }
        })

    return documents
    