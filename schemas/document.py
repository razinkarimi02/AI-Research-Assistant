from typing import TypedDict, Dict, Optional

class Document(TypedDict):
    id: str
    source: str           
    filename: str
    text: str
    page: Optional[int]
    metadata: Dict
