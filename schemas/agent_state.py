from typing import TypedDict, List
from schemas.document import Document 

class AgentState(TypedDict):
    query: str
    documents: List[Document]
    research_agent: List[str]   
    decision: str
    steps: List[str]
    iteration: int
    response: str
