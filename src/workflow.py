from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import ollama
import json
import asyncio
from src.mcp_search_client import mcp_internet_search
from schemas.agent_state import AgentState
from ingestion.file_router import ingest_files
from langchain_community.llms import Ollama
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

# BM25 → unchanged
from rank_bm25 import BM25Okapi

# PromptTemplate → langchain-core
from langchain_core.prompts import PromptTemplate

# OllamaEmbeddings → langchain-community
from langchain_community.embeddings import OllamaEmbeddings


def rag_agent(state: AgentState) -> AgentState:
    query = state["query"]
    raw_docs = state.get("documents", [])

    if not raw_docs:
        state["response"] = "No documents provided for RAG."
        return state

    lc_docs = [
        Document(
            page_content=doc["text"],
            metadata={
                "source": doc["source"],
                "filename": doc.get("filename", ""),
                "page": doc.get("page")
            }
        )
        for doc in raw_docs
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(lc_docs)


    # ============================================================
    # 1️⃣ BM25 RETRIEVAL (Top 10)
    # ============================================================
    corpus = [c.page_content for c in chunks]
    tokenized = [c.split() for c in corpus]

    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())

    bm25_top_k = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:10]

    bm25_docs = [(chunks[i], bm25_scores[i]) for i in bm25_top_k]

    # ============================================================
    # 2️⃣ SEMANTIC RETRIEVAL (Top 10)
    # ============================================================
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    sem_docs = vectorstore.similarity_search_with_score(query, k=10)
    # sem_docs → [(Document, distance)]

    # ============================================================
    # 3️⃣ NORMALIZATION + WEIGHTED FUSION
    # ============================================================
    doc_score_map = {}

    # ---- Normalize BM25 (0–1) ----
    bm25_max = max(score for _, score in bm25_docs) if bm25_docs else 1.0

    for doc, score in bm25_docs:
        norm_score = score / bm25_max
        doc_score_map[doc.page_content] = {
            "doc": doc,
            "bm25": norm_score,
            "semantic": 0.0
        }

    # ---- Normalize Semantic (distance → similarity → 0–1) ----
    semantic_scores = [(doc, 1 / (1 + dist)) for doc, dist in sem_docs]
    sem_max = max(score for _, score in semantic_scores) if semantic_scores else 1.0

    for doc, score in semantic_scores:
        norm_score = score / sem_max

        if doc.page_content not in doc_score_map:
            doc_score_map[doc.page_content] = {
                "doc": doc,
                "bm25": 0.0,
                "semantic": norm_score
            }
        else:
            doc_score_map[doc.page_content]["semantic"] = norm_score

    # ---- Weighted Fusion ----
    SEM_WEIGHT = 0.6
    BM25_WEIGHT = 0.4

    for v in doc_score_map.values():
        v["final_score"] = (
            SEM_WEIGHT * v["semantic"] +
            BM25_WEIGHT * v["bm25"]
        )

    # ============================================================
    # 4️⃣ TOP 10 FUSED → CROSS-ENCODER
    # ============================================================
    top_fused = sorted(
        doc_score_map.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )[:10]

    fused_docs = [v["doc"] for v in top_fused]

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(query, d.page_content) for d in fused_docs]
    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(fused_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    final_docs = [doc for doc, _ in reranked]

    print("Final Reranked Docs: ",final_docs)

    context = "\n\n".join(d.page_content for d in final_docs)
    # print("Final Context: ",context)

    prompt = PromptTemplate(
        template="""
        You are a precise research assistant.

        Answer the question using ONLY the context below.
        If the answer is not present, say so.

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )

    llm = Ollama(model="llama3:8b", temperature=0)

    answer = llm(
        prompt.format(context=context, question=query)
    )

    state["response"] = answer
    state["steps"] = state.get("steps", []) + ["rag_hybrid_bm25_semantic_rerank"]

    return state

def orchestrator_agent(state: AgentState) -> AgentState:
    query = state["query"]
    research_notes = state.get("research_agent", [])
    documents = state.get("documents", [])
    # print("o-doc: ",documents)
    iteration = state.get("iteration", 0)

    prompt = f"""
    You are an orchestration agent in a multi-step AI system.

    Your job is to decide the NEXT action to take based on the user query and available information.

    User query:
    {query}

    Available information:

    Internet research notes:
    {research_notes if research_notes else "None"}

    Retrieved documents (from user files):
    {len(documents)} document chunks available

    Iteration count:
    {iteration}

    Decision rules:
    - If the query can be confidently answered using existing information → end
    - If the query doesn't require any tool connection or document retrieval → end
    - If documents exist AND they are relevant to the query → rag
    - If information is missing AND documents are insufficient → research
    - Avoid unnecessary research if documents already contain the answer
    - Never exceed 5 iterations

    RESPONSE FORMAT STRICTLY:
    - Respond ONLY in valid JSON
    - The "decision" field MUST be exactly one of: "research", "rag", "end"
    - Do NOT use synonyms, abbreviations, or any other words like "search" or "lookup"
    - Ensure your JSON is parseable

    Schema:
    {{
        "decision": "research" | "rag" | "end",
        "reason": "Explain your choice in one concise sentence."
    }}
    """


    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
        )

    response = llm.ainvoke(
        prompt
    )

    try:
        content = response["message"]["content"]
        decision_json = json.loads(content)
        decision = decision_json.get("decision", "research")
    except Exception:
        decision = "research"

    # decision = "rag"
    # ---- Loop guard ----
    if iteration >= 5:
        decision = "end"

    state["decision"] = decision
    state["iteration"] = iteration + 1
    state["steps"] = state.get("steps", []) + [f"orchestrator→{decision}"]

    return state



# -------- Build Graph --------
def build_research_agent():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("orchestrator", orchestrator_agent)
    graph.add_node("research", internet_search)
    graph.add_node("rag", rag_agent)
    graph.add_node("final_answer", final_answer_agent)

    # Entry
    graph.set_entry_point("orchestrator")

    # Orchestrator routing
    graph.add_conditional_edges(
        "orchestrator",
        lambda state: state.get("decision"),
        {
            "research": "research",
            "rag": "rag",
            "end": "final_answer",
        },
    )

    # Loops back to orchestrator
    graph.add_edge("research", "orchestrator")
    graph.add_edge("rag", "orchestrator")

    # Exit
    graph.add_edge("final_answer", END)

    return graph.compile()

# -------- Node 1: Research --------
async def internet_search(state: AgentState) -> AgentState:
    query = state["query"]
    mcp= mcp_internet_search(query)
    results = await(
        mcp.mcp_search(max_results=5)
    )
    print("Proper results: ",results)

    state["response"] = results
    state.setdefault("research_agent", []).append(results)
    state["iteration"] = state.get("iteration", 0) + 1
    state["steps"] = state.get("steps", []) + ["internet_search"]

    return state


async def final_answer_agent(state: AgentState) -> AgentState:
    formatted_response = ollama.chat(
        model="llama3:8b",
        messages=[{
            "role": "user",
            "content": "This is the query: " + str(state.get("query", "")) + "\nResponse from llm: " + str(state.get("response", "")) + "\nIf the response is somewhat like this: I can answer this without using any tools, then generate a final answer based on your knowledge. Otherwise, provide a concise summary answer based on the information provided."
        }],
    )

    print("Final formatted response: ",formatted_response)
    return {
        **state,
        "final_answer": formatted_response["message"]["content"]
    }


async def run_workflow(query: str, files: list[str], logger) -> str:
    graph = build_research_agent()

    documents = ingest_files(files) if files else []

    initial_state: AgentState = {
        "query": query,
        "documents": documents,
        "research_agent": [],
        "decision": "",
        "steps": [],
        "iteration": 0,
        "response": ""
    }

    logger.info("Starting workflow for query: %s", query)

    final_state = await graph.ainvoke(initial_state)

    logger.info("Final state steps: %s", final_state["steps"])
    logger.info("Workflow completed for query: %s", query)

    return final_state.get("final_answer") or final_state.get("response", "")
