from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from schemas.agent_state import AgentState 
from rank_bm25 import BM25Okapi
from langchain_core.prompts import PromptTemplate

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

    # BM25 RETRIEVAL (Top 10)
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
    print("BM25 Docs: ",bm25_docs)

    # SEMANTIC RETRIEVAL (Top 10)
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
    )     
    vectorstore = FAISS.from_documents(chunks, embeddings)

    sem_docs = vectorstore.similarity_search_with_score(query, k=10)

    # NORMALIZATION + WEIGHTED FUSION
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

    # TOP 10 FUSED → CROSS-ENCODER

    top_fused = sorted(
        doc_score_map.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )[:10]

    fused_docs = [v["doc"] for v in top_fused]
    print("Fused Docs: ",fused_docs)

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
    print("Final Context: ",context)


    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

    
    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": query
    })

    answer = response.content

    state["response"] = answer
    state["steps"] = state.get("steps", []) + ["rag_hybrid_bm25_semantic_rerank"]

    return state