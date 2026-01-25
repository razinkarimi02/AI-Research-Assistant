from langgraph.graph import StateGraph, END
from schemas.agent_state import AgentState
from ingestion.file_router import ingest_files
from src.rag_agent import rag_agent
from src.orchestrator_agent import orchestrator_agent
from src.internet_search_agent import internet_search
from src.final_answer_agent import final_answer_agent


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

# -------- Run Workflow --------
async def run_workflow(query: str, files: list[str], logger) -> str:
    graph = build_research_agent()

    documents = ingest_files(files) if files else []

    initial_state: AgentState = {
        "query": query,
        "documents": documents,
        "research_agent": [],
        "decision": "",
        "steps": [],
        "iteration": 1,
        "response": "",
        "final_answer": ""
    }

    logger.info("Starting workflow for query: %s", query)

    final_state = await graph.ainvoke(initial_state)

    logger.info("Final state steps: %s", final_state["steps"])
    logger.info("Workflow completed for query: %s", query)

    return final_state.get("final_answer")
