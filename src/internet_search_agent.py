from schemas.agent_state import AgentState
from src.mcp_search_client import mcp_internet_search

# -------- Node 1: Research --------
async def internet_search(state: AgentState) -> AgentState:
    query = state["query"]
    mcp= mcp_internet_search(query)
    results = await(
        mcp.mcp_search(max_results=5)
    )
    # print("Proper results: ",results)

    state["response"] = results
    state.setdefault("research_agent", []).append(results)
    state["steps"] = state.get("steps", []) + ["internet_search"]

    return state