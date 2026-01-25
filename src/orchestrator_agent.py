import json
from schemas.agent_state import AgentState
from langchain_openai import ChatOpenAI


def orchestrator_agent(state: AgentState) -> AgentState:
    query = state["query"]
    research_notes = state.get("research_agent", [])
    documents = state.get("documents", [])
    iteration = state.get("iteration", 0)

    prompt = f"""
        You are an orchestration agent in a multi-step AI system.

        Your job is to decide the NEXT action to take based on the user query and the current system state.

        User query:
        {query}

        Existing response from previous agent:
        {state.get("response", "")}

        Available information:

        Internet research notes:
        {research_notes if research_notes else "None"}

        Retrieved documents (from user files):
        {len(documents)} document chunks available

        Iteration count:
        {iteration}

        Decision rules:
        - If the query is a greeting, casual message, or small talk (e.g., "hi", "hello", "hey") → end
        - If a meaningful response already exists → end
        - If the query can be confidently answered using existing information → end
        - If the query doesn't require any tool connection or document retrieval → end
        - If documents exist AND they are relevant to the query → rag
        - If information is missing AND documents are insufficient → research
        - Avoid unnecessary research if documents already contain the answer
        - Never exceed 2 iterations

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

    # print("Orchestrator Prompt: ",prompt)


    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
        )

    response = llm.invoke(
        prompt
    )

    try:
        content = response.content
        decision_json = json.loads(content)
        print("Orchestrator decision JSON:", decision_json)
        decision = decision_json.get("decision", "research")
    except Exception as e:
        print("Error:", e)
        decision = "research"



    # decision = "rag"
    # ---- Loop guard ----
    if iteration >= 2:
        decision = "end"

    state["decision"] = decision
    state["iteration"] = iteration + 1
    state["steps"] = state.get("steps", []) + [f"orchestrator→{decision}"]

    return state
