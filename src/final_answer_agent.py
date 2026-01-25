from schemas.agent_state import AgentState
from langchain_openai import ChatOpenAI

async def final_answer_agent(state: AgentState) -> AgentState:
    prompt = f"""
        You are a final answer generator.

        Your job is to return ONLY the final answer that should be shown to the user.

        STRICT RULES:
        - Do NOT explain your reasoning
        - Do NOT mention tools, models, agents, or internal decisions
        - Do NOT restate the question
        - Output plain text only

        Behavior rules:
        - If the user query is a greeting or casual message (hi, hello, hey, etc.),
        respond with a short, friendly greeting.
        - If an existing response is provided, refine or summarize it into a clear,
        user-facing final answer.
        - If the existing response is empty, answer directly using general knowledge.
        - Keep the answer concise and user-focused.

        User query:
        {state.get("query", "")}

        Existing response:
        {state.get("response", "")}
    """


    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
        )

  
    response = await llm.ainvoke(prompt)

    print("Final formatted response:", response.content)

    return {
        **state,
        "final_answer": response.content
    }
