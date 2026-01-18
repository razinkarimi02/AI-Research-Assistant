# from asyncio import tools
import json
import ollama
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import json


class mcp_internet_search():
    def __init__(self, query: str):
        self.query=query
        self.TOOL_PROMPT = """
        You can use the following tools:

        1. internet_search(query, max_results)
        2. arxiv_paper_search(query, max_results)

        Decide the BEST tool.

        Respond ONLY in JSON:
        {
        "tool": "internet_search | arxiv_paper_search | null",
        "args": {
            "query": "...",
            "max_results": 5
        }
        }
        """

    def textcontent_to_string(self,content):
        if isinstance(content, list):
            return [self.textcontent_to_string(c) for c in content]

        if hasattr(content, "text"):
            return content.text

        return content

    async def mcp_search(self, max_results: int = 5):

        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python3",  # Executable
            args=["-m", "src.mcp_tavily_server"]
        )


        async with stdio_client(server_params) as (read, write): 
            # the client session is used to initiate the connection 
            # and send requests to server 
            async with ClientSession(read, write) as session:
                # Initialize the connection (1:1 connection with the server)
                await session.initialize()

                result = await session.list_tools()
                print("Available tools:", result)
                tools = result.tools

                print("MCP tools:", [t.name for t in tools])

                llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0
                )

                # 1️⃣ Ask LLM which tool to use
                decision = await llm.ainvoke(
                    f"{self.TOOL_PROMPT}\n\nUser query: {self.query}"
                )

                # 2️⃣ Parse JSON safely
                try:
                    tool_decision = json.loads(decision.content)
                except Exception:
                    return "I can answer this without using any tools."

                tool_name = tool_decision.get("tool")
                tool_args = tool_decision.get("args", {})

                # 3️⃣ Normalize tool name
                if tool_name in (None, "null", "None", "", False):
                    return "I can answer this without using any tools."

                # 4️⃣ Allowlist tools (VERY IMPORTANT)
                ALLOWED_TOOLS = {
                    "internet_search",
                    "arxiv_paper_search"
                }

                if tool_name not in ALLOWED_TOOLS:
                    return "I can answer this without using any tools."

                # 5️⃣ Call tool
                print(f"Calling tool: {tool_name} with args {tool_args}")

                tool_response = await session.call_tool(
                    tool_name,
                    tool_args
                )

                final_result = tool_response.content
                return self.textcontent_to_string(final_result)
