# from asyncio import tools
import json
import ollama
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import json
import os

class mcp_internet_search():
    def __init__(self, query: str):
        self.query=query
        self.TOOL_PROMPT = """
            You are an AI research assistant with access to external tools.

            Available tools:

            1. internet_search(query, max_results)
            - Use for general web search, news, blogs, documentation, comparisons.

            2. arxiv_paper_search(query, max_results)
            - Use ONLY for academic or research paper discovery (ML, AI, CS, etc.).

            3. search_repositories(query)
            - Use to find GitHub repositories related to a topic, library, or project.

            4. search_code(query)
            - Use to search for specific implementations, functions, or patterns in GitHub code.

            5. get_file_contents(owner, repo, path)
            - Use to read a specific file from a GitHub repository.

            6. list_issues(owner, repo)
            - Use to inspect open issues, bugs, or discussions in a GitHub repo.

            7. list_pull_requests(owner, repo)
            - Use to inspect recent development activity or changes.

            Guidelines:
            - Choose EXACTLY ONE tool if a tool is required.
            - Prefer GitHub tools for:
            - Code, implementations, repos, issues, PRs
            - Prefer arxiv_paper_search for:
            - Academic papers, surveys, research work
            - Prefer internet_search for:
            - Everything else
            - If no tool is needed, return null.

            Respond ONLY in valid JSON.
            No explanations, no markdown, no extra text.

            Output format:
            {
            "tool": "internet_search | arxiv_paper_search | search_repositories | search_code | get_file_contents | list_issues | list_pull_requests | null",
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

        server_params = StdioServerParameters(
            command="python",
            args=["src/custom_mcp_server.py"],
        )



        github_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github@latest"],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
            }
        )

        async with stdio_client(server_params) as (custom_read, custom_write), \
                stdio_client(github_params) as (gh_read, gh_write):

            async with ClientSession(custom_read, custom_write) as custom_session, \
                    ClientSession(gh_read, gh_write) as gh_session:

                # Initialize
                await custom_session.initialize()
                await gh_session.initialize()

                # Discover tools
                custom_tools = (await custom_session.list_tools()).tools
                gh_tools = (await gh_session.list_tools()).tools

                all_tools = {t.name: ("custom", t) for t in custom_tools}
                all_tools.update({t.name: ("github", t) for t in gh_tools})

                # print("Available MCP tools:", list(all_tools.keys()))

                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0
                )

                # Tool decision (LLM)
                decision = await llm.ainvoke(
                    f"{self.TOOL_PROMPT}\n\nUser query:\n{self.query}"
                )
                # print("LLM tool decision:", decision.content)

                try:
                    tool_decision = json.loads(decision.content)
                    print("Parsed tool decision:", tool_decision)
                except Exception:
                    return decision.content

                tool_name = tool_decision.get("tool")
                tool_args = tool_decision.get("args", {})

                if not tool_name or tool_name not in all_tools:
                    return decision.content

                # Security allow-list
                ALLOWED_TOOLS = {
                    "internet_search",
                    "arxiv_paper_search",
                    "search_repositories",
                    "get_file_contents",
                    "list_issues",
                    "search_code",
                    "list_pull_requests"
                }

                if tool_name not in ALLOWED_TOOLS:
                    return decision.content

                source, _ = all_tools[tool_name]
                print(f"Calling {source} tool: {tool_name} with args {tool_args}")

                # Execute tool
                if source == "custom":
                    tool_response = await custom_session.call_tool(tool_name, tool_args)
                else:
                    tool_response = await gh_session.call_tool(
                        tool_name,
                        {
                            "query": tool_args["query"],
                            "per_page": tool_args.get("max_results", max_results),
                        }
                    )

                tool_output = self.textcontent_to_string(tool_response.content)

                # FINAL LLM CALL (THIS WAS MISSING)
                final_prompt = f"""
                    You are an AI assistant.

                    User query:
                    {self.query}

                    Tool selected by LLM response:
                    {decision}

                    Tool selected:
                    {tool_name}

                    Tool arguments:
                    {json.dumps(tool_args, indent=2)}

                    Tool response:
                    {tool_output}

                    Using the tool response above, provide a clear and concise final answer to the user.
                    Do NOT mention tools or internal decisions.
                    """

                final_answer = await llm.ainvoke(final_prompt)

                return final_answer.content
