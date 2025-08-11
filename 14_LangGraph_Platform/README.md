<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 14: Build & Serve Agentic Graphs with LangGraph</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 14: Pre-Work](https://www.notion.so/Session-14-Deploying-Agents-to-Production-21dcd547af3d80aba092fcb6c649c150?source=copy_link#247cd547af3d80709683ff380f4cba62)| [Session 14: Deploying Agents to Production](https://www.notion.so/Session-14-Deploying-Agents-to-Production-21dcd547af3d80aba092fcb6c649c150) | [Recording!](https://us02web.zoom.us/rec/share/1YepNUK3kqQnYLY8InMfHv84JeiOMyjMRWOZQ9jfjY86dDPvHMhyoz5Zo04w_tn-.91KwoSPyP6K6u0DC)  (@@5J6DVQ)| [Session 14 Slides](https://www.canva.com/design/DAGvVPg7-mw/IRwoSgDXPEqU-PKeIw8zLg/edit?utm_content=DAGvVPg7-mw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 14 Assignment: Production Agents](https://forms.gle/nZ7ugE4W9VsC1zXE8) | [AIE7 Feedback 8/7](https://forms.gle/juo8SF5y5XiojFyC9)

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph
  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangGraph Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.
</details>

#### ✅ Answer - Advanced Build Activity :

I have been working on the Advanced Build.<br>
I have deployed my <b>News MCP Server</b>, which has three tools :<br>
Note : MCP Server file [news_mcp_server.py](news_mcp_server.py)

 - Tool : 📊 get_top_headlines <br>
This tool retrieves the latest top headlines from a specific country, with an optional category filter.
It provides a curated list of the most important news stories currently trending in the selected region.

- Tool : 🔍 search_news <br>
This tool searches for news articles based on specific keywords or topics.
It can filter results by language, sort by publication date, and return a customizable number of articles that match the search criteria.

- Tool : 📰 get_news_sources <br>
This tool provides a list of available news sources for a specific country, with an optional category filter.
It helps users discover reliable news outlets and understand the available sources for different regions and topics.

<b>Changes I have done to implement it :</b><br>
- <b>app/tools : </b><br>
Function `get_tool_belt()` updated to retrieve MCP tools using `MultiServerMCPClient`. See file [tools.py](app/tools.py)

- <b>pyproject.toml : </b><br>
Additional packages installed using uv : langchain-mcp-adapters, fastapi and uvicorn.

- <b>.env : </b><br>
NEWS_API_KEY - added this new api key.




# Ship 🚢

- Running local server (`langgraph dev`)
- Short demo showing both assistants responding

# Share 🚀
- Walk through your graph in Studio
- Share 3 lessons learned and 3 lessons not learned


#### ❓ Question:

What is the purpose of the `chunk_overlap` parameter when using `RecursiveCharacterTextSplitter` to prepare documents for RAG, and what trade-offs arise as you increase or decrease its value?

#### ✅ Answer:
The chunk_overlap parameter controls how much text is shared between consecutive chunks when splitting documents for RAG.

For example : <br>
`chunk_overlap = 0` : No overlap between chunks<br>
`chunk_overlap = 100` : 100 characters overlap between consecutive chunks.<br>
`chunk_overlap = 200` : 200 characters overlap between consecutive chunks.<br>

Trade-offs when adjusting this parameter :<br>

<b>Increasing `chunkg_overlap` :</b><br>
✅ Benefits: <br>
Better context preservation: Important information at chunk boundaries is not lost.<br>
Improved retrieval: Related concepts that span chunk boundaries are captured.<br>
Better semantic coherence: Chunks maintain more complete context.<br><br>
❌ Drawbacks:<br>
Increased storage: More redundant data stored in vector database.<br>
Higher computational cost: More embeddings to process.<br>
Potential redundancy: Same information retrieved multiple times


<b>Decreasing `chunk_overlap` :</b><br>
✅ Benefits:
Reduced storage: Less redundant data.<br>
Lower computational cost: Fewer embeddings to process.<br>
Cleaner chunks: Less duplicate information
<br><br>
❌ Drawbacks:
Context loss: Important information at chunk boundaries may be lost.<br>
Poorer retrieval: Related concepts split across chunks may be missed.<br>
Reduced semantic coherence: Chunks may lack complete context.


<br>

#### ❓ Question:

Your retriever is configured with `search_kwargs={"k": 5}`. How would adjusting `k` likely affect RAGAS metrics such as Context Precision and Context Recall in practice, and why?

##### ✅ Answer:

IMPORTANT NOTE : After review the code, the current configuration of the retriever is not set at K = 5.

#### Effect of the `k` parameter on RAGAS metrics

 - Context Precission : <br>
It mesures how relevant the retrieved documents are for answering the question.<br>
Increasing the k for this metric, likely decreases precision, because when retrieving more documents, it is more likely that some will be less relevant.<br>
Otherwise, decreasing the k for this metric will increase the precision, because fewer retrieved documents means higher probability that all are very relevant.

 - Context Recall : <br>
 It mesures how complete the retrieved information is for answering the question.<br>
 Increasing the k for this metric, likely increases recall, because more documents means higher probability of capturing all necessary information.<br>
 Otherwise, decreasing the k for this metric will decrease the recall, because fewer documents means lower probability of capturing all necessary information.


 <br>

#### ❓ Question:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

##### ✅ Answer:

<b>Definition and comparison between both agents :</b><br>
The `agent` (simple_agent graph) is a simple linear flow with conditional tool execution.<br>
The decision logic is about to check if the last message contains tool calls and ends immediately when no tools are requested.<br>
It follows the following flow : `agent` → `action` (if tool calls needed) → back to `agent` → `END` (if no tool calls)

The `agent_helpful` (agent_with_helpfulness graph) has a more complex flow with helpfulness evaluation loop.<br>
The decision logic is about to evaluate the helpfulness of the response relative to the initial query and only ends when helpfulness is deemed satisfactory or loop limit is reached (more than 10 messages).<br>
It follows the following flow : `agent` → `action` (if tool calls needed) → `helpfulness` (if no tool calls) → decision point


<b>Where the helpfulness evaluator fits:</b><br>
The helpfulness evaluator (`helpfulness_node`) is positioned as a **post-response evaluation step** that runs after the agent generates a response (when no tools are needed).<br> 
It's inserted between the agent's response and the final termination decision.


<b>Routing conditions | Route back to agent (`continue`):</b>
- When helpfulness evaluation returns "N" (not helpful)
- When the response is deemed insufficient or unhelpful relative to the initial query
- This creates a feedback loop where the agent gets another chance to improve its response

<b>Routing conditions | Terminate (`end`):</b><br>
- When helpfulness evaluation returns "Y" (helpful)
- When the response is deemed satisfactory and helpful relative to the initial query
- **Safety condition:** When the loop limit is exceeded (more than 10 messages), it automatically terminates to prevent infinite loops

<b>Key differences:</b>
- The `agent_helpful` adds an additional quality control layer that the simple `agent` lacks
- It implements a self-improvement mechanism where the agent can iterate on its responses until they meet helpfulness criteria
- It includes a safety mechanism (loop limit) to prevent infinite loops
- The helpfulness evaluator uses a separate model (`gpt-4.1-mini`) to objectively assess response quality