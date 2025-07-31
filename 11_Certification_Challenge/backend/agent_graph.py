import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Annotated, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate

# --- PDF and Vectorstore ---
def process_pdf_and_build_vectorstore(pdf_path: str, openai_api_key: str) -> object:
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)
    client = QdrantClient(':memory:')
    client.create_collection(
        collection_name='insurance_policy',
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name='insurance_policy',
        embedding=embeddings,
    )
    _ = vector_store.add_documents(documents=split_documents)
    return vector_store

# --- Toolbelt ---
def get_toolbelt(vectorstore, openai_api_key, tavily_api_key):
    # Set Tavily API key as environment variable
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    tavily_tool = TavilySearchResults(max_results=5)
    
    arxiv_tool = ArxivQueryRun()
    return [tavily_tool, arxiv_tool]

# --- Prompt ---
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant that answers questions using available information and external search tools when needed.

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE:

1. If the question asks about ANY current information, recent developments, or information that would not be in this document's time period → YOU MUST USE SEARCH TOOLS. DO NOT ANSWER FROM THE DOCUMENT ALONE.

2. If the question mentions years like 2024, 2025, or asks about "current standards", "market standards", "still standard" → YOU MUST USE SEARCH TOOLS.

3. If the question has multiple parts and one part requires current information → YOU MUST USE SEARCH TOOLS for that part.

4. YOU CANNOT claim to know current standards without searching for them.

5. When you decide to search, use the search tools immediately. DO NOT try to answer from the document first.

6. Only answer from the document if the question is about specific details that would be in the document (like UMR numbers, policy details, etc.).

DECISION FRAMEWORK:
- Document-specific questions (UMR, policy numbers, etc.) → Use context only
- Current information questions → USE SEARCH TOOLS
- Mixed questions → USE SEARCH TOOLS for current parts

Question: {question}
Context: {context}"""),
    ("human", "{question}")
])

# --- Agent State ---
from typing import TypedDict
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: List[Document]

# --- Nodes ---
def pre_rag_context(state: AgentState, vectorstore, openai_api_key):
    question = state["messages"][-1].content
    print(f'The user question is : {question}')
    
    # Create the same simple RAG graph as in insurance_rag_tool
    def retrieve(state):
        retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(state["question"])
        return {"context": retrieved_docs}
    
    def generate(state):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        # Use the same simple prompt as in the notebook
        simple_rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant who answers questions based on the provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}"""),
            ("human", "{question}")
        ])
        messages = simple_rag_prompt.format_messages(question=state["question"], context=docs_content)
        model = ChatOpenAI(model="gpt-4.1-nano", temperature=0, openai_api_key=openai_api_key)
        response = model.invoke(messages)
        return {"response": response.content}
    
    # Create simple RAG graph exactly like in the notebook
    from langgraph.graph import StateGraph, START
    from typing_extensions import TypedDict
    from langchain_core.documents import Document
    
    class State(TypedDict):
        question: str
        context: List[Document]
        response: str
    
    # Use the same approach as in the notebook
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    # Execute the graph like in the notebook
    rag_result = graph.invoke({"question": question})
    print(f'The rag result is : {rag_result}')
    
    return {
        "messages": state["messages"],
        "context": rag_result["context"]  # Esto se pasa a `agent`
    }

def call_model(state, openai_api_key):
    question_msg = state["messages"][-1]
    context_docs = state.get("context", [])
    docs_content = "\n\n".join(doc.page_content for doc in context_docs)
    messages = RAG_PROMPT.format_messages(
        question=question_msg.content,
        context=docs_content
    )
    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0, openai_api_key=openai_api_key)
    response = model.invoke(messages)
    
    # Debug: Check if response has tool calls
    print(f"DEBUG - Response content: {response.content}")
    print(f"DEBUG - Has tool calls: {hasattr(response, 'tool_calls') and response.tool_calls}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"DEBUG - Tool calls: {response.tool_calls}")
    
    return {
        "messages": [response],
        "context": context_docs
    }

def should_continue(state):
    """Determine if the agent should continue or stop."""
    last_message = state["messages"][-1]
    
    # Count iterations to prevent infinite loops
    iteration_count = len([msg for msg in state["messages"] if hasattr(msg, 'tool_calls') and msg.tool_calls])
    message_count = len(state["messages"])
    print(f"DEBUG - should_continue: iteration_count = {iteration_count}")
    print(f"DEBUG - should_continue: message_count = {message_count}")
    print(f"DEBUG - should_continue: last_message type = {type(last_message)}")
    print(f"DEBUG - should_continue: last_message has content = {hasattr(last_message, 'content') and last_message.content}")
    print(f"DEBUG - should_continue: last_message has tool_calls = {hasattr(last_message, 'tool_calls') and last_message.tool_calls}")
    
    # Stop if we've reached too many iterations or messages
    if iteration_count >= 3:  # Stop after 3 tool usage iterations
        print(f"DEBUG - Stopping due to iteration limit ({iteration_count})")
        return "end"
    
    if message_count >= 8:  # Stop if we have too many messages (prevents infinite loops)
        print(f"DEBUG - Stopping due to message count limit ({message_count})")
        return "end"
    
    # Get the original question to understand what was asked
    original_question = state["messages"][0].content if state["messages"] else ""
    question_lower = original_question.lower()
    print(f"DEBUG - should_continue: original_question = {original_question[:100]}...")
    
    # Check if this is a question that requires external search (generic approach)
    requires_external_search = any(keyword in question_lower for keyword in [
        "current", "recent", "latest", "now", "today", "modern",
        "does this", "would this", "account for", "cover",
        "since", "post-", "after", "update", "development",
        "still standard", "current standards", "market standards",
        "compare to", "how does this compare"
    ])
    
    # Also check for year patterns that indicate current information needs
    import re
    year_pattern = re.search(r'20[2-9][0-9]', question_lower)
    if year_pattern:
        requires_external_search = True
        print(f"DEBUG - should_continue: Year detected: {year_pattern.group()}")
    
    print(f"DEBUG - should_continue: requires_external_search = {requires_external_search}")
    
    # If the last message has content (not just tool calls), check if it's a final answer
    if hasattr(last_message, 'content') and last_message.content and last_message.content.strip():
        content = last_message.content.lower()
        print(f"DEBUG - should_continue: content length = {len(content)}")
        print(f"DEBUG - should_continue: content preview = {content[:100]}...")
        
        # Check for final answer indicators
        final_answer_indicators = [
            "based on the search results",
            "according to the information found",
            "the answer is",
            "here's what i found",
            "based on the latest information",
            "thank you",
            "you're welcome",
            "feel free to ask",
            "don't hesitate to ask",
            "i'm here to help",
            "if you have any",
            "if you need assistance",
            "in summary",
            "therefore",
            "conclusion"
        ]
        
        if any(indicator in content for indicator in final_answer_indicators):
            print(f"DEBUG - Final answer detected: {content[:100]}...")
            return "end"
        
        # If content is substantial and no tool calls, check if it's appropriate to stop
        if len(content) > 20 and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
            # CRITICAL: If the question requires external search but we haven't used tools yet, continue
            if requires_external_search and iteration_count == 0:
                print(f"DEBUG - Question requires external search but no tools used yet, continuing...")
                return "agent"
            
            # If we have a direct answer for document-specific questions, stop
            document_specific_keywords = ['umr', 'policy number', 'policy no', 'policy #', 'coverage', 'limit', 'premium', 'deductible']
            is_document_question = any(keyword in question_lower for keyword in document_specific_keywords)
            
            # CRITICAL: Even if it's a document question, if it mentions current/future dates, we need external info
            if is_document_question and not requires_external_search:
                print(f"DEBUG - Document-specific answer detected: {content[:50]}...")
                return "end"
            
            # For other cases, if we have substantial content, it's likely a final answer
            print(f"DEBUG - Substantial answer without tool calls detected: {content[:50]}...")
            return "end"
        
        # If we have a direct answer (like "The UMR is X"), stop
        if any(phrase in content for phrase in ["is ", "are ", "was ", "were "]) and len(content) < 100:
            print(f"DEBUG - Direct answer detected: {content[:50]}...")
            return "end"
    
    # If we have tool calls, continue to action
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"DEBUG - Tool calls detected, continuing to action")
        return "action"
    
    # Default: continue to agent
    print(f"DEBUG - Default: continuing to agent")
    return "agent"

# --- Graph ---
def build_agentic_graph(vectorstore, openai_api_key, tavily_api_key):
    tool_belt = get_toolbelt(vectorstore, openai_api_key, tavily_api_key)
    print(f"DEBUG - Tool belt created with {len(tool_belt)} tools")
    for i, tool in enumerate(tool_belt):
        print(f"DEBUG - Tool {i}: {tool.__class__.__name__}")
    
    # Create model and bind tools correctly
    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0, openai_api_key=openai_api_key)
    model_with_tools = model.bind_tools(tool_belt)
    print(f"DEBUG - Model bound with tools: {len(tool_belt)} tools")
    
    tool_node = ToolNode(tool_belt)
    def call_model_node(state):
        return call_model_with_tools(state, openai_api_key, model_with_tools)
    
    # Define end node that just returns the current state
    def end_node(state):
        return state
    
    uncompiled_graph = StateGraph(AgentState)
    uncompiled_graph.add_node("pre_rag_context", lambda state: pre_rag_context(state, vectorstore, openai_api_key))
    uncompiled_graph.add_node("agent", call_model_node)
    uncompiled_graph.add_node("action", tool_node)
    uncompiled_graph.add_node("end", end_node)
    uncompiled_graph.set_entry_point("pre_rag_context")
    uncompiled_graph.add_edge("pre_rag_context", "agent")
    uncompiled_graph.add_conditional_edges("agent", should_continue)
    uncompiled_graph.add_edge("action", "agent")
    return uncompiled_graph.compile()

def call_model_with_tools(state, openai_api_key, model_with_tools):
    question_msg = state["messages"][-1]
    print(f"DEBUG - call_model_with_tools processing message: {type(question_msg)}")
    print(f"DEBUG - call_model_with_tools message content: {question_msg.content[:200]}...")
    print(f"DEBUG - call_model_with_tools total messages in state: {len(state['messages'])}")
    
    context_docs = state.get("context", [])
    docs_content = "\n\n".join(doc.page_content for doc in context_docs)
    
    # Get the original question from the first message (HumanMessage)
    original_question = state["messages"][0].content if state["messages"] else ""
    
    # Check if this is a document-specific question (like UMR, policy number, etc.)
    question_lower = original_question.lower()
    document_specific_keywords = ['umr', 'policy number', 'policy no', 'policy #', 'coverage', 'limit', 'premium', 'deductible']
    is_document_question = any(keyword in question_lower for keyword in document_specific_keywords)
    
    # If we're processing a ToolMessage (after tool execution), use the general prompt
    if hasattr(question_msg, '__class__') and 'ToolMessage' in str(question_msg.__class__):
        is_document_question = False  # Force general prompt for tool results
        # Use a specific prompt for analyzing tool results
        tool_result_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing search results to answer a question. 

IMPORTANT:
- You have received search results from external tools.
- Analyze the search results and provide a comprehensive final answer.
- Do NOT search again - use the information you have received.
- Be direct and informative in your response.
- If the search results don't contain relevant information, state that clearly.

Original Question: {question}
Search Results: {tool_results}"""),
            ("human", "Based on the search results, answer the original question.")
        ])
        
        # Extract tool results from the ToolMessage
        tool_results = question_msg.content if hasattr(question_msg, 'content') else str(question_msg)
        
        messages = tool_result_prompt.format_messages(
            question=original_question,
            tool_results=tool_results
        )
    elif is_document_question:
        # Use a more specific prompt for document questions
        doc_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an insurance document assistant. Answer questions about insurance documents concisely and directly.

IMPORTANT:
- For specific values (UMR, policy numbers, limits, etc.), provide ONLY the value or a very brief explanation.
- Be extremely concise - one sentence if possible.
- If asked for a specific value, just state the value clearly.
- Only provide additional context if specifically requested.

Question: {question}
Context: {context}"""),
            ("human", "{question}")
        ])
        messages = doc_prompt.format_messages(
            question=original_question,
            context=docs_content
        )
    else:
        # Use the general prompt for other questions
        messages = RAG_PROMPT.format_messages(
            question=original_question,
            context=docs_content
        )
    
    # Add a system message to encourage final answers after some iterations
    iteration_count = len([msg for msg in state["messages"] if hasattr(msg, 'tool_calls') and msg.tool_calls])
    if iteration_count >= 2:  # After 2 iterations, strongly encourage stopping
        messages.insert(0, ("system", "CRITICAL: You have already searched for information. Now provide a comprehensive final answer based on what you found. Do NOT search again. If search tools returned errors, provide the best answer you can with available information."))
    elif iteration_count >= 1:  # After 1 iteration, encourage stopping
        messages.insert(0, ("system", "IMPORTANT: You have already searched for information. Now provide a comprehensive final answer based on what you found. Do not search again."))
    
    response = model_with_tools.invoke(messages)

    # Debug: Check if response has tool calls
    print(f"DEBUG - Response content: {response.content}")
    print(f"DEBUG - Has tool calls: {hasattr(response, 'tool_calls') and response.tool_calls}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"DEBUG - Tool calls: {response.tool_calls}")
        
        # Detect which tools were used
        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name', '')
            print(f"DEBUG - Tool call detected: {tool_name}")
            if 'tavily' in tool_name:
                # We need to access the global sources_used variable
                # For now, we'll add this to the state so it can be accessed later
                if 'sources_used' not in state:
                    state['sources_used'] = set()
                state['sources_used'].add("tavily")
                print("DEBUG - Tavily tool detected in call_model_with_tools")
            elif 'arxiv' in tool_name:
                if 'sources_used' not in state:
                    state['sources_used'] = set()
                state['sources_used'].add("arxiv")
                print("DEBUG - Arxiv tool detected in call_model_with_tools")

    return {
        "messages": [response],
        "context": context_docs,
        "sources_used": state.get("sources_used", set())
    }

# --- Main execution ---
async def run_agentic_rag(question: str, pdf_path: str, openai_api_key: str, tavily_api_key: str) -> dict:
    vectorstore = process_pdf_and_build_vectorstore(pdf_path, openai_api_key)
    from langchain_core.messages import HumanMessage
    graph = build_agentic_graph(vectorstore, openai_api_key, tavily_api_key)
    inputs = {"messages": [HumanMessage(content=question)]}
    
    result = None
    rag_context = []  # Store RAG context separately
    sources_used = set()  # Track which sources were used
    
    try:
        async for chunk in graph.astream(inputs, stream_mode="updates", config={"recursion_limit": 10}):
            for node, values in chunk.items():
                print(f"DEBUG - Node executed: {node}")
                
                if node == "pre_rag_context":
                    # Store the context from RAG retrieval
                    rag_context = values.get("context", [])
                    sources_used.add("rag")
                    print(f"DEBUG - RAG context retrieved: {len(rag_context)} documents")
                
                if node == "agent":
                    last_msg = values["messages"][-1]
                    print(f"DEBUG - Agent node processing message: {type(last_msg)}")
                    if hasattr(last_msg, "content") and last_msg.content:
                        print(f"DEBUG - Agent node content: {last_msg.content[:200]}...")
                        
                        # Check for tool calls in this AI message
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            for tool_call in last_msg.tool_calls:
                                tool_name = tool_call.get('name', '')
                                print(f"DEBUG - Agent made tool call: {tool_name}")
                                if 'tavily' in tool_name:
                                    sources_used.add("tavily")
                                    print("DEBUG - Tavily tool detected in agent")
                                elif 'arxiv' in tool_name:
                                    sources_used.add("arxiv")
                                    print("DEBUG - Arxiv tool detected in agent")
                        
                        # Don't set result here - wait for end node
                        # Just store context for later use
                        context_to_show = None
                        
                        # If we have RAG context, use it
                        if rag_context:
                            context_to_show = [
                                {
                                    "page": doc.metadata.get("page"),
                                    "snippet": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                                    "source": doc.metadata.get("source") or doc.metadata.get("file_path")
                                } for doc in rag_context
                            ]
                
                if node == "action":
                    print("DEBUG - Action node executed - using external tools!")
                    tool_msgs = values["messages"]
                    print(f"DEBUG - Number of tool messages: {len(tool_msgs)}")
                    
                    for i, msg in enumerate(tool_msgs):
                        print(f"DEBUG - Tool message {i}: {type(msg)}")
                        if hasattr(msg, "content") and msg.content:
                            print(f"DEBUG - Tool message {i} content: {msg.content[:200]}...")
                            # Don't set result here - wait for end node
                
                if node == "end":
                    print("DEBUG - End node executed - final answer reached!")
                    
                    # Get sources_used from state if available, otherwise use global
                    state_sources_used = values.get("sources_used", set())
                    if state_sources_used:
                        print(f"DEBUG - Sources used from state: {state_sources_used}")
                        sources_used.update(state_sources_used)
                    
                    print(f"DEBUG - Sources used: {sources_used}")
                    # Get the last message from the state
                    last_msg = values["messages"][-1] if values.get("messages") else None
                    if last_msg and hasattr(last_msg, "content") and last_msg.content:
                        # Determine source based on what tools were used
                        if "tavily" in sources_used and "rag" in sources_used:
                            source = "RAG + Tavily"
                            print("DEBUG - Combined source: RAG + Tavily")
                        elif "arxiv" in sources_used and "rag" in sources_used:
                            source = "RAG + Arxiv"
                            print("DEBUG - Combined source: RAG + Arxiv")
                        elif "tavily" in sources_used:
                            source = "tavily"
                            print("DEBUG - Source: tavily")
                        elif "arxiv" in sources_used:
                            source = "arxiv"
                            print("DEBUG - Source: arxiv")
                        else:
                            source = "rag"
                            print("DEBUG - Source: rag")
                        
                        # Prepare context for display
                        context_to_show = None
                        if "rag" in sources_used and rag_context:
                            context_to_show = [
                                {
                                    "page": doc.metadata.get("page"),
                                    "snippet": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                                    "source": doc.metadata.get("source") or doc.metadata.get("file_path")
                                } for doc in rag_context
                            ]
                        
                        result = {
                            "answer": last_msg.content,
                            "source": source,
                            "context": context_to_show,
                            "raw_output": last_msg.dict() if hasattr(last_msg, "dict") else str(last_msg)
                        }
    except Exception as e:
        print(f"DEBUG - Error in graph execution: {e}")
        # Return a fallback response
        return {
            "answer": f"Error processing the question: {str(e)}. Please try again with a simpler question.",
            "source": "error",
            "context": None,
            "raw_output": {"error": str(e)}
        }
    
    return result 