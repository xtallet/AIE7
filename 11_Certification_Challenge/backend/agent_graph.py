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
    ("system", """You are a helpful AI assistant that answers questions concisely and directly.

IMPORTANT INSTRUCTIONS:
- Answer questions directly and concisely using the provided context.
- If the context contains the answer, provide it briefly and clearly.
- If the context is insufficient, use search tools to find current information.
- After using search tools 1-2 times, provide a concise final answer and STOP.
- Be direct and to the point - avoid lengthy explanations unless specifically requested.
- If search tools return errors, provide the best answer you can with available information.

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
    if iteration_count >= 3:  # Stop after 3 tool usage iterations
        print(f"DEBUG - Stopping due to iteration limit ({iteration_count})")
        return "end"
    
    # If the last message has content (not just tool calls), check if it's a final answer
    if hasattr(last_message, 'content') and last_message.content and last_message.content.strip():
        content = last_message.content.lower()
        
        # Check for final answer indicators
        final_answer_indicators = [
            "based on the search results",
            "according to the information found",
            "the current weather",
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
            "the umr for your policy is",
            "the policy number is",
            "the coverage limit is",
            "the premium is",
            "the deductible is"
        ]
        
        if any(indicator in content for indicator in final_answer_indicators):
            print(f"DEBUG - Final answer detected: {content[:100]}...")
            return "end"
        
        # If content is substantial and no tool calls, it's likely a final answer
        if len(content) > 20 and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
            print(f"DEBUG - Substantial answer without tool calls detected: {content[:50]}...")
            return "end"
        
        # If we have a direct answer (like "The UMR is X"), stop
        if any(phrase in content for phrase in ["is ", "are ", "was ", "were "]) and len(content) < 100:
            print(f"DEBUG - Direct answer detected: {content[:50]}...")
            return "end"
    
    # If we have tool calls, continue to action
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "action"
    
    # Default: continue to agent
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
    print(f"DEBUG - Model bound with tools: {len(model_with_tools.tools) if hasattr(model_with_tools, 'tools') else 'No tools'}")
    
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
    context_docs = state.get("context", [])
    docs_content = "\n\n".join(doc.page_content for doc in context_docs)
    
    # Check if this is a document-specific question (like UMR, policy number, etc.)
    question_lower = question_msg.content.lower()
    document_specific_keywords = ['umr', 'policy number', 'policy no', 'policy #', 'coverage', 'limit', 'premium', 'deductible']
    is_document_question = any(keyword in question_lower for keyword in document_specific_keywords)
    
    if is_document_question:
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
            question=question_msg.content,
            context=docs_content
        )
    else:
        # Use the general prompt for other questions
        messages = RAG_PROMPT.format_messages(
            question=question_msg.content,
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

    return {
        "messages": [response],
        "context": context_docs
    }

# --- Main execution ---
async def run_agentic_rag(question: str, pdf_path: str, openai_api_key: str, tavily_api_key: str) -> dict:
    vectorstore = process_pdf_and_build_vectorstore(pdf_path, openai_api_key)
    from langchain_core.messages import HumanMessage
    graph = build_agentic_graph(vectorstore, openai_api_key, tavily_api_key)
    inputs = {"messages": [HumanMessage(content=question)]}
    
    result = None
    rag_context = []  # Store RAG context separately
    
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        for node, values in chunk.items():
            print(f"DEBUG - Node executed: {node}")
            
            if node == "pre_rag_context":
                # Store the context from RAG retrieval
                rag_context = values.get("context", [])
                print(f"DEBUG - RAG context retrieved: {len(rag_context)} documents")
            
            if node == "agent":
                last_msg = values["messages"][-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    # Check if this was triggered by RAG tool or external tool
                    source = "rag"  # Default to RAG
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
                    
                    result = {
                        "answer": last_msg.content,
                        "source": source,
                        "context": context_to_show,
                        "raw_output": last_msg.dict() if hasattr(last_msg, "dict") else str(last_msg)
                    }
            
            if node == "action":
                print("DEBUG - Action node executed - using external tools!")
                tool_msgs = values["messages"]
                
                # Check what tool was used
                last_tool_call = None
                for msg in tool_msgs:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        last_tool_call = msg.tool_calls[-1]
                        break
                
                # Determine source based on tool used
                source = "tool"  # Default for external tools
                if last_tool_call and 'tavily' in last_tool_call.get('name', ''):
                    source = "tavily"
                    print("DEBUG - Tavily tool was used")
                elif last_tool_call and 'arxiv' in last_tool_call.get('name', ''):
                    source = "arxiv"
                    print("DEBUG - Arxiv tool was used")
                
                for msg in tool_msgs:
                    if hasattr(msg, "content") and msg.content:
                        result = {
                            "answer": msg.content,
                            "source": source,
                            "context": context_to_show if source == "rag" else None,
                            "raw_output": msg.dict() if hasattr(msg, "dict") else str(msg)
                        }
            
            if node == "end":
                print("DEBUG - End node executed - final answer reached!")
                # Get the last message from the state
                last_msg = values["messages"][-1] if values.get("messages") else None
                if last_msg and hasattr(last_msg, "content") and last_msg.content:
                    # Determine source based on what tools were used
                    source = "rag"  # Default
                    if any('tavily' in str(msg) for msg in values.get("messages", [])):
                        source = "tavily"
                    elif any('arxiv' in str(msg) for msg in values.get("messages", [])):
                        source = "arxiv"
                    
                    result = {
                        "answer": last_msg.content,
                        "source": source,
                        "context": context_to_show if source == "rag" else None,
                        "raw_output": last_msg.dict() if hasattr(last_msg, "dict") else str(last_msg)
                    }
    
    return result 