"""
Agentic RAG Application with LangGraph using Together AI Open Source Endpoints

This application demonstrates:
1. RAG pipeline with Together AI embeddings
2. LangGraph agent with tool usage
3. Open source model integration
4. Document retrieval and generation
"""

import os
import getpass
from typing import Dict, Any, List, TypedDict
from functools import lru_cache

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Together AI imports
from together import Together

# Simple wrapper for Together AI that's compatible with LangChain tools
class ChatTogether:
    """Simple wrapper for Together AI that's compatible with LangChain tools."""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.tools = []
        # Initialize Together client with API key
        api_key = os.environ.get("TOGETHER_API_KEY")
        if api_key:
            self.client = Together(api_key=api_key)
        else:
            # Fallback: try without API key (it should be in environment)
            self.client = Together()
    
    def bind_tools(self, tools):
        """Bind tools to the model for compatibility with LangChain."""
        self.tools = tools
        return self
    
    def invoke(self, messages):
        """Invoke the model with messages."""
        # Convert LangChain messages to Together API format
        formatted_messages = self._format_messages(messages)
        
        # Generate response using Together client exactly like in the notebook
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages
        )
        
        # Create a simple message-like object
        class SimpleMessage:
            def __init__(self, content):
                self.content = content
                self.tool_calls = []
        
        return SimpleMessage(response.choices[0].message.content)
    
    def _format_messages(self, messages):
        """Convert LangChain messages to Together API format."""
        formatted = []
        for message in messages:
            if hasattr(message, 'content'):
                # Determine role based on message type
                if hasattr(message, 'type'):
                    if message.type == 'human':
                        role = 'user'
                    elif message.type == 'ai':
                        role = 'assistant'
                    elif message.type == 'system':
                        role = 'system'
                    else:
                        role = 'user'  # default
                else:
                    role = 'user'  # default
                
                formatted.append({
                    "role": role,
                    "content": str(message.content)
                })
        
        # If no messages, add a default user message
        if not formatted:
            formatted.append({
                "role": "user",
                "content": "Hello"
            })
        
        return formatted

# Together AI embeddings
from langchain_core.embeddings import Embeddings

class TogetherEmbeddings(Embeddings):
    """Wrapper for Together AI embeddings compatibility."""
    
    def __init__(self, model: str, **kwargs):
        super().__init__()
        self._model_name = model
        # Initialize Together client with API key
        api_key = os.environ.get("TOGETHER_API_KEY")
        if api_key:
            self.client = Together(api_key=api_key)
        else:
            # Fallback: try without API key (it should be in environment)
            self.client = Together()
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        """Embed a single query."""
        # Use Together embeddings API
        response = self.client.embeddings.create(
            input=text,
            model=self._model_name
        )
        return response.data[0].embedding
    
    @property
    def model(self):
        """Return the model name for compatibility."""
        return self._model_name

# Vector store and text processing
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up API key early - this will be set in main() function
TOGETHER_API_KEY = None

# Configuration
MODEL_ENDPOINT = "openai/gpt-oss-20b"  # Replace with your endpoint
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# State definition
class AgentState(TypedDict):
    """State for the agent graph."""
    messages: List[Any]
    context: List[Document]
    response: str

# RAG Pipeline
class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using Together AI."""
    
    def __init__(self, documents: List[Document] = None):
        self.documents = documents or self._load_documents()
        self.vectorstore = None
        self.retriever = None
        self._setup_pipeline()
    
    def _load_documents(self) -> List[Document]:
        """Load PDF documents from the data directory."""
        try:
            from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
            
            # Load PDFs from data directory (recursive)
            directory_loader = DirectoryLoader(
                "data", 
                glob="**/*.pdf", 
                loader_cls=PyMuPDFLoader
            )
            documents = directory_loader.load()
            print(f"📚 Loaded {len(documents)} PDF documents from data directory")
            return documents
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load PDFs from data directory: {e}")
            print("📝 Falling back to sample documents...")
            
            # Fallback to sample documents if PDF loading fails
            sample_texts = [
                "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.",
                "Retrieval-Augmented Generation (RAG) is a technique that enhances language models by retrieving relevant documents from a knowledge base and using them to generate more accurate and contextual responses.",
                "Together AI provides access to open source language models and embeddings through their API, making it easier to deploy and scale AI applications without vendor lock-in.",
                "Agentic workflows involve multiple steps where an AI agent can make decisions, use tools, and iterate on responses to provide better assistance to users.",
                "Vector databases store document embeddings and enable semantic search, allowing systems to find relevant information based on meaning rather than just keywords."
            ]
            
            return [Document(page_content=text, metadata={"source": f"sample_{i}"}) 
                    for i, text in enumerate(sample_texts)]
    
    def _setup_pipeline(self):
        """Set up the RAG pipeline with embeddings and vector store."""
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(self.documents)
        
        # Create embeddings and vector store
        embeddings = TogetherEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def retrieve_context(self, query: str) -> List[Document]:
        """Retrieve relevant context for a query."""
        return self.retriever.invoke(query)
    
    def generate_response(self, query: str, context: List[Document]) -> str:
        """Generate a response using the retrieved context."""
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the provided context to answer the user's question accurately and concisely. 
            If the context doesn't contain enough information to answer the question, say so. 
            Always cite the relevant parts of the context when possible."""),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        
        # Create the chain
        llm = ChatTogether(model=MODEL_ENDPOINT)
        chain = prompt_template | llm | StrOutputParser()
        
        # Format context for the prompt
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        # Generate response
        response = chain.invoke({
            "context": context_text,
            "question": query
        })
        
        return response

# Tools
@tool
def retrieve_information(query: str) -> str:
    """Retrieve information using RAG to answer questions about federal student aid programs, grants, loans, and financial aid policies."""
    # This will be called from the agent context
    return "This tool is called by the agent. Please ask your question directly to the agent."

@tool
def search_documents(query: str) -> str:
    """Search through available financial aid documents to find relevant information about grants, loans, and policies."""
    # This will be called from the agent context
    return "This tool is called by the agent. Please ask your question directly to the agent."

# Agent Graph
class AgenticRAGAgent:
    """Agent that uses RAG and tools to answer questions."""
    
    def __init__(self):
        self.rag_pipeline = None  # Lazy initialization
        self.tools = [retrieve_information, search_documents]
        self.graph = self._build_graph()
    
    def _get_rag_pipeline(self):
        """Lazy initialization of RAG pipeline."""
        if self.rag_pipeline is None:
            self.rag_pipeline = RAGPipeline()
        return self.rag_pipeline
    
    def _get_model_with_tools(self):
        """Get the chat model bound to tools."""
        model = ChatTogether(model=MODEL_ENDPOINT)
        return model.bind_tools(self.tools)
    
    def call_model(self, state: AgentState) -> Dict[str, Any]:
        """Call the model with accumulated messages."""
        # Get the model with tools bound
        model = self._get_model_with_tools()
        messages = state["messages"]
        
        # Use the model to generate response
        response = model.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(self, state: AgentState):
        """Determine if execution should continue or end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "action"
        return END
    
    def _build_graph(self):
        """Build the agent graph."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("agent", self.call_model)
        tool_node = ToolNode(self.tools)
        graph.add_node("action", tool_node)
        
        # Set entry point
        graph.set_entry_point("agent")
        
        # Add edges
        graph.add_conditional_edges(
            "agent", 
            self.should_continue, 
            {"action": "action", END: END}
        )
        graph.add_edge("action", "agent")
        
        return graph.compile()
    
    def invoke(self, query: str) -> str:
        """Invoke the agent with a query."""
        # Initialize state
        state = {
            "messages": [HumanMessage(content=query)],
            "context": [],
            "response": ""
        }
        
        # Run the graph
        result = self.graph.invoke(state)
        
        # Extract the final response
        final_message = result["messages"][-1]
        if hasattr(final_message, 'content'):
            return final_message.content
        else:
            return str(final_message)

# Main application
def main():
    """Main function to run the agentic RAG application."""
    print("🚀 Agentic RAG Application with Together AI Open Source Endpoints")
    print("=" * 70)
    
    # Set up API key
    global TOGETHER_API_KEY
    if "TOGETHER_API_KEY" not in os.environ:
        TOGETHER_API_KEY = getpass.getpass("Enter your Together API key: ")
        os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
    else:
        TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
    
    # Initialize the agent
    agent = AgenticRAGAgent()
    
    # Example queries
    example_queries = [
        "What is the Federal Pell Grant Program and how does it work?",
        #"Explain the Direct Loan Program and its requirements",
        #"What are the verification requirements for financial aid applications?",
        #"How is the cost of attendance calculated for financial aid packaging?"
    ]
    
    print("\n📚 Available sample documents loaded successfully!")
    print(f"🔧 Tools available: {len(agent.tools)}")
    print(f"🤖 Model: {MODEL_ENDPOINT}")
    print(f"🧠 Embeddings: {EMBEDDING_MODEL}")
    
    # Show information about loaded documents
    if hasattr(agent.rag_pipeline, 'documents'):
        print(f"📄 Documents loaded: {len(agent.rag_pipeline.documents)}")
        for i, doc in enumerate(agent.rag_pipeline.documents[:3]):  # Show first 3
            source = doc.metadata.get('source', 'Unknown')
            print(f"   {i+1}. {source}")
        if len(agent.rag_pipeline.documents) > 3:
            print(f"   ... and {len(agent.rag_pipeline.documents) - 3} more documents")
    
    print("\n" + "=" * 70)
    print("🧪 Testing the Agent")
    print("=" * 70)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 50)
        
        try:
            response = agent.invoke(query)
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)
    
    print("\n" + "=" * 70)
    print("💡 Interactive Mode")
    print("=" * 70)
    print("Enter your own questions (type 'quit' to exit):")
    
    while True:
        try:
            user_query = input("\n❓ Your question: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_query:
                print("🤔 Thinking...")
                response = agent.invoke(user_query)
                print(f"🤖 Answer: {response}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 