# ##### 🏗️ XTALLET CODE
# Import Dependencies
import os
import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# Create a Simple Graph - allowing the system to dynamically fetch Arxiv papers
# Load Arxiv Class
from langchain_community.tools.arxiv.tool import ArxivQueryRun
tool_belt = [ArxivQueryRun()]

# Load the OpenAI model
from langchain_openai import ChatOpenAI

# Put on the toolbelt
def get_model():
    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    return model.bind_tools(tool_belt)

# Start preparing the Graph
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import BaseMessage
class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

# Create the Graph nodes
from langgraph.prebuilt import ToolNode

def call_model(state):
  messages = state["messages"]
  model = get_model()
  response = model.invoke(messages)
  return {"messages" : [response]}

tool_node = ToolNode(tool_belt)

# Add nodes and set entrypoint
from langgraph.graph import StateGraph, END

uncompiled_graph = StateGraph(AgentState)

uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", tool_node)

uncompiled_graph.set_entry_point("agent")

# Add the conditional edge
def should_continue(state):
  last_message = state["messages"][-1]

  if last_message.tool_calls:
    return "action"

  return END

uncompiled_graph.add_conditional_edges(
    "agent",
    should_continue
)

# Create the last edge and compile the Graph
uncompiled_graph.add_edge("action", "agent")
agent_graph = uncompiled_graph.compile()

# Let's try the Graph with an example
from langchain_core.messages import HumanMessage

inputs = {"messages" : [HumanMessage(content="What is the maximum loan amount? Please look at Arxiv for references")]}
async def run_agent_graph():
    async for chunk in agent_graph.astream(inputs, stream_mode="updates"):
        print(f'chunk : {chunk}')
        for node, values in chunk.items():
            print(f'node : {node} and values : {values}')
            print(f"Receiving update from node: '{node}'")
            print(values["messages"])
            print("\n\n")

# Helper Functions - Wall Imports
from typing import Any, Callable, List, Optional, TypedDict, Union

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph

import asyncio
import time
from functools import wraps

def retry_on_api_error(max_retries=3, delay=2):
    """Decorador para reintentar en caso de errores de API."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "APIError" in str(e) or "server had an error" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"⚠️  API Error (attempt {attempt + 1}/{max_retries}), retrying in {delay} seconds...")
                            time.sleep(delay)
                            continue
                    raise e
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Helper Functions - Agent Node Helper
@retry_on_api_error(max_retries=3, delay=2)
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Helper Functions - Agent Creation Helper Function
def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> AgentExecutor:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += ("\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason!")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"), # XTALLET - Historial de conversciones
            MessagesPlaceholder(variable_name="agent_scratchpad"), # XTALLET - Espacio de trabajo del agente
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Helper Functions - Supervisor Helper Function
def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members # XTALLET - Opciones de routing - Ej : ["FINISH", "researcher", "writer", "reviewer"]
    
    # Crear una herramienta para el routing
    from langchain_core.tools import tool
    
    @tool
    def route(next_worker: str) -> str:
        """Select the next role to act."""
        if next_worker in options:
            return next_worker
        else:
            return "FINISH"
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    
    def extract_route_result(response):
        """Extract the route result from the tool call."""
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return {"next": response.tool_calls[0]['args']['next_worker']}
        return {"next": "FINISH"}
    
    return (
        prompt
        | llm.bind_tools([route])
        | extract_route_result
    )

# Start Preparing the Research Team
import functools
import operator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
import functools

class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str

# Start Preparing the Research Team - Load LLM
def get_research_llm():
    return ChatOpenAI(model="gpt-4o-mini")

# Research Team : Search Agent
arxiv_tool = ArxivQueryRun()
def get_search_agent():
    llm = get_research_llm()
    search_agent = create_agent(
        llm,
        [arxiv_tool],
        "You are a research assistant who can search for up-to-date info using the arxiv search engine. "
        "Your job is to gather factual information about research papers and provide detailed summaries. "
        "Do NOT format content as LinkedIn posts or add emojis/hashtags. "
        "Provide clear, factual research summaries that can be used by other teams to create content.",
    )
    return search_agent

def search_node(state):
    agent = get_search_agent()
    return agent_node(state, agent, "ArxivSearch")

# Research Team : Supervisor
def get_supervisor_agent():
    llm = get_research_llm()
    supervisor_agent = create_team_supervisor(
        llm,
        ("You are a supervisor tasked with managing a conversation between the"
        " following workers:  ArxivSearch. Given the following user request,"
        " determine the subject to be researched and respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. "
        " You should never ask your team to do anything beyond research. They are not required to write content or posts."
        " You should only pass tasks to workers that are specifically research focused."
        " IMPORTANT: Only gather research information, do NOT generate LinkedIn posts or formatted content."
        " When finished with research, respond with FINISH."),
        ["ArxivSearch"],
    )
    return supervisor_agent

# Research Team - Graph Creation
def get_research_graph():
    research_graph = StateGraph(ResearchTeamState)
    research_graph.add_node("ArxivSearch", search_node)
    research_graph.add_node("supervisor", get_supervisor_agent())
    research_graph.add_edge("ArxivSearch", "supervisor")
    research_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"ArxivSearch": "ArxivSearch", "FINISH": END},
    )
    research_graph.set_entry_point("supervisor")
    return research_graph.compile()

# Display Research Graph
import nest_asyncio
nest_asyncio.apply()

from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# display(
#     Image(
#         compiled_research_graph.get_graph().draw_mermaid_png(
#             curve_style=CurveStyle.LINEAR,
#             node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
#             wrap_label_n_words=9,
#             output_file_path=None,
#             #draw_method=MermaidDrawMethod.PYPPETEER,
#             draw_method=MermaidDrawMethod.API,
#             background_color="white",
#             padding=10,
#         )
#     )
# )

# Start Preparing the Research Team - Example usage (comentado en el notebook)
# for s in research_chain.stream(
#     "How are large language models being adapted for low-resource languages in natural language processing?", {"recursion_limit": 100}
# ):
#     if "__end__" not in s:
#         print(s)
#         print("---")

# Writin Team - Graph Preparation
# Tool Creation
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional
from typing_extensions import TypedDict
import uuid
import os
from langchain_core.tools import tool

os.makedirs('./content/advanced_build/data', exist_ok=True)

def create_random_subdirectory():
    random_id = str(uuid.uuid4())[:8]  # Use first 8 characters of a UUID
    subdirectory_path = os.path.join('./content/advanced_build/data', random_id)
    os.makedirs(subdirectory_path, exist_ok=True)
    return subdirectory_path

WORKING_DIRECTORY = Path(create_random_subdirectory())

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"

@tool
def read_document(
    file_name: Annotated[str, "File path to save the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])

@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"

@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ] = {},
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)
    return f"Document edited and saved to {file_name}"

# Document Writing State
import operator
from pathlib import Path

class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str

# Document Writing Prelude Function
def prelude(state):
    written_files = []
    if not WORKING_DIRECTORY.exists():
        WORKING_DIRECTORY.mkdir()
    try:
        written_files = [
            f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
        ]
    except:
        pass
    if not written_files:
        return {**state, "current_files": "No files written."}
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n"
        + "\n".join([f" - {f}" for f in written_files]),
    }

# Document Writing Node Creation
def get_doc_writer_agent():
    llm = get_research_llm()
    doc_writer_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        ("You are a technical writing expert specialized in Machine Learning research.\n"
        "Below are files currently in your directory:\n{current_files}"),
    )
    context_aware_doc_writer_agent = prelude | doc_writer_agent
    return context_aware_doc_writer_agent

def doc_writing_node(state):
    agent = get_doc_writer_agent()
    return agent_node(state, agent, "DocWriter")

def get_note_taking_agent():
    llm = get_research_llm()
    note_taking_agent = create_agent(
        llm,
        [create_outline, read_document],
        ("You are an assistant specializing in research note-taking."
         "Your task is to read the Machine Learning paper and extract detailed"
         "supporting material that can help other agents write summaries and posts.\n{current_files}"),
    )
    context_aware_note_taking_agent = prelude | note_taking_agent
    return context_aware_note_taking_agent

def note_taking_node(state):
    agent = get_note_taking_agent()
    return agent_node(state, agent, "NoteTaker")

def get_copy_editor_agent():
    llm = get_research_llm()
    copy_editor_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        ("You are a professional copy editor specializing in LinkedIn content." 
         "Your task is to refine the provided draft text into a well-written," 
         "engaging post appropriate for a professional audience."
         "Make sure the tone is clear, confident, and informative without sounding too academic or promotional."
         "IMPORTANT: After creating the final LinkedIn post, include the complete post content"
         " directly in your response with emojis, hashtags, and engaging language."
         "Below are files currently in your directory:\n{current_files}"),
    )
    context_aware_copy_editor_agent = prelude | copy_editor_agent
    return context_aware_copy_editor_agent

def copy_editing_node(state):
    agent = get_copy_editor_agent()
    return agent_node(state, agent, "CopyEditor")

def get_empathy_editor_agent():
    llm = get_research_llm()
    empathy_editor_agent = create_agent(
        llm,
        [write_document, edit_document, read_document],
        ("You are an empathy and tone advisor for professional communications." 
         "Your task is to review the refined LinkedIn post and ensure that it feels approachable," 
         "relatable, and aligned with a human-centered tone."
        "Below are files currently in your directory:\n{current_files}"),
    )
    empathy_editor_agent = prelude | empathy_editor_agent
    return empathy_editor_agent

def empathy_node(state):
    agent = get_empathy_editor_agent()
    return agent_node(state, agent, "EmpathyEditor")

def get_doc_writing_supervisor():
    llm = get_research_llm()
    doc_writing_supervisor = create_team_supervisor(
        llm,
        ("You are a supervisor tasked with managing a conversation between the"
        " following workers: {team_members}. You should always verify the technical"
        " contents after any edits are made. "
        "Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. "
        "IMPORTANT: After the team has created the LinkedIn post, you must generate"
        " the final LinkedIn post content directly in your response, not just save it to a file."
        " Include emojis, hashtags, and engaging language appropriate for LinkedIn."
        " When the team is finished, respond with FINISH."),
        ["DocWriter", "NoteTaker", "EmpathyEditor", "CopyEditor"],
    )
    return doc_writing_supervisor

def get_authoring_graph():
    authoring_graph = StateGraph(DocWritingState)
    authoring_graph.add_node("DocWriter", doc_writing_node)
    authoring_graph.add_node("NoteTaker", note_taking_node)
    authoring_graph.add_node("CopyEditor", copy_editing_node)
    authoring_graph.add_node("EmpathyEditor", empathy_node)
    authoring_graph.add_node("supervisor", get_doc_writing_supervisor())

    authoring_graph.add_edge("DocWriter", "supervisor")
    authoring_graph.add_edge("NoteTaker", "supervisor")
    authoring_graph.add_edge("CopyEditor", "supervisor")
    authoring_graph.add_edge("EmpathyEditor", "supervisor")

    authoring_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "DocWriter": "DocWriter",
            "NoteTaker": "NoteTaker",
            "CopyEditor" : "CopyEditor",
            "EmpathyEditor" : "EmpathyEditor",
            "FINISH": END,
        },
    )

    authoring_graph.set_entry_point("supervisor")
    return authoring_graph.compile()

# Display the Graph
from IPython.display import Image, display

# display(
#     Image(
#         compiled_authoring_graph.get_graph().draw_mermaid_png(
#             curve_style=CurveStyle.LINEAR,
#             node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
#             wrap_label_n_words=9,
#             output_file_path=None,
#             #draw_method=MermaidDrawMethod.PYPPETEER,
#             draw_method=MermaidDrawMethod.API,
#             background_color="white",
#             padding=10,
#         )
#     )
# )

def enter_research_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results

def get_research_chain():
    research_chain = enter_research_chain | get_research_graph()
    return research_chain

def enter_authoring_chain(message: str, members: List[str]):
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results

def get_authoring_chain():
    authoring_chain = (
        functools.partial(enter_authoring_chain, members=["DocWriter", "NoteTaker", "EmpathyEditor", "CopyEditor"])
        | get_authoring_graph()
    )
    return authoring_chain

def get_authoring_chain_with_output():
    """Versión modificada que captura y devuelve el contenido generado."""
    def capture_output(response):
        """Captura el contenido generado y lo devuelve como respuesta."""
        if "messages" in response and response["messages"]:
            # Buscar el último mensaje que contenga el contenido del post
            for msg in reversed(response["messages"]):
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    # Verificar si el contenido parece un post de LinkedIn completo
                    # Debe tener emojis Y hashtags Y formato de LinkedIn, NO ser solo investigación
                    has_emojis = any(emoji in content for emoji in ["🚀", "💡", "📊", "🌟", "✅", "🤖"])
                    has_hashtags = "#" in content
                    has_linkedin_format = any(keyword in content.lower() for keyword in ["linkedin", "post", "share", "thoughts", "game changer", "innovation", "let's discuss", "what are your thoughts"])
                    
                    # Mejorar la detección: si tiene emojis, hashtags y formato LinkedIn, es el post final
                    # incluso si menciona investigación (porque es normal en posts de LinkedIn sobre papers)
                    if has_emojis and has_hashtags and has_linkedin_format and "saved as" not in content.lower():
                        return {"messages": [HumanMessage(content=content)]}
            
            # Si no encontramos un post específico, buscar contenido que parezca un post de LinkedIn
            # incluso si no cumple todos los criterios, pero que no sea un mensaje sobre archivos
            for msg in reversed(response["messages"]):
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    # Si el contenido no es sobre archivos guardados y tiene algún formato de LinkedIn
                    if ("saved as" not in content.lower() and 
                        "document" not in content.lower() and 
                        "file" not in content.lower() and
                        any(keyword in content.lower() for keyword in ["linkedin", "post", "share", "thoughts", "game changer", "innovation", "let's discuss", "what are your thoughts"])):
                        return {"messages": [HumanMessage(content=content)]}
            
            # Si no encontramos un post específico, buscar cualquier contenido que no sea sobre archivos
            for msg in reversed(response["messages"]):
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    # Si el contenido no es sobre archivos guardados y no es investigación
                    if ("saved as" not in content.lower() and 
                        "document" not in content.lower() and 
                        "file" not in content.lower() and
                        "outline" not in content.lower() and
                        not any(keyword in content.lower() for keyword in ["paper", "research", "study", "authors", "findings", "summary of findings", "researchers have introduced"])):
                        return {"messages": [HumanMessage(content=content)]}
            
            # Si no encontramos un post específico, devolver el último mensaje
            last_message = response["messages"][-1]
            if hasattr(last_message, 'content'):
                return {"messages": [HumanMessage(content=last_message.content)]}
        return response
    
    authoring_chain = (
        functools.partial(enter_authoring_chain, members=["DocWriter", "NoteTaker", "EmpathyEditor", "CopyEditor"])
        | get_authoring_graph()
        | capture_output
    )
    return authoring_chain

# Let's test it
# for s in authoring_chain.stream(
#     "Write a professional LinkedIn-style post based on a Machine Learning paper that introduces a novel approach to improving model interpretability for non-technical stakeholders.",
#     {"recursion_limit": 100},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("---")

# Meta-Supervisor and Full Graph
# Create the Supervidor Node
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI

def get_supervisor_node():
    llm = get_research_llm()
    supervisor_node = create_team_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following teams: {team_members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. "
        " IMPORTANT RULES:"
        " 1. First, send to Research team to gather information"
        " 2. Then send to Response team to create the LinkedIn post"
        " 3. After Response team has generated the final LinkedIn post content, you MUST respond with FINISH"
        " 4. Do not continue routing back to Research team after Response team has finished"
        " 5. If Response team returns content with emojis, hashtags, or LinkedIn-style formatting, consider it complete"
        " The flow should be: Research team -> Response team -> FINISH.",
        ["Research team", "Response team"],
    )
    return supervisor_node

# Create the State
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

def get_last_message(state: State) -> str:
    """Extract the content of the last message from the state."""
    if "messages" in state and state["messages"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content'):
            return last_message.content
        else:
            return str(last_message)
    else:
        return "No message available"

def join_graph(response: dict):
    # Extraer el último mensaje de la respuesta del grafo
    if "messages" in response and response["messages"]:
        return {"messages": [response["messages"][-1]]}
    else:
        # Si no hay mensajes, crear un mensaje vacío
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="No response generated")]}

# Create the Super Graph
def get_super_graph():
    super_graph = StateGraph(State)
    super_graph.add_node("Research team", get_last_message | get_research_chain() | join_graph)
    super_graph.add_node("Response team", get_last_message | get_authoring_chain_with_output() | join_graph)
    super_graph.add_node("supervisor", get_supervisor_node())

    # Create the edges
    super_graph.add_edge("Research team", "supervisor")
    super_graph.add_edge("Response team", "supervisor")
    super_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "Response team": "Response team",
            "Research team": "Research team",
            "FINISH": END,
        },
    )
    super_graph.set_entry_point("supervisor")
    return super_graph.compile()

# Display Super Graph
# display(
#     Image(
#         compiled_super_graph.get_graph().draw_mermaid_png(
#             curve_style=CurveStyle.LINEAR,
#             node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
#             wrap_label_n_words=9,
#             output_file_path=None,
#             #draw_method=MermaidDrawMethod.PYPPETEER,
#             draw_method=MermaidDrawMethod.API,
#             background_color="white",
#             padding=10,
#         )
#     )
# )

def get_simple_research_graph():
    """Grafo simplificado solo con research team para pruebas rápidas."""
    research_graph = StateGraph(ResearchTeamState)
    research_graph.add_node("ArxivSearch", search_node)
    research_graph.add_node("supervisor", get_supervisor_agent())
    research_graph.add_edge("ArxivSearch", "supervisor")
    research_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"ArxivSearch": "ArxivSearch", "FINISH": END},
    )
    research_graph.set_entry_point("supervisor")
    return research_graph.compile()

def get_compiled_super_graph():
    """Función principal que devuelve el grafo completo."""
    return get_super_graph()

def get_simple_graph():
    """Función para obtener un grafo simplificado para pruebas."""
    return get_simple_research_graph()

# for s in compiled_super_graph.stream(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Write a LinkedIn post based on a Machine Learning research paper about optimizing transformer models for efficiency in low-resource settings. First consult the research team. Then make sure you consult the response team. Verify factual accuracy and LinkedIn style fit, and write the final version to disk."
#             )
#         ],
#     },
#     {"recursion_limit": 30},
# ):
#     if "__end__" not in s:
#         print(s)
#         print("---")

# SAMPLE POST :
#
# 🚀 Exciting advancements in Machine Learning! 🚀
#
# In a world increasingly reliant on machine learning, especially in natural language processing, optimizing transformer models for efficiency in low-resource settings has become a critical focus. 🌍✨
#
# A recent study presents a novel approach that emphasizes the significant impact of pre-training transformer models on high-resource data before fine-tuning them for low-resource tasks. This method was shown to enhance performance dramatically—from a BLEU score of just 3.5 to 7.1 in low-resource language translation tasks, exemplifying the power of transferable knowledge.
#
# 💡 Key takeaways from the research:
# 1. **Transfer Learning**: Pre-training on abundant data can drastically improve accuracy in low-data scenarios.
# 2. **Cross-linguistic Benefits**: Interestingly, pre-training on languages different from the task at hand still yielded notable improvements, suggesting a shared learning mechanism in transformers.
# 3. **Efficiency Innovations**: Continued work in optimizing dataflow mapping in transformer architectures promises to boost computational efficiency, making advanced language processing accessible even in resource-constrained environments.
#
# These advancements are paving the way for more inclusive AI technologies that cater to diverse linguistic needs. As we strive for global understanding, it's essential to leverage every bit of knowledge to bridge the gaps.
#
# Let's harness the power of AI to create a future where language barriers are a thing of the past! 🤝🌐
#
# #MachineLearning #AI #Transformers #NaturalLanguageProcessing #Research #DataScience #Inclusion #Innovation 