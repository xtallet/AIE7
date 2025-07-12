import os
from pathlib import Path
from typing import List, Dict, Annotated, Optional
import uuid

# LangChain & LangGraph imports
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
import functools
import operator

# --- Configuración de directorios y embeddings ---
def create_random_subdirectory():
    random_id = str(uuid.uuid4())[:8]
    subdirectory_path = os.path.join('./content/social_data', random_id)
    os.makedirs(subdirectory_path, exist_ok=True)
    return subdirectory_path

WORKING_DIRECTORY = Path(create_random_subdirectory())

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Estado del grafo ---
# Se define la estructura del estado que se irá pasando entre los nodos del grafo.
class SocialPostState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    paper_metadata: Optional[dict]
    paper_content: Optional[str]
    post_draft: Optional[str]
    verified: Optional[bool]
    style_checked: Optional[bool]

# --- Herramientas ---
@tool
def fetch_paper(
    arxiv_id: Annotated[str, "Arxiv paper ID (e.g. 1706.03762)"]
) -> Annotated[dict, "Paper metadata and content"]:
    """Fetch a Machine Learning paper from Arxiv by its ID."""
    loader = ArxivLoader(query=arxiv_id)
    docs = loader.load()
    if not docs:
        return {"error": "Paper not found"}
    doc = docs[0]
    return {
        "title": doc.metadata.get("Title", ""),
        "authors": doc.metadata.get("Authors", ""),
        "summary": doc.metadata.get("Summary", ""),
        "content": doc.page_content,
        "arxiv_id": arxiv_id,
    }

@tool
def write_social_post(
    paper_metadata: Annotated[dict, "Metadata and content of the paper"],
    platform: Annotated[str, "Social media platform (e.g. Twitter, LinkedIn)"],
) -> Annotated[str, "Draft of the social media post"]:
    """Write a social media post draft about the given paper for the specified platform."""
    title = paper_metadata.get("title", "")
    summary = paper_metadata.get("summary", "")
    authors = paper_metadata.get("authors", "")
    arxiv_id = paper_metadata.get("arxiv_id", "")
    url = f"https://arxiv.org/abs/{arxiv_id}"
    if platform.lower() == "twitter":
        post = f"New ML paper: '{title}' by {authors}. {summary[:200]}... Read more: {url} #MachineLearning #AI"
    elif platform.lower() == "linkedin":
        post = f"Excited to share a new Machine Learning paper: '{title}' by {authors}.\n\n{summary}\n\nRead the full paper: {url}"
    else:
        post = f"Check out this ML paper: '{title}' by {authors}. {summary}\n{url}"
    return post

@tool
def verify_correctness(
    paper_metadata: Annotated[dict, "Metadata and content of the paper"],
    post_draft: Annotated[str, "Draft of the social media post"],
) -> Annotated[bool, "Whether the post accurately represents the paper"]:
    """Verify that the post draft accurately represents the paper's content and findings."""
    # Aquí se simula la verificación, pero en un sistema real se usaría un LLM para comparar
    # el resumen y el post, buscando inconsistencias.
    # Por simplicidad, devolvemos True si el título aparece en el post.
    title = paper_metadata.get("title", "")
    return title in post_draft

@tool
def check_platform_style(
    post_draft: Annotated[str, "Draft of the social media post"],
    platform: Annotated[str, "Social media platform (e.g. Twitter, LinkedIn)"],
) -> Annotated[bool, "Whether the post fits the platform's style and theme"]:
    """Check if the post draft fits the style and theme of the selected social media platform."""
    # Simulación: para Twitter, menos de 280 caracteres; para LinkedIn, más de 100.
    if platform.lower() == "twitter":
        return len(post_draft) <= 280
    elif platform.lower() == "linkedin":
        return len(post_draft) > 100
    return True

# --- Agentes ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def create_agent(llm, tools, system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    agent = llm.bind_tools(tools)
    return lambda state: {"messages": [agent.invoke(state["messages"])]}

# Agente para redactar el post
post_writer_agent = create_agent(
    llm,
    [write_social_post],
    "You are a social media expert. Write engaging posts about ML papers for the specified platform."
)

# Agente para verificar corrección
correctness_verifier_agent = create_agent(
    llm,
    [verify_correctness],
    "You are an expert in ML research. Verify that the post draft accurately represents the paper."
)

# Agente para verificar estilo/plataforma
style_checker_agent = create_agent(
    llm,
    [check_platform_style],
    "You are a social media strategist. Check if the post fits the style and theme of the platform."
)

# --- Nodos del grafo ---
def fetch_paper_node(state):
    arxiv_id = state["arxiv_id"]
    paper = fetch_paper.invoke(arxiv_id)
    return {"paper_metadata": paper, "messages": state["messages"]}

def post_writer_node(state):
    platform = state["platform"]
    paper_metadata = state["paper_metadata"]
    post = write_social_post.invoke(paper_metadata, platform)
    return {"post_draft": post, "messages": state["messages"] + [HumanMessage(content=post)]}

def correctness_verifier_node(state):
    paper_metadata = state["paper_metadata"]
    post_draft = state["post_draft"]
    verified = verify_correctness.invoke(paper_metadata, post_draft)
    return {"verified": verified, "messages": state["messages"]}

def style_checker_node(state):
    post_draft = state["post_draft"]
    platform = state["platform"]
    style_checked = check_platform_style.invoke(post_draft, platform)
    return {"style_checked": style_checked, "messages": state["messages"]}

# --- Grafo principal ---
graph = StateGraph(SocialPostState)
graph.add_node("fetch_paper", fetch_paper_node)
graph.add_node("post_writer", post_writer_node)
graph.add_node("correctness_verifier", correctness_verifier_node)
graph.add_node("style_checker", style_checker_node)

graph.add_edge(START, "fetch_paper")
graph.add_edge("fetch_paper", "post_writer")
graph.add_edge("post_writer", "correctness_verifier")
graph.add_edge("correctness_verifier", "style_checker")

def decide_next(state):
    if not state.get("verified", False):
        return END
    if not state.get("style_checked", False):
        return END
    return END

graph.add_conditional_edges(
    "style_checker",
    decide_next,
    {END: END}
)

graph.set_entry_point("fetch_paper")
compiled_social_post_graph = graph.compile()

# --- Ejemplo de uso ---
if __name__ == "__main__":
    arxiv_id = "1706.03762"  # Ejemplo: Attention is All You Need
    platform = "Twitter"
    initial_state = {
        "messages": [HumanMessage(content=f"Create a social media post for arXiv:{arxiv_id} on {platform}.")],
        "arxiv_id": arxiv_id,
        "platform": platform,
    }
    final_state = None
    for step in compiled_social_post_graph.stream(initial_state):
        print(step)
        final_state = step
    # Guardar el resultado en un archivo
    if final_state and "post_draft" in final_state:
        with open("hw6_social_media_post.txt", "w") as f:
            f.write(final_state["post_draft"])
        print("\nEl post generado ha sido guardado en 'hw6_social_media_post.txt'.")
    else:
        print("\nNo se generó ningún post para guardar.")

# --- Visualización del grafo (Mermaid) ---

def show_graph_mermaid(graph, filename="social_post_graph.png"):
    try:
        from langgraph.graph import CurveStyle, NodeStyles, MermaidDrawMethod
        from IPython.display import Image, display
        # Generar imagen Mermaid del grafo
        img_bytes = graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path=filename,
            draw_method=MermaidDrawMethod.API,
            background_color="white",
            padding=10,
        )
        display(Image(filename))
    except ImportError:
        print("Para visualizar el grafo necesitas IPython y langgraph >= 0.0.40.")
    except Exception as e:
        print(f"Error al mostrar el grafo: {e}")

# --- Ejemplo de visualización ---
if __name__ == "__main__":
    # ... (código anterior)
    print("\nMostrando visualización del grafo...")
    show_graph_mermaid(compiled_social_post_graph) 