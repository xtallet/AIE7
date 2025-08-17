"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    
    return graph.compile()


# 🏗️ XTALLET - Code to create the Helpfulness Agent
def create_helpfulness_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a helpfulness-checking LangGraph agent with response evaluation.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent with helpfulness evaluation
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with the accumulated messages and append its response."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def route_to_action_or_helpfulness(state: AgentState):
        """Decide whether to execute tools or run the helpfulness evaluator."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "helpfulness"
    
    def helpfulness_node(state: AgentState) -> Dict[str, Any]:
        """Evaluate helpfulness of the latest response relative to the initial query."""
        # If we've exceeded loop limit, short-circuit with END decision marker
        if len(state["messages"]) > 10:
            return {"messages": [AIMessage(content="HELPFULNESS:END")]}    

        initial_query = state["messages"][0]
        final_response = state["messages"][-1]

        prompt_template = """
  Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

  Initial Query:
  {initial_query}

  Final Response:
  {final_response}"""

        helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
        helpfulness_check_model = get_openai_model(model_name="gpt-4.1-mini")
        helpfulness_chain = (
            helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()
        )

        helpfulness_response = helpfulness_chain.invoke(
            {
                "initial_query": initial_query.content,
                "final_response": final_response.content,
            }
        )

        decision = "Y" if "Y" in helpfulness_response else "N"
        return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}

    def helpfulness_decision(state: AgentState):
        """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
        # Check loop-limit marker
        if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
            return END

        last = state["messages"][-1]
        text = getattr(last, "content", "")
        if "HELPFULNESS:Y" in text:
            return "end"
        return "continue"

    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    
    graph_helpfulness = graph.compile()
    return graph_helpfulness


# 🏗️ XTALLET - Activity 3 - Guardrails

# Import Guardrails components
from guardrails.hub import (
    RestrictToTopic,
    DetectJailbreak, 
    CompetitorCheck,
    LlmRagEvaluator,
    HallucinationPrompt,
    ProfanityFree,
    GuardrailsPII
)

from guardrails import Guard

def setup_guardrails():
    """Configure production Guardrails."""
    global topic_guard, jailbreak_guard, pii_guard, profanity_guard, factuality_guard
    
    # Topic restriction guard
    topic_guard = Guard().use(
        RestrictToTopic(
            valid_topics=["student loans", "financial aid", "education financing", "loan repayment"],
            invalid_topics=["investment advice", "crypto", "gambling", "politics"],
            on_fail="exception"
        )
    )
    
    # Jailbreak detection guard
    jailbreak_guard = Guard().use(DetectJailbreak())
    
    # PII protection guard
    pii_guard = Guard().use(
        GuardrailsPII(
            entities=["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"], 
            on_fail="fix"
        )
    )
    
    # Content moderation guard
    profanity_guard = Guard().use(
        ProfanityFree(threshold=0.8, validation_method="sentence", on_fail="exception")
    )
    
    # Factuality guard
    factuality_guard = Guard().use(
        LlmRagEvaluator(
            eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
            llm_evaluator_fail_response="hallucinated",
            llm_evaluator_pass_response="factual", 
            llm_callable="gpt-4.1-mini",
            on_fail="exception",
            on="prompt"
        )
    )

# Initialize guardrails
setup_guardrails()

def input_validation_node(state: AgentState) -> Dict[str, Any]:
    """Validate user input for safety and compliance."""
    # Extract the user message
    user_message = state["messages"][0]
    
    # Apply input guards
    try:
        # Topic restriction check
        topic_guard.validate(user_message.content)
        
        # Jailbreak detection check
        jailbreak_result = jailbreak_guard.validate(user_message.content)
        if not jailbreak_result.validation_passed:
            return {"messages": [AIMessage(content="I cannot process that request. Please ask about student loans or financial aid.")]}
        
        # PII protection check
        pii_result = pii_guard.validate(user_message.content)
        
        # Return validated input for processing (original message, not validated output)
        return {"messages": [user_message]}
        
    except Exception as e:
        # Return error message for input validation failure
        return {"messages": [AIMessage(content=f"Input validation failed: {str(e)}")]}


def input_validation_decision(state: AgentState):
    """Route based on input validation results."""
    last_message = state["messages"][-1]
    
    # Check if input validation failed
    if "validation failed" in last_message.content.lower():
        print(f"🔍 DEBUG: Input validation failed, terminating: {last_message.content}")
        return END  # Terminate execution if validation failed
    
    # Check if this is the original user message (validation passed)
    if hasattr(last_message, 'content') and not last_message.content.startswith("Input validation failed"):
        print(f"🔍 DEBUG: Input validation passed, continuing to agent: {last_message.content[:50]}...")
        return "agent"  # Continue to agent if validation passed
    
    print(f"🔍 DEBUG: Default case, terminating: {last_message.content}")
    return END  # Default: terminate


def output_validation_node(state: AgentState) -> Dict[str, Any]:
    """Validate agent output for safety and factuality."""
    # Extract the agent's response
    agent_response = state["messages"][-1]
    
    try:
        # Content moderation check
        profanity_result = profanity_guard.validate(agent_response.content)
        
        # Factuality check (if RAG was used)
        if any("retrieve_information" in str(tool_call) for tool_call in getattr(agent_response, 'tool_calls', [])):
            factuality_result = factuality_guard.validate(agent_response.content)
            if not factuality_result.validation_passed:
                return {"messages": [AIMessage(content="I need to refine my response to be more factual. Let me try again.")]}
        
        # Return validated output
        return {"messages": [AIMessage(content=profanity_result.validated_output)]}
        
    except Exception as e:
        return {"messages": [AIMessage(content=f"Output validation failed: {str(e)}")]}
    

def guardrails_decision(state: AgentState):
    """Route based on guardrails validation results."""
    last_message = state["messages"][-1]
    
    # Check for validation failures
    if "validation failed" in last_message.content.lower():
        return "refinement"
    elif "refine" in last_message.content.lower():
        return "refinement"
    else:
        return "helpfulness"
    

def create_guardrails_simple_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent with Guardrails protection.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent with input/output validation
    """
    
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, 'tool_calls', None):
            return "action"
        return "output_validation"
    
    # Build graph with guardrails
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("input_validation", input_validation_node)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("output_validation", output_validation_node)
    
    graph.set_entry_point("input_validation")
    
    # Add conditional routing for input validation
    graph.add_conditional_edges(
        "input_validation",
        input_validation_decision,  # ← Nuevo router
        {"agent": "agent", END: END}
    )
    
    graph.add_conditional_edges("agent", should_continue, {"action": "action", "output_validation": "output_validation"})
    graph.add_edge("action", "agent")
    graph.add_edge("output_validation", END)
    
    return graph.compile()


def create_guardrails_helpfulness_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a helpfulness-checking LangGraph agent with Guardrails protection.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent with helpfulness evaluation and Guardrails
    """
    
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with the accumulated messages and append its response."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def route_to_action_or_helpfulness(state: AgentState):
        """Decide whether to execute tools or run the helpfulness evaluator."""
        last_message = state["messages"][-1]
        if getattr(last_message, 'tool_calls', None):
            return "action"
        return "helpfulness"
    
    def helpfulness_node(state: AgentState) -> Dict[str, Any]:
        """Evaluate helpfulness of the latest response relative to the initial query."""
        # If we've exceeded loop limit, short-circuit with END decision marker
        if len(state["messages"]) > 10:
            return {"messages": [AIMessage(content="HELPFULNESS:END")]}    

        initial_query = state["messages"][0]
        final_response = state["messages"][-1]

        prompt_template = """
  Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

  Initial Query:
  {initial_query}

  Final Response:
  {final_response}"""

        helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
        helpfulness_check_model = get_openai_model(model_name="gpt-4.1-mini")
        helpfulness_chain = (
            helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()
        )

        helpfulness_response = helpfulness_chain.invoke(
            {
                "initial_query": initial_query.content,
                "final_response": final_response.content,
            }
        )

        decision = "Y" if "Y" in helpfulness_response else "N"
        return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}

    def helpfulness_decision(state: AgentState):
        """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
        # Check loop-limit marker
        if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
            return END

        last = state["messages"][-1]
        text = getattr(last, "content", "")
        if "HELPFULNESS:Y" in text:
            return "output_validation"
        return "continue"

    # Build graph with guardrails
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("input_validation", input_validation_node)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)
    graph.add_node("output_validation", output_validation_node)
    
    graph.set_entry_point("input_validation")
    
    # Add conditional routing for input validation
    graph.add_conditional_edges(
        "input_validation",
        input_validation_decision,  # ← Nuevo router
        {"agent": "agent", END: END}
    )
    
    graph.add_conditional_edges("agent", route_to_action_or_helpfulness, {"action": "action", "helpfulness": "helpfulness"})
    graph.add_conditional_edges("helpfulness", helpfulness_decision, {"continue": "agent", "output_validation": "output_validation", END: END})
    graph.add_edge("action", "agent")
    graph.add_edge("output_validation", END)
    
    return graph.compile()