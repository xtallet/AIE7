"""Simple LangGraph Client Agent that uses the A2A protocol to communicate with the existing server.

This implements Activity #1: Build a LangGraph Graph to "use" your application.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Annotated, TypedDict, List
from uuid import uuid4

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientAgentState(TypedDict):
    """State schema for the client agent graph."""
    messages: Annotated[List, add_messages]
    a2a_response: Any  # Response from A2A server


def send_message_to_a2a_sync(message: str, base_url: str = "http://localhost:10000") -> Any:
    """Synchronous function to send a message to the A2A server."""
    
    async def _send_message_async():
        """Async function to send message."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            try:
                # Resolve agent card
                resolver = A2ACardResolver(
                    httpx_client=httpx_client,
                    base_url=base_url,
                )
                
                agent_card = await resolver.get_agent_card()
                logger.info("Successfully fetched agent card")
                
                # Create client
                client = A2AClient(
                    httpx_client=httpx_client,
                    agent_card=agent_card
                )
                logger.info("A2A client created successfully")
                
                # Prepare message payload
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': message}],
                        'message_id': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(**send_message_payload)
                )
                
                # Send message and get response
                response = await client.send_message(request)
                return response
                
            except Exception as e:
                logger.error(f"Error in A2A communication: {e}")
                raise
    
    # Run the async function in a new thread to avoid event loop conflicts
    try:
        # Create a new event loop in a new thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_send_message_async())
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error creating event loop: {e}")
        raise


def client_agent_node(state: ClientAgentState) -> Dict[str, Any]:
    """Node that processes user input and sends it to the A2A server."""
    # Get the last user message
    last_message = state["messages"][-1]
    user_query = last_message.content
    
    logger.info(f"Processing user query: {user_query}")
    
    # Send message to A2A server
    try:
        response = send_message_to_a2a_sync(user_query)
        
        # Extract the response content
        if hasattr(response, 'root') and hasattr(response.root, 'result'):
            a2a_response = response.root.result
            
            # Log detailed information about tools used
            logger.info("=== A2A Response Details ===")
            
            # Check for artifacts (tool outputs)
            if hasattr(a2a_response, 'artifacts') and a2a_response.artifacts:
                logger.info(f"Number of artifacts: {len(a2a_response.artifacts)}")
                for i, artifact in enumerate(a2a_response.artifacts):
                    logger.info(f"Artifact {i+1}: {artifact.name} - {artifact.description}")
                    if hasattr(artifact, 'parts') and artifact.parts:
                        for part in artifact.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                logger.info(f"  Content preview: {part.root.text[:100]}...")
            
            # Check for metadata
            if hasattr(a2a_response, 'metadata') and a2a_response.metadata:
                logger.info(f"Metadata: {a2a_response.metadata}")
            
            # Check for task information
            if hasattr(a2a_response, 'id'):
                logger.info(f"Task ID: {a2a_response.id}")
            if hasattr(a2a_response, 'status'):
                logger.info(f"Task Status: {a2a_response.status}")
            
            logger.info("=== End A2A Response Details ===")
            
        else:
            a2a_response = response
            logger.info("Response structure doesn't have root.result")
        
        logger.info("Received response from A2A server")
        
        # Create AI message with the response
        ai_message = AIMessage(content=str(a2a_response))
        
        return {
            "messages": [ai_message],
            "a2a_response": a2a_response
        }
        
    except Exception as e:
        error_message = AIMessage(content=f"Error communicating with A2A server: {str(e)}")
        logger.error(f"Error in client agent node: {e}")
        
        return {
            "messages": [error_message],
            "a2a_response": None
        }


def build_client_agent_graph():
    """Build the client agent graph that uses the A2A server."""
    
    # Build the graph
    graph = StateGraph(ClientAgentState)
    
    # Add nodes
    graph.add_node("client_agent", client_agent_node)
    
    # Set entry point
    graph.set_entry_point("client_agent")
    
    return graph.compile()


def run_client_agent_example():
    """Example of how to use the client agent graph."""
    
    # Build the graph
    graph = build_client_agent_graph()
    
    # Example conversation
    initial_message = HumanMessage(content="What are the latest developments in artificial intelligence?")
    
    # Run the graph
    result = graph.invoke({
        "messages": [initial_message],
        "a2a_response": None
    })
    
    logger.info("Graph execution completed")
    logger.info(f"Final messages: {result['messages']}")
    
    return result


if __name__ == "__main__":
    # Run the example
    run_client_agent_example() 