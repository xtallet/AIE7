"""Advanced Agent using LangChain with different personas that communicates with the existing A2A server.

This implements the Advanced Build: Use a different Agent Framework to test your application.
"""
import asyncio
import logging
import random
import os
from typing import Dict, Any, List
from uuid import uuid4

import httpx
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""
    
    def __init__(self, message: str = "OpenAI API key not found in environment variables."):
        self.message = message
        super().__init__(self.message)


class A2ATool:
    """LangChain tool that communicates with the A2A server."""
    
    def __init__(self, base_url: str = "http://localhost:10000"):
        self.base_url = base_url
        self.client = None
        self.agent_card = None
        self.httpx_client = None
    
    async def initialize(self):
        """Initialize the A2A client."""
        # Create a persistent HTTP client
        self.httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        
        try:
            resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=self.base_url,
            )
            
            self.agent_card = await resolver.get_agent_card()
            logger.info("Successfully fetched agent card")
            
            self.client = A2AClient(
                httpx_client=self.httpx_client,
                agent_card=self.agent_card
            )
            logger.info("A2A client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize A2A client: {e}")
            # Clean up the HTTP client if initialization fails
            if self.httpx_client:
                await self.httpx_client.aclose()
            raise
    
    async def cleanup(self):
        """Clean up resources."""
        if self.httpx_client:
            await self.httpx_client.aclose()
    
    async def query_a2a_agent(self, query: str) -> str:
        """Query the A2A agent server for information."""
        if not self.client:
            raise RuntimeError("A2A client not initialized")
        
        try:
            # Prepare message payload
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': query}],
                    'message_id': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )
            
            # Send message and get response
            response = await self.client.send_message(request)
            
            # Extract the response content
            if hasattr(response, 'root') and hasattr(response.root, 'result'):
                a2a_response = response.root.result
                if hasattr(a2a_response, 'artifacts') and a2a_response.artifacts:
                    # Extract text content from artifacts
                    content_parts = []
                    for artifact in a2a_response.artifacts:
                        if hasattr(artifact, 'parts'):
                            for part in artifact.parts:
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    content_parts.append(part.root.text)
                    
                    if content_parts:
                        return "\n\n".join(content_parts)
                
                return str(a2a_response)
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error querying A2A agent: {e}")
            return f"Error communicating with A2A server: {str(e)}"


def create_a2a_tool(a2a_tool_instance: A2ATool):
    """Create a LangChain tool from the A2A tool instance."""
    
    @tool
    async def query_a2a_agent(query: str) -> str:
        """Query the A2A agent server for information."""
        return await a2a_tool_instance.query_a2a_agent(query)
    
    return query_a2a_agent


class PersonaAgent:
    """LangChain agent with different personas that can query the A2A server."""
    
    def __init__(self, persona: str, a2a_tool: A2ATool):
        self.persona = persona
        self.a2a_tool = a2a_tool
        self.llm = ChatOpenAI(temperature=0.7)
        
        # Create the LangChain tool from the A2A tool instance
        a2a_langchain_tool = create_a2a_tool(a2a_tool)
        
        # Define the prompt template with persona-specific instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_persona_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=[a2a_langchain_tool],
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[a2a_langchain_tool],
            verbose=True,
            max_iterations=3
        )
    
    def _get_persona_system_prompt(self) -> str:
        """Get the system prompt based on the selected persona."""
        base_prompt = """You are an AI agent with a specific persona and goal. 
        You have access to a powerful A2A agent server that can search the web, 
        find academic papers, and retrieve documents. Use this tool to get 
        comprehensive information and then present it according to your persona.
        
        Always use the query_a2a_agent tool to get information. Do not make up facts.
        Present the information in a way that matches your persona's style and needs.
        
        Your persona: {persona_instructions}
        
        Remember: You are not the A2A agent - you are using it as a tool to help 
        fulfill your persona's specific goals and communication style."""
        
        persona_instructions = {
            "academic_researcher": """You are a PhD researcher in Computer Science specializing in AI. 
            You want to understand the latest breakthroughs in transformer architectures. 
            You need detailed technical explanations with academic sources and mathematical foundations.
            Present information in a scholarly, detailed manner with technical depth.""",
            
            "tech_journalist": """You are a tech journalist writing an article about AI trends in 2024. 
            You want to find the most surprising and impactful AI developments. 
            You need concrete examples, statistics, and expert opinions with verifiable sources.
            Present information in an engaging, journalistic style with clear examples and sources.""",
            
            "curious_student": """You are a high school student fascinated by AI. 
            You want to understand how AI affects everyday life in simple terms. 
            You need practical examples, real-world applications, and resources to learn more.
            Present information in simple, accessible language with practical examples."""
        }
        
        return base_prompt.format(
            persona_instructions=persona_instructions.get(self.persona, persona_instructions["curious_student"])
        )
    
    async def run(self, query: str) -> str:
        """Run the agent with the given query."""
        try:
            result = await self.agent_executor.ainvoke({
                "input": query,
                "chat_history": []
            })
            return result["output"]
        except Exception as e:
            logger.error(f"Error running persona agent: {e}")
            return f"Error running agent: {str(e)}"


def run_random_persona():
    """Run a single random persona for testing."""
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set.")
        logger.error("Please set your OpenAI API key in the .env file or as an environment variable.")
        return
    
    async def _run():
        # Initialize A2A tool
        a2a_tool = A2ATool()
        try:
            await a2a_tool.initialize()
            
            # Select random persona
            personas = ["academic_researcher", "tech_journalist", "curious_student"]
            selected_persona = random.choice(personas)
            
            # Define query for the selected persona
            queries = {
                "academic_researcher": "What are the latest breakthroughs in transformer architectures? I need detailed technical explanations with academic sources.",
                "tech_journalist": "What are the most surprising AI developments in 2024? I need concrete examples and statistics for my article.",
                "curious_student": "How does AI affect my everyday life? Give me simple examples I can understand."
            }
            
            query = queries[selected_persona]
            
            logger.info(f"Selected persona: {selected_persona}")
            logger.info(f"Query: {query}")
            
            # Create and run the persona agent
            persona_agent = PersonaAgent(selected_persona, a2a_tool)
            result = await persona_agent.run(query)
            
            logger.info(f"\nPersona {selected_persona} response:")
            logger.info(f"{result}")
            
        finally:
            # Always clean up resources
            await a2a_tool.cleanup()
    
    asyncio.run(_run())


if __name__ == "__main__":
    # Run a random persona example
    run_random_persona()
    
    # Uncomment to run all personas
    # asyncio.run(run_persona_examples()) 