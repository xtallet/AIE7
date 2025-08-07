from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
import requests
from dice_roller import DiceRoller

load_dotenv()

mcp = FastMCP("mcp-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query"""
    search_results = client.get_search_context(query=query)
    return search_results

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)

@mcp.tool()
def get_weather_info(location: str) -> str:
    """Get current weather information for a specific location"""
    query = f"current weather {location}"
    weather_results = client.get_search_context(query=query)
    return f"Weather information for {location}:\n{weather_results}"

@mcp.tool()
def get_exchange_rate(from_currency: str, to_currency: str, amount: float = 1.0) -> str:
    """Get exchange rate between two currencies"""
    try:
        # Using the free Exchange Rate API
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        rates = data.get('rates', {})
        
        if to_currency.upper() not in rates:
            return f"Error: Currency {to_currency} not found in available rates"
        
        rate = rates[to_currency.upper()]
        converted_amount = amount * rate
        
        return f"Exchange Rate: {amount} {from_currency.upper()} = {converted_amount:.2f} {to_currency.upper()}\nRate: 1 {from_currency.upper()} = {rate:.4f} {to_currency.upper()}"
    
    except requests.RequestException as e:
        return f"Error fetching exchange rate: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")