from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import os
import requests
from datetime import datetime

load_dotenv()

mcp = FastMCP("news-mcp-server")


# News API configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_BASE_URL = "https://newsapi.org/v2"

@mcp.tool()
def get_top_headlines(country: str = "us", category: str = None, page_size: int = 10) -> str:
    """Get top headlines from News API for a specific country and optional category"""
    if not NEWS_API_KEY:
        return "Error: NEWS_API_KEY not found in environment variables"
    
    url = f"{NEWS_API_BASE_URL}/top-headlines"
    params = {
        "apiKey": NEWS_API_KEY,
        "country": country,
        "pageSize": page_size
    }
    
    if category:
        params["category"] = category
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "ok" and data["articles"]:
            headlines = []
            for i, article in enumerate(data["articles"][:page_size], 1):
                headline = f"{i}. {article['title']}"
                if article.get('description'):
                    headline += f"\n   {article['description'][:100]}..."
                headlines.append(headline)
            
            return f"Top headlines for {country.upper()}:\n" + "\n\n".join(headlines)
        else:
            return f"No headlines found for {country}"
            
    except requests.RequestException as e:
        return f"Error fetching news: {str(e)}"

@mcp.tool()
def search_news(query: str, language: str = "en", sort_by: str = "publishedAt", page_size: int = 10) -> str:
    """Search for news articles based on a query"""
    if not NEWS_API_KEY:
        return "Error: NEWS_API_KEY not found in environment variables"
    
    url = f"{NEWS_API_BASE_URL}/everything"
    params = {
        "apiKey": NEWS_API_KEY,
        "q": query,
        "language": language,
        "sortBy": sort_by,
        "pageSize": page_size
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "ok" and data["articles"]:
            articles = []
            for i, article in enumerate(data["articles"][:page_size], 1):
                article_info = f"{i}. {article['title']}"
                if article.get('description'):
                    article_info += f"\n   {article['description'][:100]}..."
                if article.get('publishedAt'):
                    published_date = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                    article_info += f"\n   Published: {published_date.strftime('%Y-%m-%d %H:%M')}"
                articles.append(article_info)
            
            return f"News articles for '{query}':\n" + "\n\n".join(articles)
        else:
            return f"No articles found for '{query}'"
            
    except requests.RequestException as e:
        return f"Error searching news: {str(e)}"

@mcp.tool()
def get_news_sources(country: str = "us", category: str = None) -> str:
    """Get available news sources for a country and optional category"""
    if not NEWS_API_KEY:
        return "Error: NEWS_API_KEY not found in environment variables"
    
    url = f"{NEWS_API_BASE_URL}/sources"
    params = {
        "apiKey": NEWS_API_KEY,
        "country": country
    }
    
    if category:
        params["category"] = category
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "ok" and data["sources"]:
            sources = []
            for i, source in enumerate(data["sources"], 1):
                source_info = f"{i}. {source['name']}"
                if source.get('description'):
                    source_info += f" - {source['description'][:100]}..."
                sources.append(source_info)
            
            return f"News sources for {country.upper()}:\n" + "\n".join(sources)
        else:
            return f"No sources found for {country}"
            
    except requests.RequestException as e:
        return f"Error fetching sources: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")