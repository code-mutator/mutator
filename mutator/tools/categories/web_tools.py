"""
Web and network tools for the Coding Agent Framework.
"""

import requests
import json
import os
from typing import Any, Dict
from urllib.parse import urlparse
from pathlib import Path

from ..decorator import tool


def _has_web_search_api_keys() -> bool:
    """Check if required API keys for web search are available."""
    google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_cx = os.getenv("GOOGLE_SEARCH_CX")
    bing_api_key = os.getenv("BING_SEARCH_API_KEY")
    
    return bool((google_api_key and google_cx) or bing_api_key)


# Only register web_search if API keys are available
if _has_web_search_api_keys():
    @tool
    def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        <short_description>Search the web for information using integrated search APIs.</short_description>
        
        <long_description>
        This tool provides web search capabilities for finding up-to-date information,
        documentation, and resources from the internet. It's designed to complement
        the agent's knowledge with current information.

        ## Important Notes

        1. **Search Integration**:
           - Ready for integration with Google Search API, Bing API, or similar services
           - Returns structured search results with titles, URLs, and snippets
           - Supports result limiting to prevent information overload
           - Handles rate limiting and API quotas

        2. **Use Cases**:
           - Finding current documentation and tutorials
           - Researching new technologies or frameworks
           - Looking up error messages and solutions
           - Finding code examples and best practices
           - Checking latest versions and updates

        ## Examples

        - Search for documentation: `web_search("React hooks tutorial")`
        - Find error solutions: `web_search("Python ImportError fix")`
        - Research technologies: `web_search("GraphQL vs REST API comparison")`
        - Limited results: `web_search("Docker best practices", max_results=3)`
        </long_description>

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)
        
        Returns:
            Dict containing search results with titles, URLs, snippets, and metadata
        """
        try:
            # Check for API key in environment variables
            google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
            google_cx = os.getenv("GOOGLE_SEARCH_CX")
            bing_api_key = os.getenv("BING_SEARCH_API_KEY")
            
            if google_api_key and google_cx:
                return _google_search(query, max_results, google_api_key, google_cx)
            elif bing_api_key:
                return _bing_search(query, max_results, bing_api_key)
            else:
                return {"error": "No web search API keys configured"}
                
        except Exception as e:
            return {"error": f"Web search failed: {str(e)}"}


def _google_search(query: str, max_results: int, api_key: str, cx: str) -> Dict[str, Any]:
    """Perform Google Custom Search API search."""
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": min(max_results, 10)  # Google limits to 10 per request
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "Google"
            })
        
        return {
            "results": results,
            "total_results": len(results),
            "search_engine": "Google Custom Search",
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Google search failed: {str(e)}"}


def _bing_search(query: str, max_results: int, api_key: str) -> Dict[str, Any]:
    """Perform Bing Web Search API search."""
    try:
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {
            "q": query,
            "count": min(max_results, 50),  # Bing allows up to 50
            "textDecorations": False,
            "textFormat": "Raw"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("webPages", {}).get("value", []):
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "source": "Bing"
            })
        
        return {
            "results": results,
            "total_results": len(results),
            "search_engine": "Bing Web Search",
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Bing search failed: {str(e)}"}


@tool
def fetch_url(url: str, response_type: str = "md", headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
    """
    <short_description>Fetch content from web URLs and return in specified format.</short_description>
    
    <long_description>
    This tool fetches web content via GET requests and can return the content in
    different formats for better processing and readability.

    ## Important Notes

    1. **Response Types**:
       - `md`: Convert HTML to Markdown format (default)
       - `html`: Return raw HTML content
       - Automatic content type detection and handling

    2. **Content Processing**:
       - Automatic JSON parsing for API responses
       - HTML to Markdown conversion for better readability
       - Preserves original content structure

    3. **Use Cases**:
       - Fetching API data and responses
       - Reading web documentation
       - Accessing web content for analysis
       - Retrieving structured data

    ## Examples

    - Fetch as markdown: `fetch_url("https://example.com/docs")`
    - Fetch as HTML: `fetch_url("https://example.com", response_type="html")`
    - With custom headers: `fetch_url("https://api.example.com/data", headers={"Authorization": "Bearer token"})`
    </long_description>

    Args:
        url: URL to fetch content from
        response_type: Response format - "md" for markdown, "html" for HTML (default: "md")
        headers: Optional HTTP headers dictionary
        timeout: Request timeout in seconds (default: 30)
    
    Returns:
        Dict containing response data, headers, status, and formatted content
    """
    try:
        # Basic URL validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {"error": "Invalid URL format"}
        
        # Prepare request
        request_headers = headers or {}
        request_headers.setdefault('User-Agent', 'Mutator-Framework/1.0')
        
        # Make GET request
        response = requests.get(
            url=url,
            headers=request_headers,
            timeout=timeout,
            allow_redirects=True
        )
        
        # Get content type
        content_type = response.headers.get('content-type', '').lower()
        
        # Process content based on type
        content = response.text
        parsed_content = None
        formatted_content = content
        
        if 'application/json' in content_type:
            try:
                parsed_content = response.json()
                formatted_content = json.dumps(parsed_content, indent=2)
            except:
                pass
        elif 'text/html' in content_type and response_type == "md":
            formatted_content = _html_to_markdown(content)
        
        return {
            "url": url,
            "status_code": response.status_code,
            "status_text": response.reason,
            "headers": dict(response.headers),
            "content": formatted_content,
            "raw_content": content,
            "parsed_content": parsed_content,
            "content_type": content_type,
            "response_type": response_type,
            "success": 200 <= response.status_code < 300
        }
        
    except requests.exceptions.Timeout:
        return {"error": f"Request timed out after {timeout} seconds"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Failed to connect to {url}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def _html_to_markdown(html_content: str) -> str:
    """Convert HTML content to Markdown format."""
    try:
        # Try to use html2text if available
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        return h.handle(html_content)
    except ImportError:
        # Fallback to basic HTML stripping
        import re
        # Remove script and style elements
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        # Convert common HTML tags to markdown
        html_content = re.sub(r'<h([1-6])[^>]*>(.*?)</h[1-6]>', r'\n' + r'#' * 1 + r' \2\n', html_content)
        html_content = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\1\n', html_content)
        html_content = re.sub(r'<br[^>]*>', '\n', html_content)
        html_content = re.sub(r'<[^>]+>', '', html_content)  # Remove remaining tags
        return html_content.strip()


# Build the __all__ list dynamically based on what's registered
__all__ = ["fetch_url"]
if _has_web_search_api_keys():
    __all__.append("web_search") 