"""
Example of using the simple @tool decorator for creating tools.

This demonstrates how to create tools using a simpler approach
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Import the tool decorator and agent
from mutator.tools.manager import tool
from mutator.agent import Mutator


# Simple tools using the @tool decorator
@tool
def calculate(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        str: Result of the calculation
    """
    try:
        # Simple eval for demo - in production, use a safer math parser
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def count_words(text: str) -> Dict[str, int]:
    """
    Count the number of words in the given text.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dict[str, int]: Word count statistics
    """
    words = text.split()
    return {
        "total_words": len(words),
        "unique_words": len(set(words)),
        "characters": len(text),
        "characters_no_spaces": len(text.replace(" ", ""))
    }


@tool
def format_text(text: str, format_type: str = "upper") -> str:
    """
    Format text in different ways.
    
    Args:
        text: Text to format
        format_type: Type of formatting (upper, lower, title, reverse)
    
    Returns:
        str: Formatted text
    """
    if format_type == "upper":
        return text.upper()
    elif format_type == "lower":
        return text.lower()
    elif format_type == "title":
        return text.title()
    elif format_type == "reverse":
        return text[::-1]
    else:
        return text


@tool
def list_files(directory: str = ".", pattern: str = "*") -> List[str]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search in
        pattern: File pattern to match (e.g., "*.py")
    
    Returns:
        List[str]: List of matching files
    """
    try:
        path = Path(directory)
        if not path.exists():
            return [f"Error: Directory '{directory}' does not exist"]
        
        files = []
        for file_path in path.glob(pattern):
            if file_path.is_file():
                files.append(str(file_path))
        
        return files
    except Exception as e:
        return [f"Error: {str(e)}"]


@tool
async def async_example(delay: int = 1) -> str:
    """
    Example of an async tool that simulates some work.
    
    Args:
        delay: Delay in seconds
    
    Returns:
        str: Result message
    """
    await asyncio.sleep(delay)
    return f"Async work completed after {delay} seconds"


async def main():
    """Demonstrate the simple tools."""
    
    # Create agent
    agent = Mutator()
    await agent.initialize()
    
    # Register our simple tools
    agent.add_tool(calculate)
    agent.add_tool(count_words)
    agent.add_tool(format_text)
    agent.add_tool(list_files)
    agent.add_tool(async_example)
    
    print("Simple Tools Example")
    print("=" * 50)
    
    # Test calculate tool
    print("\n1. Testing calculate tool:")
    result = await agent.execute_tool("calculate", {"expression": "2 + 3 * 4"})
    print(f"   Result: {result.result}")
    
    # Test count_words tool
    print("\n2. Testing count_words tool:")
    result = await agent.execute_tool("count_words", {"text": "Hello world this is a test"})
    print(f"   Result: {result.result}")
    
    # Test format_text tool
    print("\n3. Testing format_text tool:")
    result = await agent.execute_tool("format_text", {"text": "hello world", "format_type": "title"})
    print(f"   Result: {result.result}")
    
    # Test list_files tool
    print("\n4. Testing list_files tool:")
    result = await agent.execute_tool("list_files", {"directory": ".", "pattern": "*.py"})
    print(f"   Found {len(result.result)} Python files")
    
    # Test async tool
    print("\n5. Testing async tool:")
    result = await agent.execute_tool("async_example", {"delay": 2})
    print(f"   Result: {result.result}")
    
    # Show available tools
    print("\n6. Available tools:")
    tools = await agent.get_available_tools()
    for tool_info in tools:
        if tool_info["name"] in ["calculate", "count_words", "format_text", "list_files", "async_example"]:
            print(f"   - {tool_info['name']}: {tool_info['description']}")
    
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main()) 