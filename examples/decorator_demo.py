"""
Demonstration of the new @tool decorator that automatically captures
tool name from function name and description from docstring.
"""

from mutator.tools.manager import tool


# NEW SYNTAX: @tool decorator automatically captures name and description
@tool
def calculate_area(width: float, height: float) -> float:
    """
    Calculate the area of a rectangle.
    
    Args:
        width: Width of the rectangle
        height: Height of the rectangle
    
    Returns:
        Area of the rectangle
    """
    return width * height


@tool
def process_text(text: str, operation: str = "upper") -> str:
    """
    Process text with various operations.
    
    Args:
        text: Text to process
        operation: Operation to perform (upper, lower, title, reverse)
    
    Returns:
        Processed text
    """
    if operation == "upper":
        return text.upper()
    elif operation == "lower":
        return text.lower()
    elif operation == "title":
        return text.title()
    elif operation == "reverse":
        return text[::-1]
    else:
        return text


@tool
async def fetch_data(url: str, timeout: int = 30) -> str:
    """
    Fetch data from a URL (simulated).
    
    Args:
        url: URL to fetch data from
        timeout: Timeout in seconds
    
    Returns:
        Fetched data
    """
    import asyncio
    await asyncio.sleep(0.1)  # Simulate network delay
    return f"Data from {url} (timeout: {timeout}s)"


def main():
    """Demonstrate the new @tool decorator."""
    
    print("Tool Decorator Demo")
    print("=" * 50)
    
    # Show tool properties
    print("\n1. Tool Properties:")
    print(f"   calculate_area.name: {calculate_area.name}")
    print(f"   calculate_area.description: {calculate_area.description[:50]}...")
    
    print(f"\n   process_text.name: {process_text.name}")
    print(f"   process_text.description: {process_text.description[:50]}...")
    
    print(f"\n   fetch_data.name: {fetch_data.name}")
    print(f"   fetch_data.description: {fetch_data.description[:50]}...")
    
    # Show schemas
    print("\n2. Generated Schemas:")
    schema = calculate_area.get_schema()
    print(f"   calculate_area schema: {schema['function']['name']}")
    print(f"   Parameters: {list(schema['function']['parameters']['properties'].keys())}")
    print(f"   Required: {schema['function']['parameters']['required']}")
    
    print("\n3. Key Benefits:")
    print("   ✓ No need to specify tool name - automatically uses function name")
    print("   ✓ No need to specify description - automatically uses docstring")
    print("   ✓ Cleaner, more readable code")
    print("   ✓ Docstring is cleaned up (whitespace normalized)")
    print("   ✓ Supports both sync and async functions")
    print("   ✓ Automatic type inference from function annotations")
    
    print("\n4. Usage Comparison:")
    print("   OLD: @tool('calculate_area', 'Calculate the area of a rectangle')")
    print("   NEW: @tool")
    print("        def calculate_area(width: float, height: float) -> float:")
    print("            '''Calculate the area of a rectangle.'''")
    print("            return width * height")
    
    print("\nDone!")


if __name__ == "__main__":
    main() 