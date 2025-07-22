# Examples

This directory contains examples demonstrating how to use the Coding Agent Framework.

## Available Examples

### 1. Simple Tools Example (`simple_tools_example.py`)

Demonstrates the new simple tool approach using the `@tool` decorator. This is the **recommended way** to create tools in the framework.

**Key Improvement:** The `@tool` decorator now automatically captures the tool name from the function name and the description from the function's docstring, eliminating the need to specify them manually.

**Features:**
- 90% less boilerplate code compared to traditional approaches
- **Automatic tool name capture** from function name
- **Automatic description capture** from docstring
- Automatic schema generation from function signatures
- Type inference from Python type annotations
- Support for both synchronous and asynchronous functions
- CrewAI-like syntax for familiarity

**Example:**
```python
from mutator.tools.decorator import tool

@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        str: Result of the calculation
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

## Tool Development Comparison

### Simple Tools (Recommended)

Using the `@tool` decorator, creating a new tool is extremely simple:

```python
@tool
def my_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Description of what the tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default value
    
    Returns:
        Dict containing the result
    """
    return {
        "param1": param1,
        "param2": param2,
        "result": f"Processed {param1} with value {param2}"
    }
```

**Benefits:**
- **5-10 lines of code** vs 50+ lines for traditional tools
- **Automatic schema generation** from function signature
- **Type inference** from annotations
- **Default parameter handling** automatically included
- **No boilerplate** - just write the function logic
- **Both sync and async** support automatically

## Built-in Tools

The framework comes with a comprehensive set of built-in tools organized by category:

### File Operations
- `read_file` - Read file contents

- `edit_file` - Edit files by replacing line ranges
- `create_file` - Create new files

### System Operations
- `run_shell` - Execute shell commands


### Search & Discovery
- `search_files` - Search for files by name pattern
- `grep_search` - Search for text patterns in files
- `glob_search` - Search using glob patterns
- `list_directory` - List directory contents





### Web & Network
- `web_search` - Search the web for information
- `fetch_url` - Fetch content from URLs


### AI-Powered Tools
- `codebase_search` - Semantic search through codebase
- `mermaid` - Generate Mermaid diagrams

### Task Management
- `task` - Task planning and execution
- `todo_read` - Read TODO comments from code
- `todo_write` - Add TODO comments to code

## Performance Comparison

| Metric | Simple Tools | Traditional Tools | Improvement |
|--------|--------------|-------------------|-------------|
| Lines of Code | 5-10 | 50+ | **90% reduction** |
| Schema Definition | Automatic | Manual | **100% automatic** |
| Type Safety | Inferred | Manual | **Automatic inference** |
| Default Parameters | Automatic | Manual | **Built-in support** |
| Async Support | Automatic | Manual | **Zero config** |
| Development Time | Minutes | Hours | **10x faster** |

## Running Examples

To run any example:

```bash
# From the project root
PYTHONPATH=. python examples/simple_tools_example.py
```

## Creating Your Own Tools

1. **Simple approach (recommended):**
   ```python
   from mutator.tools.manager import tool
   
   @tool
   def your_tool(param: str) -> Dict[str, Any]:
       """Tool description."""
       return {"result": f"Processed: {param}"}
   ```

2. **Register with agent:**
   ```python
   agent.tool_manager.register_function(your_tool)
   ```

That's it! The framework handles everything else automatically.

## Best Practices

1. **Use descriptive tool names** - they become the tool identifiers
2. **Include comprehensive docstrings** - they help with tool discovery
3. **Use type hints** - they enable automatic schema generation
4. **Return structured data** - use dictionaries for complex results
5. **Handle errors gracefully** - return error information in the result
6. **Keep tools focused** - one tool should do one thing well 