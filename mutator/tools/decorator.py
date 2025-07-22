"""
Tool decorator for the Coding Agent Framework.

This module provides the @tool decorator that automatically generates tool schemas
from function signatures and docstrings.
"""

import asyncio
import inspect
from typing import Any, Callable, Dict

from ..core.types import ToolResult
from .schema_generator import SchemaGenerator


class SimpleTool:
    """A simple tool created from a function using the @tool decorator."""
    
    def __init__(self, name: str, func: Callable, schema: Dict[str, Any]):
        self.name = name
        self.func = func
        self.schema = schema
        self.enabled = True
        
        # Extract descriptions from schema
        function_info = schema.get("function", {})
        self.description = function_info.get("description", "No description available")
        self.short_description = function_info.get("short_description", self.description)
        
        # If short_description is the same as description, try to extract it
        if self.short_description == self.description:
            self.short_description = self._extract_short_description(self.description)
    
    def _extract_short_description(self, description: str) -> str:
        """Extract short description from description text."""
        if not description:
            return ""
        
        # Try to extract from <short_description> tags
        import re
        short_match = re.search(r'<short_description>(.*?)</short_description>', 
                               description, re.DOTALL)
        if short_match:
            return short_match.group(1).strip()
        
        # Fallback: use first line or first sentence
        lines = description.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        if len(first_line) > 150:
            # Try to find first sentence
            for ending in ['. ', '! ', '? ']:
                pos = first_line.find(ending)
                if pos > 0 and pos < 150:
                    return first_line[:pos + 1]
            
            # Truncate at word boundary
            words = first_line[:150].split()
            if len(words) > 1:
                words.pop()
                return ' '.join(words) + "..."
            else:
                return first_line[:150] + "..."
        
        return first_line
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return self.schema
    
    def get_short_schema(self) -> Dict[str, Any]:
        """Get the short schema for this tool."""
        short_schema = self.schema.copy()
        if "function" in short_schema:
            function_schema = short_schema["function"].copy()
            function_schema["description"] = self.short_description
            # Remove the long description and short_description fields to save space
            if "short_description" in function_schema:
                del function_schema["short_description"]
            short_schema["function"] = function_schema
        return short_schema
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool function."""
        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**kwargs)
            else:
                result = self.func(**kwargs)
            
            # Check if result contains an error (common pattern in tool functions)
            if isinstance(result, dict) and "error" in result:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=result["error"]
                )
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=result
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e)
            )
    
    def perform_safety_checks(self, **kwargs) -> list:
        """Perform safety checks before execution."""
        # Basic validation - can be extended
        return []
    
    def validate_parameters(self, **kwargs) -> None:
        """Validate tool parameters against schema."""
        schema = self.get_schema()
        required = schema.get("function", {}).get("parameters", {}).get("required", [])
        
        # Check required parameters
        for param in required:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
    
    def needs_confirmation(self, **kwargs) -> bool:
        """Check if this tool execution needs confirmation."""
        return False
    
    def get_confirmation_message(self, **kwargs) -> str:
        """Get confirmation message for this tool execution."""
        return f"Execute {self.name} with parameters: {kwargs}"


def tool(func: Callable = None):
    """
    Decorator to create a simple tool from a function.
    
    Automatically extracts tool name, short description, long description,
    and parameter information from the function signature and docstring.
    
    The docstring should follow this format:
    
    ```python
    @tool
    def my_tool(param1: str, param2: int = 10) -> str:
        '''
        Short description of what this tool does.
        
        Longer description with more details about the tool's functionality,
        including examples and usage notes.
        
        Args:
            param1: Description of the first parameter
            param2: Description of the second parameter
        
        Returns:
            Description of what the tool returns
        '''
        return f"Result: {param1} {param2}"
    ```
    
    Args:
        func: The function to convert to a tool
    
    Returns:
        A SimpleTool instance that can be registered with the ToolManager
    """
    def decorator(f: Callable) -> SimpleTool:
        # Generate schema using the new schema generator
        schema = SchemaGenerator.generate_schema(f)
        
        # Create and return SimpleTool
        return SimpleTool(
            name=f.__name__,
            func=f,
            schema=schema
        )
    
    # Support both @tool and @tool() syntax
    if func is None:
        return decorator
    else:
        return decorator(func) 