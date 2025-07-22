"""
Schema generation utilities for tools.

This module provides utilities to automatically generate tool schemas from function
signatures and docstrings, extracting both short and long descriptions.
"""

import inspect
import re
from typing import Any, Dict, List, Optional, Callable, Union, get_type_hints
from pathlib import Path


class DocstringParser:
    """Parser for extracting structured information from docstrings."""
    
    def __init__(self, docstring: str):
        self.docstring = docstring or ""
        self.cleaned_docstring = inspect.cleandoc(self.docstring)
    
    def extract_short_description(self) -> str:
        """Extract short description from <short_description> tags or fallback to first line."""
        if not self.cleaned_docstring:
            return ""
        
        # First try to extract from <short_description> tags
        short_match = re.search(r'<short_description>(.*?)</short_description>', 
                               self.cleaned_docstring, re.DOTALL)
        if short_match:
            return short_match.group(1).strip()
        
        # Fallback to original logic for backwards compatibility
        lines = self.cleaned_docstring.split('\n')
        
        # Find the first non-empty line that's not a header
        first_line = None
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                first_line = line
                break
        
        if not first_line:
            return ""
        
        # If the first line is too long, try to find the first sentence
        if len(first_line) > 150:
            # Look for sentence endings
            for ending in ['. ', '! ', '? ']:
                pos = first_line.find(ending)
                if pos > 0 and pos < 150:
                    return first_line[:pos + 1]
            
            # If no sentence ending found, truncate at word boundary
            words = first_line[:150].split()
            if len(words) > 1:
                words.pop()  # Remove the last potentially incomplete word
                return ' '.join(words) + "..."
            else:
                return first_line[:150] + "..."
        
        return first_line
    
    def extract_long_description(self) -> str:
        """Extract long description from <long_description> tags or fallback to full docstring."""
        if not self.cleaned_docstring:
            return ""
        
        # First try to extract from <long_description> tags
        long_match = re.search(r'<long_description>(.*?)</long_description>', 
                              self.cleaned_docstring, re.DOTALL)
        if long_match:
            return long_match.group(1).strip()
        
        # Fallback to full docstring for backwards compatibility
        return self.cleaned_docstring
    
    def extract_parameter_descriptions(self) -> Dict[str, str]:
        """Extract parameter descriptions from the Args section."""
        param_descriptions = {}
        
        # Look for Args section
        args_match = re.search(r'Args:\s*\n(.*?)(?=\n\s*(?:Returns?|Raises?|Examples?|Notes?|$))', 
                              self.cleaned_docstring, re.DOTALL | re.IGNORECASE)
        
        if args_match:
            args_section = args_match.group(1)
            # Parse parameter descriptions
            param_lines = args_section.split('\n')
            current_param = None
            
            for line in param_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a parameter definition
                param_match = re.match(r'(\w+):\s*(.+)', line)
                if param_match:
                    param_name = param_match.group(1)
                    param_desc = param_match.group(2)
                    param_descriptions[param_name] = param_desc
                    current_param = param_name
                elif current_param and line.startswith(' '):
                    # Continuation of previous parameter description
                    param_descriptions[current_param] += ' ' + line
        
        return param_descriptions
    
    def extract_return_description(self) -> str:
        """Extract return value description."""
        return_match = re.search(r'Returns?:\s*\n(.*?)(?=\n\s*(?:Args?|Raises?|Examples?|Notes?|$))', 
                                self.cleaned_docstring, re.DOTALL | re.IGNORECASE)
        
        if return_match:
            return return_match.group(1).strip()
        
        return "No return description available"


class SchemaGenerator:
    """Generates JSON schemas for tools from function signatures and docstrings."""
    
    @staticmethod
    def python_type_to_json_type(python_type: Any) -> str:
        """Convert Python type to JSON schema type."""
        if python_type == int:
            return "integer"
        elif python_type == float:
            return "number"
        elif python_type == bool:
            return "boolean"
        elif python_type == str:
            return "string"
        elif python_type == list or (hasattr(python_type, '__origin__') and python_type.__origin__ == list):
            return "array"
        elif python_type == dict or (hasattr(python_type, '__origin__') and python_type.__origin__ == dict):
            return "object"
        elif python_type == Path:
            return "string"
        elif hasattr(python_type, '__origin__'):
            # Handle Union types (like Optional)
            if python_type.__origin__ == Union:
                args = python_type.__args__
                # If it's Optional (Union with None), use the non-None type
                if len(args) == 2 and type(None) in args:
                    non_none_type = args[0] if args[1] == type(None) else args[1]
                    return SchemaGenerator.python_type_to_json_type(non_none_type)
                # For other unions, default to string
                return "string"
        
        # Default to string for unknown types
        return "string"
    
    @staticmethod
    def extract_list_item_type(python_type: Any) -> Optional[str]:
        """Extract item type from List[T] annotations."""
        if hasattr(python_type, '__origin__') and python_type.__origin__ == list:
            if hasattr(python_type, '__args__') and python_type.__args__:
                item_type = python_type.__args__[0]
                return SchemaGenerator.python_type_to_json_type(item_type)
        return None
    
    @staticmethod
    def generate_schema(func: Callable) -> Dict[str, Any]:
        """Generate a complete JSON schema for a function."""
        tool_name = func.__name__
        
        # Parse docstring
        docstring_parser = DocstringParser(func.__doc__)
        short_description = docstring_parser.extract_short_description()
        long_description = docstring_parser.extract_long_description()
        param_descriptions = docstring_parser.extract_parameter_descriptions()
        
        # Provide default descriptions if empty
        if not short_description:
            short_description = f"Tool: {tool_name}"
        if not long_description:
            long_description = f"Tool: {tool_name}"
        
        # Get function signature
        sig = inspect.signature(func)
        
        # Try to get type hints
        try:
            type_hints = get_type_hints(func)
        except (NameError, AttributeError):
            type_hints = {}
        
        # Generate parameters schema
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # Get parameter type
            param_type = type_hints.get(param_name, param.annotation)
            json_type = SchemaGenerator.python_type_to_json_type(param_type)
            
            # Get parameter description
            param_desc = param_descriptions.get(param_name, f"Parameter: {param_name}")
            
            # Build property schema
            property_schema = {
                "type": json_type,
                "description": param_desc
            }
            
            # Handle array items
            if json_type == "array":
                item_type = SchemaGenerator.extract_list_item_type(param_type)
                if item_type:
                    property_schema["items"] = {"type": item_type}
            
            # Add default value if present
            if param.default != inspect.Parameter.empty:
                property_schema["default"] = param.default
            else:
                required.append(param_name)
            
            properties[param_name] = property_schema
        
        # Build complete schema
        schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": long_description,
                "short_description": short_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        
        return schema
    
    @staticmethod
    def generate_short_schema(func: Callable) -> Dict[str, Any]:
        """Generate a schema with short description for overview purposes."""
        full_schema = SchemaGenerator.generate_schema(func)
        
        # Create a copy with short description
        short_schema = full_schema.copy()
        if "function" in short_schema:
            function_schema = short_schema["function"].copy()
            if "short_description" in function_schema:
                function_schema["description"] = function_schema["short_description"]
                del function_schema["short_description"]
            short_schema["function"] = function_schema
        
        return short_schema 