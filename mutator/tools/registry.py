"""
Centralized tool registry for the Coding Agent Framework.

This module provides a centralized registry for managing all available tools,
their metadata, categories, and availability status.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Type, Union

from .base import BaseTool
from .decorator import SimpleTool


class ToolRegistry:
    """Centralized registry for all available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Union[BaseTool, SimpleTool]] = {}
        self.tool_categories: Dict[str, Set[str]] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        self.disabled_tools: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        
        # Initialize default categories
        self._initialize_default_categories()
    
    def _initialize_default_categories(self) -> None:
        """Initialize default tool categories."""
        default_categories = [
            "file_system",
            "shell",
            "search",
            "development",
            "git",
            "web",
            "ai",
            "task_management",
            "batch_processing",
            "system"
        ]
        
        for category in default_categories:
            self.tool_categories[category] = set()
    
    def register_tool(self, tool: Union[BaseTool, SimpleTool], category: str = "custom") -> bool:
        """Register a tool in the registry."""
        tool_name = tool.name
        
        # Store the tool
        self.tools[tool_name] = tool
        
        # Add to category
        if category not in self.tool_categories:
            self.tool_categories[category] = set()
        self.tool_categories[category].add(tool_name)
        
        # Store metadata
        self.tool_metadata[tool_name] = {
            "category": category,
            "description": getattr(tool, 'description', ''),
            "short_description": getattr(tool, 'short_description', ''),
            "parameters": getattr(tool, 'parameters', {}),
            "requires_confirmation": getattr(tool, 'requires_confirmation', False),
            "is_dangerous": getattr(tool, 'is_dangerous', False)
        }
        
        self.logger.debug(f"Registered tool: {tool_name} in category: {category}")
        return True
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the registry."""
        if tool_name in self.tools:
            # Remove from tools
            del self.tools[tool_name]
            
            # Remove from categories
            for category_tools in self.tool_categories.values():
                category_tools.discard(tool_name)
            
            # Remove metadata
            if tool_name in self.tool_metadata:
                del self.tool_metadata[tool_name]
            
            self.logger.debug(f"Unregistered tool: {tool_name}")
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[Union[BaseTool, SimpleTool]]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List all available tools, optionally filtered by category."""
        if category is None:
            return list(self.tools.keys())
        
        return list(self.tool_categories.get(category, set()))
    
    def get_tool_categories(self) -> Dict[str, List[str]]:
        """Get all tool categories and their tools."""
        return {cat: list(tools) for cat, tools in self.tool_categories.items()}
    
    def get_tool_metadata(self, tool_name: str) -> Dict[str, Any]:
        """Get metadata for a specific tool."""
        return self.tool_metadata.get(tool_name, {})
    
    def disable_tool(self, tool_name: str) -> bool:
        """Disable a tool."""
        self.disabled_tools.add(tool_name)
        self.logger.debug(f"Disabled tool: {tool_name}")
        return True
    
    def enable_tool(self, tool_name: str) -> bool:
        """Enable a tool."""
        self.disabled_tools.discard(tool_name)
        self.logger.debug(f"Enabled tool: {tool_name}")
        return True
    
    def is_tool_disabled(self, tool_name: str) -> bool:
        """Check if a tool is disabled."""
        return tool_name in self.disabled_tools
    
    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tools."""
        return [name for name in self.tools.keys() if name not in self.disabled_tools]
    
    def get_disabled_tools(self) -> List[str]:
        """Get list of disabled tools."""
        return list(self.disabled_tools)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available (enabled) tools."""
        return self.get_enabled_tools()
    
    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for all registered tools."""
        return self.export_tool_schemas()
    
    def get_tool_schemas_short(self) -> Dict[str, Dict[str, Any]]:
        """Get short schemas for all registered tools."""
        schemas = {}
        
        for tool_name, tool in self.tools.items():
            # Skip disabled tools
            if tool_name in self.disabled_tools:
                continue
                
            if hasattr(tool, 'get_short_schema'):
                schemas[tool_name] = tool.get_short_schema()
            elif hasattr(tool, 'get_schema'):
                # Fallback: create short schema from full schema
                full_schema = tool.get_schema()
                short_schema = self._create_short_schema(full_schema)
                schemas[tool_name] = short_schema
            else:
                # Create basic short schema
                schemas[tool_name] = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": getattr(tool, 'short_description', getattr(tool, 'description', f'Tool: {tool_name}'))[:150],
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
        
        return schemas
    
    def _create_short_schema(self, full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create a short schema from a full schema."""
        if "function" not in full_schema:
            return full_schema
            
        function_info = full_schema["function"]
        short_schema = {
            "type": "function",
            "function": {
                "name": function_info.get("name", ""),
                "parameters": function_info.get("parameters", {})
            }
        }
        
        # Use short_description if available, otherwise truncate description
        if "short_description" in function_info:
            short_schema["function"]["description"] = function_info["short_description"]
        else:
            description = function_info.get("description", "")
            # Truncate long descriptions
            if len(description) > 150:
                short_schema["function"]["description"] = description[:150] + "..."
            else:
                short_schema["function"]["description"] = description
        
        return short_schema
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool."""
        schemas = self.export_tool_schemas()
        return schemas.get(tool_name)
    
    def categorize_tool(self, tool_name: str, category: str) -> None:
        """Add a tool to a category."""
        if tool_name in self.tools:
            # Remove from old category
            for cat, tools in self.tool_categories.items():
                tools.discard(tool_name)
            
            # Add to new category
            if category not in self.tool_categories:
                self.tool_categories[category] = set()
            self.tool_categories[category].add(tool_name)
            
            # Update metadata
            if tool_name in self.tool_metadata:
                self.tool_metadata[tool_name]["category"] = category
    
    def clear_registry(self) -> None:
        """Clear all tools from the registry."""
        self.tools.clear()
        for category_tools in self.tool_categories.values():
            category_tools.clear()
        self.tool_metadata.clear()
        self.disabled_tools.clear()
        self.logger.debug("Cleared tool registry")
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tool names in a specific category."""
        return list(self.tool_categories.get(category, set()))
    
    def search_tools(self, query: str) -> List[str]:
        """Search for tools by name or description."""
        query_lower = query.lower()
        matching_tools = []
        
        for tool_name, metadata in self.tool_metadata.items():
            if (query_lower in tool_name.lower() or 
                query_lower in metadata.get('description', '').lower()):
                matching_tools.append(tool_name)
        
        return matching_tools
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool registry."""
        return {
            "total_tools": len(self.tools),
            "enabled_tools": len(self.get_enabled_tools()),
            "disabled_tools": len(self.disabled_tools),
            "categories": len(self.tool_categories),
            "tools_by_category": {cat: len(tools) for cat, tools in self.tool_categories.items()}
        }
    
    def validate_tool_schema(self, tool: Union[BaseTool, SimpleTool]) -> bool:
        """Validate that a tool has the required schema."""
        required_attrs = ['name']
        
        for attr in required_attrs:
            if not hasattr(tool, attr):
                self.logger.error(f"Tool missing required attribute: {attr}")
                return False
        
        return True
    
    def export_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Export all tool schemas for external use."""
        schemas = {}
        
        for tool_name, tool in self.tools.items():
            if hasattr(tool, 'get_schema'):
                schemas[tool_name] = tool.get_schema()
            else:
                # Create basic schema
                schemas[tool_name] = {
                    "name": tool_name,
                    "description": getattr(tool, 'description', ''),
                    "parameters": getattr(tool, 'parameters', {})
                }
        
        return schemas
    
    def bulk_register_tools(self, tools: List[Union[BaseTool, SimpleTool]], 
                          category: str = "custom") -> None:
        """Register multiple tools at once."""
        for tool in tools:
            if self.validate_tool_schema(tool):
                self.register_tool(tool, category)
    
    def get_dangerous_tools(self) -> List[str]:
        """Get list of tools marked as dangerous."""
        return [
            name for name, metadata in self.tool_metadata.items()
            if metadata.get('is_dangerous', False)
        ]
    
    def get_tools_requiring_confirmation(self) -> List[str]:
        """Get list of tools that require confirmation."""
        return [
            name for name, metadata in self.tool_metadata.items()
            if metadata.get('requires_confirmation', False)
        ]


# Global registry instance
_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _global_registry


def register_tool(tool: Union[BaseTool, SimpleTool], category: str = "custom") -> None:
    """Register a tool in the global registry."""
    _global_registry.register_tool(tool, category)


def get_tool(tool_name: str) -> Optional[Union[BaseTool, SimpleTool]]:
    """Get a tool from the global registry."""
    return _global_registry.get_tool(tool_name)


def list_tools(category: Optional[str] = None) -> List[str]:
    """List tools from the global registry."""
    return _global_registry.list_tools(category) 