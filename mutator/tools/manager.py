"""
Simplified Tool Manager for the Coding Agent Framework.

This module provides a streamlined ToolManager that uses the centralized registry
and separate modules for different concerns.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from .base import BaseTool, ToolSafetyChecker
from .decorator import SimpleTool
from .mcp_server import MCPServer, MCPServerManager
from .registry import ToolRegistry, get_global_registry
from ..core.config import SafetyConfig, MCPServerConfig
from ..core.path_utils import normalize_path_for_response
from ..core.types import ToolCall, ToolResult, SafetyCheck, ConfirmationCallback


class ToolManager:
    """Simplified tool manager that uses the centralized registry."""
    
    def __init__(self, 
                 safety_config: Optional[SafetyConfig] = None,
                 confirmation_callback: Optional[ConfirmationCallback] = None,
                 disabled_tools: Optional[List[str]] = None,
                 working_directory: Optional[str] = None,
                 llm_client: Optional[Any] = None,
                 config: Optional[Any] = None,
                 registry: Optional[ToolRegistry] = None):
        """Initialize the tool manager."""
        self.safety_config = safety_config or SafetyConfig()
        self.confirmation_callback = confirmation_callback
        self.working_directory = str(Path(working_directory or ".").absolute())
        self.llm_client = llm_client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Use provided registry or global registry
        self.registry = registry or get_global_registry()
        
        # Store disabled tools list for compatibility
        self._disabled_tools_list = disabled_tools or []
        
        # Initialize disabled tools in registry
        if disabled_tools:
            for tool_name in disabled_tools:
                self.registry.disable_tool(tool_name)
        
        # MCP server management
        self.mcp_manager = MCPServerManager()
        
        # Execution statistics
        self.execution_stats: Dict[str, Dict[str, Any]] = {}
        
        # Batch processing support
        self.batch_processor = None
        self._initialize_batch_processor()
        
        # Register get_tool_help implementation
        self.register_get_tool_help_implementation()
    
    @property
    def tools(self) -> Dict[str, Union[BaseTool, SimpleTool]]:
        """Get all registered tools as a dictionary (for backward compatibility)."""
        tools_dict = {}
        for tool_name in self.registry.list_tools():
            # Skip disabled tools
            if tool_name in self.registry.disabled_tools:
                continue
            tool = self.registry.get_tool(tool_name)
            if tool:
                tools_dict[tool_name] = tool
        return tools_dict
    
    @property
    def disabled_tools(self) -> set:
        """Get disabled tools as a set (for backward compatibility)."""
        return set(self._disabled_tools_list)
    
    def _initialize_batch_processor(self) -> None:
        """Initialize the batch processor if LLM client is available."""
        if self.llm_client and self.config:
            try:
                # BatchProcessor class doesn't exist yet, skip initialization
                # from .batch_tools import BatchProcessor
                # self.batch_processor = BatchProcessor(self.llm_client, self, self.config)
                self.batch_processor = None
                self.logger.debug("Batch processor initialization skipped (not implemented)")
            except ImportError:
                self.logger.warning("Batch tools not available")
    
    def initialize_batch_processor(self) -> None:
        """Public method to initialize the batch processor."""
        self._initialize_batch_processor()
    
    def normalize_path(self, path: Union[str, Path]) -> str:
        """Normalize a path for tool responses."""
        return normalize_path_for_response(path, self.working_directory)
    
    def _normalize_paths_in_result(self, result: Any) -> Any:
        """Recursively normalize paths in tool results."""
        if isinstance(result, dict):
            normalized = {}
            for key, value in result.items():
                # Check if this is likely a path field
                if key in ['file_path', 'path', 'directory', 'repository_path', 'local_path', 'backup_path']:
                    if isinstance(value, (str, Path)):
                        normalized[key] = self.normalize_path(value)
                    else:
                        normalized[key] = value
                else:
                    normalized[key] = self._normalize_paths_in_result(value)
            return normalized
        elif isinstance(result, list):
            return [self._normalize_paths_in_result(item) for item in result]
        else:
            return result
    
    def register_tool(self, tool: Union[BaseTool, SimpleTool], category: Optional[str] = None) -> bool:
        """Register a tool."""
        if not isinstance(tool, (BaseTool, SimpleTool)):
            self.logger.error(f"Invalid tool type: {type(tool)}")
            return False
        
        success = self.registry.register_tool(tool, category)
        
        if success:
            # Initialize execution stats
            self.execution_stats[tool.name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0
            }
        
        return success
    
    def register_function(self, func: Callable) -> None:
        """Register a function as a tool."""
        if func is None:
            self.logger.warning("Attempted to register None as a tool, skipping")
            return
            
        if isinstance(func, SimpleTool):
            tool_name = func.name
            tool = func
        else:
            # Use the tool decorator to create a SimpleTool
            from .decorator import tool
            tool = tool(func)
            tool_name = tool.name
        
        # Skip registration if tool is disabled
        if tool_name in self.registry.disabled_tools:
            self.logger.debug(f"Skipping registration of disabled tool: {tool_name}")
            return
        
        # Register in registry
        self.registry.register_tool(tool)
        
        # Initialize execution stats
        self.execution_stats[tool_name] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
    
    def register_builtin_tools(self) -> None:
        """Register built-in tools from the builtin module."""
        from .builtin import get_builtin_tools
        
        builtin_tools = get_builtin_tools()
        for tool_name, tool in builtin_tools.items():
            self.register_tool(tool)
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        success = self.registry.unregister_tool(tool_name)
        
        if success and tool_name in self.execution_stats:
            del self.execution_stats[tool_name]
        
        return success
    
    def get_tool(self, tool_name: str) -> Optional[Union[BaseTool, SimpleTool]]:
        """Get a tool by name."""
        return self.registry.get_tool(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return self.registry.get_available_tools()  # Use get_available_tools which filters disabled tools
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.registry.get_available_tools()
    
    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for all registered tools."""
        return self.registry.get_tool_schemas()
    
    def get_tool_schemas_short(self) -> Dict[str, Dict[str, Any]]:
        """Get short schemas for all registered tools."""
        return self.registry.get_tool_schemas_short()
    
    def get_tool_schema_full(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get full schema for a specific tool."""
        return self.registry.get_tool_schema(tool_name)
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a tool."""
        metadata = self.registry.get_tool_metadata(tool_name)
        if not metadata:
            return None
        
        stats = self.execution_stats.get(tool_name, {})
        
        return {
            "name": tool_name,  # Add the tool name
            **metadata,
            "stats": stats
        }
    
    def get_tool_list(self) -> List[Dict[str, Any]]:
        """Get list of all tools with their info."""
        tools = []
        for tool_name in self.registry.list_tools():
            info = self.get_tool_info(tool_name)
            if info:
                tools.append(info)
        return tools
    
    def categorize_tool(self, tool_name: str, category: str) -> None:
        """Add a tool to a category."""
        self.registry.categorize_tool(tool_name, category)
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tool names in a category."""
        return self.registry.get_tools_by_category(category)
    
    def disable_tool(self, tool_name: str) -> bool:
        """Disable a tool."""
        success = self.registry.disable_tool(tool_name)
        if success:
            # Remove from execution stats if it exists
            if tool_name in self.execution_stats:
                del self.execution_stats[tool_name]
        return success
    
    def enable_tool(self, tool_name: str) -> bool:
        """Enable a previously disabled tool."""
        return self.registry.enable_tool(tool_name)
    
    def is_tool_disabled(self, tool_name: str) -> bool:
        """Check if a tool is disabled."""
        return self.registry.is_tool_disabled(tool_name)
    
    def get_disabled_tools(self) -> List[str]:
        """Get list of disabled tool names."""
        return self.registry.get_disabled_tools()
    
    async def execute_tool(self, tool_name_or_call: Union[str, ToolCall], parameters: Dict[str, Any] = None) -> ToolResult:
        """Execute a tool call."""
        start_time = time.time()
        
        # Handle both string tool name and ToolCall object
        if isinstance(tool_name_or_call, str):
            tool_name = tool_name_or_call
            tool_arguments = parameters or {}
        else:
            tool_call = tool_name_or_call
            tool_name = tool_call.name
            tool_arguments = tool_call.arguments
        
        # Check if it's an MCP tool
        if "." in tool_name:
            server_name, mcp_tool_name = tool_name.split(".", 1)
            server = self.mcp_manager.get_server(server_name)
            if server:
                return await server.call_tool(mcp_tool_name, tool_arguments)
        
        # Get tool from registry
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        # Update execution stats
        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0
            }
        
        self.execution_stats[tool_name]["total_calls"] += 1
        
        try:
            # Log function call start with structured format
            self.logger.info(f"Calling {tool_name}")
            if tool_arguments:
                self.logger.info("Params:")
                for key, value in tool_arguments.items():
                    # Truncate very long values for readability
                    if isinstance(value, str) and len(value) > 200:
                        display_value = value[:200] + "..."
                    else:
                        display_value = value
                    self.logger.info(f"- {key}: {display_value}")
            else:
                self.logger.info("Params: None")
            
            # Perform safety checks
            safety_checks = self._perform_safety_checks(tool, tool_arguments)
            
            # Check if any safety checks failed
            failed_checks = [check for check in safety_checks if not check.passed]
            if failed_checks:
                critical_failures = [check for check in failed_checks if check.severity == "error"]
                if critical_failures:
                    error_msg = f"Safety check failed: {critical_failures[0].message}"
                    self.logger.error(f"Output: {error_msg}")
                    return ToolResult(
                        tool_name=tool_name,
                        success=False,
                        error=error_msg,
                        safety_checks=safety_checks
                    )
            
            # Execute the tool
            result = await tool.execute(**tool_arguments)
            
            # Normalize paths in the result
            if result.success and result.result:
                result.result = self._normalize_paths_in_result(result.result)
            
            # Update execution time and stats
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.safety_checks = safety_checks
            
            # Log structured output
            if result.success:
                self.execution_stats[tool_name]["successful_calls"] += 1
                # Format output for logging
                if result.result is not None:
                    # Truncate very long results for readability
                    if isinstance(result.result, str) and len(result.result) > 500:
                        display_result = result.result[:500] + "..."
                    elif isinstance(result.result, dict):
                        display_result = str(result.result)[:500] + "..." if len(str(result.result)) > 500 else result.result
                    else:
                        display_result = result.result
                    self.logger.info(f"Output: {display_result}")
                else:
                    self.logger.info("Output: None")
            else:
                self.execution_stats[tool_name]["failed_calls"] += 1
                self.logger.error(f"Output: {result.error}")
            
            self.execution_stats[tool_name]["total_execution_time"] += execution_time
            total_calls = self.execution_stats[tool_name]["total_calls"]
            total_time = self.execution_stats[tool_name]["total_execution_time"]
            self.execution_stats[tool_name]["average_execution_time"] = total_time / total_calls
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.execution_stats[tool_name]["failed_calls"] += 1
            self.execution_stats[tool_name]["total_execution_time"] += execution_time
            
            error_msg = str(e)
            self.logger.error(f"Tool execution failed for '{tool_name}': {error_msg}")
            self.logger.error(f"Output: {error_msg}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _perform_safety_checks(self, tool: Union[BaseTool, SimpleTool], arguments: Dict[str, Any]) -> List[SafetyCheck]:
        """Perform safety checks for a tool."""
        checks = []
        
        # Use tool's own safety checks if available
        if hasattr(tool, 'perform_safety_checks'):
            checks.extend(tool.perform_safety_checks(**arguments))
        
        # Additional safety checks based on tool type
        tool_name = tool.name
        
        if 'shell' in tool_name.lower():
            command = arguments.get('command', '')
            if command:
                safety_check = ToolSafetyChecker.check_shell_command(
                    command, 
                    self.safety_config.blocked_shell_commands,
                    self.safety_config.allowed_shell_commands
                )
                checks.append(safety_check)
        
        elif any(keyword in tool_name.lower() for keyword in ['file', 'read', 'write', 'edit']):
            file_path = arguments.get('file_path', arguments.get('path', ''))
            if file_path:
                safety_check = ToolSafetyChecker.check_file_access(file_path)
                checks.append(safety_check)
        
        return checks
    
    def get_execution_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all tools."""
        return self.execution_stats.copy()
    
    async def add_mcp_server(self, config: MCPServerConfig) -> bool:
        """Add an MCP server."""
        return await self.mcp_manager.add_server(config)
    
    async def remove_mcp_server(self, server_name: str) -> bool:
        """Remove an MCP server."""
        return await self.mcp_manager.remove_server(server_name)
    
    async def start_mcp_servers(self) -> Dict[str, bool]:
        """Start all MCP servers."""
        return await self.mcp_manager.start_all()
    
    async def stop_mcp_servers(self) -> None:
        """Stop all MCP servers."""
        await self.mcp_manager.stop_all()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all tools and servers."""
        tool_stats = self.registry.get_tool_stats()
        mcp_health = await self.mcp_manager.health_check()
        
        return {
            "status": "healthy",
            "tools": tool_stats,
            "mcp_servers": mcp_health
        }
    
    def create_get_tool_help_implementation(self) -> Callable:
        """Create a custom implementation for the get_tool_help tool."""
        def get_tool_help_impl(tool_name: str) -> Dict[str, Any]:
            """Get detailed help and full description for a specific tool."""
            try:
                # Get the full schema for the requested tool
                schema = self.get_tool_schema_full(tool_name)
                if not schema:
                    return {
                        "error": f"Tool '{tool_name}' not found",
                        "available_tools": list(self.registry.list_tools())
                    }
                
                # Extract detailed information
                function_info = schema.get("function", {})
                description = function_info.get("description", "No description available")
                short_description = function_info.get("short_description", "")
                parameters = function_info.get("parameters", {})
                
                # Try to extract long description from the full description
                long_description = self._extract_long_description(description)
                
                # Get tool statistics if available
                stats = self.execution_stats.get(tool_name, {})
                
                return {
                    "tool_name": tool_name,
                    "description": long_description,  # Return the full/long description
                    "short_description": short_description,
                    "parameters": parameters,
                    "usage_stats": stats,
                    "help": f"Full documentation for {tool_name} tool"
                }
            except Exception as e:
                return {"error": f"Failed to get help for tool '{tool_name}': {str(e)}"}
        
        return get_tool_help_impl
    
    def _extract_long_description(self, description: str) -> str:
        """Extract long description from description text."""
        if not description:
            return ""
        
        # Try to extract from <long_description> tags
        import re
        long_match = re.search(r'<long_description>(.*?)</long_description>', 
                              description, re.DOTALL)
        if long_match:
            return long_match.group(1).strip()
        
        # Fallback: return full description
        return description
    
    def register_get_tool_help_implementation(self) -> None:
        """Register the get_tool_help tool with proper implementation."""
        # Create the implementation
        impl = self.create_get_tool_help_implementation()
        
        # Create a SimpleTool with the implementation
        from .decorator import SimpleTool
        import inspect
        
        # Create schema for the tool
        schema = {
            "type": "function",
            "function": {
                "name": "get_tool_help",
                "description": "Get detailed help and full description for a specific tool when you need more information about how to use it properly.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool to get help for"
                        }
                    },
                    "required": ["tool_name"]
                }
            }
        }
        
        # Create and register the tool
        tool = SimpleTool("get_tool_help", impl, schema)
        self.register_tool(tool)

    async def shutdown(self) -> None:
        """Shutdown the tool manager."""
        self.logger.debug("Shutting down tool manager...")
        
        # Stop all MCP servers
        await self.stop_mcp_servers()
        
        # Clear registry
        self.registry.clear()
        self.execution_stats.clear()
        
        self.logger.debug("Tool manager shutdown complete") 