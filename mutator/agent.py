"""
Mutator Framework - Main Agent Class

This module contains the main Mutator class that orchestrates the entire framework.
The agent manages LLM interactions, tool execution, context management, and task processing.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator, Union
from pathlib import Path

from .core.config import AgentConfig
from .core.types import AgentEvent, ExecutionMode
from .llm.client import LLMClient
from .tools.manager import ToolManager
from .context.manager import ContextManager
from .execution.executor import TaskExecutor


class Mutator:
    """
    Main agent class that coordinates all framework components.
    
    This class provides the high-level interface for interacting with the coding agent.
    It manages initialization, tool registration, context management, and task execution.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the coding agent with the given configuration.
        
        Args:
            config: Agent configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.llm_client = None
        self.tool_manager = None
        self.context_manager = None
        self.executor = None
        
        # Track initialization state
        self._initialized = False
        
        # Set up logging
        self._setup_logging()
        
        self.logger.info(f"Mutator initialized with config: {config.model_dump()}")
    
    async def initialize(self) -> None:
        """
        Initialize all agent components.
        
        This must be called before using the agent.
        """
        if self._initialized:
            self.logger.warning("Agent already initialized")
            return
        
        self.logger.info("Initializing Mutator...")
        
        # Initialize LLM client
        self.llm_client = LLMClient(self.config.llm_config)
        
        # Initialize tool manager
        self.tool_manager = ToolManager(
            disabled_tools=self.config.disabled_tools,
            config=self.config
        )
        
        # Register built-in tools
        self._register_builtin_tools()
        
        # Register tool schemas with LLM client
        self._register_tools_with_llm()
        
        # Initialize context manager
        self.context_manager = ContextManager(
            self.config.context_config,
            self.config.vector_store_config,
            self.config.working_directory
        )
        
        # Initialize executor
        from .execution.planner import TaskPlanner
        self.planner = TaskPlanner(
            llm_client=self.llm_client,
            context_manager=self.context_manager,
            config=self.config
        )
        
        self.executor = TaskExecutor(
            llm_client=self.llm_client,
            tool_manager=self.tool_manager,
            context_manager=self.context_manager,
            planner=self.planner,
            config=self.config
        )
        
        # Set up LangChain components after tools are registered
        self.executor.setup_langchain_components()
        
        self._initialized = True
        self.logger.info("Mutator initialization complete")
    
    async def execute_task(self, task: str, execution_mode: ExecutionMode = ExecutionMode.AGENT, context: Optional[Dict[str, Any]] = None) -> AsyncIterator[AgentEvent]:
        """
        Execute a coding task.
        
        Args:
            task: The task description
            execution_mode: How to execute the task (AGENT or CHAT)
            context: Additional context for the task
        
        Yields:
            AgentEvent: Events during task execution
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        self.logger.info(f"Executing task: {task}")
        
        async for event in self.executor.execute_task(task, execution_mode, context):
            yield event
    
    async def interactive_chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> AsyncIterator[AgentEvent]:
        """
        Have an interactive conversation with the agent that can use tools.
        
        Args:
            message: User message
            context: Additional context for the conversation
        
        Yields:
            AgentEvent: Events during interaction including tool calls
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        self.logger.debug(f"Interactive chat: {message}")
        
        # Pass the raw user message directly to the executor.
        # The agent should be responsible for gathering its own context via tools.
        async for event in self.executor.execute_interactive_chat(message, context):
            yield event
    
    async def get_project_context(self) -> Dict[str, Any]:
        """Get the current project context."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        return self.context_manager.get_context_summary()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of all agent components.
        
        Returns:
            Dict[str, Any]: Health status information including:
                - status: Overall health status ("healthy", "degraded", "unhealthy")
                - llm_ready: Whether LLM client is ready
                - context_ready: Whether context manager is ready
                - tool_count: Number of available tools
                - indexed_files: Number of indexed files
                - details: Detailed health information from each component
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        health_info = {
            "status": "healthy",
            "llm_ready": False,
            "context_ready": False,
            "tool_count": 0,
            "indexed_files": 0,
            "details": {}
        }
        
        try:
            # Check LLM client health
            if self.llm_client:
                llm_health = await self.llm_client.health_check()
                health_info["llm_ready"] = llm_health
                health_info["details"]["llm"] = {
                    "ready": llm_health,
                    "model": self.config.llm_config.model,
                    "provider": self.config.llm_config.provider.value if self.config.llm_config.provider else "unknown"
                }
            
            # Check context manager health
            if self.context_manager:
                context_health = self.context_manager.health_check()
                health_info["context_ready"] = context_health.get("status") == "healthy"
                health_info["indexed_files"] = context_health.get("indexed_files_count", 0)
                health_info["details"]["context"] = context_health
            
            # Check tool manager health
            if self.tool_manager:
                tool_health = await self.tool_manager.health_check()
                available_tools = self.tool_manager.list_tools()
                health_info["tool_count"] = len(available_tools)
                health_info["details"]["tools"] = tool_health
            
            # Check executor health if available
            if self.executor:
                executor_status = await self.executor.get_execution_status()
                health_info["details"]["executor"] = executor_status
            
            # Determine overall health status
            if not health_info["llm_ready"]:
                health_info["status"] = "unhealthy"
            elif not health_info["context_ready"]:
                health_info["status"] = "degraded"
            elif health_info["tool_count"] == 0:
                health_info["status"] = "degraded"
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
        
        return health_info
    
    async def search_context(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant context."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        return self.context_manager.search_context(query, limit=max_results)
    
    async def add_context_from_file(self, file_path: str) -> None:
        """Add context from a file."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        if self.context_manager:
            await self.context_manager.add_file_context(file_path)
    
    async def add_context_from_directory(self, directory_path: str, recursive: bool = True) -> None:
        """Add context from a directory."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        if self.context_manager:
            await self.context_manager.add_directory_context(directory_path, recursive)
    
    async def update_context(self) -> None:
        """Update the context with latest project state."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        if self.context_manager:
            await self.context_manager.update_context()
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        return self.tool_manager.list_tools()
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        return self.tool_manager.get_tool_info(tool_name)
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        return self.tool_manager.is_tool_available(tool_name)
    
    def disable_tool(self, tool_name: str) -> None:
        """Disable a specific tool."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        self.tool_manager.disable_tool(tool_name)
    
    def enable_tool(self, tool_name: str) -> None:
        """Enable a specific tool."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        self.tool_manager.enable_tool(tool_name)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.context_manager:
            await self.context_manager.cleanup()
        
        if self.executor:
            await self.executor.cleanup()
        
        self.logger.info("Mutator cleanup complete")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_level = getattr(logging, self.config.logging_level.upper(), logging.INFO)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('mutator.log')
            ]
        )
        
        # Set specific logger levels
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    def _register_builtin_tools(self) -> None:
        """Register built-in tools with the tool manager."""
        # Import here to avoid circular imports
        from .tools.builtin import (
            # Simple tools
            read_file, edit_file, create_file,
            run_shell,
            list_directory,
            web_search, fetch_url,
            # Migrated tools
            mermaid,
            delegate_task,
            # Batch tools
            process_search_files_by_name, process_search_files_by_content,
            process_search_files_sementic
        )
        
        # Register simple tools
        self.tool_manager.register_function(read_file)
        self.tool_manager.register_function(edit_file)
        self.tool_manager.register_function(create_file)
        
        self.tool_manager.register_function(run_shell)
        
        self.tool_manager.register_function(list_directory)
        
        # Only register web_search if it's available (API keys configured)
        if web_search is not None:
            self.tool_manager.register_function(web_search)
        self.tool_manager.register_function(fetch_url)
        
        # Register migrated tools
        self.tool_manager.register_function(mermaid)
        # Don't register the placeholder get_tool_help - use custom implementation instead
        # (get_tool_help is already registered in ToolManager.__init__)
        self.tool_manager.register_function(delegate_task)
        
        # Register batch tools
        self.tool_manager.register_function(process_search_files_by_name)
        self.tool_manager.register_function(process_search_files_by_content)
        self.tool_manager.register_function(process_search_files_sementic)
    
    def _register_tools_with_llm(self) -> None:
        """Register tool schemas with the LLM client for function calling."""
        # Get tool schemas - use short descriptions if configured
        if self.config.llm_config.use_short_tool_descriptions:
            tool_schemas = self.tool_manager.get_tool_schemas_short()
        else:
            tool_schemas = self.tool_manager.get_tool_schemas()
        
        # Register each tool schema with the LLM client
        for tool_name, schema in tool_schemas.items():
            # Extract the function part from the schema for LiteLLM
            # Schema format: {"type": "function", "function": {...}}
            # LiteLLM expects only the function part: {...}
            if "function" in schema:
                function_schema = schema["function"]
                self.llm_client.register_function(tool_name, None, function_schema)
            else:
                # Fallback to the whole schema if it's already in the right format
                self.llm_client.register_function(tool_name, None, schema)
            
        self.logger.debug(f"Registered {len(tool_schemas)} tools with LLM client for function calling (using {'short' if self.config.llm_config.use_short_tool_descriptions else 'full'} descriptions)")
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"Mutator(initialized={self._initialized}, tools={len(self.tool_manager.list_tools()) if self.tool_manager else 0})"
    
    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return f"Mutator(config={self.config}, initialized={self._initialized})" 