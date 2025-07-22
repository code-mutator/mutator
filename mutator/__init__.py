"""
Coding Agent Framework - A comprehensive framework for building AI-powered coding agents.

This framework provides a complete solution for creating intelligent coding assistants
that can understand, analyze, and modify code using Large Language Models (LLMs).

Key Features:
- Multi-LLM support via LiteLLM (OpenAI, Anthropic, Google, etc.)
- Extensible tool system with built-in file operations, shell commands, and more
- Intelligent context management with vector-based code understanding
- Task planning and execution with sub-agent delegation
- Safety features and user confirmations
- Streaming execution with real-time feedback
- CLI interface for easy interaction

Quick Start:
```python
from mutator import Mutator, AgentConfig

# Create and initialize agent
config = AgentConfig()
agent = Mutator(config)
await agent.initialize()

# Execute a task
async for event in agent.execute_task("Add type hints to my Python functions"):
    print(event)
```

For more examples, see the `examples/` directory.
"""

from .__version__ import __version__

from .core.types import (
    TaskType,
    TaskStatus,
    ExecutionMode,
    AgentEvent,
    ConversationTurn,
    ToolResult,
    LLMResponse,
    TaskResult,
)

from .core.config import (
    AgentConfig,
    LLMConfig,
    SafetyConfig,
    ContextConfig,
    ExecutionConfig,
    ConfigManager,
)

from .agent import Mutator

from .llm.client import LLMClient
from .context.manager import ContextManager
from .tools import ToolManager

# Import all built-in tools for easy access
from .tools.builtin import *

# Convenience functions
async def create_agent(project_path=None, config=None):
    """
    Create and initialize a coding agent.
    
    Args:
        project_path: Path to the project directory (optional)
        config: Agent configuration (optional, defaults to AgentConfig())
    
    Returns:
        Mutator: Initialized agent instance
    """
    if config is None:
        config = AgentConfig()
    
    if project_path is not None:
        # Convert to string and set working directory
        project_path_str = str(project_path)
        config.working_directory = project_path_str
        config.context_config.project_path = project_path_str
    
    agent = Mutator(config)
    await agent.initialize()
    return agent


async def execute_task(task, execution_mode=ExecutionMode.AGENT, project_path=None, config=None, **kwargs):
    """
    Execute a task using a coding agent.
    
    Args:
        task: Task description to execute
        execution_mode: Execution mode (AGENT or CHAT)
        project_path: Path to the project directory (optional)
        config: Agent configuration (optional)
        **kwargs: Additional arguments passed to execute_task
    
    Returns:
        AsyncIterator[AgentEvent]: Stream of execution events
    """
    agent = await create_agent(project_path=project_path, config=config)
    
    try:
        async for event in agent.execute_task(task, execution_mode=execution_mode, **kwargs):
            yield event
    finally:
        # Clean up agent resources
        if hasattr(agent, 'cleanup'):
            await agent.cleanup()


async def chat(message, project_path=None, config=None, **kwargs):
    """
    Chat with a coding agent.
    
    Args:
        message: Message to send to the agent
        project_path: Path to the project directory (optional)
        config: Agent configuration (optional)
        **kwargs: Additional arguments passed to chat
    
    Returns:
        LLMResponse: Agent response
    """
    agent = await create_agent(project_path=project_path, config=config)
    
    try:
        return await agent.chat(message, **kwargs)
    finally:
        # Clean up agent resources
        if hasattr(agent, 'cleanup'):
            await agent.cleanup()


__all__ = [
    # Version
    "__version__",
    
    # Core types
    "TaskType",
    "TaskStatus", 
    "ExecutionMode",
    "AgentEvent",
    "ConversationTurn",
    "ToolResult",
    "LLMResponse",
    "TaskResult",
    
    # Configuration
    "AgentConfig",
    "LLMConfig",
    "SafetyConfig",
    "ContextConfig",
    "ExecutionConfig",
    "ConfigManager",
    
    # Main classes
    "Mutator",
    "LLMClient",
    "ContextManager",
    "ToolManager",
    
    # Convenience functions
    "create_agent",
    "execute_task",
    "chat",
    
    # Built-in tools (imported from tools.builtin)
    "read_file",
    "edit_file", 
    "create_file",
    "run_shell",
    "search_files_by_name",
    "search_files_by_content",
    "list_directory",
    "web_search",
    "fetch_url",
    "search_files_sementic",
    "mermaid",
    "delegate_task",
    "process_search_files_by_name",
    "process_search_files_by_content",
    "process_search_files_sementic",
] 