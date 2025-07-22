"""
Task planner for the Coding Agent Framework.

This module handles the planning and orchestration of coding tasks.
The planner creates simple plans and lets the LLM decide when to use 
the task tool for sub-task execution.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.types import TaskType
from ..core.config import AgentConfig
from ..llm.client import LLMClient
from ..context.manager import ContextManager


class TaskPlanner:
    """Creates simple task plans and lets the LLM handle execution via tools."""
    
    def __init__(self, llm_client: LLMClient, context_manager: ContextManager, config: AgentConfig):
        self.llm_client = llm_client
        self.context_manager = context_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def create_task_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create a prompt for the LLM to execute the task."""
        try:
            self.logger.debug(f"Creating task prompt for: {task}")
            
            # Get relevant context
            relevant_context = []
            if context:
                search_results = self.context_manager.search_context(task, limit=10)
                relevant_context = search_results
            
            # Create the task prompt
            prompt = f"""Please help me with the following task:

Task: {task}

"""
            
            # Add relevant context if available
            if relevant_context:
                prompt += "Relevant Context:\n"
                for i, context_item in enumerate(relevant_context[:5], 1):
                    prompt += f"{i}. {context_item.get('content', str(context_item))}\n"
                prompt += "\n"
            
            # Add guidance
            guidance = self._get_task_guidance()
            prompt += guidance
            
            return prompt
                
        except Exception as e:
            self.logger.error(f"Failed to create task prompt: {str(e)}")
            return f"Error creating task prompt: {str(e)}"
    
    def _get_task_guidance(self) -> str:
        """Get guidance for task execution."""
        return """## Available Tools & Approach:

**Direct Tools (for simple operations):**
- `read_file()`, `edit_file()` - File operations
- `list_directory()`, `search_files_by_name()` - Project exploration
- `search_files_by_content()`, `codebase_search()` - Finding information
- `run_shell()` - Command execution
- `git_*` tools - Version control operations

**Task Tool (for complex operations):**
- Use `task` tool for multi-step operations across multiple files
- Perfect for refactoring, implementing features, or system changes
- Sub-agents handle complex workflows with full tool access

**Recommended Approach:**
1. Start with exploration to understand the project structure
2. Use direct tools for simple, focused operations
3. Use the task tool when work spans multiple files or requires coordination
4. Build context before making changes

Choose the most appropriate tools based on the task complexity and scope."""

    def get_task_guidance(self, task: str) -> str:
        """Get guidance for the LLM based on task type."""
        return """For effective task execution:

**Simple Operations:**
- Use direct tools (read_file, edit_file) for file operations
- Use search_files_by_content, codebase_search for finding information
- Use run_shell for simple commands
- Use git_* tools for version control operations

**Complex Operations:**
- Use the 'task' tool for multi-step operations spanning multiple files
- Ideal for refactoring, implementing features, or system changes
- Sub-agents handle complex workflows with full tool access

The AI will automatically determine the best approach based on task complexity.""" 