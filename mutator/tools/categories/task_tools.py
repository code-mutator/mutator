"""
Task and project management tools for the Coding Agent Framework.
"""

import os
import re
import json
import uuid
import asyncio
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

from ..decorator import tool


@tool
async def delegate_task(task_description: str, expected_output: str, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    <short_description>Delegate a specific task to a sub-agent and return a comprehensive summary.</short_description>
    
    <long_description>
    This tool creates a dedicated sub-agent to perform a specific task and returns a detailed summary
    of the results. The sub-agent has full access to all tools and can perform complex operations.

    ## Important Notes

    1. **Task Delegation**:
       - Creates a specialized sub-agent for the specific task
       - Sub-agent has full access to file system, search, and analysis tools
       - Task is executed in isolation with complete context provided

    2. **Expected Output**:
       - Caller must specify what kind of output they expect
       - This guides the sub-agent's focus and response format
       - Examples: "analysis summary", "list of findings", "code modifications made"

    3. **Context Data**:
       - Optional metadata about the task (file paths, line numbers, etc.)
       - Helps provide additional context to the sub-agent
       - Can include search results, file information, or any relevant data

    4. **Sub-Agent Capabilities**:
       - Full access to file system tools (read, write, edit, create)
       - Search and discovery tools (grep, semantic search, etc.)
       - Code analysis and formatting tools
       - Shell command execution capabilities

    ## Examples

    - File analysis: `delegate_task("Analyze the authentication system in auth.py", "summary of how authentication works and any security concerns")`
    - Code review: `delegate_task("Review the payment processing code", "list of potential issues and improvement suggestions")`
    - Bug investigation: `delegate_task("Investigate the memory leak in the server", "root cause analysis and proposed fix")`
    - Documentation: `delegate_task("Document the API endpoints", "structured documentation of all endpoints")`

    ## Use Cases

    - Complex analysis tasks requiring multiple tool calls
    - Code review and quality assessment
    - Bug investigation and debugging
    - Documentation generation
    - Refactoring planning
    - Security audits
    - Performance analysis
    </long_description>

    Args:
        task_description: Complete description of the task to be performed
        expected_output: Description of what output format/content is expected
        context_data: Optional metadata and context information for the task
    
    Returns:
        Dict containing the task results, summary, and execution details
    """
    try:
        # Import here to avoid circular imports
        from ...agent import Mutator
        from ...core.config import AgentConfig
        from ...core.types import ExecutionMode
        
        # Create sub-agent configuration
        config = AgentConfig()
        # Import here to avoid circular imports
        from ..decorator import get_working_directory
        config.working_directory = get_working_directory()
        
        # Create and initialize the sub-agent
        sub_agent = Mutator(config)
        await sub_agent.initialize()
        
        # Prepare the complete task prompt
        task_prompt = _prepare_task_prompt(task_description, expected_output, context_data)
        
        # Execute the task
        task_output = []
        execution_events = []
        tool_calls = []
        
        async for event in sub_agent.execute_task(
            task_prompt,
            execution_mode=ExecutionMode.AGENT
        ):
            execution_events.append(event)
            
            # Extract meaningful output
            if event.event_type == "llm_response":
                content = event.data.get("content", "")
                if content:
                    task_output.append(content)
            elif event.event_type == "tool_call_completed":
                tool_name = event.data.get("tool_name", "")
                success = event.data.get("success", False)
                result = event.data.get("result", {})
                
                tool_calls.append({
                    "tool_name": tool_name,
                    "success": success,
                    "result": result
                })
                
                if success:
                    task_output.append(f"✅ {tool_name} completed successfully")
                else:
                    error = event.data.get("error", "Unknown error")
                    task_output.append(f"❌ {tool_name} failed: {error}")
        
        # Determine success based on execution
        success = len(execution_events) > 0 and not any(
            event.event_type == "error" for event in execution_events
        )
        
        # Extract the final response (last LLM response)
        final_response = ""
        for event in reversed(execution_events):
            if event.event_type == "llm_response":
                final_response = event.data.get("content", "")
                break
        
        # Generate comprehensive summary
        summary = _generate_task_summary(task_output, tool_calls, success, final_response)
        
        return {
            "success": success,
            "task_description": task_description,
            "expected_output": expected_output,
            "final_response": final_response,
            "summary": summary,
            "tool_calls_made": len(tool_calls),
            "successful_tool_calls": len([tc for tc in tool_calls if tc["success"]]),
            "failed_tool_calls": len([tc for tc in tool_calls if not tc["success"]]),
            "execution_events": len(execution_events),
            "context_data": context_data or {}
        }
        
    except Exception as e:
        logging.error(f"Task delegation failed: {str(e)}")
        return {
            "success": False,
            "task_description": task_description,
            "expected_output": expected_output,
            "error": f"Task delegation failed: {str(e)}",
            "summary": f"Task failed with error: {str(e)}",
            "tool_calls_made": 0,
            "successful_tool_calls": 0,
            "failed_tool_calls": 0,
            "execution_events": 0,
            "context_data": context_data or {}
        }


def _prepare_task_prompt(task_description: str, expected_output: str, context_data: Dict[str, Any] = None) -> str:
    """Prepare a comprehensive task prompt for the sub-agent."""
    prompt_parts = [
        "You are a specialized sub-agent tasked with performing a specific operation.",
        "",
        f"**TASK**: {task_description}",
        "",
        f"**EXPECTED OUTPUT**: {expected_output}",
        "",
        "**INSTRUCTIONS**:",
        "1. Perform the task systematically and thoroughly",
        "2. Use all available tools as needed to gather information and complete the task",
        "3. Provide a comprehensive response that matches the expected output format",
        "4. Be specific and detailed in your analysis/findings",
        "5. If you encounter errors, handle them gracefully and report them",
        "6. Focus on delivering exactly what was requested in the expected output",
        ""
    ]
    
    # Add context data if provided
    if context_data:
        prompt_parts.extend([
            "**CONTEXT DATA**:",
            "The following additional context is provided for this task:",
            ""
        ])
        
        for key, value in context_data.items():
            if isinstance(value, (dict, list)):
                prompt_parts.append(f"- {key}: {json.dumps(value, indent=2)}")
            else:
                prompt_parts.append(f"- {key}: {value}")
        
        prompt_parts.append("")
    
    prompt_parts.extend([
        "**IMPORTANT**:",
        "- Your response should directly address the expected output",
        "- Use tools systematically to gather all necessary information",
        "- Provide concrete, actionable insights",
        "- If the task involves multiple files or components, analyze them comprehensively",
        "- End your response with a clear summary that matches the expected output format"
    ])
    
    return "\n".join(prompt_parts)


def _generate_task_summary(task_output: List[str], tool_calls: List[Dict], success: bool, final_response: str) -> str:
    """Generate a comprehensive summary of task execution."""
    summary_parts = []
    
    # Task execution status
    if success:
        summary_parts.append("✅ Task completed successfully")
    else:
        summary_parts.append("❌ Task failed")
    
    # Tool usage summary
    if tool_calls:
        successful_calls = len([tc for tc in tool_calls if tc["success"]])
        failed_calls = len([tc for tc in tool_calls if not tc["success"]])
        
        summary_parts.append(f"Tools used: {len(tool_calls)} total ({successful_calls} successful, {failed_calls} failed)")
        
        # List unique tools used
        unique_tools = list(set(tc["tool_name"] for tc in tool_calls))
        if unique_tools:
            summary_parts.append(f"Tools executed: {', '.join(unique_tools)}")
    
    # Response summary
    if final_response:
        # Extract first few lines of the response as preview
        response_lines = final_response.strip().split('\n')
        if len(response_lines) > 3:
            preview = '\n'.join(response_lines[:3]) + "..."
        else:
            preview = final_response.strip()
        
        summary_parts.append(f"Response preview: {preview}")
    
    return " | ".join(summary_parts)


__all__ = [
    "delegate_task"
] 