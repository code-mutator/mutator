"""
System operation tools for the Coding Agent Framework.
"""

import subprocess
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..decorator import tool


async def _summarize_long_output(output: str, output_type: str) -> str:
    """Summarize long output using LLM, focusing on errors and warnings."""
    try:
        # Import here to avoid circular imports
        from ...llm.client import LLMClient
        from ...core.config import LLMConfig
        
        # Create a basic LLM config
        llm_config = LLMConfig()
        client = LLMClient(llm_config)
        
        # Create summarization prompt
        prompt = f"""
Aalyze the following {output_type} output and provide a concise summary focusing on:
1. Any errors or warnings
2. Key information or results
3. Important status messages
4. Any failures or issues

<OUTPUT>
{output}
</OUTPUT>

Please provide a clear, concise summary that highlights the most important information, especially any errors or warnings.
with Examples for any error or warnings.
"""
        
        response = await client.complete(prompt)
        return response.content.strip()
        
    except Exception as e:
        return f"Failed to summarize {output_type}: {str(e)}"


def _count_lines(text: str) -> int:
    """Count the number of lines in a text string."""
    if not text:
        return 0
    # Split by newlines and filter out empty strings at the end
    lines = text.split('\n')
    # If the text ends with a newline, the last element will be empty
    if lines and lines[-1] == '':
        lines = lines[:-1]
    return len(lines)


def _get_first_n_lines(text: str, n: int) -> str:
    """Get the first n lines of text."""
    if not text:
        return ""
    lines = text.split('\n')
    return '\n'.join(lines[:n])


@tool
async def run_shell(command: str, timeout: int = 30) -> Dict[str, Any]:
    """
    <short_description>
    Execute shell commands in the user's default shell, this is really powerful
    You can achieve a lot like git commands, string replace (after dry run), install packages, run tests, etc.
    if what you want to do is complex, you can write and run a python code with python -c for example.
    </short_description>
    
    <long_description>
    This tool executes shell, timeout controls,
    and detailed output capture. It provides a secure way to run system commands while
    preventing potentially dangerous operations.

    ## Important Notes
    
    1. **Working Directory**:
       - Automatically uses the current working directory
       - Commands run in the context of the current project
       - Supports relative path references in commands

    2. **Timeout Control**:
       - Default timeout of 30 seconds prevents hanging commands
       - Configurable timeout for longer-running operations

    3. **Output Capture**:
       - Captures both stdout and stderr separately
       - Returns exit code for proper error handling
       - Preserves command output formatting
       - Automatically summarizes long output (>100 lines) using LLM

    ## Examples

    - Run a simple command: `run_shell("ls -la")`
    - Build project: `run_shell("npm install")`
    - Run with custom timeout: `run_shell("long_running_script.sh", timeout=300)`
    - Check git status: `run_shell("git status")`

    ## Use Cases

    - Running build commands
    - Executing tests
    - Git operations
    - File system operations
    - System administration tasks

    ## Safety Features

    Commands are automatically blocked if they contain:
    - `rm -rf /` - Recursive deletion of root
    - `sudo rm` - Privileged deletion commands
    - `format` - Disk formatting commands
    - `shutdown` - System shutdown commands
    - `reboot` - System restart commands
    </long_description>

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds (default: 30)
    
    Returns:
        Dict containing command output, exit code, and execution metadata
    """
    try:
        # Enhanced safety checks
        dangerous_patterns = [
            'rm -rf /', 'sudo rm', 'format', 'del /f', 'shutdown', 'reboot',
            'mkfs', 'fdisk', 'dd if=', 'chmod 777', 'chown root',
            'systemctl stop', 'service stop', 'killall', 'pkill -9'
        ]
        
        command_lower = command.lower()
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                raise ValueError(f"Command rejected for safety: {command}")
        
        # Get current working directory
        # Import here to avoid circular imports
        from ..decorator import get_working_directory
        working_directory = get_working_directory()
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_directory,
            timeout=timeout
        )
        
        # Process stdout
        stdout_lines = _count_lines(result.stdout)
        processed_stdout = result.stdout
        stdout_summary = None
        
        if stdout_lines > 100:
            # Use async function to summarize
            stdout_summary = await _summarize_long_output(result.stdout, "stdout")
            first_30_lines = _get_first_n_lines(result.stdout, 30)
            processed_stdout = f"SUMMARY (original had {stdout_lines} lines):\n<{stdout_summary}\n\n--- FIRST 30 LINES ---\n{first_30_lines}"
        
        # Process stderr
        stderr_lines = _count_lines(result.stderr)
        processed_stderr = result.stderr
        stderr_summary = None
        
        if stderr_lines > 100:
            # Use async function to summarize
            stderr_summary = await _summarize_long_output(result.stderr, "stderr")
            first_30_lines = _get_first_n_lines(result.stderr, 30)
            processed_stderr = f"SUMMARY (original had {stderr_lines} lines):\n{stderr_summary}\n\n--- FIRST 30 LINES ---\n{first_30_lines}"
        
        return {
            "exit_code": result.returncode,
            "stdout": processed_stdout,
            "stderr": processed_stderr,
            "success": result.returncode == 0,
            "working_directory": working_directory,
            "stdout_summarized": stdout_lines > 100,
            "stderr_summarized": stderr_lines > 100
        }
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Command timed out after {timeout} seconds")
    except Exception as e:
        raise RuntimeError(f"Failed to execute command: {str(e)}") from e


__all__ = [
    "run_shell"
] 