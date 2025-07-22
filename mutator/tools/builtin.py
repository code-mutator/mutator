"""
Built-in tools for the Coding Agent Framework.

This module provides a comprehensive set of tools for file operations, system interaction,
and project management. All tools are properly categorized and documented.
"""

from .categories.file_tools import (
    read_file, edit_file, create_file
)

from .categories.search_tools import (
    search_files_by_name, search_files_by_content, list_directory
)

from .categories.system_tools import (
    run_shell
)

from .categories.task_tools import (
    delegate_task
)

from .categories.ai_tools import (
    search_files_sementic, mermaid
)

from .categories.web_tools import (
    fetch_url
)

# Import web_search conditionally
try:
    from .categories.web_tools import web_search
except ImportError:
    web_search = None

from .batch_tools import (
    process_search_files_by_name, process_search_files_by_content,
    process_search_files_sementic
)

# Registry of all built-in tools
BUILTIN_TOOLS = {
    # File operations
    "read_file": read_file,
    "edit_file": edit_file,
    "create_file": create_file,
    
    # Search operations
    "search_files_by_name": search_files_by_name,
    "search_files_by_content": search_files_by_content,
    "list_directory": list_directory,
    
    # System operations
    "run_shell": run_shell,
    
    # Task management
    "delegate_task": delegate_task,
    
    # AI-powered tools
    "search_files_sementic": search_files_sementic,
    "mermaid": mermaid,
    
    # Web tools
    "fetch_url": fetch_url,
    
    # Batch tools
    "process_search_files_by_name": process_search_files_by_name,
    "process_search_files_by_content": process_search_files_by_content,
    "process_search_files_sementic": process_search_files_sementic,
}

# Add web_search if available
if web_search is not None:
    BUILTIN_TOOLS["web_search"] = web_search

# Tool categories for organization
TOOL_CATEGORIES = {
    "file_operations": [
        "read_file", "edit_file", "create_file"
    ],
    "search_operations": [
        "search_files_by_name", "search_files_by_content", "list_directory"
    ],
    "system_operations": [
        "run_shell"
    ],
    "development_tools": [
    ],
    "task_management": [
        "delegate_task"
    ],
    "ai_tools": [
        "search_files_sementic", "mermaid"
    ],
    "web_tools": [
        "fetch_url", "web_search"
    ],
    "batch_tools": [
        "process_search_files_by_name", "process_search_files_by_content",
        "process_search_files_sementic"
    ]
}

def get_builtin_tools():
    """Get all built-in tools."""
    return BUILTIN_TOOLS

def get_tool_categories():
    """Get tool categories for organization."""
    return TOOL_CATEGORIES

def get_tools_by_category(category: str):
    """Get tools in a specific category."""
    return {name: BUILTIN_TOOLS[name] for name in TOOL_CATEGORIES.get(category, [])} 