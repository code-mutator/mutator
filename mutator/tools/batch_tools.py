"""
Batch tools for the Coding Agent Framework.

This module provides functionality to create batch variants of existing tools
that group results and use delegate_task to process them.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from ..core.types import AgentEvent, ToolResult, ExecutionMode
from ..core.config import AgentConfig
from ..llm.client import LLMClient
from .decorator import tool


def _group_items(items: List[Any], max_per_group: int = 20) -> List[List[Any]]:
    """Group items into batches of max_per_group size."""
    if not items:
        return []
    
    groups = []
    for i in range(0, len(items), max_per_group):
        groups.append(items[i:i + max_per_group])
    
    return groups


@tool
async def process_search_files_by_name(pattern: str,
                                       operation_description: str,
                                       max_results: int = None,
                                       delegate_processing_results: bool = False) -> Dict[str, Any]:
    """
    <short_description>
    Find files matching a pattern and process them in groups using delegate_task.
    </short_description>
    
    <long_description>
    This tool finds files matching a pattern and processes them in groups using delegate_task.
    Each group contains up to 20 files and is processed by a dedicated sub-agent.

    ## Important Notes

    1. **delegate_processing_results
       - When delegate_processing_results=False, returns search results without processing
       - Useful for previewing what files would be processed

    2. **Processing Mode**:
       - When delegate_processing_results=True (default), files are grouped (max 20 per group)
       - Each group is processed by delegate_task with a specialized sub-agent
       - Results are aggregated and returned

    3. **Grouping Strategy**:
       - Files are grouped into batches of up to 20 items
       - Each group is processed independently
       - Results from all groups are combined

    4. **Context Preservation**:
       - Each group includes complete context about the files to process
       - Operation description is included in each delegation
       - File paths and metadata are preserved

    ## Examples

    - Preview files: `process_search_files_by_name("*.py", "Analyze Python files", delegate_processing_results=False)`
    - Process files: `process_search_files_by_name("*.py", "Add type hints to Python files", delegate_processing_results=True)`
    - Limited scope: `process_search_files_by_name("test_*.py", "Update test files", max_results=50, delegate_processing_results=False)`
    </long_description>

    Args:
        pattern: File name pattern to search for (supports wildcards)
        operation_description: Description with all the detailsof what operation to perform on matched files
        max_results: Maximum number of files to find (optional)
        delegate_processing_results: If True, a sub agent will process the results and return summary (default: False)
    
    Returns:
        Dict containing search results and processing results from delegate_task
    """
    try:
        # Import here to avoid circular imports
        from .categories.search_tools import search_files_by_name
        
        # Find matching files
        search_result = await search_files_by_name.execute(name_pattern=pattern)
        
        if not search_result.success or not search_result.result.get("matches"):
            return {
                "success": False,
                "message": "No files found matching the pattern",
                "pattern": pattern,
                "search_result": search_result.result if search_result.success else {"error": search_result.error}
            }
        
        items = search_result.result["matches"]
        
        # Apply max_results limit
        if max_results and len(items) > max_results:
            items = items[:max_results]
        
        # If not delegate_processing_results, return results directly
        if not delegate_processing_results:
            return {
                "success": True,
                "pattern": pattern,
                "operation_description": operation_description,
                "files_found": len(items),
                "matches": items,
            }
        
        # Group items for processing
        groups = _group_items(items, max_per_group=20)
        
        # Import delegate_task
        from .categories.task_tools import delegate_task
        
        # Process each group using delegate_task
        group_results = []
        for i, group in enumerate(groups):
            group_task_description = f"""
{operation_description}

Process the following {len(group)} files:
{json.dumps(group, indent=2)}

Instructions:
- {operation_description}
- Process each file according to the operation requirements
- Verify changes are correct and complete
- Handle any errors gracefully
- Provide a summary of what was accomplished for each file
"""
            
            expected_output = f"Summary of processing {len(group)} files with details of what was accomplished for each file"
            
            context_data = {
                "operation_description": operation_description,
                "group_number": i + 1,
                "total_groups": len(groups),
                "files_in_group": group,
                "search_pattern": pattern
            }
            
            # Delegate the task
            result = await delegate_task.execute(
                task_description=group_task_description,
                expected_output=expected_output,
                context_data=context_data
            )
            
            group_results.append({
                "group_number": i + 1,
                "files_processed": len(group),
                "success": result.result.get("success", False) if result.success else False,
                "summary": result.result.get("summary", "No summary available") if result.success else result.error,
                "final_response": result.result.get("final_response", "") if result.success else "",
                "tool_calls_made": result.result.get("tool_calls_made", 0) if result.success else 0
            })
        
        # Generate overall summary
        total_files = len(items)
        successful_groups = len([gr for gr in group_results if gr["success"]])
        failed_groups = len([gr for gr in group_results if not gr["success"]])
        
        return {
            "success": successful_groups > 0,
            "pattern": pattern,
            "operation_description": operation_description,
            "total_files_found": total_files,
            "total_groups": len(groups),
            "successful_groups": successful_groups,
            "failed_groups": failed_groups,
            "group_results": group_results,
            "summary": f"Processed {total_files} files in {len(groups)} groups: {successful_groups} successful, {failed_groups} failed"
        }
        
    except Exception as e:
        return {"success": False, "error": f"Batch search files failed: {str(e)}"}


@tool
async def process_search_files_by_content(pattern: str,
                                          operation_description: str,
                                          file_pattern: str = "*",
                                          max_results: int = 100,
                                          delegate_processing_results: bool = False) -> Dict[str, Any]:
    """
    <short_description>
    Search for text patterns in files and process matches in groups using delegate_task.
    </short_description>
    
    <long_description>
    This tool searches for text patterns in files and processes matches in groups using delegate_task.
    Each group contains up to 20 matches and is processed by a dedicated sub-agent.

    ## Important Notes

    1. **delegate_processing_results
       - When delegate_processing_results=False, returns search results without processing
       - Useful for previewing what matches would be processed
       - No delegate_task calls are made in delegate_processing_results mode

    2. **Processing Mode**:
       - When delegate_processing_results=True (default), matches are grouped (max 20 per group)
       - Each group is processed by delegate_task with a specialized sub-agent
       - Results are aggregated and returned

    3. **Grouping Strategy**:
       - Matches are grouped into batches of up to 20 items
       - Groups are organized by file when possible
       - Each group is processed independently

    4. **Context Preservation**:
       - Each group includes surrounding code context
       - File paths and line numbers are preserved
       - Operation description is included in each delegation

    ## Examples

    - Preview matches: `process_search_files_by_content("import old_module", "Replace with new_module", delegate_processing_results=False)`
    - Process matches: `process_search_files_by_content("import old_module", "Replace with new_module imports")`
    - Limited scope: `process_search_files_by_content("TODO.*urgent", "Address urgent TODOs", "*.py", 50)`
    </long_description>

    Args:
        pattern: Regular expression pattern to search for
        operation_description: Description with all the details of what operation to perform on matches
        file_pattern: Pattern for files to search (default: "*")
        max_results: Maximum number of matches to find
        delegate_processing_results: If True, a sub agent will process the results and return summary (default: False)
    
    Returns:
        Dict containing search results and processing results from delegate_task
    """
    try:
        # Import here to avoid circular imports
        from .categories.search_tools import search_files_by_content
        
        # Search for matches
        search_result = await search_files_by_content.execute(
            content_pattern=pattern, 
            file_pattern=file_pattern, 
            max_results=max_results
        )
        
        if not search_result.success or not search_result.result.get("matches"):
            return {
                "success": False,
                "message": "No matches found for the pattern",
                "pattern": pattern,
                "search_result": search_result.result if search_result.success else {"error": search_result.error}
            }
        
        items = search_result.result["matches"]
        
        # If not delegate_processing_results, return results directly
        if not delegate_processing_results:
            return {
                "success": True,
                "pattern": pattern,
                "operation_description": operation_description,
                "matches_found": len(items),
                "matches": items,
            }
        
        # Group items for processing
        groups = _group_items(items, max_per_group=20)
        
        # Import delegate_task
        from .categories.task_tools import delegate_task
        
        # Process each group using delegate_task
        group_results = []
        for i, group in enumerate(groups):
            group_task_description = f"""
{operation_description}

Process the following {len(group)} matches:
{json.dumps(group, indent=2)}

Instructions:
- {operation_description}
- Process each match according to the operation requirements
- Consider the surrounding code context
- Verify changes are correct and complete
- Handle any errors gracefully
- Provide a summary of what was accomplished for each match
"""
            
            expected_output = f"Summary of processing {len(group)} matches with details of what was accomplished for each match"
            
            context_data = {
                "operation_description": operation_description,
                "group_number": i + 1,
                "total_groups": len(groups),
                "matches_in_group": group,
                "search_pattern": pattern,
                "file_pattern": file_pattern
            }
            
            # Delegate the task
            result = await delegate_task.execute(
                task_description=group_task_description,
                expected_output=expected_output,
                context_data=context_data
            )
            
            group_results.append({
                "group_number": i + 1,
                "matches_processed": len(group),
                "success": result.result.get("success", False) if result.success else False,
                "summary": result.result.get("summary", "No summary available") if result.success else result.error,
                "final_response": result.result.get("final_response", "") if result.success else "",
                "tool_calls_made": result.result.get("tool_calls_made", 0) if result.success else 0
            })
        
        # Generate overall summary
        total_matches = len(items)
        successful_groups = len([gr for gr in group_results if gr["success"]])
        failed_groups = len([gr for gr in group_results if not gr["success"]])
        
        return {
            "success": successful_groups > 0,
            "pattern": pattern,
            "operation_description": operation_description,
            "total_matches_found": total_matches,
            "total_groups": len(groups),
            "successful_groups": successful_groups,
            "failed_groups": failed_groups,
            "group_results": group_results,
            "summary": f"Processed {total_matches} matches in {len(groups)} groups: {successful_groups} successful, {failed_groups} failed"
        }
        
    except Exception as e:
        return {"success": False, "error": f"Batch grep search failed: {str(e)}"}


@tool
async def process_search_files_sementic(query: str,
                                        operation_description: str,
                                        file_types: List[str] = None,
                                        max_results: int = 50,
                                        delegate_processing_results: bool = False) -> Dict[str, Any]:
    """
    <short_description>
    Perform semantic code search and process results in groups using delegate_task.
    using vector database holding functions and classes info.
    </short_description>
    
    <long_description>
    This tool performs semantic code search and processes results in groups using delegate_task.
    Each group contains up to 20 search results and is processed by a dedicated sub-agent.

    ## Important Notes

    1. **delegate_processing_results
       - When delegate_processing_results=False, returns search results without processing
       - Useful for previewing what code would be processed
       - No delegate_task calls are made in delegate_processing_results mode

    2. **Processing Mode**:
       - When delegate_processing_results=True (default), results are grouped (max 20 per group)
       - Each group is processed by delegate_task with a specialized sub-agent
       - Results are aggregated and returned

    3. **Grouping Strategy**:
       - Results are grouped into batches of up to 20 items
       - Groups are organized by file when possible
       - Each group is processed independently

    4. **Context Preservation**:
       - Each group includes code snippets and context
       - File paths and line numbers are preserved
       - Search query context is maintained

    ## Examples

    - Preview results: `process_search_files_sementic("authentication logic", "Refactor auth", delegate_processing_results=False)`
    - Process results: `process_search_files_sementic("authentication logic", "Refactor authentication to use new JWT system")`
    - Specific types: `process_search_files_sementic("database queries", "Migrate to new ORM", ["py", "js"])`
    </long_description>

    Args:
        query: Natural language query describing what to find as a functionality without the reason.
        operation_description: Description of what operation to perform on search results
        file_types: Optional list of file extensions to search (e.g., ["py", "js", "ts"])
        max_results: Maximum number of results to return from search
        delegate_processing_results: If True, a sub agent will process the results and return summary (default: False)
    
    Returns:
        Dict containing search results and processing results from delegate_task
    """
    try:
        # Import here to avoid circular imports
        from .categories.ai_tools import search_files_sementic
        
        # Perform semantic search
        search_result_obj = await search_files_sementic.execute(
            query=query, 
            file_types=file_types, 
            max_results=max_results
        )
        search_result = search_result_obj.result if search_result_obj.success else {}
        
        if not search_result.get("results"):
            return {
                "success": False,
                "message": "No results found for the query",
                "query": query,
                "search_result": search_result
            }
        
        items = search_result["results"]
        
        # If not delegate_processing_results, return results directly
        if not delegate_processing_results:
            return {
                "success": True,
                "query": query,
                "operation_description": operation_description,
                "results_found": len(items),
                "results": items,
            }
        
        # Group items for processing
        groups = _group_items(items, max_per_group=20)
        
        # Import delegate_task
        from .categories.task_tools import delegate_task
        
        # Process each group using delegate_task
        group_results = []
        for i, group in enumerate(groups):
            group_task_description = f"""
{operation_description}

Process the following {len(group)} semantic search results:
Query: {query}

Results:
{json.dumps(group, indent=2)}

Instructions:
- {operation_description}
- Process each search result according to the operation requirements
- Consider the context and relationships found
- Verify changes are correct and complete
- Handle any errors gracefully
- Provide a summary of what was accomplished for each result
"""
            
            expected_output = f"Summary of processing {len(group)} semantic search results with details of what was accomplished for each result"
            
            context_data = {
                "operation_description": operation_description,
                "group_number": i + 1,
                "total_groups": len(groups),
                "results_in_group": group,
                "search_query": query,
                "file_types": file_types
            }
            
            # Delegate the task
            result = await delegate_task.execute(
                task_description=group_task_description,
                expected_output=expected_output,
                context_data=context_data
            )
            
            group_results.append({
                "group_number": i + 1,
                "results_processed": len(group),
                "success": result.result.get("success", False) if result.success else False,
                "summary": result.result.get("summary", "No summary available") if result.success else result.error,
                "final_response": result.result.get("final_response", "") if result.success else "",
                "tool_calls_made": result.result.get("tool_calls_made", 0) if result.success else 0
            })
        
        # Generate overall summary
        total_results = len(items)
        successful_groups = len([gr for gr in group_results if gr["success"]])
        failed_groups = len([gr for gr in group_results if not gr["success"]])
        
        return {
            "success": successful_groups > 0,
            "query": query,
            "operation_description": operation_description,
            "total_results_found": total_results,
            "total_groups": len(groups),
            "successful_groups": successful_groups,
            "failed_groups": failed_groups,
            "group_results": group_results,
            "summary": f"Processed {total_results} semantic search results in {len(groups)} groups: {successful_groups} successful, {failed_groups} failed"
        }
        
    except Exception as e:
        return {"success": False, "error": f"Batch semantic search failed: {str(e)}"}


__all__ = [
    "process_search_files_by_name",
    "process_search_files_by_content", 
    "process_search_files_sementic"
] 