"""
Search and discovery tools for the Coding Agent Framework.
"""

import re
import glob
import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..decorator import tool
from ...core.path_utils import should_exclude_from_search


def _count_lines_in_file(file_path: Path) -> int:
    """
    Count the number of lines in a file.
    
    Args:
        file_path: Path to the file to count lines in
        
    Returns:
        Number of lines in the file, or 0 if file cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        # Return 0 if file cannot be read (binary files, permission errors, etc.)
        return 0


def _is_glob_pattern(pattern: str) -> bool:
    """
    Detect if a pattern is likely a glob pattern vs regex pattern.
    
    Args:
        pattern: The pattern to analyze
        
    Returns:
        True if it looks like a glob pattern, False if it looks like regex
    """
    # Common glob indicators
    glob_indicators = [
        pattern.startswith('*'),
        pattern.endswith('*'),
        '**' in pattern,
        pattern.count('*') > 0 and not any(c in pattern for c in ['^', '$', '(', ')', '[', ']', '{', '}', '|', '\\']),
        pattern.startswith('?'),
        pattern.endswith('?'),
    ]
    
    # Common regex indicators
    regex_indicators = [
        pattern.startswith('^'),
        pattern.endswith('$'),
        '\\' in pattern,
        '(' in pattern and ')' in pattern,
        '[' in pattern and ']' in pattern,
        '{' in pattern and '}' in pattern,
        '|' in pattern,
        # Only treat dot as regex indicator if it's escaped or in complex patterns
        '\\.' in pattern,
        pattern.count('.') > 1 and ('*' not in pattern or '?' not in pattern),
    ]
    
    # If it has regex indicators, treat as regex
    if any(regex_indicators):
        return False
    
    # If it has glob indicators, treat as glob
    if any(glob_indicators):
        return True
    
    # Default to treating simple strings as glob patterns for user-friendliness
    return True


def _validate_regex_pattern(pattern: str) -> Optional[str]:
    """
    Validate a regex pattern and return error message if invalid.
    
    Args:
        pattern: The regex pattern to validate
        
    Returns:
        Error message if invalid, None if valid
    """
    try:
        re.compile(pattern)
        return None
    except re.error as e:
        return f"Invalid regex pattern: {str(e)}"


def _build_tree_structure(directory: Path, max_depth: int = 3, max_children: int = 20, current_depth: int = 0, working_directory: Optional[Path] = None) -> Dict[str, Any]:
    """
    Build a tree structure for a directory with depth and children limits.
    
    Args:
        directory: Directory to build tree for
        max_depth: Maximum depth to traverse (default: 3)
        max_children: Maximum children per parent (default: 20)
        current_depth: Current depth in traversal
        working_directory: Working directory for gitignore checks
        
    Returns:
        Tree structure dictionary
    """
    if current_depth >= max_depth:
        return None
    
    if working_directory is None:
        working_directory = directory
    
    try:
        items = []
        child_count = 0
        
        # Get all items in directory
        all_items = list(directory.iterdir())
        
        # Sort items: directories first, then files, alphabetically
        all_items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
        
        for item in all_items:
            if child_count >= max_children:
                # Add truncation info if we hit the limit
                items.append({
                    "name": f"... ({len(all_items) - max_children} more items)",
                    "type": "truncated",
                    "truncated_count": len(all_items) - max_children
                })
                break
            
            # Skip files/directories that should be excluded by .gitignore
            if should_exclude_from_search(item, working_directory):
                continue
            
            item_info = {
                "name": item.name,
                "type": "file" if item.is_file() else "directory" if item.is_dir() else "other"
            }
            
            # Add detailed information
            try:
                stat = item.stat()
                item_info.update({
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "permissions": oct(stat.st_mode)[-3:],
                    "is_symlink": item.is_symlink()
                })
            except:
                # Handle permission errors gracefully
                item_info.update({
                    "size": "unknown",
                    "modified": "unknown",
                    "permissions": "unknown",
                    "is_symlink": False
                })
            
            # Add line count for files
            if item.is_file():
                item_info["line_count"] = _count_lines_in_file(item)
            
            # If it's a directory and we haven't reached max depth, recurse
            if item.is_dir() and current_depth < max_depth - 1:
                children = _build_tree_structure(item, max_depth, max_children, current_depth + 1, working_directory)
                if children:
                    item_info["children"] = children
            
            items.append(item_info)
            child_count += 1
        
        return items
        
    except Exception:
        return None


def search_files_by_name(name_pattern: str) -> Dict[str, Any]:
    """
    <short_description>Search for files by name pattern using glob or regex patterns.</short_description>
    
    <long_description>
    This tool searches for files matching a specific name pattern within the project directory.
    Supports both glob patterns (*.py, test_*.js) and regex patterns for flexible file discovery.

    ## Important Notes

    1. **Pattern Types**:
       - Glob patterns: Use wildcards like *.py, test_*.js, **/*.md
       - Regex patterns: Use regex syntax for complex matching
       - Automatic detection of pattern type based on syntax

    2. **Search Scope**:
       - Searches recursively through all subdirectories
       - Respects .gitignore patterns and standard exclusions
       - Excludes binary files, cache directories, and build artifacts

    3. **Performance**:
       - Optimized for large codebases
       - Limits results to prevent overwhelming output
       - Fast pattern matching using appropriate algorithms

    ## Examples

    - Find Python files: `search_files_by_name("*.py")`
    - Find test files: `search_files_by_name("test_*.js")`
    - Find config files: `search_files_by_name("*config*")`
    - Regex search: `search_files_by_name("^api_.*\\.py$")`

    ## Use Cases

    - Locating specific files in large projects
    - Finding files by naming conventions
    - Discovering configuration or documentation files
    - Identifying test files or build artifacts
    </long_description>

    Args:
        name_pattern: File name pattern (glob or regex)
    
    Returns:
        Dict containing matching files with metadata
    """
    try:
        # Import here to avoid circular imports
        from ..decorator import get_working_directory
        
        # Get the configured working directory
        search_path = Path(get_working_directory())
        
        matches = []
        
        # Determine if pattern is glob or regex
        is_glob = any(char in name_pattern for char in ['*', '?', '[', ']'])
        
        if is_glob:
            # Use glob pattern matching
            try:
                # Handle both simple patterns and recursive patterns
                if '**' in name_pattern:
                    pattern_parts = name_pattern.split('**/')
                    if len(pattern_parts) == 2:
                        # Pattern like "**/*.py"
                        for file_path in search_path.rglob(pattern_parts[1]):
                            if file_path.is_file() and not should_exclude_from_search(file_path, search_path):
                                matches.append({
                                    "path": str(file_path.relative_to(search_path)),
                                    "size": file_path.stat().st_size,
                                    "line_count": _count_lines_in_file(file_path)
                                })
                    else:
                        # Complex recursive pattern
                        for file_path in search_path.rglob('*'):
                            if file_path.is_file() and file_path.match(name_pattern) and not should_exclude_from_search(file_path, search_path):
                                matches.append({
                                    "path": str(file_path.relative_to(search_path)),
                                    "size": file_path.stat().st_size,
                                    "line_count": _count_lines_in_file(file_path)
                                })
                else:
                    # Simple glob pattern
                    for file_path in search_path.rglob(name_pattern):
                        if file_path.is_file() and not should_exclude_from_search(file_path, search_path):
                            matches.append({
                                "path": str(file_path.relative_to(search_path)),
                                "size": file_path.stat().st_size,
                                "line_count": _count_lines_in_file(file_path)
                            })
            except Exception as e:
                return {"error": f"Glob pattern error: {str(e)}"}
        else:
            # Use regex pattern matching
            import re
            try:
                pattern = re.compile(name_pattern)
                
                def pattern_matches(filename: str) -> bool:
                    return bool(pattern.search(filename))
                
            except re.error as e:
                return {"error": f"Invalid regex pattern: {str(e)}"}
        
        # Search function
        def search_in_directory(path: Path):
            try:
                for item in path.iterdir():
                    # Skip files/directories that should be excluded by .gitignore
                    if should_exclude_from_search(item, search_path):
                        continue
                    
                    if item.is_file():
                        if pattern_matches(item.name):
                            matches.append({
                                "path": str(item.relative_to(search_path)),
                                "size": item.stat().st_size,
                                "line_count": _count_lines_in_file(item)
                            })
                    elif item.is_dir():
                        search_in_directory(item)
            except PermissionError:
                # Skip directories we can't access
                pass
            except Exception:
                # Skip other errors but continue searching
                pass
        
        search_in_directory(search_path)
        
        return {
            "matches": matches,
            "total_matches": len(matches),
            "pattern_type": "glob" if is_glob else "regex"
        }
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


@tool
def search_files_by_content(content_pattern: str, file_pattern: str = "*", max_results: int = 50) -> Dict[str, Any]:
    """
    <short_description>Search for text patterns in files using regex, with support for both glob and regex file filtering.</short_description>
    
    <long_description>
    This tool performs text pattern matching across files, ideal for finding specific code patterns,
    variable names, function calls, or any text content within your codebase. It supports both glob
    and regex patterns for file filtering, making it a comprehensive search solution.

    ## When to Use This Tool
    - When you need to find text matches like variable names, function calls, or specific strings
    - When you know the precise pattern you're looking for (including regex patterns)
    - When you want to quickly locate all occurrences of a specific term across multiple files
    - When you need to search specific file types or patterns using glob or regex

    ## When NOT to Use This Tool
    - For semantic or conceptual searches (e.g., "how does authentication work") - use codebase_search instead
    - For finding code that implements a certain functionality without knowing the exact terms
    - When you need to understand code concepts rather than locate specific terms

    ## Important Notes

    1. **Content Pattern Matching**:
       - Uses regular expressions for powerful pattern matching
       - Supports complex patterns like `function\\(.*\\)` for function calls

    2. **File Pattern Filtering**:
       - Automatically detects glob patterns (*.py, **/*.js) vs regex patterns
       - Supports glob patterns like `*.py` for Python files, `**/*.js` for JavaScript files
       - Supports regex patterns like `.*\\.py$` for advanced file matching
       - Always searches recursively through subdirectories

    3. **Performance Optimization**:
       - Results are limited to prevent overwhelming output
       - Efficient file traversal with gitignore respect

    4. **Result Details**:
       - Returns file path, line number, and matching line content
       - Includes the actual matched text for context
       - Provides file line count for each file with matches
       - Provides truncation information when results are limited

    ## Examples

    - Find function calls in Python files: `search_files_by_content(r"registerTool\\(", file_pattern="*.py", max_results=20)`
    - Search in all JavaScript files: `search_files_by_content("class.*:", file_pattern="**/*.js", max_results=20)`
    - Complex file regex: `search_files_by_content("TODO:", file_pattern=r".*\\\\.(py|js|ts)$", max_results=50)`

    ## Use Cases
    - Finding specific function or method calls
    - Locating variable declarations or usage
    - Searching for error messages or log statements
    - Finding configuration keys or constants
    - Identifying code patterns or structures
    - Exploring files by type with content filtering
    </long_description>

    Args:
        content_pattern: Regular expression pattern to search for in file contents.
        file_pattern: Glob pattern (*.py, **/*.js) or regex pattern (.*\\\\.py$) to filter files.
        max_results: Maximum number of results to return (prevents overwhelming output).
    
    Returns:
        Dict containing search results with file paths, line numbers, matching content, and file line counts
    """
    try:
        # Import here to avoid circular imports
        from ..decorator import get_working_directory
        
        # Get the configured working directory
        search_path = Path(get_working_directory())
        
        matches = []
        
        # Compile regex pattern for content search
        flags = re.IGNORECASE
        try:
            content_regex = re.compile(content_pattern, flags)
        except re.error as e:
            return {"error": f"Invalid content regex pattern: {str(e)}"}
        
        # Detect file pattern type and get files to search
        is_glob = _is_glob_pattern(file_pattern)
        
        if is_glob:
            files = search_path.rglob(file_pattern)
        else:
            # Use regex patterns for file filtering
            regex_error = _validate_regex_pattern(file_pattern)
            if regex_error:
                return {"error": f"File pattern error: {regex_error}"}
            
            try:
                file_regex = re.compile(file_pattern, re.IGNORECASE)
            except re.error as e:
                return {"error": f"Invalid file regex pattern: {str(e)}"}
            
            # Get all files and filter with regex
            all_files = search_path.rglob("*")
            
            files = [f for f in all_files if f.is_file() and file_regex.search(str(f.relative_to(search_path)))]
        
        # Search through files
        for file_path in files:
            if not file_path.is_file():
                continue
            
            # Skip files that should be excluded by .gitignore
            if should_exclude_from_search(file_path, search_path):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.splitlines()
                file_rel_path = str(file_path.relative_to(search_path))
                file_has_matches = False
                
                for line_num, line in enumerate(lines, 1):
                    if content_regex.search(line):
                        if not file_has_matches:
                            file_has_matches = True
                        
                        matches.append({
                            "file": file_rel_path,
                            "line_number": line_num,
                            "line_content": line.strip(),
                            "match": content_regex.search(line).group(),
                            "file_line_count": len(lines)
                        })
                        
                        if len(matches) >= max_results:
                            break
                            
            except Exception:
                # Skip files that can't be read
                continue
                
            if len(matches) >= max_results:
                break
        
        return {
            "matches": matches,
            "total_matches": len(matches),
            "truncated": len(matches) >= max_results,
            "file_pattern_type": "glob" if is_glob else "regex",
        }
        
    except Exception as e:
        return {"error": f"Content search failed: {str(e)}"}



@tool
def list_directory(directory: str = ".", include_tree: bool = True, max_depth: int = 3, max_children: int = 20) -> Dict[str, Any]:
    """
    <short_description>List the contents of a directory with detailed information including hidden files and optional tree structure.</short_description>
    
    <long_description>
    This tool provides comprehensive directory listing capabilities similar to the `ls -la` command,
    with an optional tree structure view for better visualization of directory hierarchy.

    ## Important Notes

    1. **Directory Listing**:
       - Lists all files and directories in the specified path
       - Provides detailed information including size, permissions, and modification time
       - Optional tree structure with configurable depth and children limits

    2. **File Information**:
       - Includes name, type, size, permissions, modification time, and line count for files
       - Distinguishes between files, directories, and symlinks
       - Shows comprehensive metadata for each item

    3. **Hidden Files**:
       - Hidden files are always included in the listing
       - Respects platform-specific hidden file conventions
       - Useful for finding configuration files and system files

    4. **Tree Structure**:
       - Optional hierarchical tree view with configurable depth (default: 3 levels)
       - Limits children per parent to prevent overwhelming output (default: 20)
       - Shows truncation information when limits are reached
       - Maintains flat list for compatibility

    5. **Error Handling**:
       - Gracefully handles permission errors
       - Provides clear error messages for inaccessible directories
       - Continues listing even if some files are inaccessible

    ## Examples

    - List current directory: `list_directory()`
    - List with tree structure: `list_directory("src/components", include_tree=True)`
    - Flat list only: `list_directory(".", include_tree=False)`
    - Custom limits: `list_directory(".", max_depth=2, max_children=10)`

    ## Use Cases
    - Exploring project structure comprehensively
    - Checking directory contents before operations
    - Finding hidden configuration files
    - Verifying file permissions and sizes
    - Understanding complete directory organization with tree view
    - Visualizing directory hierarchy with depth control
    </long_description>

    Args:
        directory: Directory to list (default: current directory)
        include_tree: Whether to include tree structure (default: True)
        max_depth: Maximum depth for tree structure (default: 3)
        max_children: Maximum children per parent in tree (default: 20)
    
    Returns:
        Dict containing directory contents with detailed file information including line counts, metadata, and optional tree structure
    """
    try:
        # Import here to avoid circular imports
        from ..decorator import get_working_directory
        
        # Get the configured working directory
        working_dir = Path(get_working_directory())
        
        # Resolve the target directory relative to working directory
        if directory == ".":
            dir_path = working_dir
        else:
            dir_path = Path(directory)
            if not dir_path.is_absolute():
                dir_path = working_dir / directory
        
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory}"}
        
        if not dir_path.is_dir():
            return {"error": f"Path is not a directory: {directory}"}
        
        items = []
        
        for item in dir_path.iterdir():
            # Skip files/directories that should be excluded by .gitignore
            if should_exclude_from_search(item, working_dir):
                continue
            
            item_info = {
                "name": item.name,
                "type": "file" if item.is_file() else "directory" if item.is_dir() else "other"
            }
            
            # Always include detailed information
            try:
                stat = item.stat()
                item_info.update({
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "permissions": oct(stat.st_mode)[-3:],
                    "is_symlink": item.is_symlink()
                })
            except:
                # Handle permission errors gracefully
                item_info.update({
                    "size": "unknown",
                    "modified": "unknown",
                    "permissions": "unknown",
                    "is_symlink": False
                })
            
            # Add line count for files
            if item.is_file():
                item_info["line_count"] = _count_lines_in_file(item)
            
            items.append(item_info)
        
        # Sort by name for consistent output
        items.sort(key=lambda x: x["name"].lower())
        
        # Create relative path for display
        try:
            relative_path = str(dir_path.relative_to(working_dir))
            if relative_path == ".":
                display_path = "."
            else:
                display_path = relative_path
        except ValueError:
            # If path is not relative to working directory, show absolute path
            display_path = str(dir_path)
        
        result = {
            "directory": display_path,
            "items": items,
            "total_items": len(items)
        }
        
        # Add tree structure if requested
        if include_tree:
            tree_structure = _build_tree_structure(dir_path, max_depth, max_children, working_directory=working_dir)
            if tree_structure:
                result["tree"] = tree_structure
                result["tree_config"] = {
                    "max_depth": max_depth,
                    "max_children": max_children
                }
        
        return result
        
    except Exception as e:
        return {"error": f"Directory listing failed: {str(e)}"}


__all__ = [
    "search_files_by_name",
    "search_files_by_content", 
    "list_directory"
] 