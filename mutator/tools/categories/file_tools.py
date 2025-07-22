"""
File operation tools for the Coding Agent Framework.
"""

import chardet
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..decorator import tool
except ImportError:
    # Fallback for testing
    def tool(func):
        return func

# Import indentation fixing functionality
from .indentation_fixer import fix_indentation


@tool
def read_file(file_path: str, line_number: int = 1, lines_before: int = 100, lines_after: int = 200) -> Dict[str, Any]:
    """
    <short_description>
    Read a snippet of a file from the file system around a specific line with line numbers.
    Balance between reading the what is only needed for the task and reducing the risk of not reading enough.
    </short_description>
    
    <long_description>
    This tool reads a portion of a file around a specific line number, providing context
    while avoiding reading entire large files. Each line is prefixed with its line number
    for easy reference. It handles various file types and automatically detects encoding,
    providing detailed information about the file structure and content.

    ## Important Notes

    1. **File Path Handling**:
       - Supports both relative and absolute paths
       - Automatically resolves path separators for cross-platform compatibility
       - Returns relative path in results for consistency

    2. **Snippet Reading**:
       - Reads a window of lines around the specified line number
       - Default window size is 300 lines total (150 before + 150 after)
       - Automatically adjusts window if near file boundaries
       - Each line is prefixed with its line number in format: {line_number}\t|\t{content}

    3. **Line Number Prefixing**:
       - Every line in the returned content includes its actual line number
       - Format: {line_number}\t|\t{line_content}
       - Makes it easy to reference specific lines in the file
       - Line numbers are 1-indexed to match standard editor conventions

    4. **Encoding Support**:
       - Automatically detects file encoding for maximum compatibility
       - Falls back to UTF-8 if detection fails
       - Gracefully handles encoding issues with error reporting

    5. **File Validation**:
       - Checks if file exists before attempting to read
       - Verifies that the path points to a file (not a directory)
       - Provides clear error messages for common issues

    ## Examples

    - Read around line 50: `read_file("src/main.py", 50)`
    - Read with custom window: `read_file("config/settings.json", 10, 20, 30)`
    - Read from beginning: `read_file("README.md", 1, 0, 100)`

    ## Use Cases

    - Examining specific code sections with line references
    - Reading around error locations with precise line numbers
    - Analyzing specific configuration sections
    - Reviewing documentation sections
    - Inspecting data file sections with line context
    </long_description>

    Args:
        file_path: Path to the file to read (relative or absolute)
        line_number: Target line number to read around (1-indexed)
        lines_before: Number of lines to read before the target line
        lines_after: Number of lines to read after the target line
    
    Returns:
        Dict containing file snippet with line numbers, metadata, and analysis information
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Auto-detect encoding
        encoding = 'utf-8'
        try:
            with open(path, 'rb') as f:
                raw_data = f.read()
                if raw_data:
                    detected = chardet.detect(raw_data)
                    if detected['encoding'] and detected['confidence'] > 0.7:
                        encoding = detected['encoding']
        except Exception:
            pass  # Fall back to UTF-8
        
        # Read all lines to get total count and extract snippet
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        all_lines = content.splitlines()
        total_lines = len(all_lines)
        
        # Validate line number
        if line_number < 1:
            raise ValueError("Line number must be >= 1")
        
        if line_number > total_lines:
            raise ValueError(f"Line number {line_number} is beyond file length {total_lines}")
        
        # Calculate snippet boundaries (convert to 0-indexed)
        target_idx = line_number - 1
        start_idx = max(0, target_idx - lines_before)
        end_idx = min(total_lines, target_idx + lines_after + 1)
        
        # Extract snippet and prefix each line with its line number
        snippet_lines = []
        for i in range(start_idx, end_idx):
            line_num = i + 1
            line_content = all_lines[i]
            prefixed_line = f"{line_num}\t|\t{line_content}"
            snippet_lines.append(prefixed_line)
        
        snippet_content = '\n'.join(snippet_lines)

        return {
            "content": snippet_content,
            "snippet_lines": len(snippet_lines),
            "total_lines": total_lines,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {str(e)}") from e

@tool
def edit_file(file_path: str, start_line_inclusive: int, end_line_exclusive: int, new_content: str, skip_validation: bool = False) -> Dict[str, Any]:
    """
    <short_description>Edit a file by replacing a specific range of lines with new content.</short_description>
    
    <long_description>
    This tool provides precise line-based editing capabilities, allowing you to replace specific
    sections of a file while preserving the rest of the content. It provides detailed feedback
    about the changes made without creating backup files.

    ## Important Notes

    1. **Line Numbering**:
       - Uses 1-based line numbering (first line is line 1)
       - start_line_inclusive is inclusive (line is included in replacement)
       - end_line_exclusive is exclusive (line is not included in replacement)
       - Handles edge cases like editing at file boundaries

    2. **Content Replacement**:
       - Replaces the specified line range with new content
       - New content can be multiple lines or empty
       - Automatically handles line ending consistency
       - Auto-detects and preserves file encoding

    3. **Validation**:
       - Checks if file exists before editing
       - Validates line number ranges
       - Provides clear error messages for invalid operations

    ## Examples

    - Replace single line: `edit_file("script.py", 10, 11, "# Updated comment")`
    - Replace multiple lines: `edit_file("config.py", 5, 9, "new_setting = True\\nother_setting = False")`
    - Insert content: `edit_file("file.txt", 1, 1, "New first line\\n")`
    - Replace function: `edit_file("module.py", 15, 26, "def new_function():\\n    pass")`

    ## Use Cases

    - Updating specific functions or methods
    - Modifying configuration settings
    - Fixing bugs in specific code sections
    - Updating documentation sections
    - Refactoring code blocks
    </long_description>

    Args:
        file_path: Path to the file to edit (relative or absolute)
        start_line_inclusive: Starting line number (1-indexed, inclusive)
        end_line_exclusive: Ending line number (1-indexed, exclusive)
        new_content: New content to replace the specified lines
        skip_validation: should be False by default, in case you are sure the new content is balanced even with the warning retuened, trigger the tool again with skip_validation=True
    
    Returns:
        Dict containing edit result and metadata
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect encoding
        encoding = 'utf-8'
        try:
            with open(path, 'rb') as f:
                raw_data = f.read()
                if raw_data:
                    detected = chardet.detect(raw_data)
                    if detected['encoding'] and detected['confidence'] > 0.7:
                        encoding = detected['encoding']
        except Exception:
            pass  # Fall back to UTF-8
        
        # Read current content
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        lines = content.splitlines()
        
        # Validate line numbers
        if start_line_inclusive < 1 or end_line_exclusive < 1:
            raise ValueError("Line numbers must be >= 1")
        
        if start_line_inclusive > len(lines) + 1:
            raise ValueError(f"Start line {start_line_inclusive} is beyond file length {len(lines)}")
        
        if end_line_exclusive < start_line_inclusive:
            raise ValueError(f"End line {end_line_exclusive} must be >= start line {start_line_inclusive}")
        
        # Get old content for indentation analysis
        old_content = "\n".join(lines[start_line_inclusive - 1:end_line_exclusive - 1])
        
        # Fix indentation of new content based on old content pattern
        if not skip_validation:
            new_content = fix_indentation(old_content, new_content, content, file_path)
        
        new_content_bracket_analysis = analyze_bracket_context(new_content)
        old_content_bracket_analysis = analyze_bracket_context(old_content)
        if not skip_validation and new_content_bracket_analysis.stack_before != old_content_bracket_analysis.stack_before:
            return {"error": "The new content brackets is not balanced after merging, please fix it", "type": "warning"}
        if not skip_validation and new_content_bracket_analysis.stack_after != old_content_bracket_analysis.stack_after:
            return {"error": "The new content brackets is not balanced after merging, please fix it", "type": "warning"}
        
        # Adjust for 0-indexed arrays
        start_idx = start_line_inclusive - 1
        end_idx = min(end_line_exclusive - 1, len(lines))
        
        # Split new content into lines
        new_lines = new_content.splitlines()
        
        # Replace the lines
        updated_lines = lines[:start_idx] + new_lines + lines[end_idx:]
        
        # Write updated content
        updated_content = '\n'.join(updated_lines)
        with open(path, 'w', encoding=encoding) as f:
            f.write(updated_content)
        
        # Generate merged content with 10 lines before and after the edited section
        context_lines_before = 10
        context_lines_after = 10
        
        # Calculate boundaries for merged content
        merged_start_idx = max(0, start_idx - context_lines_before)
        merged_end_idx = min(len(updated_lines), start_idx + len(new_lines) + context_lines_after)
        
        # Extract merged content with line numbers
        merged_lines = []
        for i in range(merged_start_idx, merged_end_idx):
            line_num = i + 1
            line_content = updated_lines[i]
            prefixed_line = f"{line_num}\t|\t{line_content}"
            merged_lines.append(prefixed_line)
        
        merged_content = '\n'.join(merged_lines)
        merged_content = "Merged applied!, here is the new state for the updated lines with additional lines\n verify if the merge is correct:\n" + merged_content 
        
        return {
            "encoding": encoding,
            "merged_content": merged_content,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to edit file: {str(e)}") from e


@tool
def create_file(file_path: str, full_content: str = "") -> Dict[str, Any]:
    """
    <short_description>Create a new file with specified content, overwriting if it exists.</short_description>
    
    <long_description>
    This tool creates new files with optional initial content. It will always overwrite
    existing files and provides detailed feedback about the creation process.

    ## Important Notes

    1. **File Creation Behavior**:
       - Always overwrites existing files without warning
       - Creates parent directories automatically
       - No backup files are created

    2. **Content Initialization**:
       - Supports empty file creation (default)
       - Accepts multi-line content with proper formatting
       - Uses UTF-8 encoding for maximum compatibility

    3. **Path Handling**:
       - Supports both relative and absolute paths
       - Automatically creates necessary parent directories
       - Returns relative path in results for consistency

    4. **Encoding Support**:
       - Uses UTF-8 encoding for maximum compatibility
       - Consistent encoding handling across platforms

    ## Examples

    - Create empty file: `create_file("new_file.txt")`
    - Create with content: `create_file("script.py", "#!/usr/bin/env python3\\nprint('Hello, World!')")`
    - Create in subdirectory: `create_file("src/utils/helper.py", "# Helper functions")`
    - Overwrite existing: `create_file("existing.txt", "New content")`

    ## Use Cases

    - Creating new source code files
    - Initializing configuration files
    - Creating documentation files
    - Setting up project structure
    - Generating template files
    </long_description>

    Args:
        file_path: Path to the file to create (relative or absolute)
        full_content: Complete initial content for the file (default: empty)
    
    Returns:
        Dict containing creation result and metadata
    """
    try:
        path = Path(file_path)
        
        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file existed before creation
        file_existed = path.exists()
        
        # Create file with UTF-8 encoding (always overwrite)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        return {
            "content_length": len(full_content),
            "lines_created": len(full_content.splitlines()),
            "encoding": "utf-8",
            "overwritten": file_existed
        }
    except Exception as e:
        raise RuntimeError(f"Failed to create file: {str(e)}") from e


from typing import List
from dataclasses import dataclass


@dataclass
class BracketAnalysis:
    """Result of bracket analysis for a code snippet."""
    stack_before: List[str]
    stack_after: List[str]


def analyze_bracket_context(snippet: str) -> BracketAnalysis:
    """
    Analyze a code snippet to determine the bracket stack state before and after.
    
    This function helps understand the bracket context when working with partial code
    snippets that may not be balanced on their own but come from balanced files.
    
    The algorithm works by:
    1. Tracking opening and closing brackets in the snippet
    2. Matching pairs within the snippet
    3. Identifying unmatched closing brackets (likely from before the snippet)
    4. Identifying unmatched opening brackets (likely continuing after the snippet)
    
    Args:
        snippet: Code snippet to analyze
        
    Returns:
        BracketAnalysis object containing:
        - stack_before: Estimated stack of open brackets before the snippet
        - stack_after: Estimated stack of open brackets after the snippet
    
    Example:
        >>> snippet = "} else { return func(a, b);"
        >>> analysis = analyze_bracket_context(snippet)
        >>> analysis.stack_before  # ['{'] - there was an open brace before
        >>> analysis.stack_after   # ['{'] - brace still open
        >>> analysis.unmatched_closing  # ['}'] - closing brace from before
    """
    # Define bracket pairs (excluding angle brackets to avoid confusion with comparison operators)
    bracket_pairs = {
        '(': ')',
        '[': ']',
        '{': '}'
    }
    
    closing_to_opening = {v: k for k, v in bracket_pairs.items()}
    
    # Track brackets in the snippet
    stack_during = []
    unmatched_closing = []
    
    # Process each character in the snippet
    for char in snippet:
        if char in bracket_pairs:
            # Opening bracket
            stack_during.append(char)
        elif char in closing_to_opening:
            # Closing bracket
            if stack_during and stack_during[-1] == closing_to_opening[char]:
                # Matched pair within snippet
                stack_during.pop()
            else:
                # Unmatched closing bracket - likely from before snippet
                unmatched_closing.append(char)
    
    # Estimate stack before snippet
    # The unmatched closing brackets suggest what was open before
    stack_before = []
    for closing_bracket in unmatched_closing:
        opening_bracket = closing_to_opening[closing_bracket]
        stack_before.append(opening_bracket)
    
    # Stack after snippet is what remains unmatched in stack_during
    stack_after = stack_during.copy()
    
    return BracketAnalysis(
        stack_before=stack_before,
        stack_after=stack_after,
    )


__all__ = [
    "read_file", "edit_file", "create_file"
] 