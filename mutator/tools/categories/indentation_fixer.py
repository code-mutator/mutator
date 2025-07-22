"""
Indentation fixing functionality for the Coding Agent Framework.

This module provides tools to fix indentation of LLM-generated code by analyzing
the existing code patterns and applying consistent indentation rules.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple


def fix_indentation(old_content: str, new_content: str, full_content: str, file_path: str = "") -> str:
    """
    Fix indentation of new content based on the indentation pattern of old content.
    
    This function analyzes the indentation pattern used in the old content and applies
    it to the new content, ensuring consistent indentation throughout the file.
    
    Args:
        old_content: The original content being replaced
        new_content: The new content to fix indentation for
        full_content: The complete file content for context analysis
        file_path: Path to the file being edited (used for language detection)
        
    Returns:
        The new content with corrected indentation
    """
    if not old_content.strip() or not new_content.strip():
        return new_content
    
    # Detect indentation style and base indentation from old content
    base_indent, indent_style, indent_size = _detect_indentation_pattern(old_content, full_content)
    
    # Detect file language from extension
    language = _detect_language(file_path)
    
    # Split content into lines for processing
    new_lines = new_content.split('\n')
    if not new_lines:
        return new_content
    
    # Process each line to fix indentation
    fixed_lines = []
    current_indent_level = 0
    
    for i, line in enumerate(new_lines):
        stripped_line = line.strip()
        
        if not stripped_line:
            # Keep empty lines as-is
            fixed_lines.append('')
            continue
        
        # Adjust indentation level based on closing brackets/keywords
        if _is_closing_line(stripped_line, language):
            current_indent_level = max(0, current_indent_level - 1)
        
        # Reset indentation for sibling statements (Python-specific logic)
        if language == 'python' and current_indent_level > 0:
            # Check if this is a sibling statement (same level as previous opening statement)
            if _is_opening_line(stripped_line, language):
                # Look back to see if there was indented content after the last opening line
                # If so, this should be a sibling; if not, it should be nested
                if _has_indented_content_after_last_opening(fixed_lines, base_indent, indent_style, indent_size):
                    # Find the level of the last opening line to make this a sibling
                    last_opening_level = _find_last_unclosed_opening_level(fixed_lines, base_indent, indent_style, indent_size)
                    if last_opening_level is not None:
                        current_indent_level = last_opening_level
        
        # Apply indentation
        total_indent = base_indent + (indent_style * indent_size * current_indent_level)
        fixed_line = total_indent + stripped_line
        fixed_lines.append(fixed_line)
        
        # Adjust indentation level for next line based on opening brackets/keywords
        if _is_opening_line(stripped_line, language):
            current_indent_level += 1
    
    return '\n'.join(fixed_lines)


def _detect_indentation_pattern(old_content: str, full_content: str) -> Tuple[str, str, int]:
    """
    Detect the indentation pattern from old content and full file context.
    
    Returns:
        Tuple of (base_indent, indent_style, indent_size)
        - base_indent: The base indentation of the old content
        - indent_style: ' ' for spaces, '\t' for tabs
        - indent_size: Number of spaces/tabs per indentation level
    """
    old_lines = old_content.split('\n')
    
    # Find the base indentation from the first non-empty line of old content
    base_indent = ""
    for line in old_lines:
        if line.strip():
            base_indent = line[:len(line) - len(line.lstrip())]
            break
    
    # Detect indentation style and size from full content
    indent_style, indent_size = _analyze_file_indentation(full_content)
    
    return base_indent, indent_style, indent_size


def _analyze_file_indentation(content: str) -> Tuple[str, int]:
    """
    Analyze the file's indentation style and size.
    
    Returns:
        Tuple of (indent_style, indent_size)
    """
    lines = content.split('\n')
    
    # Count tabs vs spaces
    tab_count = 0
    space_indents = []
    
    for line in lines:
        if not line.strip():
            continue
            
        leading_whitespace = line[:len(line) - len(line.lstrip())]
        
        if leading_whitespace.startswith('\t'):
            tab_count += 1
        elif leading_whitespace.startswith(' '):
            space_count = len(leading_whitespace)
            if space_count > 0:
                space_indents.append(space_count)
    
    # Determine style
    if tab_count > len(space_indents):
        return '\t', 1
    
    # For spaces, find the most common indentation size
    if space_indents:
        # Find GCD-like pattern for indentation
        indent_size = _find_common_indent_size(space_indents)
        return ' ', indent_size
    
    # Default to 4 spaces
    return ' ', 4


def _find_common_indent_size(space_indents: List[int]) -> int:
    """Find the most likely indentation size from a list of space counts."""
    if not space_indents:
        return 4
    
    # Calculate differences between consecutive indentation levels
    sorted_indents = sorted(set(space_indents))
    if len(sorted_indents) < 2:
        return sorted_indents[0] if sorted_indents else 4
    
    # Find the most common difference (which represents the indent size)
    differences = []
    for i in range(1, len(sorted_indents)):
        diff = sorted_indents[i] - sorted_indents[i-1]
        if diff > 0:
            differences.append(diff)
    
    if differences:
        # Find the most common difference
        from collections import Counter
        counter = Counter(differences)
        return counter.most_common(1)[0][0]
    
    # Fallback to minimum positive indentation
    positive_indents = [i for i in space_indents if i > 0]
    if positive_indents:
        return min(positive_indents)
    
    return 4


def _detect_language(file_path: str) -> str:
    """
    Detect programming language from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Language identifier string
    """
    if not file_path:
        return "unknown"
    
    extension = Path(file_path).suffix.lower()
    
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c++': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.hxx': 'cpp',
        '.h++': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.scala': 'scala',
        '.sc': 'scala',
        '.r': 'r',
        '.R': 'r',
        '.m': 'objective-c',
        '.mm': 'objective-c',
        '.pl': 'perl',
        '.pm': 'perl',
        '.perl': 'perl',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
        '.fish': 'fish',
        '.ps1': 'powershell',
        '.psm1': 'powershell',
        '.sql': 'sql',
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        '.xml': 'xml',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini',
        '.lua': 'lua',
        '.vim': 'vim',
        '.vimrc': 'vim',
        '.dart': 'dart',
        '.ex': 'elixir',
        '.exs': 'elixir',
        '.erl': 'erlang',
        '.hrl': 'erlang',
        '.hs': 'haskell',
        '.lhs': 'haskell',
        '.ml': 'ocaml',
        '.mli': 'ocaml',
        '.clj': 'clojure',
        '.cljs': 'clojure',
        '.cljc': 'clojure',
        '.jl': 'julia',
        '.nim': 'nim',
        '.nims': 'nim',
        '.cr': 'crystal',
        '.d': 'd',
        '.pas': 'pascal',
        '.pp': 'pascal',
        '.inc': 'pascal',
        '.asm': 'assembly',
        '.s': 'assembly',
        '.S': 'assembly',
        '.f': 'fortran',
        '.f90': 'fortran',
        '.f95': 'fortran',
        '.f03': 'fortran',
        '.f08': 'fortran',
        '.for': 'fortran',
        '.ftn': 'fortran',
        '.cob': 'cobol',
        '.cbl': 'cobol',
        '.cpy': 'cobol',
        '.v': 'verilog',
        '.vh': 'verilog',
        '.vhd': 'vhdl',
        '.vhdl': 'vhdl',
        '.tcl': 'tcl',
        '.tk': 'tcl',
        '.awk': 'awk',
        '.sed': 'sed',
        '.make': 'makefile',
        '.mk': 'makefile',
        '.cmake': 'cmake',
        '.dockerfile': 'dockerfile',
        '.proto': 'protobuf',
        '.thrift': 'thrift',
        '.graphql': 'graphql',
        '.gql': 'graphql',
        '.tf': 'terraform',
        '.tfvars': 'terraform',
    }
    
    return language_map.get(extension, 'unknown')


def _remove_comments_and_strings(line: str, language: str) -> str:
    """
    Remove comments and string literals from a line to avoid false positives.
    
    Args:
        line: The line to process
        language: Programming language identifier
        
    Returns:
        Line with comments and strings removed
    """
    if not line.strip():
        return line
    
    # Handle different comment styles by language
    if language in ['python', 'ruby', 'perl', 'bash', 'zsh', 'fish', 'r', 'tcl', 'awk', 'sed', 'makefile', 'cmake', 'dockerfile', 'yaml', 'toml', 'ini']:
        # Remove hash comments (but not in strings)
        in_string = False
        quote_char = None
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                    result.append(char)
                elif char == '#':
                    # Comment starts here, ignore rest of line
                    break
                else:
                    result.append(char)
            else:
                result.append(char)
                if char == quote_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    quote_char = None
            i += 1
        return ''.join(result)
    
    elif language in ['javascript', 'typescript', 'java', 'c', 'cpp', 'csharp', 'go', 'rust', 'swift', 'kotlin', 'scala', 'dart', 'objective-c', 'd', 'verilog']:
        # Remove C-style comments and also handle # comments for mixed content
        in_string = False
        quote_char = None
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                    result.append(char)
                elif char == '#':
                    # Python-style comment (for mixed content)
                    break
                elif char == '/' and i + 1 < len(line):
                    next_char = line[i + 1]
                    if next_char == '/':
                        # Single-line comment starts here
                        break
                    elif next_char == '*':
                        # Multi-line comment starts here
                        # Find the end of the comment on this line
                        i += 2
                        found_end = False
                        while i + 1 < len(line):
                            if line[i] == '*' and line[i + 1] == '/':
                                i += 2  # Skip both '*' and '/'
                                found_end = True
                                break
                            i += 1
                        # If we didn't find the end, this comment continues to next line
                        # For single-line processing, just ignore rest of line
                        if not found_end:
                            break
                        continue
                    else:
                        result.append(char)
                else:
                    result.append(char)
            else:
                result.append(char)
                if char == quote_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    quote_char = None
            i += 1
        return ''.join(result)
    
    elif language in ['haskell', 'lua', 'elm']:
        # Remove -- comments
        in_string = False
        quote_char = None
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                    result.append(char)
                elif char == '-' and i + 1 < len(line) and line[i + 1] == '-':
                    # Comment starts here, ignore rest of line
                    break
                else:
                    result.append(char)
            else:
                result.append(char)
                if char == quote_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    quote_char = None
            i += 1
        return ''.join(result)
    
    elif language in ['sql']:
        # Remove -- and # comments
        in_string = False
        quote_char = None
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                    result.append(char)
                elif char == '#':
                    # Comment starts here, ignore rest of line
                    break
                elif char == '-' and i + 1 < len(line) and line[i + 1] == '-':
                    # Comment starts here, ignore rest of line
                    break
                else:
                    result.append(char)
            else:
                result.append(char)
                if char == quote_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    quote_char = None
            i += 1
        return ''.join(result)
    
    elif language in ['vim']:
        # Remove " comments
        in_string = False
        quote_char = None
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_string:
                if char == "'":
                    in_string = True
                    quote_char = char
                    result.append(char)
                elif char == '"':
                    # Comment starts here, ignore rest of line
                    break
                else:
                    result.append(char)
            else:
                result.append(char)
                if char == quote_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    quote_char = None
            i += 1
        return ''.join(result)
    
    elif language in ['lisp', 'scheme', 'clojure']:
        # Remove ; comments
        in_string = False
        quote_char = None
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                    result.append(char)
                elif char == ';':
                    # Comment starts here, ignore rest of line
                    break
                else:
                    result.append(char)
            else:
                result.append(char)
                if char == quote_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    quote_char = None
            i += 1
        return ''.join(result)
    
    elif language in ['erlang']:
        # Remove % comments
        in_string = False
        quote_char = None
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                    result.append(char)
                elif char == '%':
                    # Comment starts here, ignore rest of line
                    break
                else:
                    result.append(char)
            else:
                result.append(char)
                if char == quote_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    quote_char = None
            i += 1
        return ''.join(result)
    
    # For unknown languages, return as-is
    return line


def _has_indented_content_after_last_opening(fixed_lines: List[str], base_indent: str, indent_style: str, indent_size: int) -> bool:
    """
    Check if there's indented content after the last opening line.
    
    This helps determine if a new opening line should be a sibling or nested.
    If there's already indented content after the last opening line, then
    the new opening line should be a sibling. Otherwise, it should be nested.
    
    Args:
        fixed_lines: List of already processed lines
        base_indent: Base indentation string
        indent_style: Indentation style (' ' or '\t')
        indent_size: Size of each indentation level
        
    Returns:
        True if there's indented content after the last opening line
    """
    if not fixed_lines:
        return False
    
    # Find the last opening line
    last_opening_idx = None
    for i in range(len(fixed_lines) - 1, -1, -1):
        line = fixed_lines[i]
        if line.strip() and line.rstrip().endswith(':'):
            last_opening_idx = i
            break
    
    if last_opening_idx is None:
        return False
    
    # Get the indentation level of the last opening line
    opening_line = fixed_lines[last_opening_idx]
    opening_line_indent = opening_line[:len(opening_line) - len(opening_line.lstrip())]
    if opening_line_indent.startswith(base_indent):
        remaining_indent = opening_line_indent[len(base_indent):]
        if indent_style == '\t':
            opening_level = len(remaining_indent)
        else:
            opening_level = len(remaining_indent) // indent_size if remaining_indent else 0
    else:
        return False
    
    # Check if there's any line after the opening line that's more indented
    for i in range(last_opening_idx + 1, len(fixed_lines)):
        line = fixed_lines[i]
        if not line.strip():
            continue
        
        # Skip pure comment lines - they don't count as "indented content"
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            continue
        
        # Calculate the indentation level of this line
        line_indent = line[:len(line) - len(line.lstrip())]
        if line_indent.startswith(base_indent):
            remaining_indent = line_indent[len(base_indent):]
            if indent_style == '\t':
                line_level = len(remaining_indent)
            else:
                line_level = len(remaining_indent) // indent_size if remaining_indent else 0
            
            # If this line is more indented than the opening line, we have indented content
            if line_level > opening_level:
                return True
    
    return False


def _find_last_unclosed_opening_level(fixed_lines: List[str], base_indent: str, indent_style: str, indent_size: int) -> Optional[int]:
    """
    Find the indentation level of the last unclosed opening line to determine sibling level.
    
    This function analyzes the structure to find opening lines that haven't been closed yet,
    which helps determine the proper indentation level for sibling statements.
    
    Args:
        fixed_lines: List of already processed lines
        base_indent: Base indentation string
        indent_style: Indentation style (' ' or '\t')
        indent_size: Size of each indentation level
        
    Returns:
        Indentation level of the last unclosed opening line, or None if not found
    """
    if not fixed_lines:
        return None
    
    # Track opening lines and their levels
    opening_stack = []
    
    for line in fixed_lines:
        if not line.strip():
            continue
        
        # Calculate the indentation level of this line
        line_indent = line[:len(line) - len(line.lstrip())]
        if line_indent.startswith(base_indent):
            remaining_indent = line_indent[len(base_indent):]
            if indent_style == '\t':
                level = len(remaining_indent)
            else:
                level = len(remaining_indent) // indent_size if remaining_indent else 0
            
            # Check if this line ends with a colon (opening line in Python)
            if line.rstrip().endswith(':'):
                opening_stack.append(level)
            else:
                # This is a regular line - it might close some opening blocks
                # Remove any opening blocks that are at a higher level than this line
                while opening_stack and opening_stack[-1] >= level:
                    opening_stack.pop()
    
    # Return the level of the most recent unclosed opening line
    return opening_stack[-1] if opening_stack else None


def _is_opening_line(line: str, language: str) -> bool:
    """Check if a line opens a new block (increases indentation)."""
    # Remove comments and strings to avoid false positives
    clean_line = _remove_comments_and_strings(line, language).strip()
    
    if not clean_line:
        return False
    
    # Python-style blocks (colon at end indicates a block)
    if language == 'python':
        if clean_line.endswith(':'):
            return True
    elif language in ['ruby', 'lua', 'vim']:
        # Languages that use specific keywords to start blocks
        if language == 'ruby':
            ruby_opening_keywords = ['if ', 'unless ', 'while ', 'until ', 'for ', 'def ', 'class ', 'module ', 'begin', 'case ', 'when ']
            for keyword in ruby_opening_keywords:
                if clean_line.startswith(keyword):
                    return True
        elif language == 'lua':
            lua_opening_keywords = ['if ', 'while ', 'for ', 'function ', 'repeat', 'do']
            for keyword in lua_opening_keywords:
                if clean_line.startswith(keyword):
                    return True
        elif language == 'vim':
            vim_opening_keywords = ['if ', 'while ', 'for ', 'function ', 'try']
            for keyword in vim_opening_keywords:
                if clean_line.startswith(keyword):
                    return True
    else:
        # For other languages, colon might indicate a block (like in some cases)
        if clean_line.endswith(':'):
            return True
    
    # Bracket-based blocks
    if clean_line.endswith('{'):
        return True
    
    # Multi-line constructs that don't close on same line
    if clean_line.endswith('(') or clean_line.endswith('['):
        return True
    
    return False


def _is_closing_line(line: str, language: str) -> bool:
    """Check if a line closes a block (decreases indentation)."""
    # Remove comments and strings to avoid false positives
    clean_line = _remove_comments_and_strings(line, language).strip()
    
    if not clean_line:
        return False
    
    # Closing brackets
    if clean_line.startswith('}') or clean_line.startswith(']') or clean_line.startswith(')'):
        return True
    
    # Python-specific closing keywords (these start new blocks at a lower level)
    if language == 'python':
        closing_keywords = ['else:', 'elif ', 'except:', 'except ', 'finally:', 'case ']
        
        for keyword in closing_keywords:
            if clean_line.startswith(keyword):
                return True
    
    # Other language-specific closing keywords
    elif language in ['ruby']:
        ruby_closing_keywords = ['else', 'elsif ', 'rescue', 'ensure', 'when ', 'end']
        for keyword in ruby_closing_keywords:
            if clean_line.startswith(keyword):
                return True
    
    elif language in ['perl']:
        perl_closing_keywords = ['else', 'elsif ', 'elseif ']
        for keyword in perl_closing_keywords:
            if clean_line.startswith(keyword):
                return True
    
    elif language in ['lua']:
        lua_closing_keywords = ['else', 'elseif ', 'end']
        for keyword in lua_closing_keywords:
            if clean_line.startswith(keyword):
                return True
    
    elif language in ['vim']:
        vim_closing_keywords = ['else', 'elseif ', 'endif', 'endfor', 'endwhile', 'endfunction']
        for keyword in vim_closing_keywords:
            if clean_line.startswith(keyword):
                return True
    
    return False
