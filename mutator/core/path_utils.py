"""
Path utilities for the Coding Agent Framework.

This module provides utilities for handling paths, particularly for converting
absolute paths to relative paths to optimize request sizes.
"""

import os
from pathlib import Path
from typing import Optional, Union, List, Type
from pydantic import BaseModel
import json


# Global code analyzer instance for gitignore checking
_code_analyzer = None


def get_code_analyzer():
    """Get or create a global CodeAnalyzer instance for gitignore checking."""
    global _code_analyzer
    if _code_analyzer is None:
        from ..context.code_analyzer import CodeAnalyzer
        from ..core.config import ContextConfig
        
        # Create a minimal context config for gitignore checking
        config = ContextConfig()
        _code_analyzer = CodeAnalyzer(config)
    
    return _code_analyzer


def should_exclude_from_search(file_path: Path, working_directory: Optional[Path] = None) -> bool:
    """Check if a file should be excluded from search results based on .gitignore patterns.
    
    Args:
        file_path: Path to the file to check
        working_directory: Working directory (defaults to current directory)
    
    Returns:
        True if the file should be excluded, False otherwise
    """
    if working_directory is None:
        working_directory = Path.cwd()
    
    # Convert relative paths to absolute paths
    if not file_path.is_absolute():
        file_path = working_directory / file_path
    
    # Use the existing CodeAnalyzer gitignore functionality
    analyzer = get_code_analyzer()
    return analyzer.should_ignore_file(file_path, working_directory)


def get_working_directory() -> Path:
    """Get the current working directory."""
    return Path.cwd()


def to_relative_path(path: Union[str, Path], working_directory: Optional[Union[str, Path]] = None) -> str:
    """
    Convert an absolute path to a relative path based on the working directory.
    
    Args:
        path: The path to convert (can be absolute or relative)
        working_directory: The working directory to use as base (defaults to current working directory)
    
    Returns:
        str: The relative path as a string
    """
    if working_directory is None:
        working_directory = get_working_directory()
    
    path_obj = Path(path)
    working_dir_obj = Path(working_directory)
    
    # If the path is already relative, return it as is
    if not path_obj.is_absolute():
        return str(path_obj)
    
    try:
        # Try to make it relative to the working directory
        relative_path = path_obj.relative_to(working_dir_obj)
        return str(relative_path)
    except ValueError:
        # If the path is not within the working directory, return the absolute path
        return str(path_obj)


def normalize_path_for_response(path: Union[str, Path], working_directory: Optional[Union[str, Path]] = None) -> str:
    """
    Normalize a path for use in tool responses to minimize request size.
    
    This function converts absolute paths to relative paths when they are within
    the working directory, which helps reduce the size of tool responses.
    
    Args:
        path: The path to normalize
        working_directory: The working directory to use as base (defaults to current working directory)
    
    Returns:
        str: The normalized path as a string
    """
    return to_relative_path(path, working_directory) 


def find_git_root(start_path: Union[str, Path]) -> Optional[Path]:
    """
    Find the root directory of a git repository.
    
    Args:
        start_path: Directory to start searching from
        
    Returns:
        Path to git root directory or None if not found
    """
    start_path = Path(start_path).resolve()
    
    # Check if current directory is git root
    if (start_path / ".git").exists():
        return start_path
    
    # Check parent directories
    for parent in start_path.parents:
        if (parent / ".git").exists():
            return parent
    
    return None


def get_common_ignore_patterns() -> List[str]:
    """
    Get common file patterns that should be ignored in code analysis.
    
    Returns:
        List of file patterns to ignore
    """
    return [
        "*.pyc",
        "*.pyo",
        "__pycache__",
        ".git",
        ".svn",
        ".hg",
        ".DS_Store",
        "node_modules",
        "*.egg-info",
        "build",
        "dist",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        ".coverage",
        "htmlcov",
        ".env",
        ".venv",
        "venv",
        "env",
        ".idea",
        ".vscode",
        "*.log",
        "*.tmp",
        "*.bak",
        ".cache",
        ".sass-cache",
        ".parcel-cache",
        ".next",
        ".nuxt",
        ".vuepress",
        "coverage",
        ".nyc_output",
        "lib-cov",
        "*.lcov",
        ".grunt",
        ".lock-wscript",
        "*.pid",
        "*.seed",
        "*.pid.lock",
        "*.tgz",
        "*.tar.gz",
        "*.rar",
        "*.zip",
        "*.7z",
        "*.dmg",
        "*.iso",
        "*.jar",
        "*.war",
        "*.ear",
        "*.sar",
        "*.class",
        "*.so",
        "*.dll",
        "*.dylib",
        "*.exe",
        "*.out",
        "*.app",
        "*.deb",
        "*.rpm",
        "*.msi",
        "*.msm",
        "*.msp",
        "*.cab",
        "*.whl",
        "*.egg",
        "*.dmg",
        "*.pkg",
        "*.snap",
        "*.flatpak",
        "*.appimage",
        "*.deb",
        "*.rpm",
        "*.apk",
        "*.ipa",
        "*.aab",
        "*.tar.xz",
        "*.tar.bz2",
        "*.tar.gz",
        "*.tar.lz",
        "*.tar.lzma",
        "*.tar.lzo",
        "*.tar.Z",
        "*.tgz",
        "*.tbz",
        "*.tbz2",
        "*.tz",
        "*.deb",
        "*.rpm",
        "*.xz",
        "*.lz",
        "*.lzma",
        "*.lzo",
        "*.Z",
        "*.gz",
        "*.bz2",
        "*.lzma",
        "*.xz",
        "*.Z",
        "*.7z",
        "*.ace",
        "*.afa",
        "*.alz",
        "*.apk",
        "*.arc",
        "*.arj",
        "*.bz",
        "*.cab",
        "*.cpio",
        "*.deb",
        "*.dmg",
        "*.ear",
        "*.gz",
        "*.iso",
        "*.jar",
        "*.lha",
        "*.lzh",
        "*.lzma",
        "*.lzo",
        "*.rar",
        "*.rpm",
        "*.rz",
        "*.tar",
        "*.taz",
        "*.tbz",
        "*.tbz2",
        "*.tgz",
        "*.tlz",
        "*.txz",
        "*.tZ",
        "*.tz",
        "*.war",
        "*.xpi",
        "*.xz",
        "*.Z",
        "*.zip",
        "*.zoo",
        "*.zpaq",
        "*.zst",
        "*.zstd",
        "*.zz",
        "*.zzip"
    ]


def get_common_text_extensions() -> List[str]:
    """
    Get common text file extensions that should be included in code analysis.
    
    Returns:
        List of file extensions to include
    """
    return [
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".vue",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".cc",
        ".cxx",
        ".cs",
        ".php",
        ".rb",
        ".go",
        ".rs",
        ".swift",
        ".kt",
        ".scala",
        ".clj",
        ".hs",
        ".ml",
        ".fs",
        ".ex",
        ".exs",
        ".erl",
        ".hrl",
        ".elm",
        ".dart",
        ".lua",
        ".pl",
        ".pm",
        ".r",
        ".R",
        ".matlab",
        ".m",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".psm1",
        ".bat",
        ".cmd",
        ".html",
        ".htm",
        ".xhtml",
        ".xml",
        ".xsl",
        ".xslt",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".styl",
        ".stylus",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".properties",
        ".env",
        ".gitignore",
        ".dockerignore",
        ".gitattributes",
        ".editorconfig",
        ".prettierrc",
        ".eslintrc",
        ".babelrc",
        ".tsconfig",
        ".jsconfig",
        ".package",
        ".lock",
        ".md",
        ".markdown",
        ".rst",
        ".txt",
        ".log",
        ".sql",
        ".graphql",
        ".gql",
        ".proto",
        ".thrift",
        ".avro",
        ".schema",
        ".avsc",
        ".jsonl",
        ".ndjson",
        ".csv",
        ".tsv",
        ".psv",
        ".ssv",
        ".dsv",
        ".makefile",
        ".mk",
        ".cmake",
        ".gradle",
        ".sbt",
        ".cabal",
        ".stack",
        ".nix",
        ".bzl",
        ".bazel",
        ".build",
        ".ninja",
        ".meson",
        ".gyp",
        ".gypi",
        ".waf",
        ".scons",
        ".ant",
        ".maven",
        ".ivy",
        ".pom",
        ".project",
        ".classpath",
        ".settings",
        ".launch",
        ".prefs",
        ".workspace",
        ".metadata",
        ".iml",
        ".ipr",
        ".iws",
        ".idea",
        ".vscode",
        ".sublime-project",
        ".sublime-workspace",
        ".atom",
        ".brackets",
        ".textmate",
        ".tmproj",
        ".tmproj",
        ".xcodeproj",
        ".xcworkspace",
        ".pbxproj",
        ".pbxuser",
        ".mode1v3",
        ".mode2v3",
        ".perspectivev3",
        ".xcuserstate",
        ".xcscheme",
        ".xccheckout",
        ".moved-aside",
        ".xcuserstate",
        ".xcscmblueprint",
        ".xcsettings",
        ".gitkeep",
        ".gitignore_global",
        ".gitconfig",
        ".gitmodules",
        ".gitattributes",
        ".gitignore",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global",
        ".gitkeep",
        ".gitmodules",
        ".gitattributes",
        ".gitconfig",
        ".gitignore_global"
    ]


# Pydantic utility functions
def parse_pydantic_output(raw_output: str, pydantic_model: Type[BaseModel]) -> Optional[BaseModel]:
    """
    Parse raw LLM output into a Pydantic model instance.
    
    Args:
        raw_output: Raw text output from LLM
        pydantic_model: Pydantic model class to parse into
        
    Returns:
        Pydantic model instance or None if parsing fails
    """
    if not raw_output or not pydantic_model:
        return None
    
    import json
    import re
    
    # Try to extract JSON from the raw output
    json_str = extract_json_from_text(raw_output)
    
    if not json_str:
        return None
    
    try:
        # Parse JSON and create Pydantic model
        parsed_data = json.loads(json_str)
        return pydantic_model(**parsed_data)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # If direct parsing fails, try to clean and retry
        try:
            cleaned_json = clean_json_string(json_str)
            parsed_data = json.loads(cleaned_json)
            return pydantic_model(**parsed_data)
        except Exception:
            return None


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON content from text that may contain other content.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        JSON string or None if not found
    """
    import re
    
    # Try to find JSON blocks in code blocks (including nested objects)
    json_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0]
    
    # Try to find complete JSON structures with proper nesting
    # First look for objects (which are more common for structured output)
    def find_balanced_json(text, start_char, end_char):
        """Find balanced JSON structures."""
        results = []
        i = 0
        while i < len(text):
            if text[i] == start_char:
                # Found start character, now find the matching end
                count = 1
                j = i + 1
                while j < len(text) and count > 0:
                    if text[j] == start_char:
                        count += 1
                    elif text[j] == end_char:
                        count -= 1
                    j += 1
                
                if count == 0:
                    # Found balanced structure
                    results.append(text[i:j])
                    i = j
                else:
                    i += 1
            else:
                i += 1
        
        return results
    
    # Look for balanced objects first
    object_matches = find_balanced_json(text, '{', '}')
    if object_matches:
        # Return the largest match (most likely to be complete)
        return max(object_matches, key=len)
    
    # Look for balanced arrays
    array_matches = find_balanced_json(text, '[', ']')
    if array_matches:
        # Return the largest match (most likely to be complete)
        return max(array_matches, key=len)
    
    return None


def clean_json_string(json_str: str) -> str:
    """
    Clean JSON string to make it more likely to parse correctly.
    
    Args:
        json_str: JSON string to clean
        
    Returns:
        Cleaned JSON string
    """
    import re
    
    # Remove comments
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Fix common issues
    json_str = json_str.strip()
    
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix quotes
    json_str = json_str.replace("'", '"')
    
    return json_str


def format_pydantic_for_llm(pydantic_model: Type[BaseModel]) -> str:
    """
    Format Pydantic model schema for LLM prompt.
    
    Args:
        pydantic_model: Pydantic model class
        
    Returns:
        Formatted schema string for LLM
    """
    # Use model_json_schema for Pydantic V2 compatibility
    if hasattr(pydantic_model, 'model_json_schema'):
        schema = pydantic_model.model_json_schema()
    else:
        # Fallback for older Pydantic versions
        schema = pydantic_model.schema()
    
    # Create a readable format for the LLM
    example_dict = create_example_from_schema(schema)
    
    prompt = f"""
Please format your response as a JSON object that matches this exact schema:

Model: {pydantic_model.__name__}
Schema: {json.dumps(schema, indent=2)}

Example format:
```json
{json.dumps(example_dict, indent=2)}
```

IMPORTANT: Your response must be valid JSON that can be parsed directly into the {pydantic_model.__name__} model.
"""
    
    return prompt


def create_example_from_schema(schema: dict) -> dict:
    """
    Create an example dictionary from a Pydantic schema.
    
    Args:
        schema: Pydantic model schema
        
    Returns:
        Example dictionary matching the schema
    """
    example = {}
    properties = schema.get("properties", {})
    
    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type", "string")
        
        if field_type == "string":
            example[field_name] = f"example_{field_name}"
        elif field_type == "integer":
            example[field_name] = 42
        elif field_type == "number":
            example[field_name] = 3.14
        elif field_type == "boolean":
            example[field_name] = True
        elif field_type == "array":
            items_schema = field_schema.get("items", {})
            if items_schema.get("type") == "string":
                example[field_name] = ["example_item"]
            else:
                example[field_name] = []
        elif field_type == "object":
            example[field_name] = {}
        else:
            example[field_name] = None
    
    return example 