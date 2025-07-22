"""
Code analysis and extraction utilities for the Coding Agent Framework.

This module provides functionality to analyze code files, extract elements
like functions and classes, and determine file languages.
"""

import re
import fnmatch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import os

from ..core.config import ContextConfig


class CodeAnalyzer:
    """
    Analyzes code files and extracts structural elements.
    
    This class provides functionality to:
    - Analyze code files and extract functions, classes, and other elements
    - Determine programming languages from file extensions
    - Check if files should be ignored based on configuration patterns and .gitignore files
    - Chunk file content for processing
    
    File Ignoring Features:
    - Respects configuration ignore patterns from ContextConfig
    - Automatically detects and parses .gitignore files in git repositories
    - Supports standard gitignore patterns including wildcards and directory patterns
    - Handles gitignore negation patterns (patterns starting with '!')
    - Caches gitignore patterns for performance
    - Only applies gitignore patterns in actual git repositories (directories with .git)
    """
    
    def __init__(self, context_config: ContextConfig):
        """Initialize the code analyzer."""
        self.context_config = context_config
        self._language_map = self._build_language_map()
        self._language_patterns = self._build_language_patterns()
        self._gitignore_patterns: Optional[List[Tuple[str, Path]]] = None
        self._gitignore_cache_path: Optional[Path] = None
    
    def _build_language_map(self) -> Dict[str, str]:
        """Build mapping of file extensions to programming languages."""
        return {
            '.py': 'python',
            '.pl': 'perl',
            '.pm': 'perl',
            '.rb': 'ruby',
            '.php': 'php',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sh': 'bash',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.md': 'markdown',
            '.txt': 'text',
        }
    
    def _build_language_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Build patterns for extracting functions and classes by language."""
        return {
            'python': {
                'class': [r'^class\s+(\w+).*?:'],
                'function': [r'^def\s+(\w+)\s*\([^)]*\)\s*:']
            },
            'javascript': {
                'class': [r'^class\s+(\w+)'],
                'function': [
                    r'^function\s+(\w+)\s*\(',
                    r'^const\s+(\w+)\s*=\s*\(',
                    r'^let\s+(\w+)\s*=\s*\(',
                    r'^var\s+(\w+)\s*=\s*\(',
                    r'^(\w+)\s*:\s*function\s*\(',
                    r'^(\w+)\s*\([^)]*\)\s*=>\s*\{'
                ]
            },
            'typescript': {
                'class': [r'^class\s+(\w+)', r'^interface\s+(\w+)', r'^type\s+(\w+)\s*='],
                'function': [
                    r'^function\s+(\w+)\s*\(',
                    r'^const\s+(\w+)\s*=\s*\(',
                    r'^let\s+(\w+)\s*=\s*\(',
                    r'^var\s+(\w+)\s*=\s*\(',
                    r'^(\w+)\s*:\s*function\s*\(',
                    r'^(\w+)\s*\([^)]*\)\s*=>\s*\{'
                ]
            },
            'java': {
                'class': [r'^class\s+(\w+)', r'^interface\s+(\w+)', r'^enum\s+(\w+)'],
                'function': [r'^\s*(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(']
            },
            'go': {
                'class': [r'^type\s+(\w+)\s+struct'],
                'function': [r'^func\s+(\w+)\s*\(', r'^func\s+\(\w+\s+\*?\w+\)\s+(\w+)\s*\(']
            },
            'rust': {
                'class': [r'^struct\s+(\w+)', r'^enum\s+(\w+)', r'^trait\s+(\w+)'],
                'function': [r'^fn\s+(\w+)\s*\(']
            },
            'cpp': {
                'class': [r'^class\s+(\w+)', r'^struct\s+(\w+)'],
                'function': [r'^\w+\s+(\w+)\s*\(']
            },
            'c': {
                'class': [r'^struct\s+(\w+)', r'^typedef\s+struct\s+(\w+)'],
                'function': [r'^\w+\s+(\w+)\s*\(']
            },
            'csharp': {
                'class': [r'^class\s+(\w+)', r'^interface\s+(\w+)', r'^struct\s+(\w+)'],
                'function': [r'^\s*(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(']
            },
            'swift': {
                'class': [r'^class\s+(\w+)', r'^struct\s+(\w+)', r'^protocol\s+(\w+)'],
                'function': [r'^func\s+(\w+)\s*\(']
            },
            'kotlin': {
                'class': [r'^class\s+(\w+)', r'^interface\s+(\w+)', r'^data\s+class\s+(\w+)'],
                'function': [r'^fun\s+(\w+)\s*\(']
            },
            'scala': {
                'class': [r'^class\s+(\w+)', r'^object\s+(\w+)', r'^trait\s+(\w+)'],
                'function': [r'^def\s+(\w+)\s*\(']
            },
            'ruby': {
                'class': [r'^class\s+(\w+)', r'^module\s+(\w+)'],
                'function': [r'^def\s+(\w+)\s*\(']
            },
            'php': {
                'class': [r'^class\s+(\w+)', r'^interface\s+(\w+)', r'^trait\s+(\w+)'],
                'function': [r'^function\s+(\w+)\s*\(']
            },
            'perl': {
                'class': [r'^package\s+(\w+)'],
                'function': [r'^sub\s+(\w+)\s*\{']
            },
            'bash': {
                'class': [],
                'function': [r'^(\w+)\s*\(\)\s*\{', r'^function\s+(\w+)\s*\(\)\s*\{']
            }
        }
    
    def _find_gitignore_files(self, working_directory: Path) -> List[Path]:
        """Find all .gitignore files in the directory tree."""
        gitignore_files = []
        
        # Check if this is a git repository
        if not (working_directory / '.git').exists():
            return gitignore_files
        
        # Find all .gitignore files
        for root, dirs, files in os.walk(working_directory):
            root_path = Path(root)
            
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            
            if '.gitignore' in files:
                gitignore_files.append(root_path / '.gitignore')
        
        return gitignore_files
    
    def _parse_gitignore_file(self, gitignore_path: Path, working_directory: Path) -> List[Tuple[str, Path]]:
        """Parse a .gitignore file and return patterns with their base directory."""
        patterns = []
        
        if not gitignore_path.exists():
            return patterns
        
        # Get the directory containing this gitignore file
        gitignore_dir = gitignore_path.parent
        
        try:
            content = gitignore_path.read_text(encoding='utf-8', errors='ignore')
            
            for line in content.splitlines():
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Handle negation patterns (we'll keep them as-is for now)
                if line.startswith('!'):
                    patterns.append((line, gitignore_dir))
                    continue
                
                # Convert gitignore patterns to fnmatch patterns
                # Handle directory patterns (ending with /)
                if line.endswith('/'):
                    patterns.append((line.rstrip('/'), gitignore_dir))
                    patterns.append((line.rstrip('/') + '/*', gitignore_dir))
                else:
                    patterns.append((line, gitignore_dir))
                    # Add pattern for matching directories
                    patterns.append((line + '/', gitignore_dir))
                    patterns.append((line + '/*', gitignore_dir))
                
        except Exception:
            # If we can't read the file, skip it
            pass
        
        return patterns
    
    def _load_gitignore_patterns(self, working_directory: Path) -> List[Tuple[str, Path]]:
        """Load and cache gitignore patterns with their base directories."""
        if self._gitignore_patterns is not None and self._gitignore_cache_path == working_directory:
            return self._gitignore_patterns
        
        all_patterns = []
        gitignore_files = self._find_gitignore_files(working_directory)
        
        for gitignore_file in gitignore_files:
            patterns = self._parse_gitignore_file(gitignore_file, working_directory)
            all_patterns.extend(patterns)
        
        # Cache the patterns
        self._gitignore_patterns = all_patterns
        self._gitignore_cache_path = working_directory
        
        return all_patterns
    
    def _matches_gitignore_pattern(self, file_path: Path, working_directory: Path) -> bool:
        """Check if a file matches any gitignore pattern."""
        gitignore_patterns = self._load_gitignore_patterns(working_directory)
        
        if not gitignore_patterns:
            return False
        
        relative_path = file_path.relative_to(working_directory)
        
        # Track if file should be ignored
        should_ignore = False
        
        # Check against all gitignore patterns
        for pattern, base_dir in gitignore_patterns:
            # Handle negation patterns
            if pattern.startswith('!'):
                negation_pattern = pattern[1:]
                if self._matches_gitignore_pattern_relative(relative_path, negation_pattern, base_dir, working_directory):
                    should_ignore = False  # Negation pattern matches, so don't ignore
                continue
            
            # Check if pattern matches
            if self._matches_gitignore_pattern_relative(relative_path, pattern, base_dir, working_directory):
                should_ignore = True
        
        return should_ignore
    
    def _matches_gitignore_pattern_relative(self, file_relative_path: Path, pattern: str, 
                                          pattern_base_dir: Path, working_directory: Path) -> bool:
        """Check if a file matches a gitignore pattern relative to the pattern's base directory."""
        # Get the relative path from the pattern's base directory to the working directory
        try:
            base_relative_to_working = pattern_base_dir.relative_to(working_directory)
        except ValueError:
            return False
        
        # Check if the file is within the scope of this gitignore file
        if base_relative_to_working != Path('.'):
            # The gitignore is in a subdirectory, check if file is in that subdirectory
            try:
                file_relative_to_base = file_relative_path.relative_to(base_relative_to_working)
            except ValueError:
                # File is not in the subdirectory, so this pattern doesn't apply
                return False
        else:
            # Gitignore is in the root directory
            file_relative_to_base = file_relative_path
        
        # Convert to string for pattern matching
        path_str = str(file_relative_to_base)
        filename = file_relative_to_base.name
        
        # Apply the pattern matching logic
        return self._matches_pattern(path_str, pattern, filename)
    
    def _matches_pattern(self, path_str: str, pattern: str, filename: str) -> bool:
        """Check if a path matches a gitignore pattern."""
        # Direct filename match
        if fnmatch.fnmatch(filename, pattern):
            return True
        
        # Full path match
        if fnmatch.fnmatch(path_str, pattern):
            return True
        
        # Check if pattern matches any part of the path
        path_parts = path_str.split('/')
        
        # For patterns like "*.log", check if any part matches
        if '*' in pattern and not '/' in pattern:
            return any(fnmatch.fnmatch(part, pattern) for part in path_parts)
        
        # For directory patterns, check if any parent directory matches
        if pattern.endswith('/') or pattern.endswith('/*'):
            clean_pattern = pattern.rstrip('/*')
            for i in range(len(path_parts)):
                partial_path = '/'.join(path_parts[:i+1])
                if fnmatch.fnmatch(partial_path, clean_pattern):
                    return True
        
        return False
    
    def clear_gitignore_cache(self) -> None:
        """Clear the cached gitignore patterns."""
        self._gitignore_patterns = None
        self._gitignore_cache_path = None
    
    def get_ignore_patterns_info(self, working_directory: Path) -> Dict[str, Any]:
        """Get information about ignore patterns being used."""
        info = {
            'config_patterns': self.context_config.ignore_patterns.copy(),
            'gitignore_patterns': [],
            'gitignore_files': [],
            'is_git_repository': (working_directory / '.git').exists()
        }
        
        if info['is_git_repository']:
            gitignore_files = self._find_gitignore_files(working_directory)
            info['gitignore_files'] = [str(f.relative_to(working_directory)) for f in gitignore_files]
            
            # Get patterns with their base directories
            patterns_with_dirs = self._load_gitignore_patterns(working_directory)
            info['gitignore_patterns'] = [
                {
                    'pattern': pattern,
                    'base_dir': str(base_dir.relative_to(working_directory))
                }
                for pattern, base_dir in patterns_with_dirs
            ]
        
        return info
    
    def should_ignore_file(self, file_path: Path, working_directory: Path) -> bool:
        """Check if file should be ignored based on patterns and .gitignore."""
        if not file_path.is_relative_to(working_directory):
            return True
        
        relative_path = file_path.relative_to(working_directory)
        path_str = str(relative_path)
        
        # Check config ignore patterns first
        for pattern in self.context_config.ignore_patterns:
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                return True
        
        # Check gitignore patterns if this is a git repository
        if self._matches_gitignore_pattern(file_path, working_directory):
            return True
        
        return False
    
    def get_file_language(self, file_path: Path) -> str:
        """Determine the programming language of a file."""
        suffix = file_path.suffix.lower()
        return self._language_map.get(suffix, 'text')
    
    def chunk_content(self, content: str, chunk_size: int, 
                     chunk_overlap: int) -> List[Tuple[str, int, int]]:
        """Chunk file content into smaller pieces for embedding."""
        lines = content.split('\n')
        chunks = []
        
        current_chunk = []
        current_size = 0
        start_line = 0
        
        for i, line in enumerate(lines):
            line_size = len(line)
            
            # If adding this line would exceed chunk size, save current chunk
            if current_size + line_size > chunk_size and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append((chunk_content, start_line, i - 1))
                
                # Start new chunk with overlap
                overlap_lines = min(chunk_overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
                current_size = sum(len(line) for line in current_chunk)
                start_line = i - overlap_lines
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add the last chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append((chunk_content, start_line, len(lines) - 1))
        
        return chunks
    
    def extract_code_elements(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Extract code elements (functions, classes) from content using generic patterns."""
        elements = []
        
        # Get patterns for this language
        patterns = self._language_patterns.get(language, {})
        
        # Extract classes
        class_patterns = patterns.get('class', [])
        for pattern in class_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                elements.append({
                    'type': 'class',
                    'name': match.group(1),
                    'line': content[:match.start()].count('\n') + 1,
                    'signature': match.group(0)
                })
        
        # Extract functions
        function_patterns = patterns.get('function', [])
        for pattern in function_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                elements.append({
                    'type': 'function',
                    'name': match.group(1),
                    'line': content[:match.start()].count('\n') + 1,
                    'signature': match.group(0)
                })
        
        return elements
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file and return its metadata and elements."""
        if not file_path.exists():
            return {}
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            language = self.get_file_language(file_path)
            elements = self.extract_code_elements(content, language)
            
            return {
                'file_path': str(file_path),
                'language': language,
                'file_size': len(content),
                'line_count': len(content.split('\n')),
                'elements': elements,
                'last_modified': file_path.stat().st_mtime
            }
            
        except Exception as e:
            return {'error': str(e)}

