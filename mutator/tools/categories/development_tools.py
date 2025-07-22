"""
Development tools for the Coding Agent Framework.
"""

import ast
import subprocess
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..decorator import tool


def _auto_detect_language(code: str) -> str:
    """Auto-detect programming language from code content."""
    code_lower = code.lower().strip()
    
    # Python indicators
    if any(keyword in code_lower for keyword in ['def ', 'import ', 'from ', 'class ', 'if __name__']):
        return "python"
    
    # JavaScript/TypeScript indicators
    if any(keyword in code_lower for keyword in ['function ', 'const ', 'let ', 'var ', '=>', 'console.log']):
        if 'interface ' in code_lower or ': ' in code and 'string' in code_lower:
            return "typescript"
        return "javascript"
    
    # Java indicators
    if any(keyword in code_lower for keyword in ['public class', 'private ', 'public static void main']):
        return "java"
    
    # C/C++ indicators
    if any(keyword in code_lower for keyword in ['#include', 'int main', 'printf', 'cout']):
        return "c++"
    
    # Default to Python
    return "python"


__all__ = [] 