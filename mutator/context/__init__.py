"""Context management components for the Coding Agent Framework."""

# Apply comprehensive ONNX suppression and initialize environment IMMEDIATELY
from .suppress_warnings import apply_comprehensive_suppression, initialize_environment
apply_comprehensive_suppression()
initialize_environment()

from .manager import ContextManager
from .vector_store import VectorStoreManager
from .code_analyzer import CodeAnalyzer
from .indexer import CodebaseIndexer
from .search import ContextSearcher
from .git_integration import GitIntegration

__all__ = [
    "ContextManager",
    "VectorStoreManager", 
    "CodeAnalyzer",
    "CodebaseIndexer",
    "ContextSearcher",
    "GitIntegration",
    "initialize_environment"
] 