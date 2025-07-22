"""
Context manager for the Coding Agent Framework.

This module provides the ContextManager class that orchestrates codebase indexing,
vector storage, context retrieval, and project understanding through various
specialized components.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.types import ContextItem, ContextType
from ..core.config import ContextConfig, VectorStoreConfig
from .suppress_warnings import initialize_environment
from .vector_store import VectorStoreManager
from .code_analyzer import CodeAnalyzer
from .indexer import CodebaseIndexer
from .search import ContextSearcher
from .git_integration import GitIntegration


class ContextManager:
    """Manages codebase context through specialized components."""
    
    def __init__(self, 
                 context_config: ContextConfig,
                 vector_config: VectorStoreConfig,
                 working_directory: str = "."):
        """Initialize the context manager."""
        # Initialize environment FIRST to prevent ONNX runtime issues
        initialize_environment()
        
        self.context_config = context_config
        self.vector_config = vector_config
        self.working_directory = Path(working_directory)
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized components
        self._setup_components()
        
        # Track context cache
        self._context_cache: Dict[str, ContextItem] = {}
    
    def _setup_components(self) -> None:
        """Setup all specialized components."""
        # Initialize vector store manager
        self.vector_store = VectorStoreManager(self.vector_config)
        
        # Initialize code analyzer
        self.code_analyzer = CodeAnalyzer(self.context_config)
        
        # Initialize indexer
        self.indexer = CodebaseIndexer(
            self.context_config,
            self.vector_store,
            self.code_analyzer,
            self.working_directory
        )
        
        # Initialize searcher
        self.searcher = ContextSearcher(
            self.vector_store,
            self.code_analyzer,
            self.working_directory
        )
        
        # Initialize Git integration
        self.git_integration = GitIntegration(self.working_directory)
    
    def discover_project_context(self) -> List[ContextItem]:
        """Discover project-level context files."""
        context_items = []
        
        # Look for project context files
        for context_file in self.context_config.project_context_files:
            file_path = self.working_directory / context_file
            if not file_path.exists():
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8')
                context_items.append(ContextItem(
                    type=ContextType.DOCUMENTATION,
                    content=content,
                    metadata={
                        'file_path': str(file_path),
                        'file_name': context_file,
                        'type': 'project_context'
                    },
                    relevance_score=1.0,
                    source=str(file_path)
                ))
            except Exception as e:
                self.logger.warning(f"Failed to read {context_file}: {str(e)}")
        
        return context_items
    
    def index_codebase(self, force_reindex: bool = False, async_mode: bool = True) -> None:
        """Index the entire codebase for vector search."""
        self.indexer.index_codebase(force_reindex, async_mode)
    
    def search_context(self, query: str, limit: int = 10) -> List[ContextItem]:
        """Search for relevant context using vector similarity or fallback methods."""
        indexing_status = self.indexer.get_indexing_status()
        indexing_ready = (
            indexing_status['indexing_complete'] and 
            not indexing_status['indexing_in_progress'] and
            not indexing_status['indexing_error']
        )
        
        return self.searcher.search_context(query, limit, indexing_ready)
    
    def get_file_context(self, file_path: str) -> Optional[ContextItem]:
        """Get context for a specific file."""
        return self.searcher.get_file_context(file_path)
    
    def get_directory_context(self, directory_path: str) -> List[ContextItem]:
        """Get context for a directory (file listing and structure)."""
        return self.searcher.get_directory_context(directory_path)
    
    def get_git_context(self) -> List[ContextItem]:
        """Get Git repository context."""
        return self.git_integration.get_git_context()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context."""
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            
            # Get indexing status
            indexing_status = self.indexer.get_indexing_status()
            
            # Get Git stats
            git_stats = self.git_integration.get_repository_stats()
            
            return {
                **vector_stats,
                **indexing_status,
                **git_stats,
                'working_directory': str(self.working_directory)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get context summary: {str(e)}")
            return {}
    
    def clear_context(self) -> None:
        """Clear all context and cached data."""
        try:
            # Clear vector store
            self.vector_store.clear_collection()
            
            # Clear caches
            self._context_cache.clear()
            
            self.logger.debug("Context cleared successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to clear context: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the context manager."""
        vector_health = self.vector_store.health_check()
        indexing_status = self.indexer.get_indexing_status()
        
        return {
            "status": "healthy",
            **vector_health,
            **indexing_status,
            "has_git": self.git_integration.has_git_repo()
        }
    
    async def cleanup(self) -> None:
        """Clean up resources used by the context manager."""
        try:
            # Stop any ongoing indexing
            self.indexer.stop_indexing()
            
            # Clean up vector store resources if needed
            if hasattr(self.vector_store, 'cleanup'):
                await self.vector_store.cleanup()
                
        except Exception as e:
            # Log cleanup error but don't raise
            self.logger.warning(f"Warning: Error during context manager cleanup: {e}")


# Export the main class
__all__ = ["ContextManager"] 