"""
Search operations for the Coding Agent Framework.

This module handles context search operations, including vector similarity search
and fallback text-based search methods.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core.types import ContextItem, ContextType
from .vector_store import VectorStoreManager
from .code_analyzer import CodeAnalyzer


class ContextSearcher:
    """Handles context search operations."""
    
    def __init__(self, vector_store: VectorStoreManager, code_analyzer: CodeAnalyzer,
                 working_directory: Path):
        """Initialize the context searcher."""
        self.vector_store = vector_store
        self.code_analyzer = code_analyzer
        self.working_directory = working_directory
        self.logger = logging.getLogger(__name__)
    
    def search_context(self, query: str, limit: int = 10, 
                      indexing_ready: bool = True) -> List[ContextItem]:
        """Search for relevant context using vector similarity or fallback methods."""
        # Use fallback search if indexing is not ready
        if not indexing_ready:
            self.logger.debug("Using fallback search (indexing not ready)")
            return self._fallback_search(query, limit)
        
        try:
            # Try vector similarity search first
            if self.vector_store.has_embedding_model():
                results = self.vector_store.search(query, limit)
                return self._convert_vector_results(results)
            else:
                # No embedding model, use fallback
                return self._fallback_search(query, limit)
            
        except Exception as e:
            self.logger.warning(f"Vector search failed, using fallback: {str(e)}")
            return self._fallback_search(query, limit)
    
    def _convert_vector_results(self, results: Dict[str, Any]) -> List[ContextItem]:
        """Convert vector search results to ContextItem objects."""
        context_items = []
        
        if not results['documents'] or not results['documents'][0]:
            return context_items
        
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if results['distances'] else 0.0
            
            # Convert distance to relevance score (lower distance = higher relevance)
            relevance_score = max(0.0, 1.0 - distance)
            
            # Determine context type based on metadata
            context_type = self._determine_context_type(metadata)
            
            context_items.append(ContextItem(
                type=context_type,
                content=doc,
                metadata=metadata,
                relevance_score=relevance_score,
                source=metadata.get('file_path', ''),
                line_start=metadata.get('start_line'),
                line_end=metadata.get('end_line')
            ))
        
        return context_items
    
    def _determine_context_type(self, metadata: Dict[str, Any]) -> ContextType:
        """Determine the context type based on metadata."""
        if 'element_type' in metadata:
            return {
                'function': ContextType.FUNCTION,
                'class': ContextType.CLASS,
                'variable': ContextType.VARIABLE
            }.get(metadata['element_type'], ContextType.FILE)
        return ContextType.FILE
    
    def _fallback_search(self, query: str, limit: int = 10) -> List[ContextItem]:
        """Fallback search using simple text matching when vector search is unavailable."""
        context_items = []
        query_lower = query.lower()
        
        try:
            # Search in files using simple text matching
            file_count = 0
            for file_path in self.working_directory.rglob('*'):
                if file_count >= limit:
                    break
                    
                if not file_path.is_file():
                    continue
                
                if self.code_analyzer.should_ignore_file(file_path, self.working_directory):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if query_lower in content.lower():
                        context_item = self._create_text_match_context(
                            file_path, content, query_lower
                        )
                        if context_item:
                            context_items.append(context_item)
                            file_count += 1
                except Exception:
                    continue  # Skip files that can't be read
            
            return context_items
            
        except Exception as e:
            self.logger.error(f"Fallback search failed: {str(e)}")
            return []
    
    def _create_text_match_context(self, file_path: Path, content: str, 
                                  query_lower: str) -> Optional[ContextItem]:
        """Create a context item from text matching results."""
        relative_path = file_path.relative_to(self.working_directory)
        
        # Find the line containing the query
        lines = content.split('\n')
        matching_lines = []
        
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                matching_lines.append((i + 1, line.strip()))
                if len(matching_lines) >= 3:  # Limit to 3 matches per file
                    break
        
        if not matching_lines:
            return None
        
        context_content = '\n'.join([
            f"Line {line_num}: {line}" for line_num, line in matching_lines
        ])
        
        return ContextItem(
            type=ContextType.FILE,
            content=context_content,
            metadata={
                'file_path': str(relative_path),
                'language': self.code_analyzer.get_file_language(file_path),
                'search_method': 'fallback_text_match'
            },
            relevance_score=0.7,  # Fixed relevance for text matches
            source=str(relative_path),
            line_start=matching_lines[0][0],
            line_end=matching_lines[-1][0]
        )
    
    def get_file_context(self, file_path: str) -> Optional[ContextItem]:
        """Get context for a specific file."""
        try:
            full_path = self.working_directory / file_path
            if not full_path.exists():
                return None
            
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            language = self.code_analyzer.get_file_language(full_path)
            
            # Extract code elements
            elements = self.code_analyzer.extract_code_elements(content, language)
            
            return ContextItem(
                type=ContextType.FILE,
                content=content,
                metadata={
                    'file_path': file_path,
                    'language': language,
                    'file_size': len(content),
                    'elements': elements,
                    'last_modified': full_path.stat().st_mtime
                },
                relevance_score=1.0,
                source=file_path
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get file context for {file_path}: {str(e)}")
            return None
    
    def get_directory_context(self, directory_path: str) -> List[ContextItem]:
        """Get context for a directory (file listing and structure)."""
        try:
            full_path = self.working_directory / directory_path
            if not full_path.exists() or not full_path.is_dir():
                return []
            
            # Create directory structure context
            files = []
            dirs = []
            
            for item in full_path.iterdir():
                if self.code_analyzer.should_ignore_file(item, self.working_directory):
                    continue
                    
                if item.is_dir():
                    dirs.append(item.name)
                else:
                    files.append({
                        'name': item.name,
                        'size': item.stat().st_size,
                        'language': self.code_analyzer.get_file_language(item)
                    })
            
            directory_content = self._build_directory_content(directory_path, files, dirs)
            
            return [ContextItem(
                type=ContextType.DIRECTORY,
                content=directory_content,
                metadata={
                    'directory_path': directory_path,
                    'file_count': len(files),
                    'subdirectory_count': len(dirs),
                    'files': files,
                    'subdirectories': dirs
                },
                relevance_score=1.0,
                source=directory_path
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to get directory context for {directory_path}: {str(e)}")
            return []
    
    def _build_directory_content(self, directory_path: str, files: List[Dict[str, Any]], 
                                dirs: List[str]) -> str:
        """Build directory content string."""
        content = f"Directory: {directory_path}\n\n"
        content += f"Subdirectories ({len(dirs)}):\n"
        
        for d in sorted(dirs):
            content += f"  {d}/\n"
        
        content += f"\nFiles ({len(files)}):\n"
        
        for f in sorted(files, key=lambda x: x['name']):
            content += f"  {f['name']} ({f['language']}, {f['size']} bytes)\n"
        
        return content 