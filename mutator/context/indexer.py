"""
Indexing operations for the Coding Agent Framework.

This module handles codebase indexing, file processing, and background
indexing operations for the context manager.
"""

import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import fnmatch
import json
import hashlib

from git import Repo, InvalidGitRepositoryError

from ..core.config import ContextConfig, VectorStoreConfig
from .vector_store import VectorStoreManager
from .code_analyzer import CodeAnalyzer


class CodebaseIndexer:
    """Handles codebase indexing operations."""
    
    def __init__(self, context_config: ContextConfig, vector_store: VectorStoreManager,
                 code_analyzer: CodeAnalyzer, working_directory: Path):
        """Initialize the codebase indexer."""
        self.context_config = context_config
        self.vector_store = vector_store
        self.code_analyzer = code_analyzer
        self.working_directory = working_directory
        self.logger = logging.getLogger(__name__)
        
        # Metadata storage ID
        self._metadata_id = "indexing_metadata"
        
        # Indexing state
        self._indexing_in_progress = False
        self._indexing_complete = False
        self._indexing_error: Optional[str] = None
        self._indexing_thread: Optional[threading.Thread] = None
        
        # Load persisted metadata
        self._load_indexing_metadata()
        
        # Git repository handling
        self._git_repo: Optional[Repo] = None
        self._setup_git_repo()
    
    def _setup_git_repo(self) -> None:
        """Setup Git repository if available."""
        try:
            self._git_repo = Repo(self.working_directory)
            self.logger.debug("Git repository detected")
        except InvalidGitRepositoryError:
            self.logger.debug("No Git repository found")
            self._git_repo = None
    
    def _load_indexing_metadata(self) -> None:
        """Load indexing metadata from vector store."""
        try:
            # Try to get existing metadata
            results = self.vector_store.collection.get(
                ids=[self._metadata_id],
                include=['metadatas']
            )
            
            if results['metadatas'] and len(results['metadatas']) > 0:
                metadata = results['metadatas'][0]
                last_scan_str = metadata.get('last_scan_time')
                if last_scan_str:
                    self._last_scan_time = datetime.fromisoformat(last_scan_str)
                else:
                    self._last_scan_time = None
                
                # Parse indexed_files from JSON string
                indexed_files_json = metadata.get('indexed_files_json', '{}')
                try:
                    self._indexed_files = json.loads(indexed_files_json)
                except json.JSONDecodeError:
                    self._indexed_files = {}
                
                # Parse indexing_complete from string
                indexing_complete_str = metadata.get('indexing_complete', 'False')
                self._indexing_complete = indexing_complete_str.lower() in ('true', '1', 'yes')
                
                self.logger.debug(f"Loaded indexing metadata: {len(self._indexed_files)} files previously indexed")
            else:
                self._last_scan_time = None
                self._indexed_files = {}
                self._indexing_complete = False
                self.logger.debug("No previous indexing metadata found")
                
        except Exception as e:
            self.logger.warning(f"Failed to load indexing metadata: {str(e)}")
            self._last_scan_time = None
            self._indexed_files = {}
            self._indexing_complete = False
    
    def _save_indexing_metadata(self) -> None:
        """Save indexing metadata to vector store."""
        try:
            # Convert indexed_files dict to JSON string for storage
            indexed_files_json = json.dumps(self._indexed_files) if self._indexed_files else "{}"
            
            metadata = {
                'last_scan_time': self._last_scan_time.isoformat() if self._last_scan_time else None,
                'indexed_files_json': indexed_files_json,
                'indexing_complete': str(self._indexing_complete),
                'metadata_type': 'indexing_metadata'
            }
            
            # Check if metadata document already exists
            try:
                existing = self.vector_store.collection.get(
                    ids=[self._metadata_id],
                    include=['metadatas']
                )
                if existing['metadatas'] and len(existing['metadatas']) > 0:
                    # Update existing metadata
                    self.vector_store.collection.update(
                        ids=[self._metadata_id],
                        metadatas=[metadata]
                    )
                else:
                    # Create new metadata document
                    self.vector_store.collection.add(
                        ids=[self._metadata_id],
                        documents=["Indexing metadata"],
                        metadatas=[metadata]
                    )
            except Exception:
                # If update fails, try to add (might be first time)
                self.vector_store.collection.add(
                    ids=[self._metadata_id],
                    documents=["Indexing metadata"],
                    metadatas=[metadata]
                )
            
            self.logger.debug("Saved indexing metadata")
            
        except Exception as e:
            self.logger.warning(f"Failed to save indexing metadata: {str(e)}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get a hash of the file content and modification time."""
        try:
            stat = file_path.stat()
            # Use both modification time and size for a quick hash
            content = f"{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return ""
    
    def _has_file_changed(self, file_path: Path) -> bool:
        """Check if a file has changed since last indexing."""
        relative_path = str(file_path.relative_to(self.working_directory))
        current_hash = self._get_file_hash(file_path)
        
        if not current_hash:
            return True  # If we can't get hash, assume changed
        
        stored_hash = self._indexed_files.get(relative_path)
        return stored_hash != current_hash
    
    def _mark_file_indexed(self, file_path: Path) -> None:
        """Mark a file as indexed with its current hash."""
        relative_path = str(file_path.relative_to(self.working_directory))
        file_hash = self._get_file_hash(file_path)
        if file_hash:
            self._indexed_files[relative_path] = file_hash
    
    def index_codebase(self, force_reindex: bool = False, async_mode: bool = True) -> None:
        """Index the entire codebase for vector search."""
        if self._indexing_in_progress:
            self.logger.info("Indexing already in progress")
            return
        
        # Check if we need to reindex
        if not force_reindex and self._indexing_complete:
            # Check if any files have changed
            files_to_check = self._collect_files_to_index()
            changed_files = [f for f in files_to_check if self._has_file_changed(f)]
            
            if not changed_files:
                self.logger.info("No files have changed since last indexing")
                return
            else:
                self.logger.info(f"Found {len(changed_files)} changed files to reindex")
        
        if async_mode:
            self._start_background_indexing(force_reindex)
        else:
            self._index_codebase_sync(force_reindex)
    
    def _start_background_indexing(self, force_reindex: bool = False) -> None:
        """Start indexing in a background thread."""
        if self._indexing_thread and self._indexing_thread.is_alive():
            return
            
        self._indexing_in_progress = True
        self._indexing_error = None
        
        def index_worker():
            try:
                self._index_codebase_sync(force_reindex)
                self._indexing_complete = True
                self._save_indexing_metadata()
                self.logger.info("Background indexing completed successfully")
            except Exception as e:
                self._indexing_error = str(e)
                self.logger.error(f"Background indexing failed: {str(e)}")
            finally:
                self._indexing_in_progress = False
        
        self._indexing_thread = threading.Thread(target=index_worker, daemon=True)
        self._indexing_thread.start()
        self.logger.info("Started background indexing thread")
    
    def _index_codebase_sync(self, force_reindex: bool = False) -> None:
        """Synchronous codebase indexing."""
        self.logger.info("Starting codebase indexing...")
        
        # Clear existing embeddings if force reindex
        if force_reindex:
            self.vector_store.clear_collection()
            self._indexed_files = {}
            self._indexing_complete = False
        
        # Collect files to index
        all_files = self._collect_files_to_index()
        
        # Filter to only changed files if not force reindex
        if not force_reindex:
            files_to_index = [f for f in all_files if self._has_file_changed(f)]
        else:
            files_to_index = all_files
        
        if not files_to_index:
            self.logger.info("No files found to index")
            return
            
        self.logger.info(f"Found {len(files_to_index)} files to index")
        
        # Log some example files being indexed
        if files_to_index:
            example_files = [str(f.relative_to(self.working_directory)) for f in files_to_index[:5]]
            self.logger.debug(f"Example files to index: {', '.join(example_files)}")
            if len(files_to_index) > 5:
                self.logger.debug(f"... and {len(files_to_index) - 5} more files")
        
        # Process files in smaller batches
        batch_size = 5
        for i in range(0, len(files_to_index), batch_size):
            batch = files_to_index[i:i + batch_size]
            try:
                self._index_file_batch(batch)
                # Mark files as indexed
                for file_path in batch:
                    self._mark_file_indexed(file_path)
                # Force garbage collection after each batch
                import gc
                gc.collect()
            except Exception as e:
                self.logger.warning(f"Failed to index batch {i//batch_size + 1}: {str(e)}")
        
        self._last_scan_time = datetime.now()
        self._indexing_complete = True
        self._save_indexing_metadata()
        self.logger.debug("Codebase indexing completed")
    
    def _collect_files_to_index(self) -> List[Path]:
        """Collect files that should be indexed."""
        files_to_index = []
        max_files = getattr(self.context_config, 'max_files_to_index', 50)
        
        # Collect all eligible files more efficiently
        all_files = []
        
        def should_skip_directory(dir_path: Path) -> bool:
            """Check if we should skip an entire directory."""
            if not dir_path.is_relative_to(self.working_directory):
                return True
            
            relative_path = dir_path.relative_to(self.working_directory)
            path_str = str(relative_path)
            
            # Check if directory matches ignore patterns
            for pattern in self.context_config.ignore_patterns:
                if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(dir_path.name, pattern):
                    return True
            
            return False
        
        # Walk through directories more efficiently
        for root, dirs, files in os.walk(self.working_directory):
            root_path = Path(root)
            
            # Skip ignored directories and remove them from dirs to prevent traversal
            dirs[:] = [d for d in dirs if not should_skip_directory(root_path / d)]
            
            # Process files in current directory
            for file_name in files:
                file_path = root_path / file_name
                
                if not file_path.is_file():
                    continue
                
                if self.code_analyzer.should_ignore_file(file_path, self.working_directory):
                    continue
                
                try:
                    file_size = file_path.stat().st_size
                    # Skip very large files
                    if file_size > 1024 * 1024:  # Skip files > 1MB
                        continue
                    all_files.append((file_path, file_size))
                    
                    # Early exit if we have enough files
                    if len(all_files) >= max_files * 2:  # Collect a bit more for sorting
                        break
                except (OSError, PermissionError):
                    continue
            
            # Early exit if we have enough files
            if len(all_files) >= max_files * 2:
                break
        
        # Sort by size and take the smallest files first
        all_files.sort(key=lambda x: x[1])
        
        for file_path, _ in all_files[:max_files]:
            files_to_index.append(file_path)
        
        return files_to_index
    
    def _check_recent_changes(self) -> bool:
        """Check if there are recent changes in the codebase."""
        if not self._git_repo:
            return True  # No git repo, assume changes
        
        try:
            # Check if there are uncommitted changes
            if self._git_repo.is_dirty():
                return True
            
            # Check recent commits
            commits = list(self._git_repo.iter_commits(max_count=5))
            if commits and self._last_scan_time:
                latest_commit_time = datetime.fromtimestamp(commits[0].committed_date)
                if latest_commit_time > self._last_scan_time:
                    return True
            
            return False
        except Exception as e:
            self.logger.warning(f"Failed to check git changes: {str(e)}")
            return True
    
    def _index_file_batch(self, files: List[Path]) -> None:
        """Index a batch of files."""
        documents = []
        metadatas = []
        ids = []
        
        for file_path in files:
            try:
                # Remove existing documents for this file first
                relative_path = file_path.relative_to(self.working_directory)
                self._remove_file_documents(relative_path)
                
                # Read file content
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Skip empty files
                if not content.strip():
                    continue
                
                # Get file metadata
                language = self.code_analyzer.get_file_language(file_path)
                
                # Chunk the content
                chunks = self.code_analyzer.chunk_content(
                    content, 
                    self.vector_store.vector_config.chunk_size,
                    self.vector_store.vector_config.chunk_overlap
                )
                
                # Limit chunks per file
                max_chunks_per_file = 3
                chunk_count = 0
                
                # Create embeddings for each chunk
                for chunk_idx, (chunk_content, start_line, end_line) in enumerate(chunks):
                    if not chunk_content.strip() or chunk_count >= max_chunks_per_file:
                        continue
                    
                    doc_id = f"{relative_path}::{chunk_idx}"
                    
                    documents.append(chunk_content)
                    metadatas.append({
                        'file_path': str(relative_path),
                        'language': language,
                        'start_line': start_line,
                        'end_line': end_line,
                        'chunk_index': chunk_idx,
                        'file_size': len(content),
                        'indexed_at': datetime.now().isoformat(),
                        'document_type': 'code_chunk'
                    })
                    ids.append(doc_id)
                    chunk_count += 1
                
                # Extract code elements
                elements = self.code_analyzer.extract_code_elements(content, language)
                for element in elements:
                    doc_id = f"{relative_path}::{element['type']}::{element['name']}"
                    
                    documents.append(f"{element['signature']}\n\n{element.get('docstring', '')}")
                    metadatas.append({
                        'file_path': str(relative_path),
                        'language': language,
                        'element_type': element['type'],
                        'element_name': element['name'],
                        'line_number': element['line'],
                        'indexed_at': datetime.now().isoformat(),
                        'document_type': 'code_element'
                    })
                    ids.append(doc_id)
                
            except Exception as e:
                self.logger.warning(f"Failed to index {file_path}: {str(e)}")
        
        # Add to vector store
        if documents:
            self.vector_store.add_documents(documents, metadatas, ids)
    
    def _remove_file_documents(self, relative_path: Path) -> None:
        """Remove existing documents for a file from the vector store."""
        try:
            # Query for documents from this file
            results = self.vector_store.collection.get(
                where={'file_path': str(relative_path)}
            )
            
            if results['ids']:
                # Delete existing documents
                self.vector_store.collection.delete(ids=results['ids'])
                self.logger.debug(f"Removed {len(results['ids'])} existing documents for {relative_path}")
        except Exception as e:
            self.logger.warning(f"Failed to remove existing documents for {relative_path}: {str(e)}")
    
    def get_indexing_status(self) -> Dict[str, Any]:
        """Get the current indexing status."""
        return {
            'indexing_in_progress': self._indexing_in_progress,
            'indexing_complete': self._indexing_complete,
            'indexing_error': self._indexing_error,
            'last_scan_time': self._last_scan_time.isoformat() if self._last_scan_time else None,
            'has_git_repo': self._git_repo is not None,
            'indexed_files_count': len(self._indexed_files) if hasattr(self, '_indexed_files') else 0
        }
    
    def stop_indexing(self) -> None:
        """Stop any ongoing indexing operations."""
        self._indexing_in_progress = False
        if self._indexing_thread and self._indexing_thread.is_alive():
            self._indexing_thread.join(timeout=5.0) 