"""
Vector store management for the Coding Agent Framework.

This module handles vector storage operations using ChromaDB for semantic search
and context retrieval.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Apply comprehensive ONNX suppression before any ML library imports
from .suppress_warnings import suppress_onnx_warnings

import chromadb
with suppress_onnx_warnings():
    from sentence_transformers import SentenceTransformer

from ..core.config import VectorStoreConfig
from .suppress_warnings import configure_embedding_environment


class VectorStoreManager:
    """Manages vector storage operations using ChromaDB."""
    
    def __init__(self, vector_config: VectorStoreConfig):
        """Initialize the vector store manager."""
        self.vector_config = vector_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize vector store
        self._setup_vector_store()
        
        # Initialize embedding model
        self._setup_embedding_model()
    
    def _setup_vector_store(self) -> None:
        """Setup the vector store (ChromaDB by default)."""
        if self.vector_config.type.lower() != "chromadb":
            raise ValueError(f"Unsupported vector store type: {self.vector_config.type}")
        
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.vector_config.path
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.vector_config.collection_name,
                metadata={"description": "Codebase context embeddings"}
            )
            self.logger.debug(f"ChromaDB collection '{self.vector_config.collection_name}' ready")
            
        except Exception as e:
            self.logger.error(f"Failed to setup vector store: {str(e)}")
            raise
    
    def _setup_embedding_model(self) -> None:
        """Setup the sentence transformer model for embeddings."""
        try:
            # Configure environment to prevent ONNX runtime issues
            configure_embedding_environment()
            
            model_name = self.vector_config.embedding_model
            
            # Force CPU to avoid MPS/CoreML/ONNX issues on macOS
            device = 'cpu'
            
            with suppress_onnx_warnings():
                # Additional configuration to prevent ONNX runtime issues
                import torch
                torch.set_num_threads(1)  # Reduce thread contention
                
                # Disable ONNX providers at the PyTorch level
                import os
                os.environ['ONNXRUNTIME_PROVIDERS'] = 'CPUExecutionProvider'
                
                # Load model with specific configuration to avoid ONNX issues
                self.embedding_model = SentenceTransformer(
                    model_name, 
                    device=device,
                    trust_remote_code=False,
                    token=False
                )
                
                # Set model to evaluation mode and disable gradients
                self.embedding_model.eval()
                for param in self.embedding_model.parameters():
                    param.requires_grad = False
                
                # Disable ONNX optimization if available
                if hasattr(self.embedding_model, '_modules'):
                    for module in self.embedding_model._modules.values():
                        if hasattr(module, 'eval'):
                            module.eval()
            
            self.logger.debug(f"Embedding model '{model_name}' loaded on {device}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load embedding model: {str(e)}")
            # Fallback: set embedding model to None and use simple text matching
            self.embedding_model = None
            self.logger.debug("Using fallback text matching instead of embeddings")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], 
                     ids: List[str]) -> None:
        """Add documents to the vector store in batches."""
        if not documents:
            return
        
        try:
            batch_size = 1000  # Safe batch size for ChromaDB
            total_added = 0
            
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                batch_docs = documents[i:batch_end]
                batch_metas = metadatas[i:batch_end]
                batch_ids = ids[i:batch_end]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                total_added += len(batch_docs)
            
            self.logger.debug(f"Added {total_added} documents to vector store in batches")
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise
    
    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for similar documents in the vector store."""
        if not self.embedding_model:
            raise ValueError("Embedding model not available for search")
        
        try:
            with suppress_onnx_warnings():
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit
                )
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            collection_count = self.collection.count()
            
            # Get file counts by language (sample for stats)
            results = self.collection.query(
                query_texts=[""],
                n_results=min(collection_count, 1000)
            )
            
            language_counts = {}
            if results['metadatas']:
                for metadata in results['metadatas'][0]:
                    lang = metadata.get('language', 'unknown')
                    language_counts[lang] = language_counts.get(lang, 0) + 1
            
            return {
                'total_documents': collection_count,
                'languages': language_counts,
                'collection_name': self.vector_config.collection_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.chroma_client.delete_collection(self.vector_config.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.vector_config.collection_name,
                metadata={"description": "Codebase context embeddings"}
            )
            self.logger.debug("Vector store collection cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {str(e)}")
            raise
    
    def has_embedding_model(self) -> bool:
        """Check if embedding model is available."""
        return self.embedding_model is not None
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store."""
        return {
            "collection_initialized": self.collection is not None,
            "embedding_model_loaded": self.embedding_model is not None,
            "collection_count": self.collection.count() if self.collection else 0
        }
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID from the vector store."""
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                return {
                    'id': results['ids'][0],
                    'document': results['documents'][0] if results['documents'] else None,
                    'metadata': results['metadatas'][0] if results['metadatas'] else None
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get document by ID: {str(e)}")
            return None 