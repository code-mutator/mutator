"""
Comprehensive warning suppression and environment configuration for the Coding Agent Framework.

This module provides low-level suppression of ONNX runtime warnings that
bypass Python's normal logging and warning systems, along with environment
configuration for optimal ML library performance.
"""

import os
import sys
import ctypes
import warnings
import logging
import contextlib
from typing import Generator


def configure_onnx_environment_early():
    """Configure environment variables before any ONNX-related imports."""
    # Core ONNX runtime settings
    os.environ['ONNXRUNTIME_LOG_SEVERITY_LEVEL'] = '4'  # Only fatal errors
    os.environ['ORT_DISABLE_ALL_OPTIMIZATION'] = '1'
    os.environ['ORT_DISABLE_PROVIDERS'] = 'CoreMLExecutionProvider,TensorrtExecutionProvider,DmlExecutionProvider'
    os.environ['ORT_DISABLE_COREML'] = '1'
    os.environ['ONNXRUNTIME_PROVIDERS'] = 'CPUExecutionProvider'
    
    # Sentence transformers specific
    os.environ['SENTENCE_TRANSFORMERS_DISABLE_ONNX'] = '1'
    os.environ['SENTENCE_TRANSFORMERS_DEVICE'] = 'cpu'
    
    # Threading and parallelism
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # PyTorch settings
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_DISABLE_CUDA_MEMORY_CACHING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Transformers and HuggingFace
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['HUGGINGFACE_HUB_DISABLE_TELEMETRY'] = '1'


def configure_onnx_environment():
    """Configure environment variables to prevent ONNX runtime issues on macOS."""
    # This is an alias for the early configuration function for backward compatibility
    configure_onnx_environment_early()


def configure_embedding_environment():
    """Configure environment variables to prevent ONNX runtime issues."""
    # Disable ONNX runtime optimizations that cause issues on macOS
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Additional PyTorch configurations
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_DISABLE_CUDA_MEMORY_CACHING'] = '1'
    
    # Force CPU execution for sentence transformers
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['SENTENCE_TRANSFORMERS_DEVICE'] = 'cpu'
    
    # Suppress additional warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers.*")


def setup_logging_suppressions():
    """Setup logging suppressions for ONNX runtime and related libraries."""
    # Configure logging to suppress ONNX runtime errors
    onnx_logger = logging.getLogger('onnxruntime')
    onnx_logger.setLevel(logging.CRITICAL)
    onnx_capi_logger = logging.getLogger('onnxruntime.capi')
    onnx_capi_logger.setLevel(logging.CRITICAL)
    
    # Suppress sentence transformers logging
    st_logger = logging.getLogger('sentence_transformers')
    st_logger.setLevel(logging.WARNING)
    
    # Suppress transformers logging
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.WARNING)

    # Suppress warnings
    warnings.filterwarnings("ignore", message=".*Context leak detected.*")
    warnings.filterwarnings("ignore", message=".*msgtracer returned -1.*")
    warnings.filterwarnings("ignore", message=".*GetElementType is not implemented.*")
    warnings.filterwarnings("ignore", message=".*CoreMLExecutionProvider.*")
    warnings.filterwarnings("ignore", message=".*Non-zero status code returned.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=".*onnx.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=".*onnx.*")


class StderrSuppressor:
    """Context manager to suppress stderr output from C libraries."""
    
    def __init__(self, suppress_phrases=None):
        self.suppress_phrases = suppress_phrases or [
            'Context leak detected',
            'msgtracer returned -1',
            'GetElementType is not implemented',
            'CoreMLExecutionProvider',
            'Non-zero status code returned',
            'sequential_executor.cc',
            'ExecuteKernel',
            'onnxruntime',
            'Status Message: Exception:'
        ]
        self.original_stderr = None
        self.devnull = None
        
    def __enter__(self):
        try:
            # Try to redirect stderr to devnull at the file descriptor level
            self.original_stderr = os.dup(2)  # Save original stderr
            self.devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull, 2)  # Redirect stderr to devnull
        except (OSError, AttributeError):
            # Fallback to Python-level redirection
            self.original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if isinstance(self.original_stderr, int):
                # Restore original stderr file descriptor
                os.dup2(self.original_stderr, 2)
                os.close(self.original_stderr)
                if self.devnull is not None:
                    os.close(self.devnull)
            else:
                # Restore Python stderr
                if hasattr(sys.stderr, 'close'):
                    sys.stderr.close()
                sys.stderr = self.original_stderr
        except (OSError, AttributeError):
            pass  # Best effort cleanup


class SuppressONNXStderr:
    """Custom stderr handler to suppress ONNX runtime errors that bypass Python logging."""
    
    def __init__(self):
        self.original_stderr = sys.stderr
        
    def write(self, text):
        if self._should_suppress(text):
            return  # Suppress these specific messages
        self.original_stderr.write(text)
        
    def flush(self):
        self.original_stderr.flush()
    
    def _should_suppress(self, text: str) -> bool:
        """Check if the text should be suppressed."""
        suppress_phrases = [
            'Context leak detected',
            'msgtracer returned -1',
            'GetElementType is not implemented',
            'CoreMLExecutionProvider',
            'Non-zero status code returned while running',
            'sequential_executor.cc',
            'ExecuteKernel',
            'onnxruntime',
            'Status Message: Exception:'
        ]
        return any(phrase in text for phrase in suppress_phrases)


def apply_stderr_suppression():
    """Apply stderr suppression for ONNX runtime errors."""
    sys.stderr = SuppressONNXStderr()


@contextlib.contextmanager
def suppress_onnx_warnings() -> Generator[None, None, None]:
    """Context manager to suppress ONNX runtime warnings."""
    with StderrSuppressor():
        yield


def apply_comprehensive_suppression():
    """Apply comprehensive suppression of ONNX runtime warnings."""
    # Configure environment first
    configure_onnx_environment_early()
    
    # Python-level warning suppression
    warnings.filterwarnings("ignore", message=".*Context leak detected.*")
    warnings.filterwarnings("ignore", message=".*msgtracer returned -1.*")
    warnings.filterwarnings("ignore", message=".*GetElementType is not implemented.*")
    warnings.filterwarnings("ignore", message=".*CoreMLExecutionProvider.*")
    warnings.filterwarnings("ignore", message=".*Non-zero status code returned.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=".*onnx.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=".*onnx.*")
    
    # Logging suppression
    for logger_name in ['onnxruntime', 'onnxruntime.capi', 'sentence_transformers']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)


def initialize_environment():
    """Initialize all environment configurations for the context manager."""
    configure_onnx_environment()
    configure_embedding_environment()
    setup_logging_suppressions()
    apply_stderr_suppression()


# Apply suppression immediately when module is imported
apply_comprehensive_suppression() 