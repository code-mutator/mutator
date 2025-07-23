#!/usr/bin/env python3
"""
Basic usage examples for the Coding Agent Framework - Lightweight Version

This script demonstrates basic framework functionality without heavy ML operations.
Designed to run quickly without loading large models or doing intensive indexing.

To run this script:
    cd /path/to/framework
    PYTHONPATH=. python examples/basic_usage.py
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

# Import the framework
from mutator import (
    AgentConfig, ConfirmationLevel
)
from mutator.core.config import (
    LLMConfig, ContextConfig, SafetyConfig, ExecutionConfig
)


async def example_basic_configuration():
    """Example: Basic configuration without heavy operations."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Configuration")
    print("=" * 60)
    
    try:
        # Create custom configuration
        config = AgentConfig(
            llm_config=LLMConfig(
                model="gpt-3.5-turbo",
                max_tokens=1000,
                temperature=0.2,
                # Example of using base_url for custom endpoints
                # base_url="https://api.example.com/v1"
            ),
            context_config=ContextConfig(
                project_path=".",
                max_context_files=5,
                ignore_patterns=["*.pyc", "__pycache__", ".git", "*.log"]
            ),
            safety_config=SafetyConfig(
                confirmation_level=ConfirmationLevel.NONE,
                allowed_shell_commands=["ls", "cat", "find"],
                blocked_shell_commands=["rm", "sudo", "wget", "curl"]
            ),
            execution_config=ExecutionConfig(
                default_mode="chat",
                max_iterations=5,
                retry_on_failure=False,
                continue_on_tool_failure=True
            )
        )
        
        print("✅ Configuration created successfully:")
        print(f"  LLM Model: {config.llm_config.model}")
        print(f"  Max Tokens: {config.llm_config.max_tokens}")
        print(f"  Safety Level: {config.safety_config.confirmation_level}")
        print(f"  Max Context Files: {config.context_config.max_context_files}")
        print(f"  Working Directory: {config.working_directory}")
        
    except Exception as e:
        print(f"Error in configuration: {str(e)}")


async def example_config_validation():
    """Example: Configuration validation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Configuration Validation")
    print("=" * 60)
    
    try:
        # Test valid configuration
        valid_config = AgentConfig(
            llm_config=LLMConfig(
                model="gpt-4",
                max_tokens=2000,
                temperature=0.1
            )
        )
        print("✅ Valid configuration created")
        
        # Test configuration serialization
        config_dict = valid_config.to_dict()
        print(f"✅ Configuration serialized: {len(config_dict)} keys")
        
        # Test configuration from dict
        restored_config = AgentConfig.from_dict(config_dict)
        print(f"✅ Configuration restored from dict")
        print(f"  Model: {restored_config.llm_config.model}")
        
    except Exception as e:
        print(f"Error in validation: {str(e)}")


async def example_file_operations():
    """Example: Basic file operations without heavy indexing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Basic File Operations")
    print("=" * 60)
    
    try:
        # Check if setup.py exists
        setup_file = Path("setup.py")
        if setup_file.exists():
            print(f"✅ Found setup.py ({setup_file.stat().st_size} bytes)")
            
            # Read first few lines
            with open(setup_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:5]
                print(f"✅ Read first {len(lines)} lines")
                for i, line in enumerate(lines, 1):
                    print(f"  {i}: {line.strip()}")
        else:
            print("⚠️  setup.py not found")
        
        # List Python files in current directory (without recursion)
        python_files = list(Path(".").glob("*.py"))
        print(f"\n✅ Found {len(python_files)} Python files in current directory:")
        for py_file in python_files[:5]:  # Show first 5
            print(f"  - {py_file.name}")
            
    except Exception as e:
        print(f"Error in file operations: {str(e)}")


async def example_project_structure():
    """Example: Analyze project structure without ML models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Project Structure Analysis")
    print("=" * 60)
    
    try:
        project_root = Path(".")
        
        # Count different file types
        file_counts = {}
        total_files = 0
        
        for file_path in project_root.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix:
                    file_counts[suffix] = file_counts.get(suffix, 0) + 1
                    total_files += 1
                
                # Stop if we've counted too many files (performance)
                if total_files > 100:
                    break
        
        print(f"✅ Analyzed {total_files} files")
        print("File type distribution:")
        for suffix, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {suffix}: {count} files")
            
        # Check for common project files
        project_files = [
            "README.md", "setup.py", "requirements.txt", "pyproject.toml",
            "package.json", "Cargo.toml", ".gitignore"
        ]
        
        print("\nProject files found:")
        for proj_file in project_files:
            if Path(proj_file).exists():
                print(f"  ✅ {proj_file}")
            else:
                print(f"  ❌ {proj_file}")
                
    except Exception as e:
        print(f"Error in project analysis: {str(e)}")


async def example_environment_check():
    """Example: Check environment and dependencies."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Environment Check")
    print("=" * 60)
    
    try:
        import os
        import sys
        
        print(f"✅ Python version: {sys.version.split()[0]}")
        print(f"✅ Platform: {sys.platform}")
        print(f"✅ Working directory: {os.getcwd()}")
        
        # Check for API keys
        api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        print("\nAPI Key Status:")
        for key in api_keys:
            if os.getenv(key):
                print(f"  ✅ {key}: Set")
            else:
                print(f"  ❌ {key}: Not set")
        
        # Check Python path
        print(f"\nPython path includes {len(sys.path)} directories")
        
        # Check if we can import key modules without initializing them
        try:
            from mutator.core.types import TaskType
            print("✅ Core types import successful")
        except ImportError as e:
            print(f"❌ Core types import failed: {e}")
            
        try:
            from mutator.core.config import AgentConfig
            print("✅ Config import successful")
        except ImportError as e:
            print(f"❌ Config import failed: {e}")
            
    except Exception as e:
        print(f"Error in environment check: {str(e)}")


async def main():
    """Run lightweight examples."""
    print("Coding Agent Framework - Lightweight Examples")
    print("=" * 60)
    print("Running without heavy ML models for better performance")
    print("=" * 60)
    
    # Configure minimal logging
    logging.basicConfig(level=logging.CRITICAL)  # Only critical errors
    
    # Suppress all warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore")
    
    # Run lightweight examples
    examples = [
        example_basic_configuration,
        example_config_validation,
        example_file_operations,
        example_project_structure,
        example_environment_check,
    ]
    
    for example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"Example failed: {str(e)}")
        
        # Small delay between examples
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("All lightweight examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Set environment variables to reduce resource usage
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    
    print("🚀 Starting lightweight examples (no heavy ML models)...")
    asyncio.run(main()) 