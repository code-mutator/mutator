#!/usr/bin/env python3
"""
Full usage examples for the Coding Agent Framework.

This script demonstrates all framework features including:
- Agent initialization with ML models
- Task execution with context indexing
- Chat interactions with vector search
- Project analysis with semantic search
- Tool usage with full context
- Configuration management

WARNING: This script loads heavy ML models and may be slow.
For faster testing, use basic_usage.py instead.

To run this script:
    cd /path/to/framework
    PYTHONPATH=. python examples/full_examples.py
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

# Import the framework
from mutator import (
    Mutator, create_agent, execute_task, chat,
    TaskType, ExecutionMode, AgentConfig, ConfirmationLevel
)
from mutator.core.config import (
    LLMConfig, ContextConfig, SafetyConfig, ExecutionConfig
)


async def example_basic_task_execution():
    """Example: Basic task execution with full ML models."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Task Execution")
    print("=" * 60)
    
    agent = None
    try:
        # Check if API key is available
        import os
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            print("⚠️  No API key found. This example requires an OpenAI or Anthropic API key.")
            print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
            print("Example: export OPENAI_API_KEY='your-key-here'")
            return
        
        # Create agent with default configuration
        agent = await create_agent(project_path=".")
        
        # Execute a simple task
        task = "Analyze the project structure and list all Python files"
        
        print(f"Executing task: {task}")
        print("-" * 40)
        
        # Execute task
        events = []
        async for event in agent.execute_task(
            task="Create a simple Python script that prints 'Hello, World!'",
            execution_mode=ExecutionMode.AGENT
        ):
            events.append(event)
            print(f"Event: {event.event_type}")
            if event.event_type == "llm_response":
                print(f"  Response: {event.data['content'][:100]}...")
            elif event.event_type == "tool_call_started":
                print(f"  Tool Call: {event.data['tool_name']}")
            elif event.event_type == "tool_call_completed":
                print(f"  Tool Result: {event.data['success']}")
            elif event.event_type == "task_completed":
                print(f"  Final Status: {event.data.get('status', 'completed')}")
        
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"Error in basic task execution: {str(e)}")
        if "AuthenticationError" in str(e) or "api_key" in str(e):
            print("💡 This error indicates missing API credentials.")
            print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
    finally:
        # Clean up resources
        if agent:
            await agent.cleanup()


async def example_chat_interaction():
    """Example: Chat interaction with the agent."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Chat Interaction")
    print("=" * 60)
    
    agent = None
    try:
        # Create agent
        agent = await create_agent(project_path=".")
        
        # List of questions to ask
        questions = [
            "What is this project about?",
            "What are the main Python files in this project?",
            "What dependencies does this project have?",
            "Are there any configuration files in the project?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            response = await agent.chat(question)
            print(f"A: {response.content[:200]}...")  # Truncate for brevity
        
    except Exception as e:
        print(f"Error in chat interaction: {str(e)}")
    finally:
        # Clean up resources
        if agent:
            await agent.cleanup()


async def example_project_analysis():
    """Example: Project analysis and context search."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Project Analysis")
    print("=" * 60)
    
    agent = None
    try:
        agent = await create_agent(project_path=".")
        
        # Get project context
        print("Getting project context...")
        context = await agent.get_project_context()
        
        print(f"Project: {context.get('project_name', 'Unknown')}")
        print(f"Files found: {len(context.get('files', []))}")
        print(f"Project type: {context.get('project_type', 'Unknown')}")
        
        # Search for specific content
        print("\nSearching for 'async' in project...")
        results = await agent.search_context("async", max_results=3)
        
        for i, result in enumerate(results, 1):
            # Fix: ContextItem is a Pydantic model, use attribute access
            file_path = getattr(result, 'source', 'Unknown') or result.metadata.get('file_path', 'Unknown')
            content = getattr(result, 'content', '')[:100]
            print(f"{i}. {file_path}")
            print(f"   Preview: {content}...")
        
        # Search for imports
        print("\nSearching for 'import' statements...")
        import_results = await agent.search_context("import", max_results=5)
        
        for result in import_results:
            # Fix: ContextItem is a Pydantic model, use attribute access
            file_path = getattr(result, 'source', 'Unknown') or result.metadata.get('file_path', 'Unknown')
            print(f"   - {file_path}")
        
    except Exception as e:
        print(f"Error in project analysis: {str(e)}")
    finally:
        # Clean up resources
        if agent:
            await agent.cleanup()


async def example_tool_usage():
    """Example: Direct tool usage."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Direct Tool Usage")
    print("=" * 60)
    
    agent = None
    try:
        agent = await create_agent(project_path=".")
        
        # List available tools
        print("Available tools:")
        tools = await agent.get_available_tools()
        for tool in tools:
            # Fix: Handle both function schema and direct tool format
            if isinstance(tool, dict):
                if tool.get("type") == "function":
                    func_info = tool.get("function", {})
                    name = func_info.get("name", "unknown")
                    desc = func_info.get("description", "No description")
                    print(f"  - {name}: {desc}")
                else:
                    # Handle direct tool format
                    name = tool.get("name", "unknown")
                    desc = tool.get("description", "No description")
                    print(f"  - {name}: {desc}")
            else:
                # Handle other formats
                print(f"  - {str(tool)}")
        
        # Use read_file tool directly
        print(f"\nReading setup.py file...")
        result = await agent.execute_tool(
            "read_file",
            {"file_path": "setup.py"},
            ExecutionMode.CHAT
        )
        
        if result.success:
            content = result.result.get("content", "")
            lines = len(content.split('\n'))
            size = result.result.get("file_size", 0)
            print(f"✓ File read successfully: {lines} lines, {size} bytes")
            print(f"First 200 characters: {content[:200]}...")
        else:
            print(f"✗ Failed to read file: {result.error}")
        
        # Use search_fsearch_files_by_nameiles tool
        print(f"\nSearching for Python files...")
        search_result = await agent.execute_tool(
            "search_files_by_name",
            {
                "pattern": r"\.py$",
                "directory": ".",
                "file_pattern": "*.py",
                "include_content": False
            },
            ExecutionMode.CHAT
        )
        
        if search_result.success:
            results = search_result.result.get("results", [])
            print(f"✓ Found {len(results)} Python files")
            for result in results[:5]:  # Show first 5
                print(f"  - {result.get('file_path', 'Unknown')}")
        else:
            print(f"✗ Search failed: {search_result.error}")
        
    except Exception as e:
        print(f"Error in tool usage: {str(e)}")
    finally:
        # Clean up resources
        if agent:
            await agent.cleanup()


async def example_convenience_functions():
    """Example: Using convenience functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Convenience Functions")
    print("=" * 60)
    
    try:
        # Simple task execution
        print("Using execute_task convenience function...")
        events = await execute_task(
            "List all Python files in the current directory",
            project_path=".",
            execution_mode=ExecutionMode.CHAT
        )
        
        print(f"Executed task with {len(events)} events")
        for event in events[-3:]:  # Show last 3 events
            print(f"  {event.event_type}: {event.timestamp}")
        
        # Simple chat
        print("\nUsing chat convenience function...")
        response = await chat(
            "What is the purpose of this project?",
            project_path="."
        )
        
        print(f"Chat response: {response[:150]}...")
        
    except Exception as e:
        print(f"Error in convenience functions: {str(e)}")


async def main():
    """Run all examples with full ML models."""
    print("Coding Agent Framework - Full Examples")
    print("=" * 60)
    print("⚠️  WARNING: This loads heavy ML models and may be slow!")
    print("=" * 60)
    
    # Configure logging to reduce noise
    logging.basicConfig(level=logging.WARNING)
    
    # Suppress specific noisy loggers
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*Context leak detected.*")
    
    # Run examples
    examples = [
        example_basic_task_execution,
        example_chat_interaction,
        example_project_analysis,
        example_tool_usage,
        example_convenience_functions
    ]
    
    for example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"Example failed: {str(e)}")
        
        # Delay and cleanup between examples
        await asyncio.sleep(2)
        import gc
        gc.collect()
    
    print("\n" + "=" * 60)
    print("All full examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Set environment variables to reduce resource usage
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    
    # Check for API key
    api_key_available = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    
    if not api_key_available:
        print("⚠️  Warning: No API key found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        print("\nThe examples will run but may fail at LLM calls...")
        print("Press Ctrl+C to stop, or Enter to continue")
        try:
            input()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            exit(0)
    else:
        print("✅ API key detected - examples should work properly!")
    
    print("\n🚀 Starting full examples (loading ML models, may be slow)...")
    asyncio.run(main()) 