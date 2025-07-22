#!/usr/bin/env python3
"""
Demonstration of the simplified planning and execution system.

This example shows how the framework now uses simplified planning where:
1. The planner creates simple single-step plans
2. The LLM decides when to use the 'task' tool for complex operations
3. The 'task' tool handles sub-task delegation automatically

The key benefit is that the LLM makes intelligent decisions about when
to break down tasks, rather than having complex Python logic try to
predict this.
"""

import asyncio
import os
from pathlib import Path

from mutator.agent import Mutator
from mutator.core.config import AgentConfig
from mutator.core.types import ExecutionMode, TaskType


async def demonstrate_simplified_planning():
    """Demonstrate the simplified planning system."""
    
    print("=== Simplified Planning and Execution Demo ===\n")
    
    # Create a basic configuration
    config = AgentConfig()
    
    # Create and initialize the agent
    agent = Mutator(config)
    await agent.initialize()
    
    print("Agent initialized with simplified planning system\n")
    
    # Example 1: Simple task (LLM will handle directly)
    print("1. Simple Task Example:")
    print("   Task: 'Read the README.md file and summarize its contents'")
    print("   Expected: LLM will use read_file tool directly\n")
    
    simple_task = "Read the README.md file and summarize its contents"
    
    print("   Executing simple task...")
    async for event in agent.execute_task(simple_task, ExecutionMode.CHAT):
        if event.event_type == "complexity_analysis":
            print(f"   Complexity Analysis: {event.data['reasoning']}")
        elif event.event_type == "plan_created":
            print(f"   Plan Created: {event.data['step_statistics']['total']} step(s)")
        elif event.event_type == "llm_response":
            print(f"   LLM Response: {event.data['content'][:100]}...")
        elif event.event_type == "task_completed":
            print(f"   Task Status: {event.data['status']}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Complex task (LLM may choose to use task tool)
    print("2. Complex Task Example:")
    print("   Task: 'Create a comprehensive test suite for all Python files in the project'")
    print("   Expected: LLM may use the 'task' tool to delegate this complex operation\n")
    
    complex_task = ("Create a comprehensive test suite for all Python files in the project. "
                   "Include unit tests, integration tests, and ensure good coverage. "
                   "Follow testing best practices and create appropriate test structure.")
    
    print("   Executing complex task...")
    async for event in agent.execute_task(complex_task, ExecutionMode.AGENT):
        if event.event_type == "complexity_analysis":
            print(f"   Complexity Analysis: {event.data['reasoning']}")
        elif event.event_type == "plan_created":
            print(f"   Plan Created: {event.data['step_statistics']['total']} step(s)")
        elif event.event_type == "tool_call_started":
            tool_name = event.data['tool_name']
            print(f"   Tool Called: {tool_name}")
            if tool_name == "task":
                print("     -> LLM decided to delegate to sub-agent!")
        elif event.event_type == "tool_call_completed":
            tool_name = event.data['tool_name']
            if tool_name == "task":
                result = event.data['result']
                print(f"     -> Sub-agent status: {result.get('status', 'unknown')}")
        elif event.event_type == "task_completed":
            print(f"   Task Status: {event.data['status']}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: List processing task
    print("3. List Processing Example:")
    print("   Task: 'Update the docstrings in all Python files to follow Google style'")
    print("   Expected: LLM may use the 'task' tool to handle multiple files efficiently\n")
    
    list_task = ("Update the docstrings in all Python files in the project to follow Google style. "
                "Ensure each function, class, and module has proper documentation with "
                "Args, Returns, and Examples sections where appropriate.")
    
    print("   Executing list processing task...")
    async for event in agent.execute_task(list_task, ExecutionMode.AGENT):
        if event.event_type == "complexity_analysis":
            print(f"   Complexity Analysis: {event.data['reasoning']}")
        elif event.event_type == "tool_call_started":
            tool_name = event.data['tool_name']
            print(f"   Tool Called: {tool_name}")
            if tool_name == "task":
                print("     -> LLM chose to use task tool for list processing!")
        elif event.event_type == "task_completed":
            print(f"   Task Status: {event.data['status']}")
    
    print("\n" + "="*60 + "\n")
    
    print("Key Benefits of Simplified Planning:")
    print("1. ✅ LLM makes intelligent decisions about task complexity")
    print("2. ✅ No complex Python logic trying to predict LLM needs")
    print("3. ✅ 'task' tool provides powerful sub-agent delegation")
    print("4. ✅ Simpler codebase that's easier to maintain")
    print("5. ✅ More flexible - LLM can adapt to unexpected scenarios")
    
    await agent.cleanup()


async def demonstrate_tool_decision_making():
    """Show how the LLM decides between direct tools and task tool."""
    
    print("\n=== Tool Decision Making Demo ===\n")
    
    config = AgentConfig()
    agent = Mutator(config)
    await agent.initialize()
    
    # Get available tools to show the LLM's options
    tools = await agent.get_available_tools()
    tool_names = [tool['name'] for tool in tools]
    
    print(f"Available tools: {', '.join(tool_names)}")
    print("\nThe LLM will choose between:")
    print("- Direct tools (read_file, search_files_by_name, etc.) for simple operations")
    print("- 'task' tool for complex multi-step operations")
    print("\nThis decision is made intelligently based on the task requirements.\n")
    
    # Examples of different scenarios
    scenarios = [
        {
            "description": "Single file operation",
            "task": "Read the setup.py file",
            "expected_tool": "read_file"
        },
        {
            "description": "Simple search operation", 
            "task": "Find all TODO comments in Python files",
            "expected_tool": "search_files_by_name or search_files_by_content"
        },
        {
            "description": "Complex multi-file operation",
            "task": "Refactor all import statements across the project to use absolute imports",
            "expected_tool": "task (sub-agent delegation)"
        },
        {
            "description": "System-wide changes",
            "task": "Set up a complete CI/CD pipeline with testing, linting, and deployment",
            "expected_tool": "task (sub-agent delegation)"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['description']}:")
        print(f"   Task: '{scenario['task']}'")
        print(f"   Expected Tool Choice: {scenario['expected_tool']}")
        print(f"   → The LLM will analyze this and choose the most appropriate approach")
        print()
    
    await agent.cleanup()


if __name__ == "__main__":
    print("Simplified Planning and Execution System Demo")
    print("=" * 50)
    
    # Run the demonstrations
    asyncio.run(demonstrate_simplified_planning())
    asyncio.run(demonstrate_tool_decision_making())
    
    print("\nDemo completed! The simplified system allows the LLM to make")
    print("intelligent decisions about when to use direct tools vs. sub-agents.") 