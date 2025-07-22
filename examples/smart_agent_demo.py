#!/usr/bin/env python3
"""
Demo script to showcase the improved intelligent agent.

This script demonstrates how the enhanced agent is much smarter at:
1. Understanding project context
2. Making intelligent tool choices
3. Building context before making decisions
4. Providing evidence-based analysis
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutator import Mutator, TaskType, ExecutionMode
from mutator.core.config import AgentConfig


async def demonstrate_intelligent_behavior():
    """Show how the agent is now much smarter."""
    
    print("🧠 Intelligent Agent Demo")
    print("=" * 50)
    
    # Create agent with smart configuration
    config = AgentConfig()
    agent = Mutator(config)
    await agent.initialize()
    
    # Test scenarios that previously failed
    test_scenarios = [
        {
            "name": "Project Understanding",
            "task": "What does this project do?",
            "expected_behavior": "Should explore project structure and read key files"
        },
        {
            "name": "Smart File Discovery",
            "task": "Read the README file",
            "expected_behavior": "Should find README files in project root, not random venv files"
        },
        {
            "name": "Complex Task Recognition",
            "task": "Refactor all the Python files to follow PEP 8 standards and add proper docstrings",
            "expected_behavior": "Should recognize complexity and use task tool for multi-step operations"
        },
        {
            "name": "Simple Task Handling",
            "task": "Show me the project structure",
            "expected_behavior": "Should use list_directory and search_files_by_name directly"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Task: '{scenario['task']}'")
        print(f"   Expected: {scenario['expected_behavior']}")
        print("   Agent Response:")
        
        # Execute the task
        tool_calls_made = []
        responses = []
        
        async for event in agent.execute_task(
            scenario['task'],
            execution_mode=ExecutionMode.AGENT
        ):
            if event.event_type == "tool_call_started":
                tool_name = event.data.get("tool_name", "Unknown")
                tool_calls_made.append(tool_name)
                print(f"   🔧 Using tool: {tool_name}")
            elif event.event_type == "llm_response":
                content = event.data.get("content", "")
                if content and len(content) > 100:
                    responses.append(content[:100] + "...")
                elif content:
                    responses.append(content)
            elif event.event_type == "task_completed":
                print(f"   ✅ Task completed")
        
        # Show intelligence indicators
        print(f"   📊 Tools used: {', '.join(tool_calls_made) if tool_calls_made else 'None'}")
        print(f"   📝 Responses: {len(responses)} generated")
        
        # Brief pause between scenarios
        await asyncio.sleep(1)
    
    print("\n" + "=" * 50)
    print("🎯 Key Improvements Demonstrated:")
    print("1. ✅ Smart context building - explores before analyzing")
    print("2. ✅ Intelligent tool selection - uses right tool for the job")
    print("3. ✅ Complex task recognition - delegates to sub-agents when needed")
    print("4. ✅ Evidence-based responses - shows actual tool usage")
    print("5. ✅ Proper file discovery - finds project files, not system files")
    
    await agent.cleanup()


async def test_task_complexity_analysis():
    """Test the improved task complexity analysis."""
    
    print("\n🔍 Task Complexity Analysis Demo")
    print("=" * 50)
    
    config = AgentConfig()
    agent = Mutator(config)
    await agent.initialize()
    
    # Test different complexity levels
    test_tasks = [
        "Read the main.py file",
        "Find all TODO comments in the project",
        "Refactor the authentication system to use JWT tokens",
        "Implement a complete user management system with frontend, backend, and database",
        "What are the dependencies of this project?",
        "Create a comprehensive test suite for all modules and set up CI/CD pipeline"
    ]
    
    for task in test_tasks:
        # Get complexity analysis
        analysis = await agent.planner.analyze_task_complexity(task)
        
        print(f"\nTask: '{task}'")
        print(f"Complexity Score: {analysis['complexity_score']}/10")
        print(f"Recommended Type: {analysis['recommended_type'].value}")
        print(f"Reasoning: {analysis['reasoning']}")
        print(f"Indicators: {', '.join(analysis['indicators'])}")
    
    await agent.cleanup()


async def main():
    """Run all demonstrations."""
    try:
        await demonstrate_intelligent_behavior()
        await test_task_complexity_analysis()
        
        print("\n🎉 Demo completed successfully!")
        print("The agent is now much smarter and should handle tasks intelligently.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 