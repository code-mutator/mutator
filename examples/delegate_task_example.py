#!/usr/bin/env python3
"""
Example demonstrating the delegate_task functionality.

This example shows how to use the delegate_task tool to delegate specific tasks
to sub-agents for processing.
"""

import asyncio
from pathlib import Path
from mutator.tools.categories.task_tools import delegate_task


async def main():
    """Demonstrate delegate_task usage."""
    
    print("=== Delegate Task Example ===\n")
    
    # Example 1: File analysis task
    print("1. File Analysis Task")
    print("-" * 50)
    
    task_description = """
    Analyze the following Python code for potential improvements:
    
    def calculate_total(items):
        total = 0
        for item in items:
            if item > 0:
                total = total + item
        return total
    
    Please provide suggestions for:
    - Code optimization
    - Readability improvements
    - Best practices
    """
    
    expected_output = "Code analysis with specific suggestions for improvement"
    
    context_data = {
        "language": "python",
        "analysis_type": "code_review",
        "focus_areas": ["optimization", "readability", "best_practices"]
    }
    
    result = await delegate_task.execute(
        task_description=task_description,
        expected_output=expected_output,
        context_data=context_data
    )
    
    print(f"Task Success: {result.success}")
    print(f"Task Name: {result.tool_name}")
    if result.success:
        print(f"Result: {result.result}")
    else:
        print(f"Error: {result.error}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Documentation task
    print("2. Documentation Task")
    print("-" * 50)
    
    task_description = """
    Create API documentation for the following function:
    
    def process_user_data(user_id, data_type="profile", include_history=False):
        '''Process user data based on type and options.'''
        # Implementation details...
        pass
    
    Include:
    - Function description
    - Parameter explanations
    - Return value description
    - Usage examples
    """
    
    expected_output = "Complete API documentation in markdown format"
    
    context_data = {
        "documentation_format": "markdown",
        "function_name": "process_user_data",
        "include_examples": True
    }
    
    result = await delegate_task.execute(
        task_description=task_description,
        expected_output=expected_output,
        context_data=context_data
    )
    
    print(f"Task Success: {result.success}")
    print(f"Task Name: {result.tool_name}")
    if result.success:
        print(f"Result: {result.result}")
    else:
        print(f"Error: {result.error}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Batch processing simulation
    print("3. Batch Processing Simulation")
    print("-" * 50)
    
    # Simulate search results that would come from batch tools
    search_results = [
        {"file": "auth.py", "line": 45, "content": "def authenticate(user):"},
        {"file": "user.py", "line": 23, "content": "def get_user_profile(id):"},
        {"file": "session.py", "line": 12, "content": "def create_session(user_id):"}
    ]
    
    task_description = f"""
    Review the following functions for security best practices:
    
    Search Results:
    {search_results}
    
    For each function:
    1. Identify potential security vulnerabilities
    2. Suggest improvements
    3. Rate the security level (1-10)
    
    Focus on authentication, authorization, and data validation.
    """
    
    expected_output = "Security analysis report with ratings and recommendations for each function"
    
    context_data = {
        "analysis_type": "security_review",
        "functions_count": len(search_results),
        "focus_areas": ["authentication", "authorization", "data_validation"],
        "search_results": search_results
    }
    
    result = await delegate_task.execute(
        task_description=task_description,
        expected_output=expected_output,
        context_data=context_data
    )
    
    print(f"Task Success: {result.success}")
    print(f"Task Name: {result.tool_name}")
    if result.success:
        print(f"Result: {result.result}")
    else:
        print(f"Error: {result.error}")
    
    print("\n" + "="*60 + "\n")
    print("Note: These examples will fail without proper LLM configuration.")
    print("The delegate_task tool requires a working LLM setup to function.")
    print("In a real environment with proper configuration, the sub-agent would")
    print("perform the analysis and return detailed results.")


if __name__ == "__main__":
    asyncio.run(main()) 