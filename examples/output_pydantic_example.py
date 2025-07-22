#!/usr/bin/env python3
"""
Example demonstrating the output_pydantic feature in mutator Coding Agent Framework.

This example shows how to use Pydantic models to get structured, validated output
from the coding agent functionality.
"""

import asyncio
import os
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from mutator.agent import Mutator
from mutator.core.config import AgentConfig
from mutator.core.types import ExecutionMode


# Example 1: Simple Blog Post Model
class BlogPost(BaseModel):
    """A simple blog post model for structured output."""
    title: str = Field(..., description="The title of the blog post")
    content: str = Field(..., description="The main content of the blog post")
    author: str = Field(..., description="The author of the blog post")
    published: bool = Field(default=False, description="Whether the blog post is published")
    tags: List[str] = Field(default_factory=list, description="List of tags for the blog post")


# Example 2: Code Analysis Model
class CodeAnalysis(BaseModel):
    """A model for code analysis results."""
    language: str = Field(..., description="Programming language of the analyzed code")
    complexity: int = Field(..., ge=1, le=10, description="Complexity score from 1-10")
    issues: List[str] = Field(default_factory=list, description="List of issues found in the code")
    suggestions: List[str] = Field(default_factory=list, description="List of improvement suggestions")
    maintainability: str = Field(..., description="Maintainability rating (High/Medium/Low)")


# Example 3: Task Summary Model
class TaskSummary(BaseModel):
    """A model for task execution summary."""
    task_description: str = Field(..., description="Description of the completed task")
    steps_completed: List[str] = Field(default_factory=list, description="List of steps completed")
    files_modified: List[str] = Field(default_factory=list, description="List of files that were modified")
    outcome: str = Field(..., description="Overall outcome of the task")
    execution_time: Optional[float] = Field(None, description="Time taken to complete the task")


# Example 4: Project Structure Model
class FileInfo(BaseModel):
    """Information about a file in the project."""
    path: str = Field(..., description="Relative path to the file")
    type: str = Field(..., description="Type of file (e.g., python, javascript, etc.)")
    size: int = Field(..., description="Size of the file in bytes")
    description: str = Field(..., description="Brief description of the file's purpose")


class ProjectStructure(BaseModel):
    """Model representing the structure of a project."""
    project_name: str = Field(..., description="Name of the project")
    description: str = Field(..., description="Brief description of the project")
    main_language: str = Field(..., description="Primary programming language")
    files: List[FileInfo] = Field(default_factory=list, description="List of important files")
    dependencies: List[str] = Field(default_factory=list, description="List of main dependencies")


async def example_blog_post():
    """Example 1: Generate a blog post with structured output."""
    print("=" * 60)
    print("Example 1: Blog Post Generation with Pydantic Output")
    print("=" * 60)
    
    # Create agent configuration
    config = AgentConfig(
        working_directory=".",
        llm_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
    )
    
    # Create and initialize agent
    agent = Mutator(config)
    await agent.initialize()
    
    # Define the task
    task = """
    Create a blog post about the benefits of using Python for data science.
    The blog post should be engaging, informative, and suitable for beginners.
    Include relevant tags for categorization.
    """
    
    print(f"Task: {task}")
    print("\nExecuting task with BlogPost Pydantic model...")
    
    # Execute task with Pydantic output
    events = []
    async for event in agent.execute_task(
        task,
        execution_mode=ExecutionMode.AGENT,
        output_pydantic=BlogPost
    ):
        events.append(event)
        if event.event_type == "llm_response":
            print(f"LLM Response: {event.data.get('content', '')[:100]}...")
    
    # Find the completion event
    completion_events = [e for e in events if e.event_type == "task_completed"]
    if completion_events:
        result_data = completion_events[0].data["result"]
        print(f"\nTask completed successfully!")
        print(f"Output format: {result_data['output_format']}")
        
        # If we have structured output, access it
        if result_data.get("pydantic"):
            # Note: In a real scenario, you'd get the actual Pydantic object
            print(f"Structured output available: {result_data['pydantic']}")
        elif result_data.get("json_dict"):
            print(f"JSON output available: {result_data['json_dict']}")
        else:
            print(f"Raw output: {result_data['raw'][:200]}...")


async def example_code_analysis():
    """Example 2: Analyze code with structured output."""
    print("\n" + "=" * 60)
    print("Example 2: Code Analysis with Pydantic Output")
    print("=" * 60)
    
    # Create agent configuration
    config = AgentConfig(
        working_directory=".",
        llm_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.3
        }
    )
    
    # Create and initialize agent
    agent = Mutator(config)
    await agent.initialize()
    
    # Define the task
    task = """
    Analyze the following Python code and provide a detailed analysis:
    
    ```python
    def calculate_factorial(n):
        if n == 0:
            return 1
        else:
            return n * calculate_factorial(n - 1)
    
    def main():
        numbers = [5, 10, 15, 20]
        for num in numbers:
            result = calculate_factorial(num)
            print(f"Factorial of {num} is {result}")
    
    if __name__ == "__main__":
        main()
    ```
    
    Provide a complexity rating, identify any issues, and suggest improvements.
    """
    
    print(f"Task: Analyzing Python code for factorial calculation")
    print("\nExecuting task with CodeAnalysis Pydantic model...")
    
    # Execute task with Pydantic output
    events = []
    async for event in agent.execute_task(
        task,
        execution_mode=ExecutionMode.AGENT,
        output_pydantic=CodeAnalysis
    ):
        events.append(event)
        if event.event_type == "llm_response":
            print(f"Analysis in progress...")
    
    # Find the completion event
    completion_events = [e for e in events if e.event_type == "task_completed"]
    if completion_events:
        result_data = completion_events[0].data["result"]
        print(f"\nCode analysis completed!")
        print(f"Output format: {result_data['output_format']}")
        
        if result_data.get("json_dict"):
            analysis = result_data["json_dict"]
            print(f"Language: {analysis.get('language', 'Unknown')}")
            print(f"Complexity: {analysis.get('complexity', 'Unknown')}/10")
            print(f"Issues found: {len(analysis.get('issues', []))}")
            print(f"Suggestions: {len(analysis.get('suggestions', []))}")
        else:
            print(f"Raw output: {result_data['raw'][:200]}...")


async def example_project_structure():
    """Example 3: Analyze project structure with nested Pydantic models."""
    print("\n" + "=" * 60)
    print("Example 3: Project Structure Analysis with Nested Pydantic Models")
    print("=" * 60)
    
    # Create agent configuration
    config = AgentConfig(
        working_directory=".",
        llm_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.2
        }
    )
    
    # Create and initialize agent
    agent = Mutator(config)
    await agent.initialize()
    
    # Define the task
    task = """
    Analyze the current project structure and provide a detailed overview.
    Look at the main files, identify the programming language, and list key dependencies.
                Focus on the mutator_framework directory structure.
    """
    
    print(f"Task: Analyzing project structure")
    print("\nExecuting task with ProjectStructure Pydantic model...")
    
    # Execute task with Pydantic output
    events = []
    async for event in agent.execute_task(
        task,
        execution_mode=ExecutionMode.AGENT,
        output_pydantic=ProjectStructure
    ):
        events.append(event)
        if event.event_type == "llm_response":
            print(f"Structure analysis in progress...")
    
    # Find the completion event
    completion_events = [e for e in events if e.event_type == "task_completed"]
    if completion_events:
        result_data = completion_events[0].data["result"]
        print(f"\nProject structure analysis completed!")
        print(f"Output format: {result_data['output_format']}")
        
        if result_data.get("json_dict"):
            structure = result_data["json_dict"]
            print(f"Project Name: {structure.get('project_name', 'Unknown')}")
            print(f"Main Language: {structure.get('main_language', 'Unknown')}")
            print(f"Files analyzed: {len(structure.get('files', []))}")
            print(f"Dependencies: {len(structure.get('dependencies', []))}")
        else:
            print(f"Raw output: {result_data['raw'][:200]}...")


async def example_without_pydantic():
    """Example 4: Compare with regular output (without Pydantic)."""
    print("\n" + "=" * 60)
    print("Example 4: Regular Output (Without Pydantic)")
    print("=" * 60)
    
    # Create agent configuration
    config = AgentConfig(
        working_directory=".",
        llm_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.5
        }
    )
    
    # Create and initialize agent
    agent = Mutator(config)
    await agent.initialize()
    
    # Define the task
    task = "Write a simple hello world program in Python and explain how it works."
    
    print(f"Task: {task}")
    print("\nExecuting task WITHOUT Pydantic output...")
    
    # Execute task without Pydantic output
    events = []
    async for event in agent.execute_task(
        task,
        execution_mode=ExecutionMode.AGENT
        # No output_pydantic parameter
    ):
        events.append(event)
        if event.event_type == "llm_response":
            print(f"Response generated...")
    
    # Find the completion event
    completion_events = [e for e in events if e.event_type == "task_completed"]
    if completion_events:
        result_data = completion_events[0].data["result"]
        print(f"\nTask completed!")
        print(f"Output format: {result_data['output_format']}")
        print(f"Raw output: {result_data['raw'][:300]}...")


async def main():
    """Main function to run all examples."""
    print("mutator Coding Agent Framework - output_pydantic Examples")
    print("=" * 60)
    
    # Check if we have the required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        print("   These examples require an OpenAI API key to work.")
        print("   Set your API key with: export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        # Run examples
        await example_blog_post()
        await example_code_analysis()
        await example_project_structure()
        await example_without_pydantic()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nKey Benefits of output_pydantic:")
        print("✅ Structured, validated output")
        print("✅ Type safety and auto-completion")
        print("✅ Easy integration with other systems")
        print("✅ Consistent data format")
        print("✅ Backwards compatible with raw output")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("Make sure you have a valid OpenAI API key set.")


if __name__ == "__main__":
    asyncio.run(main()) 