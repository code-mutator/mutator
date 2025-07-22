#!/usr/bin/env python3
"""
Example demonstrating batch tools for bulk operations.

This example shows how to use batch tools to handle large numbers of items
that would otherwise overwhelm the LLM's context window. Perfect for tasks like:
- Migrating for loops to Java streams
- Updating coding patterns across many files
- Applying transformations to numerous matches
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

from mutator import Mutator
from mutator.core.config import AgentConfig


async def migrate_for_loops_example():
    """Example: Migrate for loops to Java streams across a large codebase."""
    print("🔄 Batch For Loop Migration Example")
    print("=" * 60)
    
    # Create agent
    agent = Mutator()
    await agent.initialize()
    
    # Define the migration prompt template
    migration_prompt = """
    You are a Java code migration expert. I need to migrate a for loop to use Java streams.
    
    Here is the for loop that needs migration:
    
    {item}
    
    Please:
    1. Analyze the for loop pattern
    2. Convert it to use Java streams API
    3. Provide the migrated code
    4. Explain any benefits or considerations
    
    Keep the functionality identical while making it more readable and efficient.
    """
    
    # Search for for loops in Java files
    print("\n🔍 Searching for for loops in Java files...")
    
    # Use process_search_files_by_content to find for loops and process each with dedicated conversations
    result = await agent.execute_tool("process_search_files_by_content", {
        "pattern": r"for\s*\(",  # Regex pattern to find for loops
        "operation_description": "Convert for loops to enhanced for loops where applicable",
        "file_pattern": "*.java",
        "max_results": 50,  # Limit initial search
        "prompt_template": migration_prompt,
        "batch_size": 3,  # Process 3 similar matches together
        "batch_max_results": 20,  # Process up to 20 matches
        "batch_strategy": "grouped"  # Group by file for context
    })
    
    if result.success:
        batch_result = result.result
        print(f"✅ Found {batch_result.get('total_items_found', 0)} for loops")
        print(f"📦 Processed {batch_result.get('total_items_processed', 0)} in {batch_result.get('batch_count', 0)} batches")
        
        # Display some results
        if batch_result.get('batch_results'):
            for i, batch in enumerate(batch_result['batch_results'][:3]):  # Show first 3 batches
                print(f"\n📋 Batch {i+1} Results:")
                if batch.get('results'):
                    for j, item_result in enumerate(batch['results'][:2]):  # Show first 2 items
                        if item_result.get('success'):
                            print(f"  ✅ Item {j+1}: {item_result['response'][:100]}...")
                        else:
                            print(f"  ❌ Item {j+1}: Error - {item_result.get('error', 'Unknown')}")
    else:
        print(f"❌ Search failed: {result.error}")


# batch_todo_analysis function has been removed


async def batch_file_refactoring_example():
    """Example: Refactor multiple files with a specific pattern."""
    print("\n🔧 Batch File Refactoring Example")
    print("=" * 60)
    
    # Create agent
    agent = Mutator()
    await agent.initialize()
    
    # Define the refactoring prompt template
    refactoring_prompt = """
    You are a code refactoring expert. I need to refactor this file to follow modern best practices.
    
    File information:
    {item}
    
    Please analyze the file and provide:
    1. Current issues or code smells
    2. Specific refactoring recommendations
    3. Priority of changes (Critical, High, Medium, Low)
    4. Estimated effort for refactoring
    5. Any potential risks or dependencies
    
    Focus on improving maintainability, readability, and performance.
    """
    
    print("\n🔍 Finding Python files for refactoring analysis...")
    
    # Use batch_glob_search to find Python files and analyze each one
    result = await agent.execute_tool("batch_glob_search", {
        "pattern": "*.py",
        "directory": ".",
        "recursive": True,
        "prompt_template": refactoring_prompt,
        "batch_size": 2,  # Process 2 files together
        "max_results": 10,  # Analyze up to 10 files
        "batch_strategy": "sequential"  # Process files sequentially
    })
    
    if result.success:
        batch_result = result.result
        print(f"✅ Found {batch_result.get('total_items_found', 0)} Python files")
        print(f"📦 Analyzed {batch_result.get('total_items_processed', 0)} in {batch_result.get('batch_count', 0)} batches")
        
        # Display refactoring recommendations
        if batch_result.get('batch_results'):
            for i, batch in enumerate(batch_result['batch_results'][:2]):  # Show first 2 batches
                print(f"\n📋 Batch {i+1} Refactoring Analysis:")
                if batch.get('results'):
                    for j, item_result in enumerate(batch['results']):
                        if item_result.get('success'):
                            print(f"  ✅ File {j+1}: {item_result['response'][:120]}...")
                        else:
                            print(f"  ❌ File {j+1}: Error - {item_result.get('error', 'Unknown')}")
    else:
        print(f"❌ Refactoring analysis failed: {result.error}")


async def batch_security_audit_example():
    """Example: Perform security audit on multiple code patterns."""
    print("\n🔒 Batch Security Audit Example")
    print("=" * 60)
    
    # Create agent
    agent = Mutator()
    await agent.initialize()
    
    # Define the security audit prompt template
    security_audit_prompt = """
    You are a cybersecurity expert conducting a code audit. 
    
    Here is a potential security-related code pattern:
    
    {item}
    
    Please analyze this code for security vulnerabilities:
    1. Identify any security issues (SQL injection, XSS, authentication bypass, etc.)
    2. Assess the risk level (Critical, High, Medium, Low)
    3. Provide specific remediation steps
    4. Suggest secure coding alternatives
    5. Any compliance considerations (OWASP, etc.)
    
    Be thorough but practical in your recommendations.
    """
    
    print("\n🔍 Searching for potential security issues...")
    
    # Search for common security-related patterns
    security_patterns = [
        r"sql.*=.*\+",  # SQL concatenation
        r"eval\s*\(",   # eval() usage
        r"exec\s*\(",   # exec() usage
        r"input\s*\(",  # raw input
        r"os\.system",  # system calls
    ]
    
    # Use process_search_files_by_content to find security patterns
    result = await agent.execute_tool("process_search_files_by_content", {
        "pattern": "|".join(security_patterns),  # Combined regex pattern
        "operation_description": "Audit and fix security vulnerabilities in Python code",
        "file_pattern": "*.py",
        "max_results": 30
    })
    
    if result.success:
        batch_result = result.result
        print(f"✅ Found {batch_result.get('total_items_found', 0)} potential security issues")
        print(f"📦 Audited {batch_result.get('total_items_processed', 0)} in {batch_result.get('batch_count', 0)} batches")
        
        # Display security audit results
        if batch_result.get('batch_results'):
            for i, batch in enumerate(batch_result['batch_results'][:2]):  # Show first 2 batches
                print(f"\n📋 Batch {i+1} Security Audit:")
                if batch.get('results'):
                    for j, item_result in enumerate(batch['results']):
                        if item_result.get('success'):
                            print(f"  ⚠️  Issue {j+1}: {item_result['response'][:100]}...")
                        else:
                            print(f"  ❌ Issue {j+1}: Error - {item_result.get('error', 'Unknown')}")
    else:
        print(f"❌ Security audit failed: {result.error}")


async def main():
    """Run all batch tool examples."""
    print("🚀 Batch Tools Examples")
    print("=" * 80)
    print("These examples demonstrate how to use batch tools for bulk operations")
    print("that would otherwise overwhelm the LLM's context window.")
    print()
    
    try:
        # Run examples
        await migrate_for_loops_example()
        # batch_todo_analysis_example() removed
        await batch_file_refactoring_example()
        await batch_security_audit_example()
        
        print("\n✅ All batch tool examples completed!")
        print("\n💡 Key Benefits of Batch Tools:")
        print("  • Handle hundreds/thousands of items without context limits")
        print("  • Each item gets dedicated AI attention")
        print("  • Intelligent grouping and batching strategies")
        print("  • Parallel processing for faster execution")
        print("  • Perfect for code migrations, refactoring, and analysis")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main()) 