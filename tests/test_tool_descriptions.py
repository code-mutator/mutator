#!/usr/bin/env python3
"""
Test script to verify the short tool descriptions feature works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the framework to the path
sys.path.insert(0, str(Path(__file__).parent))

from mutator.agent import Mutator
from mutator.core.config import AgentConfig


async def test_short_descriptions():
    """Test the short description functionality."""
    print("Testing short tool descriptions feature...")
    
    # Create agent with short descriptions enabled (default)
    config = AgentConfig()
    config.llm_config.use_short_tool_descriptions = True
    
    agent = Mutator(config)
    await agent.initialize()
    
    # Test the short description extraction directly
    print("\nTesting short description extraction...")
    tool_manager = agent.tool_manager
    
    # Get the full schema for mermaid
    full_schema = tool_manager.get_tool_schema_full("mermaid")
    if full_schema:
        full_desc = full_schema["function"]["description"]
        print(f"Full description length: {len(full_desc)}")
        print(f"Full description first 200 chars: {full_desc[:200]}...")
        
        # Test the extraction method
        short_desc = tool_manager._extract_short_description(full_desc)
        print(f"Short description length: {len(short_desc)}")
        print(f"Short description: {short_desc}")
    
    # Get available tools
    tools = await agent.get_available_tools()
    
    print(f"\nFound {len(tools)} tools")
    
    # Find a tool with a long description (like mermaid)
    mermaid_tool = None
    for tool in tools:
        if tool["name"] == "mermaid":
            mermaid_tool = tool
            break
    
    if mermaid_tool:
        print(f"\nMermaid tool description from get_available_tools:")
        print(f"Length: {len(mermaid_tool['description'])} characters")
        print(f"Description: {mermaid_tool['description'][:200]}...")
        
        # Test get_tool_help functionality
        print(f"\nTesting get_tool_help for mermaid...")
        try:
            result = await agent.execute_tool("get_tool_help", {"tool_name": "mermaid"})
            if result.success:
                full_desc = result.result.get("description", "")
                print(f"Full description length: {len(full_desc)} characters")
                print(f"Full description preview: {full_desc[:200]}...")
            else:
                print(f"Error getting tool help: {result.error}")
        except Exception as e:
            print(f"Exception getting tool help: {e}")
    
    # Test with full descriptions
    print(f"\n" + "="*50)
    print("Testing with full descriptions...")
    
    config.llm_config.use_short_tool_descriptions = False
    agent2 = Mutator(config)
    await agent2.initialize()
    
    tools_full = await agent2.get_available_tools()
    
    mermaid_tool_full = None
    for tool in tools_full:
        if tool["name"] == "mermaid":
            mermaid_tool_full = tool
            break
    
    if mermaid_tool_full:
        print(f"\nMermaid tool full description:")
        print(f"Length: {len(mermaid_tool_full['description'])} characters")
        print(f"Description preview: {mermaid_tool_full['description'][:200]}...")
    
    # Compare token savings
    if mermaid_tool and mermaid_tool_full:
        short_len = len(mermaid_tool['description'])
        full_len = len(mermaid_tool_full['description'])
        savings = full_len - short_len
        savings_percent = (savings / full_len) * 100
        
        print(f"\nToken savings comparison:")
        print(f"Short description: {short_len} characters")
        print(f"Full description: {full_len} characters")
        print(f"Savings: {savings} characters ({savings_percent:.1f}%)")
    
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_short_descriptions()) 