#!/usr/bin/env python3
"""
Example usage of the short tool descriptions feature.
"""

import asyncio
import sys
from pathlib import Path

# Add the framework to the path
sys.path.insert(0, str(Path(__file__).parent))

from mutator.agent import Mutator
from mutator.core.config import AgentConfig


async def example_usage():
    """Example of using short descriptions with get_tool_help."""
    
    print("Example: Short Tool Descriptions with get_tool_help")
    print("=" * 60)
    
    # Create agent with short descriptions (default behavior)
    config = AgentConfig()
    config.llm_config.use_short_tool_descriptions = True  # This is the default
    
    agent = Mutator(config)
    await agent.initialize()
    
    print("\n1. Available tools (with short descriptions):")
    tools = await agent.get_available_tools()
    
    # Show a few examples
    for tool in tools[:5]:
        print(f"   • {tool['name']}: {tool['description']}")
    print(f"   ... and {len(tools) - 5} more tools")
    
    print(f"\n2. Total characters in all tool descriptions: {sum(len(tool['description']) for tool in tools):,}")
    
    print("\n3. When the LLM needs more details about a tool, it can use get_tool_help:")
    
    # Simulate LLM requesting help for a complex tool
    result = await agent.execute_tool("get_tool_help", {"tool_name": "mermaid"})
    
    if result.success:
        full_description = result.result["description"]
        print(f"   ✅ Full description retrieved: {len(full_description):,} characters")
        print(f"   📝 Preview: {full_description[:150]}...")
    else:
        print(f"   ❌ Error: {result.error}")
    
    print("\n4. Configuration options:")
    print("   • use_short_tool_descriptions = True  (default, saves ~96.7% tokens)")
    print("   • use_short_tool_descriptions = False (full descriptions always)")
    
    print("\n5. Benefits:")
    print("   ✅ Massive token savings (96.7% reduction)")
    print("   ✅ Faster API responses")
    print("   ✅ Lower API costs")
    print("   ✅ LLM can get full details when needed")
    print("   ✅ No loss of functionality")
    
    print("\n" + "=" * 60)
    print("The LLM will automatically use get_tool_help when it needs")
    print("more information about how to use a specific tool properly.")


if __name__ == "__main__":
    asyncio.run(example_usage()) 