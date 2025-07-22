#!/usr/bin/env python3
"""
Example demonstrating how to disable built-in tools in the Coding Agent Framework.

This example shows how to disable specific built-in tools to avoid conflicts
with custom implementations or external integrations.
"""

import asyncio
from mutator.core.config import AgentConfig
from mutator.agent import Mutator


async def example_disable_web_tools():
    """
    Example: Disable web tools when using custom web integrations.
    
    This is useful when you want to integrate with custom web APIs instead of
    using the built-in web tools to avoid conflicts.
    """
    print("=== Example: Disabling Web Tools ===")
    
    # Create configuration with disabled web tools
    config = AgentConfig(
        disabled_tools=[
            "web_search",
            "fetch_url"
        ]
    )
    
    # Create agent with disabled tools
    agent = Mutator(config)
    await agent.initialize()
    
    # List available tools
    available_tools = agent.tool_manager.list_tools()
    print(f"Total available tools: {len(available_tools)}")
    
    # Check that web tools are disabled
    web_tools = ["web_search", "fetch_url"]
    for tool in web_tools:
        if tool in available_tools:
            print(f"❌ {tool} is still available (should be disabled)")
        else:
            print(f"✅ {tool} is disabled")
    
    # Check that other tools are still available
    other_tools = ["read_file", "run_shell", "search_files_by_name"]
    for tool in other_tools:
        if tool in available_tools:
            print(f"✅ {tool} is available")
        else:
            print(f"❌ {tool} is not available (should be enabled)")
    
    print()


async def example_disable_shell_tools():
    """
    Example: Disable shell tools for security reasons.
    
    This is useful in environments where you want to restrict shell access
    for security purposes.
    """
    print("=== Example: Disabling Shell Tools ===")
    
    # Create configuration with disabled shell tools
    config = AgentConfig(
        disabled_tools=[
            "run_shell"
        ]
    )
    
    # Create agent with disabled tools
    agent = Mutator(config)
    await agent.initialize()
    
    # List available tools
    available_tools = agent.tool_manager.list_tools()
    print(f"Total available tools: {len(available_tools)}")
    
    # Check that shell tools are disabled
    shell_tools = ["run_shell"]
    for tool in shell_tools:
        if tool in available_tools:
            print(f"❌ {tool} is still available (should be disabled)")
        else:
            print(f"✅ {tool} is disabled")
    
    # Check that file tools are still available
    file_tools = ["read_file", "edit_file", "create_file"]
    for tool in file_tools:
        if tool in available_tools:
            print(f"✅ {tool} is available")
        else:
            print(f"❌ {tool} is not available (should be enabled)")
    
    print()


async def example_runtime_tool_management():
    """
    Example: Manage tool availability at runtime.
    
    This shows how to disable and enable tools dynamically during execution.
    """
    print("=== Example: Runtime Tool Management ===")
    
    # Create agent with default configuration
    agent = Mutator()
    await agent.initialize()
    
    # Check initial tool availability
    print("Initial state:")
    print(f"web_search available: {'web_search' in agent.tool_manager.list_tools()}")
    print(f"web_search disabled: {agent.tool_manager.is_tool_disabled('web_search')}")
    
    # Disable web_search at runtime
    agent.tool_manager.disable_tool("web_search")
    print("\nAfter disabling web_search:")
    print(f"web_search available: {'web_search' in agent.tool_manager.list_tools()}")
    print(f"web_search disabled: {agent.tool_manager.is_tool_disabled('web_search')}")
    
    # Enable web_search again
    agent.tool_manager.enable_tool("web_search")
    print("\nAfter enabling web_search:")
    print(f"web_search available: {'web_search' in agent.tool_manager.list_tools()}")
    print(f"web_search disabled: {agent.tool_manager.is_tool_disabled('web_search')}")
    
    # Show all disabled tools
    disabled_tools = agent.tool_manager.get_disabled_tools()
    print(f"\nCurrently disabled tools: {disabled_tools}")
    
    print()


async def example_configuration_file():
    """
    Example: Using configuration file to disable tools.
    
    This shows how to use a configuration file to specify disabled tools.
    """
    print("=== Example: Configuration File ===")
    
    # Create a sample configuration dictionary
    config_data = {
        "agent_name": "MyCustomAgent",
        "disabled_tools": [
            "web_search",
            "fetch_url",
            "run_shell"
        ],
        "llm_config": {
            "model": "gpt-4",
            "temperature": 0.1
        }
    }
    
    # Create configuration from dictionary
    config = AgentConfig.from_dict(config_data)
    
    # Create agent with configuration
    agent = Mutator(config)
    await agent.initialize()
    
    print(f"Agent name: {config.agent_name}")
    print(f"Disabled tools: {config.disabled_tools}")
    print(f"Total available tools: {len(agent.tool_manager.list_tools())}")
    
    # Verify disabled tools
    disabled_count = 0
    for tool in config.disabled_tools:
        if tool not in agent.tool_manager.list_tools():
            disabled_count += 1
    
    print(f"Successfully disabled {disabled_count}/{len(config.disabled_tools)} tools")
    
    print()


async def example_selective_tool_categories():
    """
    Example: Disable entire categories of tools.
    
    This shows how to disable tools by category for specific use cases.
    """
    print("=== Example: Selective Tool Categories ===")
    
    # Disable all shell tools for security
    shell_tools = ["run_shell"]
    
    # Disable web tools for offline usage
    web_tools = ["web_search", "fetch_url"]
    
    # Combine all disabled tools
    disabled_tools = shell_tools + web_tools
    
    config = AgentConfig(disabled_tools=disabled_tools)
    agent = Mutator(config)
    await agent.initialize()
    
    print(f"Disabled {len(disabled_tools)} tools across multiple categories")
    print(f"Remaining available tools: {len(agent.tool_manager.list_tools())}")
    
    # Show remaining tool categories
    remaining_tools = agent.tool_manager.list_tools()
    file_tools = [t for t in remaining_tools if t.startswith(('read_', 'write_', 'edit_', 'create_'))]
    search_tools = [t for t in remaining_tools if 'search' in t]
    dev_tools = [t for t in remaining_tools if t.startswith(('format_', 'lint_', 'analyze_', 'review_'))]
    
    print(f"File tools available: {len(file_tools)}")
    print(f"Search tools available: {len(search_tools)}")
    print(f"Development tools available: {len(dev_tools)}")
    
    print()


async def main():
    """Run all examples."""
    print("Coding Agent Framework - Disabled Tools Examples")
    print("=" * 50)
    print()
    
    await example_disable_web_tools()
    await example_disable_shell_tools()
    await example_runtime_tool_management()
    await example_configuration_file()
    await example_selective_tool_categories()
    
    print("All examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main()) 