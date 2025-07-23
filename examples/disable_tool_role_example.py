#!/usr/bin/env python3
"""
Example demonstrating the disable_tool_role parameter.

This example shows how to use the disable_tool_role parameter
for LLM providers that don't support tool role messages. When enabled,
tool messages are converted to user messages with tool_call_id prefix.
"""

import asyncio
from mutator import AgentConfig, LLMConfig, Mutator


async def main():
    """Demonstrate disable_tool_role usage."""
    
    # Example 1: Normal tool role behavior (default)
    print("=== Example 1: Normal Tool Role (Enabled) ===")
    
    config_normal = AgentConfig(
        llm_config=LLMConfig(
            model="gpt-4",
            system_prompt="You are a helpful coding assistant.",
            disable_tool_role=False  # Default behavior
        )
    )
    
    agent_normal = Mutator(config_normal)
    await agent_normal.initialize()
    
    # This will use tool messages normally
    print(f"Tool role disabled: {config_normal.llm_config.disable_tool_role}")
    print(f"System prompt: {config_normal.llm_config.system_prompt}")
    print()
    
    # Example 2: Disabled tool role behavior
    print("=== Example 2: Tool Role Disabled ===")
    
    config_disabled = AgentConfig(
        llm_config=LLMConfig(
            model="gpt-4",
            system_prompt="You are a helpful coding assistant.",
            disable_tool_role=True  # Tool messages will be converted to user messages
        )
    )
    
    agent_disabled = Mutator(config_disabled)
    await agent_disabled.initialize()
    
    # This will convert tool messages to user messages
    print(f"Tool role disabled: {config_disabled.llm_config.disable_tool_role}")
    print(f"System prompt: {config_disabled.llm_config.system_prompt}")
    print()
    
    # Example 3: Demonstrate message building difference
    print("=== Example 3: Message Building Comparison ===")
    
    # Simulate tool messages for demonstration
    sample_tool_messages = [
        {"role": "user", "content": "Can you read the file test.py?"},
        {"role": "assistant", "content": "I'll read the file for you.", "tool_calls": [{"id": "call_123", "name": "read_file", "arguments": {"file_path": "test.py"}}]},
        {"role": "tool", "content": "def hello():\n    print('Hello, World!')", "tool_call_id": "call_123"}
    ]
    
    # Normal behavior
    normal_messages = agent_normal.llm_client._prepare_messages(sample_tool_messages)
    print("Normal behavior messages:")
    for i, msg in enumerate(normal_messages):
        print(f"  {i+1}. Role: {msg['role']}, Content: {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
        if 'tool_call_id' in msg:
            print(f"      Tool Call ID: {msg['tool_call_id']}")
    print()
    
    # Disabled behavior  
    disabled_messages = agent_disabled.llm_client._prepare_messages(sample_tool_messages)
    print("Disabled tool role messages:")
    for i, msg in enumerate(disabled_messages):
        print(f"  {i+1}. Role: {msg['role']}, Content: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    print()
    
    # Example 4: Configuration from dictionary
    print("=== Example 4: Configuration from Dictionary ===")
    
    config_dict = {
        "llm_config": {
            "model": "gpt-4",
            "system_prompt": "You are a helpful coding assistant.",
            "disable_tool_role": True,
            "temperature": 0.7
        }
    }
    
    config_from_dict = AgentConfig.from_dict(config_dict)
    print(f"Config from dict - disable_tool_role: {config_from_dict.llm_config.disable_tool_role}")
    print()
    
    # Example 5: Combined with disable_system_prompt
    print("=== Example 5: Combined with disable_system_prompt ===")
    
    config_both_disabled = AgentConfig(
        llm_config=LLMConfig(
            model="gpt-4",
            system_prompt="You are a helpful coding assistant.",
            disable_system_prompt=True,
            disable_tool_role=True
        )
    )
    
    agent_both_disabled = Mutator(config_both_disabled)
    await agent_both_disabled.initialize()
    
    print(f"System prompt disabled: {config_both_disabled.llm_config.disable_system_prompt}")
    print(f"Tool role disabled: {config_both_disabled.llm_config.disable_tool_role}")
    
    # Show how both system and tool messages are converted
    mixed_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you help me?"},
        {"role": "assistant", "content": "I'll help you.", "tool_calls": [{"id": "call_456", "name": "get_help", "arguments": {}}]},
        {"role": "tool", "content": "Here's some help content", "tool_call_id": "call_456"}
    ]
    
    both_disabled_messages = agent_both_disabled.llm_client._prepare_messages(mixed_messages)
    print("Both disabled messages:")
    for i, msg in enumerate(both_disabled_messages):
        print(f"  {i+1}. Role: {msg['role']}, Content: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
    print()
    
    # Example 6: Use case scenarios
    print("=== Example 6: Use Case Scenarios ===")
    print("Use disable_tool_role=True when:")
    print("1. Your LLM provider doesn't support tool role messages")
    print("2. You want to include tool results as part of the conversation flow")
    print("3. You need to ensure compatibility across different LLM providers")
    print("4. You want to experiment with different tool result presentation")
    print("5. Your provider treats tool messages differently than expected")
    print()
    
    print("Use disable_tool_role=False (default) when:")
    print("1. Your LLM provider supports tool role messages")
    print("2. You want to maintain clear separation between tool results and user messages")
    print("3. You want to follow standard chat completion patterns with tools")
    print("4. You need proper tool_call_id tracking for multi-turn tool conversations")


if __name__ == "__main__":
    asyncio.run(main()) 