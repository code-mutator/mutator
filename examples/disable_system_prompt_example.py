#!/usr/bin/env python3
"""
Example demonstrating the disable_system_prompt parameter.

This example shows how to use the disable_system_prompt parameter
for LLM providers that don't support system prompts. When enabled,
system messages are converted to user messages.
"""

import asyncio
from mutator import AgentConfig, LLMConfig, Mutator


async def main():
    """Demonstrate disable_system_prompt usage."""
    
    # Example 1: Normal system prompt behavior (default)
    print("=== Example 1: Normal System Prompt (Enabled) ===")
    
    config_normal = AgentConfig(
        llm_config=LLMConfig(
            model="gpt-4",
            system_prompt="You are a helpful coding assistant.",
            disable_system_prompt=False  # Default behavior
        )
    )
    
    agent_normal = Mutator(config_normal)
    await agent_normal.initialize()
    
    # This will use system messages normally
    print(f"System prompt disabled: {config_normal.llm_config.disable_system_prompt}")
    print(f"System prompt: {config_normal.llm_config.system_prompt}")
    print()
    
    # Example 2: Disabled system prompt behavior
    print("=== Example 2: System Prompt Disabled ===")
    
    config_disabled = AgentConfig(
        llm_config=LLMConfig(
            model="gpt-4",
            system_prompt="You are a helpful coding assistant.",
            disable_system_prompt=True  # System prompts will be converted to user messages
        )
    )
    
    agent_disabled = Mutator(config_disabled)
    await agent_disabled.initialize()
    
    # This will convert system messages to user messages
    print(f"System prompt disabled: {config_disabled.llm_config.disable_system_prompt}")
    print(f"System prompt: {config_disabled.llm_config.system_prompt}")
    print()
    
    # Example 3: Demonstrate message building difference
    print("=== Example 3: Message Building Comparison ===")
    
    # Normal behavior
    messages_normal = agent_normal.llm_client._build_messages(
        user_message="Hello, can you help me with Python?",
        include_history=False
    )
    print("Normal behavior messages:")
    for i, msg in enumerate(messages_normal):
        print(f"  {i+1}. Role: {msg['role']}, Content: {msg['content'][:50]}...")
    print()
    
    # Disabled behavior  
    messages_disabled = agent_disabled.llm_client._build_messages(
        user_message="Hello, can you help me with Python?",
        include_history=False
    )
    print("Disabled system prompt messages:")
    for i, msg in enumerate(messages_disabled):
        print(f"  {i+1}. Role: {msg['role']}, Content: {msg['content'][:100]}...")
    print()
    
    # Example 4: Configuration from dictionary
    print("=== Example 4: Configuration from Dictionary ===")
    
    config_dict = {
        "llm_config": {
            "model": "gpt-4",
            "system_prompt": "You are a helpful coding assistant.",
            "disable_system_prompt": True,
            "temperature": 0.7
        }
    }
    
    config_from_dict = AgentConfig.from_dict(config_dict)
    print(f"Config from dict - disable_system_prompt: {config_from_dict.llm_config.disable_system_prompt}")
    print()
    
    # Example 5: Use case scenarios
    print("=== Example 5: Use Case Scenarios ===")
    print("Use disable_system_prompt=True when:")
    print("1. Your LLM provider doesn't support system messages")
    print("2. You want to include system instructions as part of the conversation")
    print("3. You need to ensure compatibility across different LLM providers")
    print("4. You want to experiment with different prompting strategies")
    print()
    
    print("Use disable_system_prompt=False (default) when:")
    print("1. Your LLM provider supports system messages")
    print("2. You want to maintain clear separation between system and user messages")
    print("3. You want to follow standard chat completion patterns")


if __name__ == "__main__":
    asyncio.run(main()) 