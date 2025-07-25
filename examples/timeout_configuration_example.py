#!/usr/bin/env python3
"""
Example demonstrating timeout configuration in the Mutator framework.

This example shows how to configure timeouts at different levels and how
they propagate through the system.
"""

import asyncio
import os
from mutator.core.config import AgentConfig, LLMConfig, ExecutionConfig
from mutator.llm.client import LLMClient


async def demonstrate_timeout_configuration():
    """Demonstrate different timeout configuration scenarios."""
    
    print("=== Timeout Configuration Examples ===\n")
    
    # Example 1: Default configuration
    print("1. Default configuration:")
    config1 = AgentConfig()
    print(f"   Agent timeout: {config1.timeout}s")
    print(f"   LLM timeout: {config1.llm_config.timeout}s")
    print(f"   Execution timeout: {config1.execution_config.timeout}s")
    print()
    
    # Example 2: Custom agent timeout (propagates to LLM and execution)
    print("2. Custom agent timeout (120s):")
    config2 = AgentConfig(timeout=120)
    print(f"   Agent timeout: {config2.timeout}s")
    print(f"   LLM timeout: {config2.llm_config.timeout}s (propagated)")
    print(f"   Execution timeout: {config2.execution_config.timeout}s (propagated)")
    print()
    
    # Example 3: Explicit LLM timeout overrides agent timeout
    print("3. Explicit LLM timeout overrides agent timeout:")
    llm_config = LLMConfig(timeout=45)
    config3 = AgentConfig(timeout=120, llm_config=llm_config)
    print(f"   Agent timeout: {config3.timeout}s")
    print(f"   LLM timeout: {config3.llm_config.timeout}s (explicit, not propagated)")
    print(f"   Execution timeout: {config3.execution_config.timeout}s (propagated)")
    print()
    
    # Example 4: Mixed explicit configurations
    print("4. Mixed explicit configurations:")
    llm_config = LLMConfig(timeout=30)
    execution_config = ExecutionConfig(timeout=600)
    config4 = AgentConfig(
        timeout=150,
        llm_config=llm_config,
        execution_config=execution_config
    )
    print(f"   Agent timeout: {config4.timeout}s")
    print(f"   LLM timeout: {config4.llm_config.timeout}s (explicit)")
    print(f"   Execution timeout: {config4.execution_config.timeout}s (explicit)")
    print()
    
    # Example 5: Demonstrating LLMClient uses the configured timeout
    print("5. LLMClient timeout integration:")
    config5 = AgentConfig(timeout=90)
    llm_client = LLMClient(config5.llm_config)
    print(f"   Agent timeout: {config5.timeout}s")
    print(f"   LLMClient timeout: {llm_client.config.timeout}s (same as LLM config)")
    print()
    
    print("=== Configuration Tips ===")
    print("• Set 'timeout' at the AgentConfig level for a global timeout")
    print("• The global timeout propagates to LLMConfig and ExecutionConfig if they use defaults")
    print("• Explicit timeouts in sub-configurations always take precedence")
    print("• Different components can have different timeout requirements:")
    print("  - LLM timeout: API request timeout (typically 30-120s)")
    print("  - Execution timeout: Overall task execution timeout (typically 300-1800s)")
    print()


async def demonstrate_timeout_from_file():
    """Demonstrate loading timeout configuration from a file."""
    
    print("=== Loading Timeout Configuration from File ===\n")
    
    # Create a sample configuration dictionary
    config_data = {
        "agent_name": "TimeoutDemo",
        "timeout": 180,  # 3 minutes
        "llm_config": {
            "model": "gpt-4",
            "timeout": 60  # Explicit LLM timeout
        },
        "execution_config": {
            # No timeout specified, will use agent timeout
        }
    }
    
    # Load configuration from dictionary (simulating file load)
    config = AgentConfig.from_dict(config_data)
    
    print(f"Loaded configuration:")
    print(f"   Agent timeout: {config.timeout}s")
    print(f"   LLM timeout: {config.llm_config.timeout}s")
    print(f"   Execution timeout: {config.execution_config.timeout}s")
    print()
    
    # Show JSON representation
    print("JSON representation:")
    print(config.to_json())
    print()


if __name__ == "__main__":
    # Run the demonstrations
    asyncio.run(demonstrate_timeout_configuration())
    asyncio.run(demonstrate_timeout_from_file()) 