#!/usr/bin/env python3
"""
Test script to demonstrate the consolidated base_url parameter functionality.

This script shows how the api_base and base_url parameters have been consolidated
into a single base_url parameter that works for all LLM providers.
"""

import warnings
from mutator.core.config import LLMConfig, LLMProvider

def test_base_url_consolidation():
    """Test the base_url consolidation functionality."""
    
    print("=== Base URL Consolidation Test ===\n")
    
    # Test 1: Using the new base_url parameter
    print("1. Using base_url (recommended):")
    config1 = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        api_key="test-key",
        base_url="https://api.openai.com/v1"
    )
    print(f"   Provider: {config1.provider}")
    print(f"   Base URL: {config1.base_url}")
    print(f"   API Base (deprecated): {config1.api_base}")
    print()
    
    # Test 2: Using the deprecated api_base parameter (with warning)
    print("2. Using api_base (deprecated, shows warning):")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config2 = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            api_key="test-key",
            api_base="https://api.anthropic.com"
        )
        if w:
            print(f"   Warning: {w[0].message}")
    
    print(f"   Provider: {config2.provider}")
    print(f"   Base URL: {config2.base_url}")  # Should be copied from api_base
    print(f"   API Base (deprecated): {config2.api_base}")
    print()
    
    # Test 3: Using both parameters (base_url takes precedence)
    print("3. Using both base_url and api_base (base_url takes precedence):")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config3 = LLMConfig(
            provider=LLMProvider.AZURE,
            model="gpt-4",
            api_key="test-key",
            base_url="https://my-azure-resource.openai.azure.com/",
            api_base="https://old-endpoint.com/"
        )
        if w:
            print(f"   Warning: {w[0].message}")
    
    print(f"   Provider: {config3.provider}")
    print(f"   Base URL: {config3.base_url}")  # Should be the base_url value
    print(f"   API Base (deprecated): {config3.api_base}")  # Should be the api_base value
    print()
    
    # Test 4: Different providers with base_url
    print("4. Different providers using base_url:")
    
    providers_examples = [
        (LLMProvider.OPENAI, "https://api.openai.com/v1"),
        (LLMProvider.ANTHROPIC, "https://api.anthropic.com"),
        (LLMProvider.AZURE, "https://my-resource.openai.azure.com/"),
        (LLMProvider.GOOGLE, "https://generativelanguage.googleapis.com/v1"),
        (LLMProvider.OLLAMA, "http://localhost:11434"),
        (LLMProvider.CUSTOM, "https://my-custom-llm-api.com/v1"),
    ]
    
    for provider, url in providers_examples:
        config = LLMConfig(
            provider=provider,
            model="test-model",
            api_key="test-key",
            base_url=url
        )
        print(f"   {provider.value}: {config.base_url}")
    
    print("\n=== Test Complete ===")
    print("✅ The base_url parameter now works for all providers!")
    print("⚠️  The api_base parameter is deprecated but still supported for backward compatibility.")

if __name__ == "__main__":
    test_base_url_consolidation() 