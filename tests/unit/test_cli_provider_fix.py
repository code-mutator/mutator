"""
Test CLI provider and API key resolution fix.

This test ensures that when the CLI switches providers, the API key
is properly resolved for the new provider.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from mutator.cli import _update_config_with_overrides
from mutator.core.config import AgentConfig, LLMProvider


class TestCLIProviderFix:
    """Test that CLI properly handles provider switching with API key resolution."""
    
    def test_update_config_with_overrides_no_changes(self):
        """Test that function returns config unchanged when no overrides provided."""
        config = AgentConfig()
        original_provider = config.llm_config.provider
        original_model = config.llm_config.model
        
        result = _update_config_with_overrides(config, None, None)
        
        assert result.llm_config.provider == original_provider
        assert result.llm_config.model == original_model
        assert result is config  # Should be same object
    
    def test_update_config_with_overrides_model_only(self):
        """Test that function updates only model when only model provided."""
        config = AgentConfig()
        original_provider = config.llm_config.provider
        
        result = _update_config_with_overrides(config, "gpt-4", None)
        
        assert result.llm_config.provider == original_provider
        assert result.llm_config.model == "gpt-4"
    
    def test_update_config_with_overrides_provider_only(self):
        """Test that function updates provider and resets API key when provider changed."""
        config = AgentConfig()
        original_model = config.llm_config.model
        
        # Set a dummy API key to verify it gets reset
        config.llm_config.api_key = "sk-test-key"
        
        result = _update_config_with_overrides(config, None, "anthropic")
        
        assert result.llm_config.provider == LLMProvider.ANTHROPIC
        assert result.llm_config.model == original_model
        # API key should be reset and re-resolved
        assert result.llm_config.api_key != "sk-test-key"
    
    def test_update_config_with_overrides_both_model_and_provider(self):
        """Test that function updates both model and provider."""
        config = AgentConfig()
        
        # Set a dummy API key to verify it gets reset
        config.llm_config.api_key = "sk-test-key"
        
        result = _update_config_with_overrides(config, "claude-3-sonnet-20240229", "anthropic")
        
        assert result.llm_config.provider == LLMProvider.ANTHROPIC
        assert result.llm_config.model == "claude-3-sonnet-20240229"
        # API key should be reset and re-resolved
        assert result.llm_config.api_key != "sk-test-key"
    
    def test_update_config_with_overrides_none_config(self):
        """Test that function creates new config when none provided."""
        result = _update_config_with_overrides(None, "gpt-4", "openai")
        
        assert isinstance(result, AgentConfig)
        assert result.llm_config.provider == LLMProvider.OPENAI
        assert result.llm_config.model == "gpt-4"
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'sk-openai-test-key',
        'ANTHROPIC_API_KEY': 'sk-ant-test-key'
    })
    def test_api_key_resolution_with_provider_switch(self):
        """Test that API key is properly resolved when switching providers."""
        # Start with OpenAI config
        config = AgentConfig()
        config.llm_config.provider = LLMProvider.OPENAI
        
        # This should pick up the OpenAI API key
        original_api_key = config.llm_config.api_key
        assert original_api_key == "sk-openai-test-key"
        
        # Switch to Anthropic
        result = _update_config_with_overrides(config, None, "anthropic")
        
        # Should now have Anthropic API key
        assert result.llm_config.provider == LLMProvider.ANTHROPIC
        assert result.llm_config.api_key == "sk-ant-test-key"
        assert result.llm_config.api_key != original_api_key
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-test-key-2'
    })
    def test_api_key_resolution_anthropic_only(self):
        """Test that Anthropic API key is properly resolved when only Anthropic key is set."""
        # Start with default config (OpenAI)
        config = AgentConfig()
        
        # Switch to Anthropic
        result = _update_config_with_overrides(config, None, "anthropic")
        
        # Should have Anthropic API key
        assert result.llm_config.provider == LLMProvider.ANTHROPIC
        assert result.llm_config.api_key == "sk-ant-test-key-2"
    
    def test_provider_case_insensitive(self):
        """Test that provider names are case insensitive."""
        config = AgentConfig()
        
        # Test various cases
        for provider_name in ["ANTHROPIC", "Anthropic", "anthropic", "AnThRoPiC"]:
            result = _update_config_with_overrides(config, None, provider_name)
            assert result.llm_config.provider == LLMProvider.ANTHROPIC
    
    def test_invalid_provider_raises_error(self):
        """Test that invalid provider names raise appropriate errors."""
        config = AgentConfig()
        
        with pytest.raises(ValueError):
            _update_config_with_overrides(config, None, "invalid_provider") 