"""
Tests for the LLM client provider-specific parameter handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from mutator.llm.client import LLMClient
from mutator.core.config import LLMConfig, LLMProvider
from mutator.core.types import LLMResponse, ToolCall


class TestLLMClientProviderParams:
    """Test provider-specific parameter handling in LLM client."""
    
    @pytest.fixture
    def openai_config(self):
        """OpenAI configuration."""
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key",
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2
        )
    
    @pytest.fixture
    def anthropic_config(self):
        """Anthropic configuration."""
        return LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            api_key="test-key",
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,  # Should be ignored for Anthropic
            presence_penalty=0.2   # Should be ignored for Anthropic
        )
    
    @pytest.fixture
    def google_config(self):
        """Google configuration."""
        return LLMConfig(
            provider=LLMProvider.GOOGLE,
            model="gemini-pro",
            api_key="test-key",
            temperature=0.7,
            top_p=0.9
        )
    
    @pytest.fixture
    def function_schema(self):
        """Sample function schema."""
        return {
            "name": "test_function",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "First argument"}
                },
                "required": ["arg1"]
            }
        }
    
    def test_openai_provider_params(self, openai_config, function_schema):
        """Test that OpenAI provider includes all supported parameters."""
        client = LLMClient(openai_config)
        client.register_function("test_function", Mock(), function_schema)
        
        params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 60
        }
        
        client._add_provider_specific_params(params, {})
        
        # Check that OpenAI-specific parameters are included
        assert "tools" in params
        assert "tool_choice" in params
        assert "top_p" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params
        
        # Check tools format (OpenAI format)
        expected_tool = {
            "type": "function",
            "function": function_schema
        }
        assert params["tools"] == [expected_tool]
        assert params["tool_choice"] == "auto"
        assert params["top_p"] == 0.9
        assert params["frequency_penalty"] == 0.1
        assert params["presence_penalty"] == 0.2
    
    def test_anthropic_provider_params(self, anthropic_config, function_schema):
        """Test that Anthropic provider only includes supported parameters."""
        client = LLMClient(anthropic_config)
        client.register_function("test_function", Mock(), function_schema)
        
        params = {
            "model": "anthropic/claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 60
        }
        
        client._add_provider_specific_params(params, {})
        
        # Check that Anthropic supports tools but not functions
        assert "tools" in params
        assert "functions" not in params
        assert "tool_choice" not in params
        
        # Check that Anthropic supports top_p but not penalties
        assert "top_p" in params
        assert "frequency_penalty" not in params
        assert "presence_penalty" not in params
        
        # Check tools format (Anthropic format)
        expected_tool = {
            "name": "test_function",
            "description": "A test function",
            "input_schema": function_schema["parameters"]
        }
        assert params["tools"] == [expected_tool]
        assert params["top_p"] == 0.9
    
    def test_google_provider_params(self, google_config, function_schema):
        """Test that Google provider includes supported parameters."""
        client = LLMClient(google_config)
        client.register_function("test_function", Mock(), function_schema)
        
        params = {
            "model": "google/gemini-pro",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 60
        }
        
        client._add_provider_specific_params(params, {})
        
        # Check that Google supports tools but not functions
        assert "tools" in params
        assert "functions" not in params
        assert "tool_choice" not in params
        
        # Check that Google supports top_p but not penalties
        assert "top_p" in params
        assert "frequency_penalty" not in params
        assert "presence_penalty" not in params
        
        # Check tools format (Google format)
        expected_tool = {
            "function_declarations": [{
                "name": "test_function",
                "description": "A test function",
                "parameters": function_schema["parameters"]
            }]
        }
        assert params["tools"] == [expected_tool]
        assert params["top_p"] == 0.9
    
    def test_no_functions_registered(self, anthropic_config):
        """Test that no function/tool parameters are added when no functions are registered."""
        client = LLMClient(anthropic_config)
        
        params = {
            "model": "anthropic/claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 60
        }
        
        client._add_provider_specific_params(params, {})
        
        # Check that no function/tool parameters are added
        assert "functions" not in params
        assert "tool_choice" not in params
        assert "tools" not in params
        
        # But other supported parameters should still be there
        assert "top_p" in params
    
    def test_azure_provider_params(self, function_schema):
        """Test that Azure provider uses OpenAI-compatible parameters."""
        config = LLMConfig(
            provider=LLMProvider.AZURE,
            model="gpt-4",
            api_key="test-key",
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2
        )
        
        client = LLMClient(config)
        client.register_function("test_function", Mock(), function_schema)
        
        params = {
            "model": "azure/gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 60
        }
        
        client._add_provider_specific_params(params, {})
        
        # Azure should use OpenAI-compatible parameters (tools format)
        assert "tools" in params
        assert "tool_choice" in params
        assert "top_p" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params
    
    def test_ollama_provider_params(self, function_schema):
        """Test that Ollama provider uses OpenAI-compatible parameters."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama2",
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2
        )
        
        client = LLMClient(config)
        client.register_function("test_function", Mock(), function_schema)
        
        params = {
            "model": "ollama/llama2",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 60
        }
        
        client._add_provider_specific_params(params, {})
        
        # Ollama should use OpenAI-compatible parameters (tools format)
        assert "tools" in params
        assert "tool_choice" in params
        assert "top_p" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params
    
    def test_custom_provider_params(self, function_schema):
        """Test that custom provider only includes basic parameters."""
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="custom-model",
            api_key="test-key",
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2
        )
        
        client = LLMClient(config)
        client.register_function("test_function", Mock(), function_schema)
        
        params = {
            "model": "custom-model",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 60
        }
        
        client._add_provider_specific_params(params, {})
        
        # Custom provider should use OpenAI-compatible parameters (tools format)
        assert "tools" in params
        assert "tool_choice" in params
        assert "top_p" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params
    
    def test_none_values_not_added(self, anthropic_config):
        """Test that None values are not added to parameters."""
        config = anthropic_config
        config.top_p = None
        
        client = LLMClient(config)
        
        params = {
            "model": "anthropic/claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 60
        }
        
        client._add_provider_specific_params(params, {})
        
        # None values should not be added
        assert "top_p" not in params
        assert "frequency_penalty" not in params
        assert "presence_penalty" not in params
    
    @pytest.mark.asyncio
    async def test_anthropic_no_unsupported_params_error(self, anthropic_config):
        """Test that Anthropic calls don't include unsupported parameters."""
        client = LLMClient(anthropic_config)
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.function_call = None  # No function call
        mock_response.choices[0].message.tool_calls = None  # No tool calls
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.dict.return_value = {"prompt_tokens": 10, "completion_tokens": 20}
        mock_response.model = "claude-3-sonnet-20240229"
        
        with patch('mutator.llm.client.acompletion', new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response
            
            messages = [{"role": "user", "content": "test message"}]
            response = await client.complete_with_messages(messages)
            
            # Check that the call was made
            assert mock_completion.called
            call_args = mock_completion.call_args[1]
            
            # Verify unsupported parameters are not included
            assert "functions" not in call_args
            assert "tool_choice" not in call_args
            assert "frequency_penalty" not in call_args
            assert "presence_penalty" not in call_args
            
            # Verify supported parameters are included
            assert "model" in call_args
            assert "messages" in call_args
            assert "temperature" in call_args
            assert "max_tokens" in call_args
            assert "top_p" in call_args
            
            # Verify response is successful
            assert response.success
            assert response.content == "Test response"


class TestLLMClientToolCallExtraction:
    """Test tool call extraction for different providers."""
    
    @pytest.fixture
    def anthropic_config(self):
        """Anthropic configuration."""
        return LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            api_key="test-key"
        )
    
    @pytest.fixture
    def openai_config(self):
        """OpenAI configuration."""
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="test-key"
        )
    
    def test_extract_openai_tool_calls(self, openai_config):
        """Test extracting tool calls from OpenAI response format."""
        client = LLMClient(openai_config)
        
        # Mock OpenAI response with tool calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_message = Mock()
        mock_message.function_call = None  # No function call
        mock_message.tool_calls = [Mock()]
        mock_message.tool_calls[0].id = "call_123"
        mock_message.tool_calls[0].function = Mock()
        mock_message.tool_calls[0].function.name = "test_function"
        mock_message.tool_calls[0].function.arguments = '{"arg1": "value1"}'
        mock_response.choices[0].message = mock_message
        
        tool_calls = client._extract_tool_calls(mock_response)
        
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].name == "test_function"
        assert tool_calls[0].arguments == {"arg1": "value1"}
    
    def test_extract_anthropic_tool_calls(self, anthropic_config):
        """Test extracting tool calls from Anthropic response format."""
        client = LLMClient(anthropic_config)
        
        # Mock Anthropic response with tool calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_message = Mock()
        mock_message.function_call = None  # No function call
        mock_message.tool_calls = [Mock()]
        mock_message.tool_calls[0].id = "call_456"
        mock_message.tool_calls[0].name = "test_function"
        mock_message.tool_calls[0].input = {"arg1": "value1"}
        mock_response.choices[0].message = mock_message
        
        tool_calls = client._extract_tool_calls(mock_response)
        
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_456"
        assert tool_calls[0].name == "test_function"
        assert tool_calls[0].arguments == {"arg1": "value1"}
    
    def test_extract_no_tool_calls(self, openai_config):
        """Test extracting tool calls when there are none."""
        client = LLMClient(openai_config)
        
        # Mock response with no tool calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_message = Mock()
        mock_message.function_call = None  # No function call
        mock_message.tool_calls = None
        mock_response.choices[0].message = mock_message
        
        tool_calls = client._extract_tool_calls(mock_response)
        
        assert len(tool_calls) == 0
    
    def test_extract_malformed_tool_calls(self, openai_config):
        """Test handling of malformed tool calls."""
        client = LLMClient(openai_config)
        
        # Mock response with malformed tool calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_message = Mock()
        mock_message.function_call = None  # No function call
        mock_message.tool_calls = [Mock()]
        mock_message.tool_calls[0].id = "call_789"
        mock_message.tool_calls[0].function = Mock()
        mock_message.tool_calls[0].function.name = "test_function"
        mock_message.tool_calls[0].function.arguments = 'invalid json'
        mock_response.choices[0].message = mock_message
        
        with patch.object(client.logger, 'error') as mock_error:
            tool_calls = client._extract_tool_calls(mock_response)
            
            # Should handle the error gracefully
            assert len(tool_calls) == 0
            # Should log at least one error (could be more due to fallback handling)
            assert mock_error.call_count >= 1 