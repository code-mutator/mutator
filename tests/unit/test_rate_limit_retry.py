"""
Test rate limit retry logic in LLM client.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from mutator.llm.client import LLMClient
from mutator.core.config import LLMConfig
from mutator.core.types import LLMResponse


class TestRateLimitRetry:
    """Test rate limit detection and retry logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = LLMConfig(
            model="gpt-3.5-turbo",
            max_retries=3,
            retry_delay=0.1,  # Short delay for testing
            debug=True
        )
        self.client = LLMClient(self.config)
    
    def test_rate_limit_error_detection(self):
        """Test that various rate limit errors are properly detected."""
        rate_limit_errors = [
            "litellm.RateLimitError: RateLimitError: OpenAIException - You exceeded your current quota, please check your plan and billing details.",
            "Rate limit reached for requests",
            "429 Too Many Requests",
            "quota exceeded",
            "throttled",
            "rate_limit_error",
            "You exceeded your current quota",
            "check your plan and billing details",
            "ratelimiterror",
            "too many requests"
        ]
        
        for error_msg in rate_limit_errors:
            assert self.client._is_rate_limit_error(error_msg, error_msg), f"Should detect rate limit in: {error_msg}"
    
    def test_non_rate_limit_error_detection(self):
        """Test that non-rate-limit errors are not detected as rate limits."""
        non_rate_limit_errors = [
            "Invalid API key",
            "Model not found",
            "Network connection error",
            "Invalid request format",
            "unauthorized",
            "authentication failed",
            "invalid model"
        ]
        
        for error_msg in non_rate_limit_errors:
            assert not self.client._is_rate_limit_error(error_msg, error_msg), f"Should NOT detect rate limit in: {error_msg}"
    
    def test_rate_limit_error_is_retryable(self):
        """Test that rate limit errors are marked as retryable."""
        rate_limit_error = "litellm.RateLimitError: You exceeded your current quota"
        mock_exception = Exception(rate_limit_error)
        
        assert self.client._is_retryable_error(mock_exception, rate_limit_error)
    
    def test_quota_exceeded_is_retryable(self):
        """Test that quota exceeded errors are now retryable (bug fix)."""
        quota_error = "quota exceeded"
        mock_exception = Exception(quota_error)
        
        # This should now be retryable (was previously non-retryable)
        assert self.client._is_retryable_error(mock_exception, quota_error)
    
    @pytest.mark.asyncio
    async def test_rate_limit_retry_with_backoff(self):
        """Test that rate limit errors trigger retry with proper backoff."""
        # Mock the acompletion function to raise rate limit error twice, then succeed
        # Create a more realistic mock that avoids JSON parsing issues
        from types import SimpleNamespace
        
        mock_response = SimpleNamespace()
        mock_choice = SimpleNamespace()
        mock_message = SimpleNamespace()
        mock_message.content = "Success"
        mock_message.tool_calls = None
        mock_message.function_call = None
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model = "gpt-3.5-turbo"
        
        call_count = 0
        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("litellm.RateLimitError: You exceeded your current quota")
            return mock_response
        
        with patch('mutator.llm.client.acompletion', side_effect=mock_acompletion):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                response = await self.client.complete_with_messages([{"role": "user", "content": "test"}])
                
                # Should have succeeded after retries
                assert response.success
                assert response.content == "Success"
                
                # Should have called sleep twice (for the two failures)
                assert mock_sleep.call_count == 2
                
                # Check that backoff times were appropriate for rate limits
                sleep_calls = mock_sleep.call_args_list
                # First retry: 60 seconds base wait
                assert sleep_calls[0][0][0] == 60.0
                # Second retry: 120 seconds (60 * 2^1)
                assert sleep_calls[1][0][0] == 120.0
    
    @pytest.mark.asyncio
    async def test_non_rate_limit_retry_with_standard_backoff(self):
        """Test that non-rate-limit errors use standard backoff."""
        # Mock the acompletion function to raise connection error twice, then succeed
        from types import SimpleNamespace
        
        mock_response = SimpleNamespace()
        mock_choice = SimpleNamespace()
        mock_message = SimpleNamespace()
        mock_message.content = "Success"
        mock_message.tool_calls = None
        mock_message.function_call = None
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model = "gpt-3.5-turbo"
        
        call_count = 0
        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Connection timeout")
            return mock_response
        
        with patch('mutator.llm.client.acompletion', side_effect=mock_acompletion):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                response = await self.client.complete_with_messages([{"role": "user", "content": "test"}])
                
                # Should have succeeded after retries
                assert response.success
                assert response.content == "Success"
                
                # Should have called sleep twice (for the two failures)
                assert mock_sleep.call_count == 2
                
                # Check that backoff times were standard (not rate limit backoff)
                sleep_calls = mock_sleep.call_args_list
                # First retry: 0.1 seconds (retry_delay * 2^0)
                assert sleep_calls[0][0][0] == 0.1
                # Second retry: 0.2 seconds (retry_delay * 2^1)
                assert sleep_calls[1][0][0] == 0.2
    
    @pytest.mark.asyncio
    async def test_exhausted_retries_return_error(self):
        """Test that exhausted retries return an error response."""
        # Mock the acompletion function to always raise rate limit error
        async def mock_acompletion(*args, **kwargs):
            raise Exception("litellm.RateLimitError: You exceeded your current quota")
        
        with patch('mutator.llm.client.acompletion', side_effect=mock_acompletion):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                response = await self.client.complete_with_messages([{"role": "user", "content": "test"}])
                
                # Should have failed after all retries
                assert not response.success
                assert "You exceeded your current quota" in response.error
    
    @pytest.mark.asyncio
    async def test_stream_completion_retry_logic(self):
        """Test that stream completion also has retry logic."""
        # Mock the acompletion function to raise rate limit error once, then succeed
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello"
        
        call_count = 0
        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("litellm.RateLimitError: You exceeded your current quota")
            
            # Return an async generator for streaming
            async def mock_stream():
                yield mock_chunk
            
            return mock_stream()
        
        with patch('mutator.llm.client.acompletion', side_effect=mock_acompletion):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                chunks = []
                async for chunk in self.client.stream_completion("test"):
                    chunks.append(chunk)
                
                # Should have succeeded after retry
                assert chunks == ["Hello"]
                
                # Should have called sleep once (for the first failure)
                assert mock_sleep.call_count == 1
                
                # Should have used rate limit backoff
                assert mock_sleep.call_args[0][0] == 60.0 