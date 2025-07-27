"""
LLM client for the Coding Agent Framework.

This module provides the LLMClient class that handles communication
with various language models using litellm, including function calling,
code extraction, and conversation management.
"""

import asyncio
import json
import os
import re
import time
import logging
import sys
import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator, Callable
from datetime import datetime

import litellm
from litellm import completion, acompletion

# Suppress litellm info logs by default
logging.getLogger("litellm").setLevel(logging.WARNING)

from ..core.types import (
    LLMResponse,
    ToolCall,
    ConversationTurn,
    TaskType,
)
from ..core.config import LLMConfig


class LLMClient:
    """Client for interacting with language models through litellm."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM client with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_litellm()
        self._conversation_history: List[ConversationTurn] = []
        self._function_schemas: Dict[str, Dict[str, Any]] = {}
        
        # Set up logging
        if self.config.debug:
            self.logger.setLevel(logging.DEBUG)
        
        self.logger.debug(f"LLM client initialized with model: {self.config.model}")
    
    def _setup_litellm(self) -> None:
        """Setup litellm configuration."""
        # Set up API keys from config
        if self.config.api_key:
            if self.config.provider.value == "openai":
                os.environ["OPENAI_API_KEY"] = self.config.api_key
            elif self.config.provider.value == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = self.config.api_key
            elif self.config.provider.value == "azure":
                os.environ["AZURE_API_KEY"] = self.config.api_key
            elif self.config.provider.value == "google":
                os.environ["GOOGLE_API_KEY"] = self.config.api_key
        
        # Set up base URL configurations for all providers
        if self.config.base_url:
            provider = self.config.provider.value
            
            if provider == "azure":
                # Azure uses AZURE_API_BASE
                os.environ["AZURE_API_BASE"] = self.config.base_url
            elif provider == "openai":
                # OpenAI uses OPENAI_API_BASE
                os.environ["OPENAI_API_BASE"] = self.config.base_url
            elif provider == "anthropic":
                # Anthropic uses ANTHROPIC_API_BASE
                os.environ["ANTHROPIC_API_BASE"] = self.config.base_url
            elif provider == "google":
                # Google uses GOOGLE_API_BASE
                os.environ["GOOGLE_API_BASE"] = self.config.base_url
            elif provider == "huggingface":
                # HuggingFace uses HUGGINGFACE_API_BASE
                os.environ["HUGGINGFACE_API_BASE"] = self.config.base_url
            elif provider == "ollama":
                # Ollama uses OLLAMA_API_BASE
                os.environ["OLLAMA_API_BASE"] = self.config.base_url
            elif provider == "custom":
                # For custom providers, we'll pass it directly in the API call
                # but also set a generic API_BASE for compatibility
                os.environ["API_BASE"] = self.config.base_url
        
        # Set up Azure API version if provided
        if self.config.provider.value == "azure" and self.config.api_version:
            os.environ["AZURE_API_VERSION"] = self.config.api_version
        
        # Configure litellm settings
        litellm.set_verbose = self.config.debug
        if self.config.timeout:
            litellm.request_timeout = self.config.timeout
        
        # Set up custom headers if provided
        if self.config.custom_headers:
            litellm.headers = self.config.custom_headers
    
    def register_function(self, name: str, function: Callable[[Dict[str, Any]], Any], schema: Dict[str, Any]) -> None:
        """Register a function for function calling."""
        self._function_schemas[name] = schema
        self.logger.debug(f"Registered function: {name}")
    
    def register_functions(self, functions: Dict[str, Tuple[Callable[[Dict[str, Any]], Any], Dict[str, Any]]]) -> None:
        """Register multiple functions for function calling."""
        for name, (function, schema) in functions.items():
            self.register_function(name, function, schema)
    
    def clear_functions(self) -> None:
        """Clear all registered functions."""
        self._function_schemas.clear()
        self.logger.debug("Cleared all registered functions")
    
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get all registered function schemas."""
        return list(self._function_schemas.values())
    
    def _prepare_model_string(self) -> str:
        """Prepare the model string for litellm."""
        if self.config.provider.value == "azure":
            return f"azure/{self.config.model}"
        elif self.config.provider.value == "anthropic":
            return f"anthropic/{self.config.model}"
        elif self.config.provider.value == "google":
            return f"google/{self.config.model}"
        elif self.config.provider.value == "huggingface":
            return f"huggingface/{self.config.model}"
        elif self.config.provider.value == "ollama":
            return f"ollama/{self.config.model}"
        else:
            return self.config.model
    
    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare messages for the API call."""
        prepared_messages = []
        
        for message in messages:
            # Handle disable_system_prompt
            if message.get("role") == "system" and self.config.disable_system_prompt:
                # Convert system message to user message with prefix
                prepared_messages.append({
                    "role": "user",
                    "content": f"System instructions: {message['content']}"
                })
            # Handle disable_tool_role
            elif message.get("role") == "tool" and self.config.disable_tool_role:
                # Convert tool message to user message with tool_call_id prefix
                tool_call_id = message.get("tool_call_id", "unknown")
                content = message.get("content", "")
                prepared_messages.append({
                    "role": "user",
                    "content": f"Tool result for call_id {tool_call_id}: {content}"
                })
            else:
                prepared_messages.append(message)
        
        return prepared_messages
    
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Complete a prompt with the LLM."""
        messages = [{"role": "user", "content": prompt}]
        return await self.complete_with_messages(messages, **kwargs)
    
    async def complete_with_messages(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        """Complete with a list of messages."""
        max_retries = kwargs.get("max_retries", self.config.max_retries)
        retry_delay = kwargs.get("retry_delay", self.config.retry_delay)
        
        for attempt in range(max_retries + 1):
            try:
                # Prepare messages
                prepared_messages = self._prepare_messages(messages)
                
                # Add system prompt if provided and not disabled
                if self.config.system_prompt and not self.config.disable_system_prompt:
                    prepared_messages.insert(0, {"role": "system", "content": self.config.system_prompt})
                
                # Prepare basic parameters
                params = {
                    "model": self._prepare_model_string(),
                    "messages": prepared_messages,
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "timeout": kwargs.get("timeout", self.config.timeout),
                }
                
                # Add base_url if configured (especially important for custom providers)
                if self.config.base_url:
                    params["api_base"] = self.config.base_url
                
                # Add provider-specific parameters
                self._add_provider_specific_params(params, kwargs)
                
                # Make the API call
                start_time = time.time()
                response = await acompletion(**params)
                execution_time = time.time() - start_time
                
                # Extract response content
                content = response.choices[0].message.content or ""
                
                # Extract tool calls if present
                tool_calls = self._extract_tool_calls(response)
                
                # Create LLMResponse
                llm_response = LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=response.choices[0].finish_reason,
                    usage=response.usage.dict() if response.usage else None,
                    model=response.model,
                    success=True
                )
                
                self.logger.debug(f"LLM response completed in {execution_time:.2f}s")
                return llm_response
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if this is a timeout error and log response body
                if self._is_timeout_error(e, error_msg):
                    response_body, status_code = self._extract_timeout_response_body(e)
                    
                    # Determine timeout type for better diagnostics
                    timeout_type = self._determine_timeout_type(e, error_msg)
                    
                    self.logger.error(f"LiteLLM timeout error (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                    self.logger.error(f"Timeout type: {timeout_type}")
                    self.logger.error(f"HTTP Status Code: {status_code}")
                    self.logger.error(f"Response body: {response_body}")
                    self.logger.error(f"Configured timeout: {self.config.timeout} seconds ({self.config.timeout/60:.1f} minutes)")
                    
                    # Additional diagnostic info
                    if hasattr(e, '__class__'):
                        self.logger.error(f"Exception type: {e.__class__.__name__}")
                    if hasattr(e, 'response') and e.response and hasattr(e.response, 'headers'):
                        self.logger.debug(f"Response headers: {dict(e.response.headers)}")
                    
                    # Log comprehensive exception details
                    self._log_exception_details(e)
                    
                    # Add troubleshooting hints based on timeout type
                    self._log_timeout_troubleshooting_hints(timeout_type)
                
                # Enhanced logging for rate limit debugging
                if self._is_rate_limit_error(error_msg, error_msg):
                    self.logger.warning(f"Rate limit error detected (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                    self.logger.debug(f"Full exception details: {repr(e)}")
                elif not self._is_timeout_error(e, error_msg):  # Avoid duplicate logging for timeouts
                    self.logger.error(f"LLM completion failed (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                
                # Check if this is a retryable error
                if attempt < max_retries and self._is_retryable_error(e, error_msg):
                    # Check if it's a rate limit error for special handling
                    if self._is_rate_limit_error(error_msg, error_msg):
                        # For rate limit errors, use more aggressive backoff
                        # Start with 60 seconds base wait time for rate limits
                        base_wait = 60
                        exponential_factor = 2 ** attempt
                        wait_time = base_wait * exponential_factor
                        
                        # Cap the wait time at 10 minutes to be reasonable
                        wait_time = min(wait_time, 600)
                        
                        self.logger.info(f"Rate limit detected, waiting {wait_time:.1f} seconds before retry (attempt {attempt + 1}/{max_retries + 1})...")
                        self.logger.info("This is normal behavior - the system will automatically retry after the rate limit period.")
                        await asyncio.sleep(wait_time)
                    else:
                        # For other retryable errors, use standard exponential backoff
                        wait_time = retry_delay * (2 ** attempt)
                        self.logger.info(f"Retryable error detected, waiting {wait_time:.1f} seconds before retry (attempt {attempt + 1}/{max_retries + 1})...")
                        await asyncio.sleep(wait_time)
                    continue
                
                # If we've exhausted retries or it's not retryable, return error
                self.logger.error(f"LLM completion failed after {attempt + 1} attempts: {error_msg}")
                return LLMResponse(
                    content="",
                    success=False,
                    error=error_msg
                )
    
    def _add_provider_specific_params(self, params: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        """Add provider-specific parameters to the API call."""
        provider = self.config.provider.value
        
        # Parameters supported by OpenAI and compatible providers
        if provider in ["openai", "azure", "huggingface", "ollama", "custom"]:
            # Add function schemas if available - use tools format for OpenAI (functions deprecated)
            if self._function_schemas:
                # Convert function schemas to tools format for OpenAI
                tools = []
                for schema in self._function_schemas.values():
                    tool = {
                        "type": "function",
                        "function": schema
                    }
                    tools.append(tool)
                params["tools"] = tools
                params["tool_choice"] = kwargs.get("tool_choice", "auto")
            
            # Add OpenAI-specific parameters
            if self.config.top_p is not None:
                params["top_p"] = self.config.top_p
            if self.config.frequency_penalty is not None:
                params["frequency_penalty"] = self.config.frequency_penalty
            if self.config.presence_penalty is not None:
                params["presence_penalty"] = self.config.presence_penalty
        
        # Parameters supported by Anthropic
        elif provider == "anthropic":
            # Anthropic supports tools instead of functions
            if self._function_schemas:
                # Convert function schemas to Anthropic tools format
                tools = []
                for schema in self._function_schemas.values():
                    tool = {
                        "name": schema["name"],
                        "description": schema["description"],
                        "input_schema": schema["parameters"]
                    }
                    tools.append(tool)
                params["tools"] = tools
            
            # Anthropic supports top_p but not frequency/presence penalties
            if self.config.top_p is not None:
                params["top_p"] = self.config.top_p
        
        # Parameters supported by Google
        elif provider == "google":
            # Google supports tools and top_p
            if self._function_schemas:
                # Convert function schemas to Google tools format
                tools = []
                for schema in self._function_schemas.values():
                    tool = {
                        "function_declarations": [{
                            "name": schema["name"],
                            "description": schema["description"],
                            "parameters": schema["parameters"]
                        }]
                    }
                    tools.append(tool)
                params["tools"] = tools
            
            if self.config.top_p is not None:
                params["top_p"] = self.config.top_p
        
        # For other providers, only add basic parameters
        else:
            if self.config.top_p is not None:
                params["top_p"] = self.config.top_p
    
    def _extract_tool_calls(self, response: Any) -> List[ToolCall]:
        """Extract tool calls from the LLM response."""
        tool_calls = []
        
        if not hasattr(response, 'choices') or not response.choices:
            return tool_calls
        
        message = response.choices[0].message
        provider = self.config.provider.value
        
        # Handle function calls (older format - OpenAI)
        if hasattr(message, 'function_call') and message.function_call:
            function_call = message.function_call
            try:
                arguments = json.loads(function_call.arguments) if function_call.arguments else {}
                tool_calls.append(ToolCall(
                    id=f"call_{int(time.time() * 1000)}",
                    name=function_call.name,
                    arguments=arguments
                ))
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse function call arguments: {e}")
        
        # Handle tool calls (newer format - OpenAI, Anthropic, Google)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    if provider == "anthropic":
                        # Anthropic tool call format
                        if hasattr(tool_call, 'input'):
                            arguments = tool_call.input if isinstance(tool_call.input, dict) else {}
                        else:
                            arguments = json.loads(tool_call.arguments) if hasattr(tool_call, 'arguments') and tool_call.arguments else {}
                        
                        tool_calls.append(ToolCall(
                            id=getattr(tool_call, 'id', f"call_{int(time.time() * 1000)}"),
                            name=getattr(tool_call, 'name', 'unknown_tool'),
                            arguments=arguments
                        ))
                    elif provider == "google":
                        # Google tool call format
                        if hasattr(tool_call, 'function_call'):
                            function_call = tool_call.function_call
                            arguments = function_call.args if hasattr(function_call, 'args') else {}
                            tool_calls.append(ToolCall(
                                id=getattr(tool_call, 'id', f"call_{int(time.time() * 1000)}"),
                                name=function_call.name,
                                arguments=arguments
                            ))
                    else:
                        # Standard tool call format (OpenAI, Azure, etc.)
                        if hasattr(tool_call, 'function'):
                            function = tool_call.function
                            arguments = json.loads(function.arguments) if function.arguments else {}
                            tool_calls.append(ToolCall(
                                id=tool_call.id,
                                name=function.name,
                                arguments=arguments
                            ))
                        else:
                            # Direct tool call format - handle both name and function.name
                            if hasattr(tool_call, 'name'):
                                name = tool_call.name
                            elif hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                                name = tool_call.function.name
                            else:
                                name = 'unknown_tool'
                            
                            arguments = json.loads(tool_call.arguments) if hasattr(tool_call, 'arguments') and tool_call.arguments else {}
                            tool_calls.append(ToolCall(
                                id=getattr(tool_call, 'id', f"call_{int(time.time() * 1000)}"),
                                name=name,
                                arguments=arguments
                            ))
                except (json.JSONDecodeError, AttributeError) as e:
                    self.logger.error(f"Failed to parse tool call: {e}")
                    # Log the structure of the tool_call for debugging
                    self.logger.debug(f"Tool call structure: {dir(tool_call)}")
                    self.logger.debug(f"Tool call type: {type(tool_call)}")
                    self.logger.debug(f"Tool call repr: {repr(tool_call)}")
                    if hasattr(tool_call, '__dict__'):
                        self.logger.debug(f"Tool call dict: {tool_call.__dict__}")
                    
                    # Try to extract whatever we can from the tool call
                    try:
                        # Fallback extraction
                        tool_id = getattr(tool_call, 'id', f"call_{int(time.time() * 1000)}")
                        tool_name = getattr(tool_call, 'name', None)
                        
                        if not tool_name and hasattr(tool_call, 'function'):
                            tool_name = getattr(tool_call.function, 'name', None)
                        
                        if not tool_name:
                            tool_name = 'unknown_tool'
                        
                        # Try to get arguments
                        arguments = {}
                        if hasattr(tool_call, 'input'):
                            arguments = tool_call.input if isinstance(tool_call.input, dict) else {}
                        elif hasattr(tool_call, 'arguments'):
                            try:
                                arguments = json.loads(tool_call.arguments) if tool_call.arguments else {}
                            except json.JSONDecodeError:
                                arguments = {}
                        elif hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                            try:
                                arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                            except json.JSONDecodeError:
                                arguments = {}
                        
                        tool_calls.append(ToolCall(
                            id=tool_id,
                            name=tool_name,
                            arguments=arguments
                        ))
                        self.logger.debug(f"Successfully extracted tool call: {tool_name}")
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback tool call extraction failed: {fallback_error}")
        
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Extracted {len(tool_calls)} tool calls: {[tc.name for tc in tool_calls]}")
        
        return tool_calls
    
    def _extract_code_blocks(self, content: str) -> List[Tuple[str, str]]:
        """Extract code blocks from response content."""
        # Pattern to match code blocks with optional language
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        code_blocks = []
        for language, code in matches:
            # Clean up the code
            code = code.strip()
            if code:
                code_blocks.append((language or 'text', code))
        
        return code_blocks
    
    def _detect_list_processing(self, content: str) -> bool:
        """Detect if the response indicates list processing is needed."""
        indicators = [
            "for each", "one by one", "process each", "iterate through",
            "for every", "loop through", "process individually",
            "each item", "per item", "item by item", "step by step"
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in indicators)
    
    def _classify_task_complexity(self, content: str) -> TaskType:
        """Classify task complexity based on content."""
        complex_indicators = [
            "plan", "multiple steps", "complex", "several", "various",
            "first", "then", "next", "finally", "step 1", "step 2",
            "architecture", "design", "refactor", "migrate", "implement"
        ]
        
        content_lower = content.lower()
        complexity_score = sum(1 for indicator in complex_indicators if indicator in content_lower)
        
        return TaskType.COMPLEX if complexity_score >= 2 else TaskType.SIMPLE
    
    def add_conversation_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn to the history."""
        self._conversation_history.append(turn)
        
        # Limit conversation history size
        max_history = 50
        if len(self._conversation_history) > max_history:
            self._conversation_history = self._conversation_history[-max_history:]
    
    def get_conversation_history(self) -> List[ConversationTurn]:
        """Get the conversation history."""
        return self._conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()
        self.logger.debug("Cleared conversation history")
    
    def get_conversation_messages(self) -> List[Dict[str, Any]]:
        """Convert conversation history to messages format."""
        messages = []
        
        for turn in self._conversation_history:
            if turn.role == "user":
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == "assistant":
                messages.append({"role": "assistant", "content": turn.content})
            elif turn.role == "system":
                messages.append({"role": "system", "content": turn.content})
        
        return messages
    
    async def chat_completion(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Perform a chat completion with context awareness."""
        # Build messages from conversation history
        messages = self.get_conversation_messages()
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Get response
        response = await self.complete_with_messages(messages)
        
        # Add to conversation history
        user_turn = ConversationTurn(
            id=f"user_{int(time.time() * 1000)}",
            role="user",
            content=message,
            metadata=context or {}
        )
        self.add_conversation_turn(user_turn)
        
        assistant_turn = ConversationTurn(
            id=f"assistant_{int(time.time() * 1000)}",
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
            metadata={"finish_reason": response.finish_reason}
        )
        self.add_conversation_turn(assistant_turn)
        
        return response
    
    async def analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze a task to determine its complexity and requirements."""
        analysis_prompt = f"""
        Analyze the following task and provide a structured analysis:
        
        Task: {task}
        
        Please analyze:
        1. Task complexity (simple/complex)
        2. Required tools or capabilities
        3. Estimated steps needed
        4. Any potential challenges
        5. Whether it requires list processing
        
        Provide your analysis in a clear, structured format.
        """
        
        response = await self.complete(analysis_prompt)
        
        # Extract analysis from response
        analysis = {
            "complexity": self._classify_task_complexity(response.content),
            "requires_list_processing": self._detect_list_processing(response.content),
            "code_blocks": self._extract_code_blocks(response.content),
            "full_analysis": response.content
        }
        
        return analysis
    
    async def generate_plan(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a plan for executing a task."""
        context_str = ""
        if context:
            context_str = f"\n\nContext:\n{json.dumps(context, indent=2)}"
        
        plan_prompt = f"""
        Create a detailed execution plan for the following task:
        
        Task: {task}{context_str}
        
        Please provide:
        1. A step-by-step plan
        2. Required tools for each step
        3. Dependencies between steps
        4. Estimated time for each step
        5. Potential risks or challenges
        
        Format your response as a structured plan.
        """
        
        response = await self.complete(plan_prompt)
        
        plan = {
            "task": task,
            "plan_content": response.content,
            "complexity": self._classify_task_complexity(response.content),
            "requires_list_processing": self._detect_list_processing(response.content),
            "code_blocks": self._extract_code_blocks(response.content),
            "generated_at": datetime.now().isoformat()
        }
        
        return plan
    
    async def stream_completion(self, prompt: str, 
                              progress_callback: Optional[Callable[[str, float], None]] = None,
                              **kwargs) -> AsyncGenerator[str, None]:
        """Stream completion with progress updates and retry logic."""
        max_retries = kwargs.get("max_retries", self.config.max_retries)
        retry_delay = kwargs.get("retry_delay", self.config.retry_delay)
        
        for attempt in range(max_retries + 1):
            try:
                messages = [{"role": "user", "content": prompt}]
                prepared_messages = self._prepare_messages(messages)
                
                # Add system prompt if provided and not disabled
                if self.config.system_prompt and not self.config.disable_system_prompt:
                    prepared_messages.insert(0, {"role": "system", "content": self.config.system_prompt})
                
                params = {
                    "model": self._prepare_model_string(),
                    "messages": prepared_messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": True,
                    "timeout": self.config.timeout,
                }
                
                # Add base_url if configured
                if self.config.base_url:
                    params["api_base"] = self.config.base_url
                
                # Add function schemas if available - use tools format for OpenAI (functions deprecated)
                if self._function_schemas:
                    # Convert function schemas to tools format for OpenAI
                    tools = []
                    for schema in self._function_schemas.values():
                        tool = {
                            "type": "function",
                            "function": schema
                        }
                        tools.append(tool)
                    params["tools"] = tools
                
                chunk_count = 0
                full_content = ""
                
                async for chunk in await acompletion(**params):
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            content = delta.content
                            full_content += content
                            chunk_count += 1
                            
                            # Call progress callback if provided
                            if progress_callback:
                                # Estimate progress (this is approximate)
                                estimated_progress = min(chunk_count / 100.0, 0.9)
                                progress_callback(content, estimated_progress)
                            
                            yield content
                
                # Final progress update
                if progress_callback:
                    progress_callback("", 1.0)
                
                # If we get here, the stream completed successfully
                return
                    
            except Exception as e:
                error_msg = str(e)
                
                # Check if this is a timeout error and log response body
                if self._is_timeout_error(e, error_msg):
                    response_body, status_code = self._extract_timeout_response_body(e)
                    
                    # Determine timeout type for better diagnostics
                    timeout_type = self._determine_timeout_type(e, error_msg)
                    
                    self.logger.error(f"LiteLLM timeout error in stream completion (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                    self.logger.error(f"Timeout type: {timeout_type}")
                    self.logger.error(f"HTTP Status Code: {status_code}")
                    self.logger.error(f"Response body: {response_body}")
                    self.logger.error(f"Configured timeout: {self.config.timeout} seconds ({self.config.timeout/60:.1f} minutes)")
                    
                    # Additional diagnostic info
                    if hasattr(e, '__class__'):
                        self.logger.error(f"Exception type: {e.__class__.__name__}")
                    if hasattr(e, 'response') and e.response and hasattr(e.response, 'headers'):
                        self.logger.debug(f"Response headers: {dict(e.response.headers)}")
                    
                    # Log comprehensive exception details
                    self._log_exception_details(e)
                    
                    # Add troubleshooting hints based on timeout type
                    self._log_timeout_troubleshooting_hints(timeout_type)
                
                # Enhanced logging for rate limit debugging
                if self._is_rate_limit_error(error_msg, error_msg):
                    self.logger.warning(f"Rate limit error in stream completion (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                    self.logger.debug(f"Full exception details: {repr(e)}")
                elif not self._is_timeout_error(e, error_msg):  # Avoid duplicate logging for timeouts
                    self.logger.error(f"Stream completion failed (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                
                # Check if this is a retryable error
                if attempt < max_retries and self._is_retryable_error(e, error_msg):
                    # Check if it's a rate limit error for special handling
                    if self._is_rate_limit_error(error_msg, error_msg):
                        # For rate limit errors, use more aggressive backoff
                        base_wait = 60
                        exponential_factor = 2 ** attempt
                        wait_time = base_wait * exponential_factor
                        wait_time = min(wait_time, 600)  # Cap at 10 minutes
                        
                        self.logger.info(f"Rate limit detected in stream, waiting {wait_time:.1f} seconds before retry (attempt {attempt + 1}/{max_retries + 1})...")
                        self.logger.info("This is normal behavior - the system will automatically retry after the rate limit period.")
                        await asyncio.sleep(wait_time)
                    else:
                        # For other retryable errors, use standard exponential backoff
                        wait_time = retry_delay * (2 ** attempt)
                        self.logger.info(f"Retryable error in stream, waiting {wait_time:.1f} seconds before retry (attempt {attempt + 1}/{max_retries + 1})...")
                        await asyncio.sleep(wait_time)
                    continue
                
                # If we've exhausted retries or it's not retryable, yield error
                self.logger.error(f"Stream completion failed after {attempt + 1} attempts: {error_msg}")
                yield f"Error: {error_msg}"
                return
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "has_functions": len(self._function_schemas) > 0,
            "function_count": len(self._function_schemas),
            "conversation_length": len(self._conversation_history)
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in text."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def validate_config(self) -> List[str]:
        """Validate the current configuration."""
        issues = []
        
        if not self.config.model:
            issues.append("Model not specified")
        
        if not self.config.api_key and self.config.provider.value != "ollama":
            issues.append("API key not provided")
        
        if self.config.temperature < 0 or self.config.temperature > 2:
            issues.append("Temperature should be between 0 and 2")
        
        if self.config.max_tokens and self.config.max_tokens < 1:
            issues.append("Max tokens should be positive")
        
        return issues
    
    async def health_check(self) -> bool:
        """Perform a health check on the LLM client."""
        try:
            # Validate configuration
            issues = self.validate_config()
            if issues:
                self.logger.warning(f"Configuration issues: {issues}")
                return False
            
            # Try a simple completion to test connectivity
            test_response = await self.complete("Hello", max_tokens=5)
            return test_response.success
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
    
    def _is_retryable_error(self, exception: Exception, error_msg: str) -> bool:
        """Determine if an error is retryable."""
        # Check for rate limit errors (always retryable)
        if self._is_rate_limit_error(error_msg, error_msg):
            return True
        
        # Check for temporary/transient errors
        retryable_patterns = [
            "timeout", "timed out", "connection error", "connection timeout",
            "server error", "internal server error", "service unavailable",
            "bad gateway", "gateway timeout", "network error",
            "temporary failure", "try again", "retry",
            "500", "502", "503", "504"  # HTTP status codes
        ]
        
        error_lower = error_msg.lower()
        
        # Check for retryable patterns
        for pattern in retryable_patterns:
            if pattern in error_lower:
                return True
        
        # Check for specific exception types
        if hasattr(exception, '__class__'):
            exception_name = exception.__class__.__name__.lower()
            if any(name in exception_name for name in ['timeout', 'connection', 'network', 'ratelimit']):
                return True
        
        # Non-retryable errors (excluding rate limit related ones)
        non_retryable_patterns = [
            "invalid api key", "unauthorized", "authentication",
            "invalid model", "model not found", "permission denied",
            "invalid request", "bad request", "malformed",
            "billing", "payment required", "quota exceeded",
            "current quota", "billing details", "plan and billing",
            "insufficient quota", "quota limit reached", "account quota",
            "subscription quota", "usage exceeded", "billing limit"
        ]
        
        for pattern in non_retryable_patterns:
            if pattern in error_lower:
                return False
        
        return False
    
    def _is_rate_limit_error(self, error_msg: str, detailed_error: str) -> bool:
        """Comprehensive detection of rate limiting errors from various providers.
        
        This method distinguishes between:
        1. Billing/quota issues (non-retryable) - e.g., "exceeded your current quota"
        2. Rate limiting (retryable) - e.g., "too many requests per minute"
        """
        combined_error = f"{error_msg} {detailed_error}".lower()
        
        # First check for non-retryable quota/billing issues
        # These are NOT rate limits but billing/quota problems that won't resolve with retries
        non_retryable_quota_patterns = [
            "exceeded your current quota", "check your plan and billing details",
            "billing details", "plan and billing", "current quota",
            "billing quota", "monthly quota", "insufficient quota",
            "quota limit reached", "payment required", "billing limit",
            "usage exceeded", "account quota", "subscription quota"
        ]
        
        for pattern in non_retryable_quota_patterns:
            if pattern in combined_error:
                # This is a billing/quota issue, not a rate limit - should NOT retry
                return False
        
        # HTTP status codes for rate limiting
        if "429" in combined_error:
            # But only if it's not a quota issue (double check)
            if any(pattern in combined_error for pattern in non_retryable_quota_patterns):
                return False
            return True
        
        # True rate limit patterns (temporary throttling that can be retried)
        rate_limit_patterns = [
            "rate limit", "rate_limit", "rate-limit",
            "too many requests", "too_many_requests", "too-many-requests",
            "throttled", "throttling", "throttle",
            "request limit", "request_limit", "request-limit",
            "api limit", "api_limit", "api-limit",
            "service limit", "service_limit", "service-limit",
            "concurrent requests", "concurrent_requests", "concurrent-requests",
            "requests per minute", "tokens per minute", "rpm limit", "tpm limit",
            "rate limit reached"
        ]
        
        for pattern in rate_limit_patterns:
            if pattern in combined_error:
                return True
        
        # Provider-specific rate limit patterns (excluding quota/billing issues)
        provider_patterns = {
            "openai": [
                "rate limit reached", "requests per minute", "tokens per minute",
                "rpm limit", "tpm limit"
            ],
            "anthropic": [
                "rate_limit_error", "request rate limit"
            ],
            "google": [
                "rate quota exceeded", "requests per minute exceeded",
                "resource exhausted"
            ],
            "azure": [
                "throttled", "requests per second", "tokens per second",
                "deployment quota"
            ]
        }
        
        for provider, patterns in provider_patterns.items():
            for pattern in patterns:
                if pattern in combined_error:
                    return True
        
        # Time-based rate limit patterns (but not billing limits)
        time_patterns = [
            "per second", "per minute", "per hour", "per day"
        ]
        
        for pattern in time_patterns:
            if pattern in combined_error and ("rate" in combined_error or "throttle" in combined_error):
                return True
        
        # Numeric pattern detection for rate limits (e.g., "exceeded your rate limit of 3 requests")
        import re
        numeric_rate_pattern = r"exceeded.*rate.*limit.*of\s+\d+"
        if re.search(numeric_rate_pattern, combined_error):
            return True
        
        # Check for litellm.RateLimitError specifically, but only if it's not a quota issue
        if "ratelimiterror" in combined_error or "rate_limit_error" in combined_error:
            # Double check it's not a quota issue
            if not any(pattern in combined_error for pattern in non_retryable_quota_patterns):
                return True
        
        return False
    
    def _determine_timeout_type(self, exception: Exception, error_msg: str) -> str:
        """Determine the specific type of timeout for better diagnostics."""
        error_lower = error_msg.lower()
        
        # Check for specific exception types first (more reliable than string matching)
        if hasattr(exception, '__class__'):
            exception_name = exception.__class__.__name__.lower()
            
            if isinstance(exception, asyncio.TimeoutError):
                return "AsyncIO Timeout (client-side timeout)"
            
            try:
                import requests
                if isinstance(exception, requests.exceptions.ConnectTimeout):
                    return "Requests Connect Timeout (failed to establish connection)"
                elif isinstance(exception, requests.exceptions.ReadTimeout):
                    return "Requests Read Timeout (no response received)"
                elif isinstance(exception, requests.exceptions.Timeout):
                    return "Requests Timeout (general timeout)"
            except ImportError:
                pass
                
            try:
                import httpx
                if isinstance(exception, httpx.ConnectTimeout):
                    return "HTTPX Connect Timeout (failed to establish connection)"
                elif isinstance(exception, httpx.ReadTimeout):
                    return "HTTPX Read Timeout (no response received)"
                elif isinstance(exception, httpx.TimeoutException):
                    return "HTTPX Timeout (general timeout)"
            except ImportError:
                pass
        
        # Check for specific timeout types based on error message patterns
        # Order matters - check more specific patterns first
        if "connection timeout" in error_lower or "connect timeout" in error_lower:
            return "Connection Timeout (failed to establish connection)"
        elif "read timeout" in error_lower:
            return "Read Timeout (connection established but no response received)"
        elif "gateway timeout" in error_lower or "504" in error_msg:
            return "Gateway Timeout (upstream server timeout)"
        elif "request timeout" in error_lower or "408" in error_msg:
            return "Request Timeout (server-side timeout)"
        
        # Check if it's likely a litellm timeout
        if "litellm" in error_lower:
            # More specific LiteLLM timeout detection
            if "request_timeout" in error_lower or "timeout" in error_lower:
                return "LiteLLM Request Timeout (configured timeout exceeded)"
            else:
                return "LiteLLM Timeout (internal timeout)"
        
        # Check for exception name patterns
        if hasattr(exception, '__class__'):
            exception_name = exception.__class__.__name__.lower()
            if 'timeout' in exception_name:
                return f"{exception.__class__.__name__} (exception-based timeout)"
        
        # Generic timeout patterns
        if any(pattern in error_lower for pattern in ["timeout", "timed out"]):
            return "Generic Timeout (unspecified type)"
        
        return "Unknown Timeout Type"

    def _is_timeout_error(self, exception: Exception, error_msg: str) -> bool:
        """Check if an error is specifically a timeout error."""
        timeout_patterns = [
            "timeout", "timed out", "connection timeout", "gateway timeout",
            "read timeout", "connect timeout", "request timeout"
        ]
        
        error_lower = error_msg.lower()
        
        # Check for timeout patterns
        for pattern in timeout_patterns:
            if pattern in error_lower:
                return True
        
        # Check for specific exception types
        if hasattr(exception, '__class__'):
            exception_name = exception.__class__.__name__.lower()
            if 'timeout' in exception_name:
                return True
        
        # Check for asyncio timeout errors
        if isinstance(exception, asyncio.TimeoutError):
            return True
            
        # Check for requests timeout errors
        try:
            import requests
            if isinstance(exception, requests.exceptions.Timeout):
                return True
        except ImportError:
            pass
            
        # Check for httpx timeout errors
        try:
            import httpx
            if isinstance(exception, (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout)):
                return True
        except ImportError:
            pass
        
        return False
    
    def _extract_timeout_response_body(self, exception: Exception) -> tuple[str, Optional[int]]:
        """Extract response body and HTTP status code from timeout exception if available."""
        response_body = "No response body available"
        status_code = None
        
        try:
            # Try to get response body from various exception attributes
            if hasattr(exception, 'response') and exception.response:
                response = exception.response
                
                # Extract status code if available
                if hasattr(response, 'status_code'):
                    status_code = response.status_code
                elif hasattr(response, 'status'):
                    status_code = response.status
                
                # Extract response body
                if hasattr(response, 'text'):
                    response_body = response.text
                elif hasattr(response, 'content'):
                    content = response.content
                    if isinstance(content, bytes):
                        response_body = content.decode('utf-8', errors='ignore')
                    else:
                        response_body = str(content)
                elif hasattr(response, 'json'):
                    try:
                        response_body = str(response.json())
                    except Exception:
                        response_body = str(response)
                else:
                    response_body = str(response)
            
            # Try to get response from OpenAI-style exceptions
            elif hasattr(exception, 'body') and exception.body:
                response_body = str(exception.body)
                # Try to extract status code from OpenAI-style exceptions
                if hasattr(exception, 'status_code'):
                    status_code = exception.status_code
            
            # Try to get response from requests-style exceptions
            elif hasattr(exception, 'args') and exception.args:
                for arg in exception.args:
                    if isinstance(arg, dict):
                        if 'response' in arg or 'body' in arg:
                            response_body = str(arg)
                        if 'status_code' in arg:
                            status_code = arg['status_code']
                        break
            
            # Fallback: try to extract any JSON-like content from the error message
            import re
            json_match = re.search(r'\{.*\}', str(exception))
            if json_match:
                response_body = json_match.group(0)
                
        except Exception as e:
            self.logger.debug(f"Failed to extract response body from timeout exception: {e}")
            response_body = f"Failed to extract response body: {str(e)}"
        
        # If no response body was found, provide context about why
        if response_body == "No response body available":
            timeout_type = self._determine_timeout_type(exception, str(exception))
            if "AsyncIO Timeout" in timeout_type or "client-side" in timeout_type:
                response_body = "No response body available (client-side timeout - no HTTP response received)"
            elif "Connect Timeout" in timeout_type or "connection" in timeout_type:
                response_body = "No response body available (connection timeout - failed to establish connection)"
            elif "LiteLLM" in timeout_type or "Request Timeout" in timeout_type:
                response_body = "No response body available (timeout occurred before server response)"
            else:
                response_body = "No response body available (timeout occurred before HTTP response)"
        
        return response_body, status_code

    def _log_timeout_troubleshooting_hints(self, timeout_type: str) -> None:
        """Log troubleshooting hints based on the timeout type."""
        
        if "client-side" in timeout_type or "AsyncIO" in timeout_type:
            self.logger.info("Troubleshooting hint: Client-side timeout suggests:")
            self.logger.info("  • Network connectivity issues")
            self.logger.info("  • Consider increasing the timeout value")
            self.logger.info("  • Check if the LLM provider is experiencing issues")
            
        elif "Connect Timeout" in timeout_type or "connection" in timeout_type:
            self.logger.info("Troubleshooting hint: Connection timeout suggests:")
            self.logger.info("  • Unable to establish connection to the LLM provider")
            self.logger.info("  • Check network connectivity and DNS resolution")
            self.logger.info("  • Verify the provider's base URL and endpoints")
            self.logger.info("  • Consider checking firewall or proxy settings")
            
        elif "Read Timeout" in timeout_type:
            self.logger.info("Troubleshooting hint: Read timeout suggests:")
            self.logger.info("  • Connection established but no response received")
            self.logger.info("  • The LLM provider may be overloaded")
            self.logger.info("  • Consider using a smaller model or reducing max_tokens")
            self.logger.info("  • Increase the timeout for complex requests")
            
        elif "Request Timeout" in timeout_type or "server-side" in timeout_type:
            self.logger.info("Troubleshooting hint: Server-side timeout suggests:")
            self.logger.info("  • The request took too long for the server to process")
            self.logger.info("  • Consider reducing request complexity (shorter prompts, fewer tools)")
            self.logger.info("  • Try using a faster model if available")
            self.logger.info("  • Increase timeout if processing complex tasks")
            
        elif "Gateway Timeout" in timeout_type:
            self.logger.info("Troubleshooting hint: Gateway timeout (504) suggests:")
            self.logger.info("  • Upstream server (LLM provider) is not responding")
            self.logger.info("  • This is typically a temporary issue")
            self.logger.info("  • Retry the request after a short delay")
            self.logger.info("  • Check the provider's status page for ongoing issues")
            
        elif "LiteLLM" in timeout_type:
            self.logger.info("Troubleshooting hint: LiteLLM timeout suggests:")
            self.logger.info("  • Internal LiteLLM processing timeout")
            self.logger.info("  • Consider increasing the timeout configuration")
            self.logger.info("  • Check if request parameters are causing delays")
            
        else:
            self.logger.info("Troubleshooting hint: General timeout suggestions:")
            self.logger.info("  • Increase timeout configuration if requests are legitimately slow")
            self.logger.info("  • Check network connectivity to the LLM provider")
            self.logger.info("  • Consider using a different model or provider")
            self.logger.info("  • Monitor provider status for ongoing issues")
        
        # Always provide general guidance
        self.logger.info(f"Current timeout setting: {self.config.timeout}s - adjust via LLMConfig(timeout=X)")
        
        # Check if timeout is unusually high and suggest review
        if self.config.timeout > 300:  # 5 minutes
            self.logger.warning(f"Note: Timeout is set to {self.config.timeout}s ({self.config.timeout/60:.1f} minutes)")
            self.logger.warning("Consider if such a long timeout is necessary for your use case")

    def _log_exception_details(self, exception: Exception) -> None:
        """Log comprehensive exception details for debugging."""
        import traceback
        
        self.logger.error("=== Exception Details ===")
        
        # Basic exception info
        self.logger.error(f"Exception class: {exception.__class__.__module__}.{exception.__class__.__name__}")
        self.logger.error(f"Exception message: {str(exception)}")
        
        # Exception arguments
        if hasattr(exception, 'args') and exception.args:
            self.logger.error(f"Exception args: {exception.args}")
        
        # Exception attributes
        exception_attrs = []
        for attr in dir(exception):
            if not attr.startswith('_') and attr not in ['args', 'with_traceback']:
                try:
                    value = getattr(exception, attr)
                    if not callable(value):
                        exception_attrs.append(f"{attr}={repr(value)}")
                except Exception:
                    exception_attrs.append(f"{attr}=<unable to access>")
        
        if exception_attrs:
            self.logger.error(f"Exception attributes: {', '.join(exception_attrs)}")
        
        # Stack trace
        if hasattr(exception, '__traceback__') and exception.__traceback__:
            self.logger.error("Exception traceback:")
            tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
            for line in tb_lines:
                for sub_line in line.rstrip().split('\n'):
                    if sub_line.strip():
                        self.logger.error(f"  {sub_line}")
        
        # Special handling for common exception types
        if hasattr(exception, 'response') and exception.response:
            response = exception.response
            self.logger.error("=== HTTP Response Details ===")
            
            # Response status
            if hasattr(response, 'status_code'):
                self.logger.error(f"Response status code: {response.status_code}")
            elif hasattr(response, 'status'):
                self.logger.error(f"Response status: {response.status}")
            
            # Response headers
            if hasattr(response, 'headers'):
                try:
                    headers = dict(response.headers)
                    self.logger.error(f"Response headers: {headers}")
                except Exception as e:
                    self.logger.error(f"Response headers (failed to parse): {e}")
            
            # Response URL
            if hasattr(response, 'url'):
                self.logger.error(f"Response URL: {response.url}")
            
            # Response content (truncated)
            if hasattr(response, 'text'):
                try:
                    text = response.text
                    if len(text) > 500:
                        text = text[:500] + "... (truncated)"
                    self.logger.error(f"Response text: {text}")
                except Exception as e:
                    self.logger.error(f"Response text (failed to read): {e}")
            elif hasattr(response, 'content'):
                try:
                    content = response.content
                    if isinstance(content, bytes):
                        if len(content) > 500:
                            content = content[:500] + b"... (truncated)"
                        content_str = content.decode('utf-8', errors='ignore')
                    else:
                        content_str = str(content)
                        if len(content_str) > 500:
                            content_str = content_str[:500] + "... (truncated)"
                    self.logger.error(f"Response content: {content_str}")
                except Exception as e:
                    self.logger.error(f"Response content (failed to read): {e}")
        
        # Additional context for specific exception types
        if 'requests' in str(type(exception)):
            self.logger.error("=== Requests Library Context ===")
            if hasattr(exception, 'request') and exception.request:
                request = exception.request
                if hasattr(request, 'method'):
                    self.logger.error(f"Request method: {request.method}")
                if hasattr(request, 'url'):
                    self.logger.error(f"Request URL: {request.url}")
                if hasattr(request, 'headers'):
                    # Don't log sensitive headers
                    safe_headers = {}
                    for k, v in dict(request.headers).items():
                        if k.lower() in ['authorization', 'api-key', 'x-api-key']:
                            safe_headers[k] = '<redacted>'
                        else:
                            safe_headers[k] = v
                    self.logger.error(f"Request headers: {safe_headers}")
        
        self.logger.error("=== End Exception Details ===")

    def _build_messages(self, user_message: str, system_message: Optional[str] = None, include_history: bool = True) -> List[Dict[str, Any]]:
        """Build messages list for completion."""
        messages = []
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.get_conversation_messages())
        
        # Handle system prompt based on configuration
        if self.config.disable_system_prompt:
            # When system prompts are disabled, merge system content into user message
            user_content = user_message
            
            # Use provided system message or config system prompt
            system_content = system_message or getattr(self.config, 'system_prompt', None)
            
            if system_content:
                user_content = f"System instructions: {system_content}\n\nUser request: {user_message}"
            
            messages.append({"role": "user", "content": user_content})
        else:
            # Normal system prompt behavior
            system_content = system_message or getattr(self.config, 'system_prompt', None)
            
            if system_content:
                messages.append({"role": "system", "content": system_content})
            
            messages.append({"role": "user", "content": user_message})
        
        return messages 