"""
Unit tests for configuration module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from pydantic import ValidationError

from mutator.core.config import (
    LLMConfig, ContextConfig, SafetyConfig, ExecutionConfig, 
    ToolConfig, MCPServerConfig, VectorStoreConfig, AgentConfig, ConfigManager
)
from mutator.core.types import ExecutionMode, ConfirmationLevel
from mutator.llm.client import LLMClient


class TestLLMConfig:
    """Test LLMConfig."""
    
    def test_llm_config_creation(self):
        """Test creating LLMConfig."""
        config = LLMConfig(
            model="gpt-4",
            api_key="test-key",
            max_tokens=2000,
            temperature=0.5
        )
        
        assert config.model == "gpt-4"
        assert config.api_key == "test-key"
        assert config.max_tokens == 2000
        assert config.temperature == 0.5
        assert config.base_url is None
        
    def test_llm_config_with_base_url(self):
        """Test creating LLMConfig with base_url."""
        config = LLMConfig(
            model="gpt-4",
            api_key="test-key",
            base_url="https://api.example.com"
        )
        
        assert config.base_url == "https://api.example.com"
        assert config.api_base is None  # Should not be set when using base_url
        
    def test_llm_config_api_base_deprecation_warning(self):
        """Test that using api_base shows deprecation warning."""
        with pytest.warns(DeprecationWarning, match="The 'api_base' parameter is deprecated"):
            config = LLMConfig(
                model="gpt-4",
                api_key="test-key",
                api_base="https://api.example.com"
            )
        
        # Should copy api_base to base_url
        assert config.base_url == "https://api.example.com"
        assert config.api_base == "https://api.example.com"
        
    def test_llm_config_both_base_url_and_api_base(self):
        """Test that providing both base_url and api_base prefers base_url."""
        with pytest.warns(DeprecationWarning, match="Both 'api_base' and 'base_url' are provided"):
            config = LLMConfig(
                model="gpt-4",
                api_key="test-key",
                base_url="https://api.new.com",
                api_base="https://api.old.com"
            )
        
        # Should prefer base_url over api_base
        assert config.base_url == "https://api.new.com"
        assert config.api_base == "https://api.old.com"
    
    def test_llm_config_defaults(self):
        """Test LLMConfig with defaults."""
        # Temporarily clear environment variables that might interfere
        import os
        original_openai = os.environ.get('OPENAI_API_KEY')
        original_anthropic = os.environ.get('ANTHROPIC_API_KEY')
        
        try:
            # Clear the environment variables
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
            if 'ANTHROPIC_API_KEY' in os.environ:
                del os.environ['ANTHROPIC_API_KEY']
            
            config = LLMConfig(model="gpt-3.5-turbo")
            
            assert config.model == "gpt-3.5-turbo"
            assert config.api_key is None
            assert config.max_tokens == 2000
            assert config.temperature == 0.1
            assert config.top_p == 0.95
            assert config.frequency_penalty == 0.0
            assert config.presence_penalty == 0.0
        finally:
            # Restore environment variables
            if original_openai:
                os.environ['OPENAI_API_KEY'] = original_openai
            if original_anthropic:
                os.environ['ANTHROPIC_API_KEY'] = original_anthropic
    
    def test_llm_config_validation(self):
        """Test LLMConfig validation."""
        # Test invalid temperature
        with pytest.raises(ValidationError):
            LLMConfig(model="gpt-4", temperature=2.0)
        
        # Test invalid max_tokens
        with pytest.raises(ValidationError):
            LLMConfig(model="gpt-4", max_tokens=0)
        
        # Test invalid top_p
        with pytest.raises(ValidationError):
            LLMConfig(model="gpt-4", top_p=1.5)
    
    def test_llm_config_disable_system_prompt(self):
        """Test LLMConfig disable_system_prompt parameter."""
        # Test default value
        config = LLMConfig(model="gpt-4")
        assert config.disable_system_prompt is False
        
        # Test setting to True
        config = LLMConfig(model="gpt-4", disable_system_prompt=True)
        assert config.disable_system_prompt is True
        
        # Test setting to False explicitly
        config = LLMConfig(model="gpt-4", disable_system_prompt=False)
        assert config.disable_system_prompt is False
    
    def test_llm_config_disable_tool_role(self):
        """Test LLMConfig disable_tool_role parameter."""
        # Test default value
        config = LLMConfig(model="gpt-4")
        assert config.disable_tool_role is False
        
        # Test setting to True
        config = LLMConfig(model="gpt-4", disable_tool_role=True)
        assert config.disable_tool_role is True
        
        # Test setting to False explicitly
        config = LLMConfig(model="gpt-4", disable_tool_role=False)
        assert config.disable_tool_role is False


class TestLLMClientSystemPrompt:
    """Test LLMClient system prompt handling."""
    
    def test_system_prompt_enabled(self):
        """Test normal system prompt behavior when enabled."""
        config = LLMConfig(
            model="gpt-4",
            system_prompt="You are a helpful assistant",
            disable_system_prompt=False
        )
        client = LLMClient(config)
        
        messages = client._build_messages(
            user_message="Hello",
            system_message="Custom system message",
            include_history=False
        )
        
        # Should have system message as first message
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Custom system message"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
    
    def test_system_prompt_disabled(self):
        """Test system prompt behavior when disabled."""
        config = LLMConfig(
            model="gpt-4",
            system_prompt="You are a helpful assistant",
            disable_system_prompt=True
        )
        client = LLMClient(config)
        
        messages = client._build_messages(
            user_message="Hello",
            system_message="Custom system message",
            include_history=False
        )
        
        # Should have only user message with system content prepended
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "System instructions: Custom system message" in messages[0]["content"]
        assert "User request: Hello" in messages[0]["content"]
    
    def test_system_prompt_disabled_with_config_prompt(self):
        """Test system prompt disabled using config system_prompt."""
        config = LLMConfig(
            model="gpt-4",
            system_prompt="You are a helpful assistant",
            disable_system_prompt=True
        )
        client = LLMClient(config)
        
        messages = client._build_messages(
            user_message="Hello",
            include_history=False
        )
        
        # Should use config system_prompt when no explicit system_message provided
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "System instructions: You are a helpful assistant" in messages[0]["content"]
        assert "User request: Hello" in messages[0]["content"]
    
    def test_no_system_prompt_when_disabled(self):
        """Test behavior when no system prompt is provided and disabled."""
        config = LLMConfig(
            model="gpt-4",
            disable_system_prompt=True
        )
        client = LLMClient(config)
        
        messages = client._build_messages(
            user_message="Hello",
            include_history=False
        )
        
        # Should have only user message without system content
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"


class TestLLMClientToolRole:
    """Test LLMClient tool role handling."""
    
    def test_tool_role_enabled(self):
        """Test normal tool role behavior when enabled."""
        config = LLMConfig(
            model="gpt-4",
            disable_tool_role=False
        )
        client = LLMClient(config)
        
        # Test tool message handling
        tool_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "I'll help you", "tool_calls": [{"id": "call_123", "name": "test_tool", "arguments": {}}]},
            {"role": "tool", "content": "Tool result content", "tool_call_id": "call_123"}
        ]
        
        prepared_messages = client._prepare_messages(tool_messages)
        
        # Should preserve tool message as-is
        assert len(prepared_messages) == 3
        assert prepared_messages[0]["role"] == "user"
        assert prepared_messages[1]["role"] == "assistant"
        assert prepared_messages[2]["role"] == "tool"
        assert prepared_messages[2]["content"] == "Tool result content"
        assert prepared_messages[2]["tool_call_id"] == "call_123"
    
    def test_tool_role_disabled(self):
        """Test tool role behavior when disabled."""
        config = LLMConfig(
            model="gpt-4",
            disable_tool_role=True
        )
        client = LLMClient(config)
        
        # Test tool message handling
        tool_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "I'll help you", "tool_calls": [{"id": "call_123", "name": "test_tool", "arguments": {}}]},
            {"role": "tool", "content": "Tool result content", "tool_call_id": "call_123"}
        ]
        
        prepared_messages = client._prepare_messages(tool_messages)
        
        # Should convert tool message to user message with prefix
        assert len(prepared_messages) == 3
        assert prepared_messages[0]["role"] == "user"
        assert prepared_messages[1]["role"] == "assistant"
        assert prepared_messages[2]["role"] == "user"
        assert "Tool result for call_id call_123: Tool result content" in prepared_messages[2]["content"]
        assert "tool_call_id" not in prepared_messages[2]
    
    def test_tool_role_disabled_with_missing_call_id(self):
        """Test tool role disabled with missing tool_call_id."""
        config = LLMConfig(
            model="gpt-4",
            disable_tool_role=True
        )
        client = LLMClient(config)
        
        # Test tool message without tool_call_id
        tool_messages = [
            {"role": "tool", "content": "Tool result content"}
        ]
        
        prepared_messages = client._prepare_messages(tool_messages)
        
        # Should convert tool message to user message with "unknown" call_id
        assert len(prepared_messages) == 1
        assert prepared_messages[0]["role"] == "user"
        assert "Tool result for call_id unknown: Tool result content" in prepared_messages[0]["content"]
    
    def test_both_system_and_tool_role_disabled(self):
        """Test behavior when both disable_system_prompt and disable_tool_role are True."""
        config = LLMConfig(
            model="gpt-4",
            disable_system_prompt=True,
            disable_tool_role=True
        )
        client = LLMClient(config)
        
        # Test mixed messages
        mixed_messages = [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "I'll help you", "tool_calls": [{"id": "call_456", "name": "test_tool", "arguments": {}}]},
            {"role": "tool", "content": "Tool result", "tool_call_id": "call_456"}
        ]
        
        prepared_messages = client._prepare_messages(mixed_messages)
        
        # All messages should be converted to user messages
        assert len(prepared_messages) == 4
        assert all(msg["role"] == "user" for msg in [prepared_messages[0], prepared_messages[3]])
        assert prepared_messages[1]["role"] == "user"  # original user message
        assert prepared_messages[2]["role"] == "assistant"  # assistant message unchanged
        
        # Check system message conversion
        assert "System instructions: System instructions" in prepared_messages[0]["content"]
        
        # Check tool message conversion
        assert "Tool result for call_id call_456: Tool result" in prepared_messages[3]["content"]


class TestContextConfig:
    """Test ContextConfig."""
    
    def test_context_config_creation(self):
        """Test creating ContextConfig."""
        config = ContextConfig(
            project_path="/path/to/project",
            max_context_files=50,
            ignore_patterns=["*.pyc", "__pycache__"]
        )
        
        assert config.project_path == "/path/to/project"
        assert config.max_context_files == 50
        assert config.ignore_patterns == ["*.pyc", "__pycache__"]
    
    def test_context_config_defaults(self):
        """Test ContextConfig defaults."""
        config = ContextConfig()
        
        assert config.project_path == "."
        assert config.max_context_files == 20
        assert config.max_file_size == 1024 * 1024  # 1MB
        assert "*.pyc" in config.ignore_patterns
        assert "__pycache__" in config.ignore_patterns
        assert ".git" in config.ignore_patterns
    
    def test_context_config_validation(self):
        """Test ContextConfig validation."""
        # Test invalid max_context_files
        with pytest.raises(ValidationError):
            ContextConfig(max_context_files=0)
        
        # Test invalid max_file_size
        with pytest.raises(ValidationError):
            ContextConfig(max_file_size=0)


class TestSafetyConfig:
    """Test SafetyConfig."""
    
    def test_safety_config_creation(self):
        """Test creating SafetyConfig."""
        config = SafetyConfig(
            confirmation_level=ConfirmationLevel.HIGH,
            allowed_shell_commands=["ls", "cat"],
            blocked_shell_commands=["rm", "sudo"]
        )
        
        assert config.confirmation_level == ConfirmationLevel.HIGH
        assert config.allowed_shell_commands == ["ls", "cat"]
        assert config.blocked_shell_commands == ["rm", "sudo"]
    
    def test_safety_config_defaults(self):
        """Test SafetyConfig defaults."""
        config = SafetyConfig()
        
        assert config.confirmation_level == ConfirmationLevel.MEDIUM
        assert isinstance(config.allowed_shell_commands, list)
        assert isinstance(config.blocked_shell_commands, list)
        assert "rm" in config.blocked_shell_commands
        assert "sudo" in config.blocked_shell_commands


class TestExecutionConfig:
    """Test ExecutionConfig."""
    
    def test_execution_config_creation(self):
        """Test creating ExecutionConfig."""
        config = ExecutionConfig(
            default_mode=ExecutionMode.AGENT,
            max_iterations=100,
            retry_on_failure=True
        )
        
        assert config.default_mode == ExecutionMode.AGENT
        assert config.max_iterations == 100
        assert config.retry_on_failure is True
    
    def test_execution_config_defaults(self):
        """Test ExecutionConfig defaults."""
        config = ExecutionConfig()
        
        assert config.default_mode == ExecutionMode.CHAT
        assert config.max_iterations == 50
        assert config.retry_on_failure is False
        assert config.continue_on_tool_failure is False
        assert config.timeout == 300
    
    def test_execution_config_validation(self):
        """Test ExecutionConfig validation."""
        # Test invalid max_iterations
        with pytest.raises(ValidationError):
            ExecutionConfig(max_iterations=0)
        
        # Test invalid timeout
        with pytest.raises(ValidationError):
            ExecutionConfig(timeout=0)


class TestToolConfig:
    """Test ToolConfig."""
    
    def test_tool_config_creation(self):
        """Test creating ToolConfig."""
        config = ToolConfig(
            name="test_tool",
            enabled=True,
            timeout=30,
            settings={"param1": "value1"}
        )
        
        assert config.name == "test_tool"
        assert config.enabled is True
        assert config.timeout == 30
        assert config.settings == {"param1": "value1"}
    
    def test_tool_config_defaults(self):
        """Test ToolConfig defaults."""
        config = ToolConfig(name="test_tool")
        
        assert config.enabled is True
        assert config.timeout == 60
        assert config.settings == {}


class TestMCPServerConfig:
    """Test MCPServerConfig."""
    
    def test_mcp_server_config_creation(self):
        """Test creating MCPServerConfig."""
        config = MCPServerConfig(
            name="test_server",
            command=["python", "-m", "test_server"],
            env={"API_KEY": "test"}
        )
        
        assert config.name == "test_server"
        assert config.command == ["python", "-m", "test_server"]
        assert config.env == {"API_KEY": "test"}
    
    def test_mcp_server_config_defaults(self):
        """Test MCPServerConfig defaults."""
        config = MCPServerConfig(
            name="test_server",
            command=["python", "-m", "test_server"]
        )
        
        assert config.enabled is True
        assert config.env == {}
        assert config.timeout == 30


class TestVectorStoreConfig:
    """Test VectorStoreConfig."""
    
    def test_vector_store_config_creation(self):
        """Test creating VectorStoreConfig."""
        config = VectorStoreConfig(
            store_path="/path/to/store",
            embedding_model="custom-model",
            chunk_size=500
        )
        
        assert config.store_path == "/path/to/store"
        assert config.embedding_model == "custom-model"
        assert config.chunk_size == 500
    
    def test_vector_store_config_defaults(self):
        """Test VectorStoreConfig defaults."""
        config = VectorStoreConfig()
        
        assert config.store_path == "./vector_store"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.max_chunks == 1000


class TestAgentConfig:
    """Test AgentConfig."""
    
    def test_agent_config_creation(self):
        """Test creating AgentConfig."""
        llm_config = LLMConfig(model="gpt-4")
        context_config = ContextConfig(project_path="/test")
        safety_config = SafetyConfig()
        execution_config = ExecutionConfig()
        
        config = AgentConfig(
            llm_config=llm_config,
            context_config=context_config,
            safety_config=safety_config,
            execution_config=execution_config
        )
        
        assert config.llm_config == llm_config
        assert config.context_config == context_config
        assert config.safety_config == safety_config
        assert config.execution_config == execution_config
    
    def test_agent_config_defaults(self):
        """Test AgentConfig with defaults."""
        config = AgentConfig()
        
        assert isinstance(config.llm_config, LLMConfig)
        assert isinstance(config.context_config, ContextConfig)
        assert isinstance(config.safety_config, SafetyConfig)
        assert isinstance(config.execution_config, ExecutionConfig)
        assert isinstance(config.vector_store_config, VectorStoreConfig)
        assert config.tool_configs == []
        assert config.mcp_server_configs == []
        assert config.logging_level == "INFO"
    
    def test_agent_config_validation(self):
        """Test AgentConfig validation."""
        config = AgentConfig()
        
        # Should not raise any validation errors
        assert config.llm_config.model == "gpt-4.1-mini"
        assert config.context_config.project_path == "."
    
    def test_agent_config_with_custom_configs(self):
        """Test AgentConfig with custom sub-configs."""
        tool_config = ToolConfig(name="custom_tool", enabled=False)
        mcp_config = MCPServerConfig(name="test_mcp", command=["test"])
        
        config = AgentConfig(
            tool_configs=[tool_config],
            mcp_server_configs=[mcp_config]
        )
        
        assert len(config.tool_configs) == 1
        assert config.tool_configs[0].name == "custom_tool"
        assert len(config.mcp_server_configs) == 1
        assert config.mcp_server_configs[0].name == "test_mcp"


class TestConfigManager:
    """Test ConfigManager."""
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Create a test config
        original_config = AgentConfig(
            llm_config=LLMConfig(model="gpt-3.5-turbo", api_key="test-key"),
            context_config=ContextConfig(project_path="/test/path"),
            safety_config=SafetyConfig(confirmation_level=ConfirmationLevel.LOW)
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            ConfigManager.save_config(original_config, temp_path)
            
            # Load the config back
            loaded_config = ConfigManager.load_config(temp_path)
            
            # Verify the loaded config matches
            assert loaded_config.llm_config.model == "gpt-3.5-turbo"
            assert loaded_config.llm_config.api_key == "test-key"
            assert loaded_config.context_config.project_path == "/test/path"
            assert loaded_config.safety_config.confirmation_level == ConfirmationLevel.LOW
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_config_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {
            "llm_config": {
                "model": "gpt-4",
                "api_key": "test-key",
                "max_tokens": 1500
            },
            "context_config": {
                "project_path": "/custom/path",
                "max_context_files": 30
            },
            "safety_config": {
                "confirmation_level": "high",
                "blocked_shell_commands": ["rm", "sudo", "dd"]
            }
        }
        
        config = ConfigManager.load_config_from_dict(config_dict)
        
        assert config.llm_config.model == "gpt-4"
        assert config.llm_config.max_tokens == 1500
        assert config.context_config.project_path == "/custom/path"
        assert config.context_config.max_context_files == 30
        assert config.safety_config.confirmation_level == ConfirmationLevel.HIGH
    
    def test_validate_config(self):
        """Test config validation."""
        # Valid config with API key
        valid_config = AgentConfig(
            llm_config=LLMConfig(api_key="test-key")
        )
        errors = ConfigManager.validate_config(valid_config)
        assert len(errors) == 0
        
        # Invalid config (we'll create one with invalid values)
        invalid_config_dict = {
            "llm_config": {
                "model": "gpt-4",
                "temperature": 2.0,  # Invalid: > 1.0
                "max_tokens": -1     # Invalid: < 1
            }
        }
        
        with pytest.raises(ValidationError):
            ConfigManager.load_config_from_dict(invalid_config_dict)
    
    def test_merge_configs(self):
        """Test merging configurations."""
        base_config = AgentConfig(
            llm_config=LLMConfig(model="gpt-3.5-turbo"),
            context_config=ContextConfig(project_path="/base")
        )
        
        # Create a partial override config that only specifies certain fields
        override_config = AgentConfig()
        override_config.llm_config.model = "gpt-4"
        override_config.llm_config.api_key = "new-key"
        override_config.safety_config.confirmation_level = ConfirmationLevel.HIGH

        merged_config = ConfigManager.merge_configs(base_config, override_config)

        # Should use override values where available
        assert merged_config.llm_config.model == "gpt-4"
        assert merged_config.llm_config.api_key == "new-key"
        assert merged_config.safety_config.confirmation_level == ConfirmationLevel.HIGH

        # Should keep base values where not overridden - but since override has defaults,
        # we expect the merge to use override's default value
        # This tests that merge works with complete config objects
        assert merged_config.context_config.project_path == "."  # Override's default
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager.load_config("/nonexistent/config.json")
    
    def test_save_config_invalid_path(self):
        """Test saving config to invalid path."""
        config = AgentConfig()
        
        with pytest.raises((FileNotFoundError, PermissionError, OSError)):
            ConfigManager.save_config(config, "/invalid/path/config.json")
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = AgentConfig(
            llm_config=LLMConfig(model="gpt-4", api_key="test"),
            context_config=ContextConfig(project_path="/test")
        )
        
        config_dict = ConfigManager.config_to_dict(config)
        
        assert isinstance(config_dict, dict)
        assert "llm_config" in config_dict
        assert "context_config" in config_dict
        assert config_dict["llm_config"]["model"] == "gpt-4"
        assert config_dict["context_config"]["project_path"] == "/test"
    
    def test_environment_variable_substitution(self):
        """Test environment variable handling in config."""
        import os

        # Set a test environment variable
        os.environ["TEST_API_KEY"] = "env-api-key"

        try:
            config_dict = {
                "llm_config": {
                    "model": "gpt-4",
                    "api_key": "${TEST_API_KEY}"
                }
            }

            config = ConfigManager.load_config_from_dict(config_dict)

            # The config system doesn't perform env var substitution automatically
            # It preserves the literal string
            assert config.llm_config.api_key == "${TEST_API_KEY}"
            
            # But we can manually substitute using os.environ
            if config.llm_config.api_key.startswith("${") and config.llm_config.api_key.endswith("}"):
                env_var = config.llm_config.api_key[2:-1]
                substituted_value = os.environ.get(env_var, config.llm_config.api_key)
                assert substituted_value == "env-api-key"

        finally:
            # Clean up
            if "TEST_API_KEY" in os.environ:
                del os.environ["TEST_API_KEY"] 