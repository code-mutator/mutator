"""
Configuration management for the Coding Agent Framework.

This module provides comprehensive configuration management including
LLM settings, tool configurations, MCP server settings, vector store
configuration, safety settings, and context management.
"""

import os
import json
import yaml
import toml
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from enum import Enum
import logging

from .types import ExecutionMode, TaskType, ConfirmationLevel

# Ensure environment variables from a local .env file are available as early as
# possible (before model validation happens).  We intentionally load this at
# import-time so that any configuration classes depending on `os.getenv` see the
# variables.
try:
    from dotenv import load_dotenv  # type: ignore

    # Do not overwrite already-exported vars (override=False) – that lets the
    # real environment take precedence while still supporting a project-level
    # .env file for convenience.
    load_dotenv(override=False)
except ModuleNotFoundError:
    # python-dotenv is optional; skip if it isn't installed.
    pass


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    CHROMADB = "chromadb"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    FAISS = "faiss"


class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4.1-mini"
    api_key: Optional[str] = None
    
    # Consolidated base URL parameter for all providers
    base_url: Optional[str] = None
    
    # Deprecated - kept for backward compatibility
    api_base: Optional[str] = Field(None, deprecated=True, description="Deprecated: Use base_url instead")
    
    api_version: Optional[str] = None
    max_tokens: int = 2000  # Changed from 4000 to 2000 to match tests
    temperature: float = 0.1
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 5
    retry_delay: float = 1.0
    stream: bool = False
    
    # Function calling settings
    function_calling: bool = True
    parallel_function_calls: bool = True
    
    # Tool description settings
    use_short_tool_descriptions: bool = True  # When True, use abbreviated tool descriptions to save tokens
    
    # Model-specific settings
    reasoning_effort: Optional[str] = None  # For o1 models
    system_prompt: Optional[str] = None
    
    # System prompt handling
    disable_system_prompt: bool = False  # When True, system messages are converted to user messages
    
    # Debug settings
    debug: bool = False
    
    # Custom headers for requests
    custom_headers: Optional[Dict[str, str]] = None
    
    @model_validator(mode='after')
    def _handle_base_url_consolidation(self):
        """Handle consolidation of api_base and base_url parameters."""
        # If both are provided, warn and prefer base_url
        if self.api_base is not None and self.base_url is not None:
            warnings.warn(
                "Both 'api_base' and 'base_url' are provided. Using 'base_url' and ignoring 'api_base'. "
                "The 'api_base' parameter is deprecated, please use 'base_url' instead.",
                DeprecationWarning,
                stacklevel=2
            )
        # If only api_base is provided, copy to base_url and warn
        elif self.api_base is not None and self.base_url is None:
            warnings.warn(
                "The 'api_base' parameter is deprecated and will be removed in a future version. "
                "Please use 'base_url' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self.base_url = self.api_base
        
        return self
    
    @model_validator(mode='after')
    def _check_api_key(self):
        """Validate API key is provided for cloud providers."""
        if self.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.AZURE, LLMProvider.GOOGLE]:
            provider_str = self.provider.value
            
            # Only try to get from environment if no explicit API key is provided
            if not self.api_key:
                env_api_key = self._get_api_key_from_env(provider_str)
                if env_api_key:
                    self.api_key = env_api_key
            
            if not self.api_key:
                # Postpone the hard failure until a model call is actually made – this lets
                # the CLI start (e.g., to show help) even if the key is absent, and it also
                # supports workflows where the key is injected later at runtime.
                logger = logging.getLogger(__name__)
                logger.warning(
                    "API key for provider '%s' is not set. The agent will attempt to read "
                    "the key again when the first model call is made and will fail at that "
                    "time if it is still missing.",
                    self.provider,
                )
        return self
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        """Validate max_tokens is positive."""
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v
    
    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v):
        """Validate top_p is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("top_p must be between 0 and 1")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def auto_detect_provider(cls, values):
        """Auto-detect provider from model name if not explicitly set."""
        if isinstance(values, dict):
            model = values.get('model', '')
            provider = values.get('provider')
            
            # Only auto-detect if provider is not explicitly set
            if not provider or provider == LLMProvider.OPENAI:
                if model.startswith('claude-'):
                    values['provider'] = LLMProvider.ANTHROPIC
                elif model.startswith('gpt-') or model.startswith('text-'):
                    values['provider'] = LLMProvider.OPENAI
                elif model.startswith('gemini-') or model.startswith('palm-'):
                    values['provider'] = LLMProvider.GOOGLE
                elif 'azure' in model.lower():
                    values['provider'] = LLMProvider.AZURE
        
        return values
    
    @classmethod
    def from_litellm_model(cls, model: str, **kwargs) -> 'LLMConfig':
        """Create config from a litellm model string."""
        # Parse provider from model string
        if "/" in model:
            provider_part, model_part = model.split("/", 1)
            provider_map = {
                "openai": LLMProvider.OPENAI,
                "anthropic": LLMProvider.ANTHROPIC,
                "azure": LLMProvider.AZURE,
                "google": LLMProvider.GOOGLE,
                "huggingface": LLMProvider.HUGGINGFACE,
                "ollama": LLMProvider.OLLAMA,
            }
            provider = provider_map.get(provider_part, LLMProvider.CUSTOM)
        else:
            provider = LLMProvider.OPENAI
            model_part = model
        
        return cls(
            provider=provider,
            model=model_part,
            **kwargs
        )

    @staticmethod
    def _get_api_key_from_env(provider_str: str) -> Optional[str]:
        """Return the first non-empty API key found for the given provider.

        The logic is intentionally lenient – it tries several common environment
        variable names that developers might use in addition to the official
        <PROVIDER>_API_KEY pattern.  This prevents the framework from failing
        early when the user has set a slightly different variable name.
        """

        provider_upper = provider_str.upper()

        # Official env-var pattern comes first
        candidates: List[str] = [f"{provider_upper}_API_KEY"]

        # Widely-used fall-backs per provider
        if provider_upper == "OPENAI":
            candidates += ["OPENAI_KEY"]
        elif provider_upper == "ANTHROPIC":
            candidates += ["ANTHROPIC_KEY"]
        elif provider_upper == "AZURE":
            # People sometimes re-use OpenAI style keys for Azure
            candidates += ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"]
        elif provider_upper == "GOOGLE":
            candidates += ["GOOGLE_API_KEY", "VERTEX_AI_API_KEY", "GOOGLE_VERTEX_AI_API_KEY"]

        # Generic catch-all – very last resort
        candidates.append("API_KEY")

        env_lower_map = {k.lower(): v for k, v in os.environ.items()}

        for name in candidates:
            # Fast path – exact name
            value = os.getenv(name)
            if value:
                return value

            # Case-insensitive fallback
            lower_name = name.lower()
            if lower_name in env_lower_map and env_lower_map[lower_name]:
                return env_lower_map[lower_name]

        return None


class ToolConfig(BaseModel):
    """Configuration for individual tools."""
    name: str
    enabled: bool = True
    timeout: int = 60  # seconds
    settings: Dict[str, Any] = Field(default_factory=dict)
    confirmation_level: ConfirmationLevel = ConfirmationLevel.NONE
    max_execution_time: int = 30  # seconds
    parameters: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, str] = Field(default_factory=dict)
    allowed_paths: List[str] = Field(default_factory=list)
    blocked_paths: List[str] = Field(default_factory=list)
    allowed_commands: List[str] = Field(default_factory=list)
    blocked_commands: List[str] = Field(default_factory=list)


class MCPServerConfig(BaseModel):
    """Configuration for MCP (Model Context Protocol) servers."""
    name: str
    command: List[str]  # Changed from str to List[str] to match tests
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    timeout: int = 30
    max_retries: int = 3
    
    @field_validator('command')
    @classmethod
    def validate_command(cls, v):
        """Validate that the command exists."""
        if isinstance(v, str):
            # Convert string to list for backward compatibility
            v = [v]
        if v and not Path(v[0]).is_file() and not any(
            Path(p) / v[0] for p in os.environ.get('PATH', '').split(os.pathsep)
            if (Path(p) / v[0]).is_file()
        ):
            raise ValueError(f"Command '{v[0]}' not found in PATH")
        return v


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    type: VectorStoreType = VectorStoreType.CHROMADB
    path: str = "./vector_store"
    store_path: str = "./vector_store"  # Added to match tests
    collection_name: str = "codebase"
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks: int = 1000  # Added to match tests
    
    # ChromaDB specific
    persist_directory: Optional[str] = None
    
    # Pinecone specific
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    
    # Weaviate specific
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    
    # Qdrant specific
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None


class SafetyConfig(BaseModel):
    """Configuration for safety and confirmation settings."""
    confirmation_level: ConfirmationLevel = ConfirmationLevel.MEDIUM  # Changed from default_confirmation_level
    require_confirmation_for_writes: bool = True
    require_confirmation_for_deletes: bool = True
    require_confirmation_for_shell: bool = True
    require_confirmation_for_git: bool = True
    
    # File operation safety
    backup_before_modify: bool = True
    backup_directory: str = "./.agent_backups"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Shell command safety
    allowed_shell_commands: List[str] = Field(default_factory=lambda: [
        "ls", "cat", "grep", "find", "git", "npm", "pip", "python", "node"
    ])
    blocked_shell_commands: List[str] = Field(default_factory=lambda: [
        "rm", "rmdir", "del", "format", "shutdown", "reboot", "su", "sudo"
    ])
    
    # Path safety
    allowed_paths: List[str] = Field(default_factory=lambda: ["."])
    blocked_paths: List[str] = Field(default_factory=lambda: [
        "/etc", "/sys", "/proc", "/dev", "/root", "/home/*/.ssh"
    ])
    
    # Network safety
    allow_network_access: bool = False
    allowed_domains: List[str] = Field(default_factory=list)
    blocked_domains: List[str] = Field(default_factory=list)


class ExecutionConfig(BaseModel):
    """Configuration for execution settings."""
    default_mode: ExecutionMode = ExecutionMode.CHAT
    max_iterations: int = 50
    timeout: int = 300  # seconds
    retry_on_failure: bool = False
    continue_on_tool_failure: bool = False
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Parallel execution settings
    max_parallel_tasks: int = 1
    enable_streaming: bool = False
    
    # Task management
    task_timeout: int = 600  # seconds
    subtask_timeout: int = 120  # seconds
    
    @field_validator('max_iterations')
    @classmethod
    def validate_max_iterations(cls, v):
        """Validate max_iterations is positive."""
        if v <= 0:
            raise ValueError("max_iterations must be positive")
        return v
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v


class ContextConfig(BaseModel):
    """Configuration for context management."""
    project_path: str = "."
    max_context_files: int = 20  # Changed from max_context_items to max_context_files
    max_file_size: int = 1024 * 1024  # 1MB
    ignore_patterns: List[str] = Field(default_factory=lambda: [
        "*.pyc", "*.pyo", "__pycache__", ".git", ".env", "node_modules",
        "venv*", ".venv*", "env*", ".env*", "build", "dist", "*.log", "*.tmp",
        "*.sqlite", "*.sqlite3", "*.db", # Exclude database files
        ".tox", ".pytest_cache", "*.egg-info", ".coverage", ".mypy_cache"
    ])
    
    # Context window and processing
    context_window_size: int = 8000  # tokens
    max_files_to_index: int = 50  # Reduced from 10000 to 50 for better performance
    
    # Context prioritization
    prioritize_recent_files: bool = True
    prioritize_modified_files: bool = True
    prioritize_open_files: bool = True
    
    # Memory management
    compress_old_context: bool = True
    context_compression_threshold: int = 20
    
    # Project context files
    project_context_files: List[str] = Field(default_factory=lambda: [
        "README.md", "package.json", "requirements.txt", "setup.py",
        "pyproject.toml", "Cargo.toml", "go.mod", "pom.xml"
    ])
    
    @field_validator('max_context_files')
    @classmethod
    def validate_max_context_files(cls, v):
        """Validate max_context_files is positive."""
        if v <= 0:
            raise ValueError("max_context_files must be positive")
        return v
    
    @field_validator('max_file_size')
    @classmethod
    def validate_max_file_size(cls, v):
        """Validate max_file_size is positive."""
        if v <= 0:
            raise ValueError("max_file_size must be positive")
        return v


class AgentConfig(BaseSettings):
    """Main configuration for the coding agent."""
    
    # Basic settings
    agent_name: str = "Mutator"
    execution_mode: ExecutionMode = ExecutionMode.CHAT
    default_task_type: TaskType = TaskType.SIMPLE
    working_directory: str = "."
    
    # Component configurations - changed field names to match tests
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    context_config: ContextConfig = Field(default_factory=ContextConfig)
    safety_config: SafetyConfig = Field(default_factory=SafetyConfig)
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    vector_store_config: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    tool_configs: List[ToolConfig] = Field(default_factory=list)
    mcp_server_configs: List[MCPServerConfig] = Field(default_factory=list)
    
    # Tool management settings
    disabled_tools: List[str] = Field(default_factory=list, description="List of built-in tool names to disable")
    
    # Advanced settings
    max_iterations: int = 10
    timeout: int = 300  # seconds
    debug: bool = False
    logging_level: str = "INFO"
    
    # Custom settings
    custom_instructions: Optional[str] = None
    custom_tools: Dict[str, Any] = Field(default_factory=dict)
    
    # Backward compatibility fields
    llm: Optional[LLMConfig] = None
    tools: Optional[List[ToolConfig]] = None
    mcp_servers: Optional[List[MCPServerConfig]] = None
    vector_store: Optional[VectorStoreConfig] = None
    safety: Optional[SafetyConfig] = None
    context: Optional[ContextConfig] = None
    
    class Config:
        env_file = ".env"
        env_prefix = "AGENT_"
        case_sensitive = False
    
    @model_validator(mode='before')
    @classmethod
    def handle_backward_compatibility(cls, values):
        """Handle backward compatibility for old field names."""
        if isinstance(values, dict):
            # Map old field names to new ones
            if 'llm' in values and 'llm_config' not in values:
                values['llm_config'] = values.pop('llm')
            if 'context' in values and 'context_config' not in values:
                values['context_config'] = values.pop('context')
            if 'safety' in values and 'safety_config' not in values:
                values['safety_config'] = values.pop('safety')
            if 'vector_store' in values and 'vector_store_config' not in values:
                values['vector_store_config'] = values.pop('vector_store')
            if 'tools' in values and 'tool_configs' not in values:
                values['tool_configs'] = values.pop('tools')
            if 'mcp_servers' in values and 'mcp_server_configs' not in values:
                values['mcp_server_configs'] = values.pop('mcp_servers')
        return values
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return self.model_dump_json()
    
    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        elif path.suffix.lower() == '.toml':
            with open(path, 'w') as f:
                toml.dump(self.to_dict(), f)
        else:
            # Default to JSON
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'AgentConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix.lower() == '.toml':
            with open(path, 'r') as f:
                data = toml.load(f)
        else:
            # Try to auto-detect format
            with open(path, 'r') as f:
                content = f.read()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        data = yaml.safe_load(content)
                    except yaml.YAMLError:
                        data = toml.loads(content)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create configuration from dictionary."""
        return cls(**data)


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration manager."""
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[AgentConfig] = None
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> AgentConfig:
        """Load configuration from file or create default."""
        if config_path:
            self.config_path = Path(config_path)
        
        if self.config_path and self.config_path.exists():
            self.config = AgentConfig.from_file(self.config_path)
        else:
            # Try to find config file
            found_path = self.find_config_file()
            if found_path:
                self.config = AgentConfig.from_file(found_path)
                self.config_path = found_path
            else:
                self.config = AgentConfig()
        
        return self.config
    
    def get_config(self) -> AgentConfig:
        """Get current configuration, loading if necessary."""
        if self.config is None:
            return self.load_config()
        return self.config
    
    def save_config(self, config: AgentConfig, path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file."""
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ValueError("No save path specified")
        
        config.save_to_file(save_path)
        self.config = config
        self.config_path = save_path
    
    def update_config(self, updates: Dict[str, Any]) -> AgentConfig:
        """Update current configuration with new values."""
        current = self.get_config()
        updated_data = current.to_dict()
        updated_data.update(updates)
        self.config = AgentConfig.from_dict(updated_data)
        return self.config
    
    def validate_config(self, config: AgentConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate LLM config
        provider_str = config.llm_config.provider.value if hasattr(config.llm_config.provider, 'value') else str(config.llm_config.provider)
        if not config.llm_config.api_key and not os.getenv(f"{provider_str.upper()}_API_KEY"):
            issues.append(f"API key required for {config.llm_config.provider} provider")
        
        # Validate paths
        if not Path(config.working_directory).exists():
            issues.append(f"Working directory does not exist: {config.working_directory}")
        
        # Validate tool configs
        for tool_config in config.tool_configs:
            if tool_config.timeout <= 0:
                issues.append(f"Tool '{tool_config.name}' has invalid timeout: {tool_config.timeout}")
        
        # Validate MCP servers
        for mcp_config in config.mcp_server_configs:
            if not mcp_config.command:
                issues.append(f"MCP server '{mcp_config.name}' has no command specified")
        
        return issues
    
    def get_default_config_paths(self) -> List[Path]:
        """Get list of default configuration file paths to search."""
        return [
            Path("agent_config.json"),
            Path("agent_config.yaml"),
            Path("agent_config.yml"),
            Path("agent_config.toml"),
            Path(".agent_config.json"),
            Path(".agent_config.yaml"),
            Path(".agent_config.yml"),
            Path(".agent_config.toml"),
        ]
    
    def find_config_file(self) -> Optional[Path]:
        """Find configuration file in default locations."""
        for path in self.get_default_config_paths():
            if path.exists():
                return path
        return None
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> AgentConfig:
        """Load configuration from file (static method for CLI usage)."""
        return AgentConfig.from_file(config_path)
    
    @staticmethod
    def load_config_from_dict(data: Dict[str, Any]) -> AgentConfig:
        """Load configuration from dictionary."""
        return AgentConfig.from_dict(data)
    
    @staticmethod
    def save_config(config: AgentConfig, path: Union[str, Path]) -> None:
        """Save configuration to file (static method for CLI usage)."""
        config.save_to_file(path)
    
    @staticmethod
    def validate_config(config: AgentConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate LLM config
        provider_str = config.llm_config.provider.value if hasattr(config.llm_config.provider, 'value') else str(config.llm_config.provider)
        if not config.llm_config.api_key and not os.getenv(f"{provider_str.upper()}_API_KEY"):
            issues.append(f"API key required for {config.llm_config.provider} provider")
        
        # Validate paths
        if not Path(config.working_directory).exists():
            issues.append(f"Working directory does not exist: {config.working_directory}")
        
        # Validate tool configs
        for tool_config in config.tool_configs:
            if tool_config.timeout <= 0:
                issues.append(f"Tool '{tool_config.name}' has invalid timeout: {tool_config.timeout}")
        
        # Validate MCP servers
        for mcp_config in config.mcp_server_configs:
            if not mcp_config.command:
                issues.append(f"MCP server '{mcp_config.name}' has no command specified")
        
        return issues
    
    @staticmethod
    def merge_configs(base_config: AgentConfig, override_config: AgentConfig) -> AgentConfig:
        """Merge two configurations, with override taking precedence."""
        base_dict = base_config.to_dict()
        override_dict = override_config.to_dict()
        
        # Deep merge dictionaries
        def deep_merge(base: dict, override: dict) -> dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(base_dict, override_dict)
        return AgentConfig.from_dict(merged_dict)
    
    @staticmethod
    def config_to_dict(config: AgentConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return config.to_dict()


# Export configuration classes
__all__ = [
    "LLMProvider", "VectorStoreType", "LLMConfig", "ToolConfig", "MCPServerConfig",
    "VectorStoreConfig", "SafetyConfig", "ExecutionConfig", "ContextConfig",
    "AgentConfig", "ConfigManager"
] 