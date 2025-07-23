# Mutator Framework

A comprehensive Python framework for building AI-powered coding agents that can execute complex coding tasks using Large Language Models (LLMs).

## Features

### Core Capabilities
- **LLM Integration**: Support for multiple LLM providers via LiteLLM (OpenAI, Anthropic, Google, etc.)
- **LangChain-Powered Execution**: Built on LangChain for robust agent orchestration, tool calling, and streaming
- **Intelligent Task Planning**: Simplified planning system where the LLM decides when to break down complex tasks
- **Extensible Tool System**: Modern @tool decorator for easy tool creation with built-in file operations, shell commands, and Git integration
- **Context Management**: Vector-based project context with ChromaDB for intelligent code understanding
- **Dual Execution Modes**: Chat mode (read-only) and Agent mode (full code modification)
- **Safety Features**: Configurable safety checks and user confirmations
- **Sub-Agent Delegation**: Automatic delegation of complex tasks to specialized sub-agents
- **Streaming Support**: Real-time streaming of agent responses and tool execution via LangGraph

### Advanced Features
- **MCP Server Integration**: Support for Model Context Protocol servers
- **Configuration Management**: Flexible configuration system with validation
- **Event-Driven Architecture**: Real-time execution monitoring and control
- **CLI Interface**: Rich command-line interface with interactive features
- **Extensible Design**: Easy to add custom tools and integrations
- **Tool Management**: Disable built-in tools to avoid conflicts with custom implementations


## Installation

```bash
pip install CodeMutator
```

### Development Installation

```bash
git clone https://github.com/code-mutator/mutator.git
cd mutator
pip install -e .
```

## Quick Start

### Basic Usage

```python
import asyncio
from mutator import Mutator, AgentConfig, ExecutionMode

async def main():
    # Create and initialize agent
    config = AgentConfig(working_directory="./my_project")
    agent = Mutator(config)
    await agent.initialize()
    
    # Execute a simple task
    async for event in agent.execute_task(
        "Add error handling to the main.py file",
        execution_mode=ExecutionMode.AGENT
    ):
        print(f"{event.event_type}: {event.data}")

asyncio.run(main())
```

### Chat Mode

```python
import asyncio
from mutator import Mutator, AgentConfig

async def main():
    config = AgentConfig(working_directory="./my_project")
    agent = Mutator(config)
    await agent.initialize()
    
    # Chat without making changes
    response = await agent.chat("What does the main function do?")
    print(response.content)

asyncio.run(main())
```

### CLI Usage

```bash
# Execute a task
mutator execute "Add unit tests for the authentication module" --project ./my_project

# Interactive chat
mutator chat --project ./my_project

# Single chat message
mutator chat "Explain the database schema" --project ./my_project

# Check agent status
mutator status --project ./my_project

# List available tools
mutator tools
```

## Configuration

### API Key Setup

The framework supports multiple LLM providers. Set up your API keys using environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google
export GOOGLE_API_KEY="your-google-api-key"

# Azure OpenAI
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2023-05-15"
```

### Provider Usage

```bash
# Use OpenAI (default)
mutator chat --model gpt-4-turbo-preview

# Use Anthropic Claude
mutator chat --provider anthropic --model claude-3-sonnet-20240229

# Use Google Gemini
mutator chat --provider google --model gemini-pro

# Use Azure OpenAI
mutator chat --provider azure --model gpt-4
```

### Creating Configuration

```bash
# Create default configuration
mutator config create --output my_config.json

# Validate configuration
mutator config validate my_config.json

# Show configuration
mutator config show my_config.json
```

### Configuration Structure

```json
{
  "llm_config": {
    "model": "gpt-4-turbo-preview",
    "provider": "openai",
    "api_key": "your-api-key",
    "max_tokens": 2000,
    "temperature": 0.1
  },
  "context_config": {
    "project_path": "./",
    "max_context_files": 20,
    "ignore_patterns": ["*.pyc", "__pycache__", ".git"]
  },
  "safety_config": {
    "confirmation_level": "medium",
    "allowed_shell_commands": ["ls", "cat", "git"],
    "blocked_shell_commands": ["rm", "sudo", "wget"]
  },
  "execution_config": {
    "default_mode": "agent",
    "max_iterations": 50,
    "retry_on_failure": true
  },
  "disabled_tools": [
    "git_status",
    "git_add",
    "run_shell"
  ]
}
```

### Provider-Specific Configuration Examples

#### OpenAI Configuration
```json
{
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4-turbo-preview",
    "api_key": "your-openai-api-key",
    "max_tokens": 4000,
    "temperature": 0.1
  }
}
```

#### Anthropic Configuration
```json
{
  "llm_config": {
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229",
    "api_key": "your-anthropic-api-key",
    "max_tokens": 4000,
    "temperature": 0.1
  }
}
```

#### Google Configuration
```json
{
  "llm_config": {
    "provider": "google",
    "model": "gemini-pro",
    "api_key": "your-google-api-key",
    "max_tokens": 2048,
    "temperature": 0.1
  }
}
```

#### Azure OpenAI Configuration
```json
{
  "llm_config": {
    "provider": "azure",
    "model": "gpt-4",
    "api_key": "your-azure-api-key",
    "base_url": "https://your-resource.openai.azure.com/",
    "api_version": "2023-05-15",
    "max_tokens": 4000,
    "temperature": 0.1
  }
}
```

## Examples

### Complex Task Execution

```python
import asyncio
from mutator import Mutator, AgentConfig, ExecutionMode

async def refactor_codebase():
    config = AgentConfig(working_directory="./my_project")
    agent = Mutator(config)
    await agent.initialize()
    
    task = """
    Refactor the user authentication system:
    1. Extract authentication logic into a separate service
    2. Add comprehensive error handling
    3. Implement rate limiting
    4. Add unit tests for all new components
    5. Update documentation
    """
    
    async for event in agent.execute_task(
        task, 
        execution_mode=ExecutionMode.AGENT
    ):
        if event.event_type == "task_started":
            print(f"Starting: {event.data}")
        elif event.event_type == "tool_call_completed":
            print(f"Completed: {event.data['tool_name']}")
        elif event.event_type == "task_completed":
            print("Task completed successfully!")

asyncio.run(refactor_codebase())
```

### Custom Tool Creation (Modern @tool Decorator)

```python
from mutator import Mutator, AgentConfig
from mutator.tools.decorator import tool

@tool
def calculate_complexity(file_path: str) -> dict:
    """Calculate code complexity metrics for a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        lines = len(content.splitlines())
        functions = content.count('def ')
        classes = content.count('class ')
        
        return {
            "file_path": file_path,
            "lines_of_code": lines,
            "functions": functions,
            "classes": classes,
            "complexity_score": lines + functions * 2 + classes * 3
        }
    except Exception as e:
        return {"error": f"Failed to analyze file: {str(e)}"}

@tool
def format_json(data: str, indent: int = 2) -> dict:
    """Format JSON data with proper indentation."""
    try:
        import json
        parsed = json.loads(data)
        formatted = json.dumps(parsed, indent=indent)
        return {
            "formatted_json": formatted,
            "original_length": len(data),
            "formatted_length": len(formatted)
        }
    except Exception as e:
        return {"error": f"Invalid JSON: {str(e)}"}

async def main():
    # Create agent
    config = AgentConfig()
    agent = Mutator(config)
    await agent.initialize()
    
    # Register custom tools
    agent.tool_manager.register_function(calculate_complexity)
    agent.tool_manager.register_function(format_json)
    
    # Use the tools
    result = await agent.tool_manager.execute_tool("calculate_complexity", {"file_path": "main.py"})
    print(f"Complexity analysis: {result}")
    
    # Or let the LLM use them in tasks
    async for event in agent.execute_task(
        "Analyze the complexity of all Python files in the src directory and format the results as JSON"
    ):
        print(f"{event.event_type}: {event.data}")

asyncio.run(main())
```

### Project Analysis

```python
import asyncio
from mutator import Mutator, AgentConfig

async def analyze_project():
    config = AgentConfig(working_directory="./my_project")
    agent = Mutator(config)
    await agent.initialize()
    
    # Get project context
    context = await agent.context_manager.get_project_context()
    print(f"Project: {context.get('project_name', 'Unknown')}")
    print(f"Files: {len(context.get('files', []))}")
    
    # Search for specific patterns
    results = await agent.context_manager.search_context("authentication", max_results=5)
    for result in results:
        print(f"Found in {result['file_path']}: {result['content'][:100]}...")

asyncio.run(analyze_project())
```

### Configuration Management

```python
import asyncio
from mutator import AgentConfig, LLMConfig, SafetyConfig
from mutator.core.types import ConfirmationLevel

async def custom_configuration():
    # Create custom configuration
    config = AgentConfig(
        llm_config=LLMConfig(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.2
        ),
        safety_config=SafetyConfig(
            confirmation_level=ConfirmationLevel.HIGH,
            allowed_shell_commands=["git", "ls", "cat"],
            blocked_shell_commands=["rm", "sudo", "curl"]
        ),
        disabled_tools=["web_search", "fetch_url"]  # Disable web tools
    )
    
    # Create agent with custom config
    agent = Mutator(config)
    await agent.initialize()
    
    # Execute task with custom settings
    async for event in agent.execute_task(
        "Review the codebase and suggest improvements"
    ):
        print(f"{event.event_type}: {event.data}")

asyncio.run(custom_configuration())
```

## Architecture

### Core Components

1. **Mutator**: Main orchestrator class
2. **LLMClient**: Interface to language models
3. **ToolManager**: Tool registration and execution with @tool decorator support
4. **ContextManager**: Project understanding and vector search
5. **TaskPlanner**: Simplified task analysis and prompt creation
6. **TaskExecutor**: Task execution with LLM-driven tool selection

### Execution Flow

```
Task Input → Complexity Analysis → LLM Execution → Tool Selection → Sub-Agent Delegation (if needed) → Results
```

### Modern Tool System

The framework uses a modern `@tool` decorator system:

```python
from mutator.tools.decorator import tool

@tool
def my_tool(param1: str, param2: int = 10) -> dict:
    """Tool description here."""
    # Tool implementation
    return {"result": "success"}
```

Features:
- **Automatic schema generation** from function signatures
- **Type inference** from annotations
- **Default parameter handling**
- **Async support** for both sync and async functions
- **Error handling** with structured responses

### Safety Features

- **Confirmation Levels**: None, Low, Medium, High
- **Command Filtering**: Allowed/blocked shell commands
- **Path Validation**: Prevent dangerous file operations
- **Interactive Mode**: User confirmation for tool execution
- **Tool Disabling**: Disable specific tools to prevent conflicts

## Advanced Usage

### Sub-Agent Task Delegation

The framework includes a powerful `task` tool that automatically delegates complex operations to sub-agents:

```python
# The LLM will automatically use the task tool for complex operations
async for event in agent.execute_task(
    "Implement a complete REST API with authentication, rate limiting, and comprehensive tests"
):
    if event.event_type == "tool_call_started" and event.data.get("tool_name") == "task":
        print("Delegating to sub-agent for complex task execution")
```

### Event Monitoring

```python
async def monitor_execution():
    config = AgentConfig()
    agent = Mutator(config)
    await agent.initialize()
    
    async for event in agent.execute_task("Create a new API endpoint"):
        if event.event_type == "tool_call_started":
            print(f"Executing: {event.data['tool_name']}")
        elif event.event_type == "complexity_analysis":
            print(f"Task complexity: {event.data['recommended_type']}")
        elif event.event_type == "task_failed":
            print(f"Task failed: {event.data['error']}")
            break
```

### Context Management

```python
# Custom context configuration
from mutator.core.config import ContextConfig

context_config = ContextConfig(
    project_path="./my_project",
    max_context_files=50,
    ignore_patterns=["*.pyc", "__pycache__", ".git", "node_modules"],
    file_size_limit=1024 * 1024  # 1MB
)

config = AgentConfig(context_config=context_config)
agent = Mutator(config)
```

### Disabled Tools

You can disable specific built-in tools to avoid conflicts with custom implementations:

```python
from mutator import Mutator, AgentConfig

# Disable git tools when using GitHub API
config = AgentConfig(
    disabled_tools=[
        "git_status",
        "git_add", 
        "git_commit",
        "git_log"
    ]
)

agent = Mutator(config)
await agent.initialize()

# Git tools are now disabled, use your custom GitHub API tools instead
```

Common use cases:
- **GitHub API Integration**: Disable git tools to use GitHub API instead
- **Security**: Disable shell tools in restricted environments
- **Performance**: Disable unused tools to reduce overhead

## CLI Reference

### Main Commands

```bash
# Execute tasks
mutator execute "task description" [options]

# Chat with agent
mutator chat [message] [options]

# Check status
mutator status [options]

# List tools
mutator tools [options]

# Configuration management
mutator config create [options]
mutator config validate <config-file>
mutator config show <config-file>
```

### Options

```bash
--project, -p       Path to project directory
--config, -c        Path to configuration file
--mode, -m          Execution mode (chat/agent)
--type, -t          Task type (simple/complex)
--verbose, -v       Verbose output
--interactive, -i   Interactive mode with confirmations
```

### Examples

```bash
# Execute with custom config
mutator execute "Add logging to all functions" --config my_config.json --project ./src

# Interactive chat session
mutator chat --project ./my_app

# Single chat message
mutator chat "What's the purpose of the main.py file?" --project ./my_app

# Check agent status
mutator status --project ./my_app

# List available tools
mutator tools --config my_config.json

# Create configuration
mutator config create --output ./configs/dev_config.json

# Validate configuration
mutator config validate ./configs/dev_config.json
```

## Best Practices

### Task Design
- **Be Specific**: Clear, actionable task descriptions
- **Let the LLM Decide**: The framework automatically determines when to use sub-agents
- **Provide Context**: Include relevant project information

### Tool Development
- **Use @tool Decorator**: Leverage the modern tool system for easy development
- **Type Annotations**: Use type hints for automatic schema generation
- **Error Handling**: Return structured error responses
- **Documentation**: Include clear docstrings for tool descriptions

### Safety
- **Use Confirmation Levels**: Appropriate safety for your environment
- **Limit Shell Commands**: Restrict dangerous operations
- **Review Generated Code**: Always review before deploying
- **Disable Unused Tools**: Reduce attack surface by disabling unnecessary tools

### Performance
- **Optimize Context**: Limit context size for faster processing
- **Use Appropriate Models**: Balance capability with cost
- **Monitor Resource Usage**: Track API calls and processing time
- **Disable Unused Tools**: Reduce initialization overhead

## Contributing

We welcome contributions to the Mutator Framework! Please follow these guidelines to ensure a smooth development experience.

### Development Setup

#### Prerequisites
- Python 3.8+ (Python 3.11+ recommended for Apple Silicon Macs)
- Git

#### Quick Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/code-mutator/mutator.git
   cd mutator
   ```

2. **Create Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Upgrade pip
   pip install --upgrade pip
   
   # Install project in development mode with dev dependencies
   pip install -e ".[dev]"
   ```

4. **Verify Installation**
   ```bash
   # Test CLI
   mutator --help
   
   # Run tests
   pytest tests/
   ```

#### Apple Silicon (ARM64) Setup

If you're on Apple Silicon (M1/M2/M3 Mac) and encounter architecture compatibility issues:

1. **Use Virtual Environment (Recommended)**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate
   
   # Install with ARM64 compatibility
   pip install --upgrade pip
   pip install -e .
   ```

2. **Alternative: Use Homebrew Python**
   ```bash
   # Install Homebrew Python (if not already installed)
   brew install python@3.11
   
   # Create virtual environment with Homebrew Python
   /opt/homebrew/bin/python3.11 -m venv .venv
   source .venv/bin/activate
   
   # Install dependencies
   pip install -e .
   ```

3. **Verify Architecture**
   ```bash
   python -c "import platform; print(f'Architecture: {platform.machine()}')"
   # Should output: arm64
   ```

#### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow the existing code style and patterns
   - Add type hints where appropriate
   - Include docstrings for new functions/classes

3. **Run Tests**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test file
   pytest tests/unit/test_tools.py
   
   # Run with coverage
   pytest tests/ --cov=mutator
   ```

4. **Code Quality Checks**
   ```bash
   # Format code (if black is installed)
   black mutator/ tests/
   
   # Sort imports (if isort is installed)
   isort mutator/ tests/
   
   # Type checking (if mypy is installed)
   mypy mutator/
   ```

5. **Test CLI Functionality**
   ```bash
   # Test with virtual environment
   source .venv/bin/activate
   mutator status
   mutator tools
   
   # Or use the wrapper script
   ./run_mutator.sh status
   ./run_mutator.sh tools
   ```

#### Adding New Tools

When adding new tools, use the modern `@tool` decorator:

```python
from mutator.tools.decorator import tool
from typing import Optional

@tool
def my_new_tool(
    required_param: str,
    optional_param: Optional[int] = None
) -> dict:
    """
    Brief description of what the tool does.
    
    Args:
        required_param: Description of required parameter
        optional_param: Description of optional parameter
        
    Returns:
        Dictionary with results or error information
    """
    try:
        # Tool implementation
        result = f"Processed {required_param}"
        return {
            "success": True,
            "result": result,
            "optional_used": optional_param is not None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

#### Testing Guidelines

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test tool interactions and workflows
3. **CLI Tests**: Test command-line interface functionality
4. **Architecture Tests**: Ensure compatibility across platforms

Example test structure:
```python
import pytest
from mutator.tools.your_tool import your_function

def test_your_function_success():
    """Test successful execution of your function."""
    result = your_function("test_input")
    assert result["success"] is True
    assert "result" in result

def test_your_function_error_handling():
    """Test error handling in your function."""
    result = your_function(None)  # Should cause error
    assert result["success"] is False
    assert "error" in result
```

#### Troubleshooting Development Issues

**Architecture Errors on Apple Silicon:**
```bash
# If you see "mach-o file, but is an incompatible architecture" errors:
# 1. Delete existing virtual environment
rm -rf .venv

# 2. Create new virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Reinstall with no cache
pip install --no-cache-dir -e .
```

**Import Errors:**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall in development mode
pip install -e .
```

**CLI Not Working:**
```bash
# Use the wrapper script
./run_mutator.sh --help

# Or activate virtual environment first
source .venv/bin/activate
mutator --help
```

#### Submitting Changes

1. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new tool for X functionality"
   ```

2. **Push to Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Include clear description of changes
   - Reference any related issues
   - Include tests for new functionality
   - Update documentation if needed

#### Commit Message Format

Use conventional commit format:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

Examples:
```
feat: add web scraping tool with rate limiting
fix: resolve architecture compatibility on Apple Silicon
docs: update contribution guidelines with venv setup
test: add integration tests for tool manager
```

### Development Environment Verification

After setup, verify everything works:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Check CLI
mutator --help
mutator status

# 3. Run tests
pytest tests/ -v

# 4. Test tool functionality
python -c "
from mutator import Mutator, AgentConfig
import asyncio

async def test():
    config = AgentConfig()
    agent = Mutator(config)
    await agent.initialize()
    tools = await agent.tool_manager.list_tools()
    print(f'Available tools: {len(tools)}')

asyncio.run(test())
"
```

If all steps complete successfully, you're ready to contribute!

## License
see LICENSE file for details.
