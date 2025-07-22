# Coding Agent Framework - Project Setup

## Issue Fixed

The project was moved from one folder to another, which caused the package installation to fail with the error:
```
ModuleNotFoundError: No module named 'mutator_framework'
```

## What Was Fixed

1. **Package Structure**: Reorganized the project to match the expected `mutator_framework` package structure
2. **Setup Configuration**: Fixed `setup.py` to properly reference the package structure and entry points
3. **Dependencies**: Ensured all dependencies (including `chardet`) are properly installed
4. **Virtual Environment**: Set up proper virtual environment usage for ARM64 architecture

## Project Structure

The project now has the correct structure:
```
mutator/
├── mutator_framework/          # Main package directory
│   ├── __init__.py
│   ├── __version__.py
│   ├── agent.py
│   ├── cli.py
│   ├── context/
│   ├── core/
│   ├── execution/
│   ├── llm/
│   └── tools/
├── venv_arm64/                      # Virtual environment
├── setup.py                        # Package configuration
├── requirements.txt                 # Dependencies
├── run_mutator.sh             # Convenience script
└── PROJECT_SETUP.md                # This file
```

## How to Run the Project

### Method 1: Using the Virtual Environment (Recommended)

1. **Activate the virtual environment**:
   ```bash
   source venv_arm64/bin/activate
   ```

2. **Run mutator**:
   ```bash
   mutator --help
   mutator tools
   mutator execute "your task here"
   ```

### Method 2: Using the Convenience Script

1. **Run directly without activating the virtual environment**:
   ```bash
   ./run_mutator.sh --help
   ./run_mutator.sh tools
   ./run_mutator.sh execute "your task here"
   ```

### Method 3: Direct Path (Alternative)

```bash
./venv_arm64/bin/mutator --help
```

## Available Commands

- `mutator --help` - Show help information
- `mutator tools` - List available tools
- `mutator agent` - Interactive coding agent with full read-write access
- `mutator chat` - Interactive chat with the coding agent in read-only mode
- `mutator status` - Show status of the agent and project
- `mutator config` - Configuration management

## Dependencies

All required dependencies are listed in `requirements.txt` and should be automatically installed when setting up the virtual environment. Key dependencies include:

- `litellm` - LLM integration
- `pydantic` - Data validation
- `langchain` - Agent orchestration
- `chromadb` - Vector database
- `typer` - CLI framework
- `rich` - Rich text and beautiful formatting
- `chardet` - Character encoding detection

## Troubleshooting

If you encounter issues:

1. **Architecture Issues**: Make sure you're using the ARM64 virtual environment on Apple Silicon Macs
2. **Missing Dependencies**: Reinstall dependencies with `pip install -r requirements.txt`
3. **Path Issues**: Use the convenience script `./run_mutator.sh` to avoid path conflicts
4. **Package Not Found**: Ensure the virtual environment is activated before running commands

## Development Setup

If you need to reinstall the package:

```bash
source venv_arm64/bin/activate
pip uninstall mutator -y
pip install -e .
```

The project is now fully functional and ready to use! 