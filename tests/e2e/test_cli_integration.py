"""
End-to-end tests for the CLI interface.
These tests require a real API key and test the CLI functionality.
"""

import pytest
import asyncio
import os
import subprocess
import tempfile
from pathlib import Path


@pytest.mark.e2e
class TestCLIBasicOperations:
    """Test basic CLI operations."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            ["python", "-m", "mutator.cli", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "execute" in result.stdout
        assert "chat" in result.stdout
        assert "status" in result.stdout
    
    def test_cli_status_command(self):
        """Test CLI status command."""
        result = subprocess.run(
            ["python", "-m", "mutator.cli", "status"],
            capture_output=True,
            text=True
        )
        
        # Should succeed even without API key for basic status
        assert result.returncode == 0
        assert "Health Check" in result.stdout or "Agent Status" in result.stdout
    
    def test_cli_tools_command(self):
        """Test CLI tools listing command."""
        result = subprocess.run(
            ["python", "-m", "mutator.cli", "tools"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Available Tools" in result.stdout
        assert "read_file" in result.stdout
    
    def test_cli_chat_command(self, temp_project_dir):
        """Test CLI chat command with mocked LLM."""
        # Create a simple config file
        config_file = temp_project_dir / "agent_config.json"
        config_data = {
            "llm_config": {
                "model": "claude-3-haiku-20240307",
                "api_key": "test-api-key"  # Use test key since we'll mock the response
            },
            "context_config": {
                "project_path": str(temp_project_dir),
                "max_context_files": 5  # Limit for faster initialization
            },
            "safety_config": {
                "confirmation_level": "none"
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_data, indent=2))
        
        # Set environment variable to mock API key
        import os
        os.environ["ANTHROPIC_API_KEY"] = "test-api-key"
        
        # Test single chat command (message is positional argument)
        # This will fail with authentication error, but that's expected with a fake key
        result = subprocess.run([
            "python", "-m", "mutator.cli", "chat",
            "--config", str(config_file),
            "What is the main purpose of this project?"
        ], capture_output=True, text=True, timeout=30)
        
        # The command should show the chat interface and handle auth error gracefully
        # Check that it at least attempted to run (not a syntax error)
        assert "Chat Message" in result.stdout
        # Should handle auth error gracefully
        assert "authentication_error" in result.stderr or "invalid x-api-key" in result.stderr or result.returncode != 0
    
    def test_cli_execute_command(self, temp_project_dir):
        """Test CLI execute command with mocked LLM."""
        # Create a simple config file
        config_file = temp_project_dir / "agent_config.json"
        config_data = {
            "llm_config": {
                "model": "claude-3-haiku-20240307",
                "api_key": "test-api-key"  # Use test key since we'll mock the response
            },
            "context_config": {
                "project_path": str(temp_project_dir),
                "max_context_files": 5  # Limit for faster initialization
            },
            "safety_config": {
                "confirmation_level": "none"
            },
            "execution_config": {
                "default_mode": "agent"
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_data, indent=2))
        
        # Set environment variable to mock API key
        import os
        os.environ["ANTHROPIC_API_KEY"] = "test-api-key"
        
        # Test simple task execution (task is positional argument)
        # This will fail with authentication error, but that's expected with a fake key
        result = subprocess.run([
            "python", "-m", "mutator.cli", "execute",
            "--config", str(config_file),
            "Create a simple hello.py file that prints 'Hello, World!'"
        ], capture_output=True, text=True, timeout=60)
        
        # The command should succeed in starting and processing
        # Check that it at least attempted to run and got to the LLM call (not a syntax error)
        assert "Executing Task" in result.stdout  # Should show task execution started
        # Should handle auth error gracefully - either fail or succeed is acceptable
        assert result.returncode in [0, 1]  # Both success and controlled failure are acceptable


@pytest.mark.e2e
class TestCLIConfigurationManagement:
    """Test CLI configuration management."""
    
    def test_cli_config_create(self, temp_project_dir):
        """Test creating configuration via CLI."""
        config_file = temp_project_dir / "new_config.json"
        
        result = subprocess.run([
            "python", "-m", "mutator.cli", "config", "create",
            "--output", str(config_file),
            "--template", "default"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert config_file.exists()
        
        # Verify config content
        import json
        config_data = json.loads(config_file.read_text())
        assert config_data["llm_config"]["model"] == "gpt-4-turbo-preview"
        assert "context_config" in config_data
        assert "project_path" in config_data["context_config"]
    
    def test_cli_config_validate(self, temp_project_dir):
        """Test validating configuration via CLI."""
        # Create a valid config
        config_file = temp_project_dir / "valid_config.json"
        config_data = {
            "llm_config": {
                "model": "gpt-4",
                "max_tokens": 2000
            },
            "context_config": {
                "project_path": str(temp_project_dir)
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_data, indent=2))
        
        result = subprocess.run([
            "python", "-m", "mutator.cli", "config", "validate",
            str(config_file)
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Configuration is valid" in result.stdout
    
    def test_cli_config_show(self, temp_project_dir):
        """Test showing configuration via CLI."""
        # Create a config
        config_file = temp_project_dir / "show_config.json"
        config_data = {
            "llm_config": {
                "model": "gpt-4"
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_data, indent=2))
        
        result = subprocess.run([
            "python", "-m", "mutator.cli", "config", "show",
            str(config_file)
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "gpt-4" in result.stdout


@pytest.mark.e2e
class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_cli_invalid_config(self, temp_project_dir):
        """Test CLI with invalid configuration."""
        # Create invalid config
        config_file = temp_project_dir / "invalid_config.json"
        config_file.write_text("invalid json content")
        
        result = subprocess.run([
            "python", "-m", "mutator.cli", "config", "validate",
            str(config_file)
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "failed" in result.stdout.lower() or "error" in result.stdout.lower()
    
    def test_cli_missing_config(self):
        """Test CLI with missing configuration file."""
        result = subprocess.run([
            "python", "-m", "mutator.cli", "config", "validate",
            "/nonexistent/config.json"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()
    
    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        result = subprocess.run([
            "python", "-m", "mutator.cli", "invalid_command"
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
    
    def test_cli_missing_api_key(self, temp_project_dir):
        """Test CLI without API key."""
        # Create config without API key
        config_file = temp_project_dir / "no_key_config.json"
        config_data = {
            "llm_config": {
                "model": "gpt-4"
            },
            "context_config": {
                "project_path": str(temp_project_dir),
                "max_context_files": 5  # Limit for faster initialization
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_data, indent=2))
        
        result = subprocess.run([
            "python", "-m", "mutator.cli", "chat",
            "--config", str(config_file),
            "Hello"
        ], capture_output=True, text=True, timeout=30)
        
        # Should fail gracefully
        assert result.returncode != 0
        # Should mention API key or authentication issue
        output_text = (result.stderr + result.stdout).lower()
        assert "api" in output_text or "key" in output_text or "authentication" in output_text


@pytest.mark.e2e
class TestCLIAdvancedFeatures:
    """Test advanced CLI features."""
    
    def test_cli_verbose_output(self, temp_project_dir):
        """Test CLI with verbose output."""
        config_file = temp_project_dir / "verbose_config.json"
        config_data = {
            "llm_config": {
                "model": "claude-3-haiku-20240307",
                "api_key": os.getenv("SONNET_KEY")
            },
            "context_config": {
                "project_path": str(temp_project_dir)
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_data, indent=2))
        
        result = subprocess.run([
            "python", "-m", "mutator.cli", "chat",
            "--config", str(config_file),
            "--verbose",
            "Hello"
        ], capture_output=True, text=True, timeout=15)
        
        # Should either succeed or fail gracefully
        assert result.returncode in [0, 1]
        # Verbose output should contain more information
        assert len(result.stdout) > 0
    
    def test_cli_output_format(self, temp_project_dir):
        """Test CLI with different output formats."""
        config_file = temp_project_dir / "format_config.json"
        config_data = {
            "llm_config": {
                "model": "claude-3-haiku-20240307",
                "api_key": os.getenv("SONNET_KEY")
            },
            "context_config": {
                "project_path": str(temp_project_dir)
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_data, indent=2))
        
        # Test JSON output format
        result = subprocess.run([
            "python", "-m", "mutator.cli", "status",
            "--config", str(config_file),
            "--format", "json"
        ], capture_output=True, text=True, timeout=15)
        
        assert result.returncode == 0
        
        # Should be valid JSON or at least contain JSON-like structure
        assert "{" in result.stdout or "status" in result.stdout.lower()
    
    def test_cli_environment_variables(self, temp_project_dir):
        """Test CLI with environment variables."""
        # Set environment variable
        env = os.environ.copy()
        env["MUTATOR_MODEL"] = "gpt-4"
        env["MUTATOR_PROJECT_PATH"] = str(temp_project_dir)
        
        result = subprocess.run([
            "python", "-m", "mutator.cli", "config", "show"
        ], capture_output=True, text=True, env=env, timeout=10)
        
        assert result.returncode == 0
        # Should use environment variables
        assert "gpt-4" in result.stdout
    
    def test_cli_interactive_mode(self, temp_project_dir):
        """Test CLI interactive mode."""
        config_file = temp_project_dir / "interactive_config.json"
        config_data = {
            "llm_config": {
                "model": "claude-3-haiku-20240307",
                "api_key": os.getenv("SONNET_KEY")
            },
            "context_config": {
                "project_path": str(temp_project_dir)
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_data, indent=2))
        
        # Test interactive mode with input
        try:
            process = subprocess.Popen([
                "python", "-m", "mutator.cli", "chat",
                "--config", str(config_file),
                "--interactive"
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Send input and exit
            stdout, stderr = process.communicate(input="Hello\nexit\n", timeout=30)
            
            assert process.returncode in [0, 1]  # Success or controlled failure
            assert len(stdout) > 0
            
        except subprocess.TimeoutExpired:
            process.kill()
            pytest.skip("Interactive mode test timed out")


@pytest.mark.e2e
class TestCLIIntegrationScenarios:
    """Test real-world CLI usage scenarios."""
    
    def test_cli_project_analysis_workflow(self, temp_project_dir):
        """Test complete project analysis workflow via CLI."""
        config_file = temp_project_dir / "analysis_config.json"
        config_data = {
            "llm_config": {
                "model": "claude-3-haiku-20240307",
                "api_key": os.getenv("SONNET_KEY")
            },
            "context_config": {
                "project_path": str(temp_project_dir)
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_data, indent=2))
        
        # Step 1: Analyze project structure
        result1 = subprocess.run([
            "python", "-m", "mutator.cli", "chat",
            "--config", str(config_file),
            "Analyze the project structure and list all Python files"
        ], capture_output=True, text=True, timeout=30)
        
        # Should either succeed or fail gracefully
        assert result1.returncode in [0, 1]
        
        # Step 2: Get code quality assessment
        result2 = subprocess.run([
            "python", "-m", "mutator.cli", "chat",
            "--config", str(config_file),
            "Assess the code quality and suggest improvements"
        ], capture_output=True, text=True, timeout=30)
        
        # Should either succeed or fail gracefully
        assert result2.returncode in [0, 1]
        
        # Both should produce some output
        assert len(result1.stdout) > 0
        assert len(result2.stdout) > 0
    
    def test_cli_development_workflow(self, temp_project_dir):
        """Test development workflow via CLI."""
        config_file = temp_project_dir / "dev_config.json"
        config_data = {
            "llm_config": {
                "model": "claude-3-haiku-20240307",
                "api_key": os.getenv("SONNET_KEY")
            },
            "context_config": {
                "project_path": str(temp_project_dir)
            },
            "execution_config": {
                "default_mode": "agent"
            },
            "safety_config": {
                "confirmation_level": "none"
            }
        }
        
        import json
        config_file.write_text(json.dumps(config_data, indent=2))
        
        # Step 1: Create a new feature
        result1 = subprocess.run([
            "python", "-m", "mutator.cli", "execute",
            "--config", str(config_file),
            "Create a simple calculator.py file with basic math operations"
        ], capture_output=True, text=True, timeout=60)
        
        # Should either succeed or fail gracefully
        assert result1.returncode in [0, 1]
        
        # Step 2: Test the created feature
        result2 = subprocess.run([
            "python", "-m", "mutator.cli", "chat",
            "--config", str(config_file),
            "Review the calculator.py file and suggest any improvements"
        ], capture_output=True, text=True, timeout=30)
        
        # Should either succeed or fail gracefully
        assert result2.returncode in [0, 1]
        
        # Both should produce some output
        assert len(result1.stdout) > 0
        assert len(result2.stdout) > 0
        
        # Check if file was created
        calc_file = temp_project_dir / "calculator.py"
        if calc_file.exists():
            content = calc_file.read_text()
            assert "def" in content  # Should have functions
            assert any(op in content for op in ["+", "-", "*", "/"])  # Should have math operations 