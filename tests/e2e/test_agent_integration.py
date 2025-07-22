"""
End-to-end integration tests for the Coding Agent Framework.
These tests require a real API key and test the full agent functionality.
"""

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from mutator import Mutator, AgentConfig
from mutator.core.config import (
    LLMConfig, ContextConfig, SafetyConfig, ExecutionConfig
)
from mutator.core.types import ExecutionMode, ConfirmationLevel, TaskType, LLMResponse


@pytest.mark.e2e
class TestAgentBasicOperations:
    """Test basic agent operations end-to-end."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, test_config, temp_project_dir):
        """Test agent initialization with mocked components."""
        # Update config to use test API key
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        
        # Mock the heavy operations
        with patch('mutator.context.vector_store.SentenceTransformer') as mock_transformer, \
             patch('mutator.context.vector_store.chromadb.PersistentClient') as mock_chroma, \
             patch('mutator.llm.client.LLMClient.health_check') as mock_llm_health:
            
            # Set up mocks
            mock_transformer.return_value = Mock()
            mock_chroma.return_value = Mock()
            mock_collection = Mock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
            mock_llm_health.return_value = True
            
            agent = Mutator(test_config)
            await agent.initialize()
            
            # Test health check
            health = await agent.health_check()
            assert health["status"] == "healthy"
            
            # Cleanup
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_simple_chat_interaction(self, test_config, temp_project_dir):
        """Test simple chat interaction with mocked LLM."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        test_config.execution_config.default_mode = ExecutionMode.CHAT
        
        # Mock the heavy operations
        with patch('mutator.context.vector_store.SentenceTransformer') as mock_transformer, \
             patch('mutator.context.vector_store.chromadb.PersistentClient') as mock_chroma, \
             patch('mutator.llm.client.LLMClient.complete') as mock_complete:
            
            # Set up mocks
            mock_transformer.return_value = Mock()
            mock_chroma.return_value = Mock()
            mock_collection = Mock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
            
            # Mock LLM response
            mock_response = LLMResponse(
                content="The README.md file provides an overview of the test project, including setup instructions and usage examples.",
                success=True,
                tool_calls=[],
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                model="claude-3-haiku-20240307"
            )
            mock_complete.return_value = mock_response
            
            agent = Mutator(test_config)
            await agent.initialize()
            
            try:
                # Test simple question
                response = await agent.chat("What is the purpose of the README.md file in this project?")
                
                assert response.success is True
                assert len(response.content) > 0
                assert "readme" in response.content.lower() or "test project" in response.content.lower()
                
            finally:
                if hasattr(agent, 'cleanup'):
                    await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_file_analysis_task(self, test_config, temp_project_dir):
        """Test file analysis task."""
        # Create test files
        (temp_project_dir / "src").mkdir(exist_ok=True)
        test_file = temp_project_dir / "src" / "example.py"
        test_file.write_text("""
def calculate_sum(a, b):
    return a + b

def calculate_product(a, b):
    return a * b

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = calculate_sum(a, b)
        self.history.append(f"Added {a} + {b} = {result}")
        return result
""")

        # Update config
        api_key = os.getenv("SONNET_KEY")
        if not api_key or not api_key.startswith("sk-ant-api"):
            pytest.skip("Valid SONNET_KEY required for this test")
            
        test_config.llm_config.api_key = api_key
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)

        agent = Mutator(test_config)
        await agent.initialize()

        try:
            response = await agent.chat(
                "Analyze the code in src/example.py and tell me what functions and classes are defined."
            )

            assert response.content is not None
            assert len(response.content) > 0
            # Should mention the functions and class
            content_lower = response.content.lower()
            assert "calculate_sum" in content_lower or "function" in content_lower

        except Exception as e:
            if "authentication" in str(e).lower() or "401" in str(e):
                pytest.skip(f"API authentication failed: {e}")
            else:
                raise
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_code_search_functionality(self, test_config, temp_project_dir):
        """Test code search functionality."""
        # Create test files
        (temp_project_dir / "src").mkdir(exist_ok=True)
        test_file = temp_project_dir / "src" / "utils.py"
        test_file.write_text("""
def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def hash_password(password):
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()
""")

        # Update config
        api_key = os.getenv("SONNET_KEY")
        if not api_key or not api_key.startswith("sk-ant-api"):
            pytest.skip("Valid SONNET_KEY required for this test")
            
        test_config.llm_config.api_key = api_key
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)

        agent = Mutator(test_config)
        await agent.initialize()

        try:
            response = await agent.chat(
                "Find all functions related to email validation in the codebase."
            )

            assert response.content is not None
            assert len(response.content) > 0
            # Should find the email validation function
            content_lower = response.content.lower()
            assert "email" in content_lower

        except Exception as e:
            if "authentication" in str(e).lower() or "401" in str(e):
                pytest.skip(f"API authentication failed: {e}")
            else:
                raise
        finally:
            await agent.cleanup()


@pytest.mark.e2e
class TestAgentCodeGeneration:
    """Test code generation capabilities."""
    
    @pytest.mark.asyncio
    async def test_simple_code_generation(self, test_config, temp_project_dir):
        """Test simple code generation task."""
        # Get the real API key
        api_key = os.getenv("SONNET_KEY")
        
        # Skip if no API key is provided at all
        if not api_key:
            pytest.skip("SONNET_KEY environment variable required for e2e test")
            
        print(f"DEBUG: Using API Key: {api_key[:20]}...")
        
        test_config.llm_config.api_key = api_key
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        
        print(f"DEBUG: Config API Key length: {len(test_config.llm_config.api_key)}")
        print(f"DEBUG: Config API Key starts with: {test_config.llm_config.api_key[:20]}")
        print(f"DEBUG: Model: {test_config.llm_config.model}")

        agent = Mutator(test_config)
        await agent.initialize()

        task = "Create a simple Python function that calculates the factorial of a number"

        # Collect events from the async generator
        events = []
        async for event in agent.execute_task(task, ExecutionMode.AGENT):
            events.append(event)
            print(f"DEBUG EVENT: {event.event_type} - {event.data}")

        # Verify events - CHECK FOR ACTUAL SUCCESS
        assert len(events) > 0
        event_types = [event.event_type for event in events]
        print(f"DEBUG: All event types: {event_types}")
        
        # Check if the task actually succeeded
        task_completed_events = [e for e in events if e.event_type == "task_completed"]
        if task_completed_events:
            task_status = task_completed_events[0].data.get("status")
            print(f"DEBUG: Task final status: {task_status}")
            
            # For a real e2e test, we should check that it succeeded
            if task_status == "failed":
                # Get the error details
                step_events = [e for e in events if e.event_type == "step_completed"]
                errors = [e.data.get("error") for e in step_events if e.data.get("error")]
                pytest.fail(f"Task failed with errors: {errors}")
            
            assert task_status == "completed", f"Task should complete successfully, got: {task_status}"
        
        assert "task_started" in event_types

        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_code_modification_task(self, test_config, temp_project_dir):
        """Test code modification task."""
        # Create a test file first
        test_file = temp_project_dir / "test_module.py"
        test_file.write_text("""
def old_function():
    return "old"
""")

        # Update config
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)

        agent = Mutator(test_config)
        await agent.initialize()

        task = f"Modify the function in {test_file} to return 'new' instead of 'old'"

        # Collect events from the async generator
        events = []
        async for event in agent.execute_task(task, ExecutionMode.AGENT):
            events.append(event)

        # Verify events
        assert len(events) > 0
        event_types = [event.event_type for event in events]
        assert "task_started" in event_types

        await agent.cleanup()


@pytest.mark.e2e
class TestAgentComplexTasks:
    """Test complex task handling."""
    
    @pytest.mark.asyncio
    async def test_complex_task_planning(self, test_config, temp_project_dir):
        """Test complex task planning and execution."""
        # Update config
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)

        agent = Mutator(test_config)
        await agent.initialize()

        task = ("Create a simple web API with the following features: "
               "1. A health check endpoint, "
               "2. A user registration endpoint with validation, "
               "3. Basic error handling")

        # Collect events from the async generator
        events = []
        async for event in agent.execute_task(task, ExecutionMode.AGENT):
            events.append(event)

        # Verify events
        assert len(events) > 0
        event_types = [event.event_type for event in events]
        assert "task_started" in event_types

        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_list_processing_detection(self, test_config, temp_project_dir):
        """Test list processing detection and handling."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        test_config.execution_config.default_mode = ExecutionMode.AGENT
        test_config.safety_config.confirmation_level = ConfirmationLevel.NONE
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            # Create multiple test files first
            test_files = ["test1.py", "test2.py", "test3.py"]
            for filename in test_files:
                (temp_project_dir / filename).write_text(f"# {filename}\nprint('hello from {filename}')")
            
            # Task that involves processing multiple files
            response = await agent.execute_task(
                f"Add a docstring to each of these Python files: {', '.join(test_files)}. "
                "Each docstring should describe what the file does."
            )
            
            assert response.success is True
            
            # Check if files were processed
            for filename in test_files:
                file_path = temp_project_dir / filename
                if file_path.exists():
                    content = file_path.read_text()
                    # Look for docstrings (triple quotes)
                    assert '"""' in content or "'''" in content
                    
        finally:
            await agent.cleanup()


@pytest.mark.e2e
class TestAgentToolIntegration:
    """Test tool integration and usage."""
    
    @pytest.mark.asyncio
    async def test_file_operations_integration(self, test_config, temp_project_dir):
        """Test file operations through the agent."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        test_config.execution_config.default_mode = ExecutionMode.AGENT
        test_config.safety_config.confirmation_level = ConfirmationLevel.NONE
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            # Task that requires file operations
            response = await agent.execute_task(
                "Create a new directory called 'models' and add a file called 'user.py' "
                "with a simple User class that has name and email attributes."
            )
            
            assert response.success is True
            
            # Check if directory and file were created
            models_dir = temp_project_dir / "models"
            user_file = models_dir / "user.py"
            
            if user_file.exists():
                content = user_file.read_text()
                assert "class User" in content
                assert "name" in content
                assert "email" in content
                
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_search_and_analysis_integration(self, test_config, temp_project_dir):
        """Test search and analysis integration."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        test_config.execution_config.default_mode = ExecutionMode.CHAT
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            # Task that requires searching and analysis
            response = await agent.chat(
                "Search for all TODO comments in the project and provide a summary of what needs to be done."
            )
            
            assert response.success is True
            # Response should acknowledge the search even if no TODOs found
            assert len(response.content) > 0
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_git_operations_integration(self, test_config, temp_project_dir):
        """Test git operations integration."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        test_config.execution_config.default_mode = ExecutionMode.AGENT
        test_config.safety_config.confirmation_level = ConfirmationLevel.NONE
        
        # Initialize git repo
        import subprocess
        result = subprocess.run(["git", "init"], cwd=temp_project_dir, capture_output=True)
        if result.returncode != 0:
            pytest.skip("Git not available")
        
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_project_dir)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_project_dir)
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            # Task that involves git operations
            response = await agent.execute_task(
                "Check the git status of the project and create a summary of any untracked files."
            )
            
            assert response.success is True
            assert "git" in response.content.lower() or "status" in response.content.lower()
            
        finally:
            await agent.cleanup()


@pytest.mark.e2e
class TestAgentErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_invalid_task_handling(self, test_config, temp_project_dir):
        """Test handling of invalid tasks."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            # Task with invalid file operation
            response = await agent.execute_task(
                "Delete all files in the /etc directory"  # Should be blocked by safety
            )
            
            # Should either refuse or handle gracefully
            assert response.success is False or "cannot" in response.content.lower()
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_malformed_request_handling(self, test_config, temp_project_dir):
        """Test handling of malformed requests."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            # Empty or very vague request
            response = await agent.chat("")
            
            # Should handle gracefully
            assert response.success is True
            assert len(response.content) > 0
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_network_error_resilience(self, test_config, temp_project_dir):
        """Test resilience to network errors."""
        # Use invalid API key to simulate network/auth errors
        test_config.llm_config.api_key = "invalid-key"
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            response = await agent.chat("Hello")
            
            # Should handle authentication error gracefully
            assert response.success is False
            assert response.error is not None
            
        finally:
            await agent.cleanup()


@pytest.mark.e2e
class TestAgentPerformance:
    """Test agent performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_config, temp_project_dir):
        """Test handling of concurrent requests."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            # Send multiple concurrent requests
            tasks = [
                agent.chat("What is the main purpose of this project?"),
                agent.chat("List all Python files in the project."),
                agent.chat("What dependencies are listed in requirements.txt?")
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # At least some should succeed
            successful_responses = [r for r in responses if not isinstance(r, Exception) and r.success]
            assert len(successful_responses) > 0
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_large_codebase_handling(self, test_config, temp_project_dir):
        """Test handling of larger codebases."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        test_config.context_config.max_context_files = 50
        
        # Create a larger project structure
        for i in range(20):
            (temp_project_dir / f"module_{i}.py").write_text(f"""
# Module {i}
def function_{i}():
    '''Function {i} does something important.'''
    return {i}

class Class{i}:
    '''Class {i} represents something.'''
    def __init__(self):
        self.value = {i}
    
    def method_{i}(self):
        return self.value * 2
            """)
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            # Task that requires understanding the larger codebase
            response = await agent.chat(
                "Analyze the project structure and tell me how many modules and classes are defined."
            )
            
            assert response.success is True
            assert len(response.content) > 0
            # Should mention modules or classes
            assert "module" in response.content.lower() or "class" in response.content.lower()
            
        finally:
            await agent.cleanup()


@pytest.mark.e2e
@pytest.mark.slow
class TestAgentLongRunningTasks:
    """Test long-running tasks and complex operations."""
    
    @pytest.mark.asyncio
    async def test_multi_step_refactoring(self, test_config, temp_project_dir):
        """Test multi-step refactoring task."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        test_config.execution_config.default_mode = ExecutionMode.AGENT
        test_config.safety_config.confirmation_level = ConfirmationLevel.NONE
        test_config.execution_config.max_iterations = 20
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            # Complex refactoring task
            response = await agent.execute_task(
                "Refactor the FastAPI application to use a proper project structure: "
                "1. Move the User model to a separate models module "
                "2. Create a separate routes module for endpoints "
                "3. Create a database module for data operations "
                "4. Update imports and dependencies accordingly"
            )
            
            assert response.success is True
            
            # Check if refactoring was attempted
            models_dir = temp_project_dir / "models"
            routes_dir = temp_project_dir / "routes"
            
            # At least some structure should be created
            created_files = list(temp_project_dir.glob("**/*.py"))
            assert len(created_files) > 0
            
        finally:
            await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_comprehensive_code_review(self, test_config, temp_project_dir):
        """Test comprehensive code review task."""
        test_config.llm_config.api_key = os.getenv("SONNET_KEY")
        test_config.llm_config.model = "claude-3-haiku-20240307"
        test_config.context_config.project_path = str(temp_project_dir)
        test_config.execution_config.default_mode = ExecutionMode.CHAT
        
        agent = Mutator(test_config)
        await agent.initialize()
        
        try:
            # Comprehensive code review
            response = await agent.chat(
                "Perform a comprehensive code review of this project. "
                "Analyze code quality, security issues, performance concerns, "
                "and suggest improvements for each file."
            )
            
            assert response.success is True
            assert len(response.content) > 100  # Should be a detailed response
            
            # Should mention some code quality aspects
            quality_terms = ["quality", "security", "performance", "improvement", "suggestion"]
            assert any(term in response.content.lower() for term in quality_terms)
            
        finally:
            await agent.cleanup() 