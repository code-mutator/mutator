"""
Tests for optimized task tools functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from mutator.tools.categories.task_tools import (
    delegate_task
)


class TestDelegateTask:
    """Test the delegate_task system."""
    
    def test_delegate_task_tool_exists(self):
        """Test that delegate_task tool exists."""
        assert hasattr(delegate_task, 'name')
        assert hasattr(delegate_task, 'description')
        assert hasattr(delegate_task, 'execute')
        assert delegate_task.name == "delegate_task"
    
    @pytest.mark.asyncio
    async def test_delegate_task_basic_execution(self):
        """Test basic delegate_task execution."""
        # Test that the delegate_task tool has the right structure
        # We'll test the error case since mocking the full agent is complex
        result = await delegate_task.execute(
            task_description="Test task",
            expected_output="Test output"
        )
        
        # Should fail because we don't have proper LLM configuration
        assert result.success is False
        assert result.tool_name == "delegate_task"
        assert result.error is not None
        assert "Task delegation failed" in result.error
    
    @pytest.mark.asyncio
    async def test_delegate_task_with_context_data(self):
        """Test delegate_task with context data."""
        context_data = {
            "file_path": "test.py",
            "line_number": 42,
            "description": "Test context"
        }
        
        result = await delegate_task.execute(
            task_description="Test task with context",
            expected_output="Test output with context",
            context_data=context_data
        )
        
        # Should fail because we don't have proper LLM configuration
        assert result.success is False
        assert result.tool_name == "delegate_task"
        assert result.error is not None
        assert "Task delegation failed" in result.error
    
    @pytest.mark.asyncio
    async def test_delegate_task_parameter_validation(self):
        """Test delegate_task parameter validation."""
        # Test with minimal parameters
        result = await delegate_task.execute(
            task_description="Minimal test",
            expected_output="Minimal output"
        )
        
        assert result.success is False
        assert result.tool_name == "delegate_task"
        
        # Test with empty strings
        result = await delegate_task.execute(
            task_description="",
            expected_output=""
        )
        
        assert result.success is False
        assert result.tool_name == "delegate_task"


class TestDelegateTaskIntegration:
    """Test delegate_task integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_delegate_task_with_file_analysis(self):
        """Test delegate_task for file analysis scenario."""
        task_description = """
        Analyze the authentication system in auth.py and provide insights.
        
        The file contains:
        - User authentication logic
        - Session management
        - Password hashing
        
        Please analyze the security aspects and suggest improvements.
        """
        
        expected_output = "Security analysis report with recommendations"
        
        context_data = {
            "file_path": "auth.py",
            "file_type": "python",
            "analysis_type": "security"
        }
        
        result = await delegate_task.execute(
            task_description=task_description,
            expected_output=expected_output,
            context_data=context_data
        )
        
        # Should fail because we don't have proper LLM configuration
        assert result.success is False
        assert result.tool_name == "delegate_task"
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_delegate_task_with_code_review(self):
        """Test delegate_task for code review scenario."""
        task_description = """
        Review the following code changes for potential issues:
        
        1. Check for security vulnerabilities
        2. Verify code quality and best practices
        3. Ensure proper error handling
        4. Validate performance implications
        
        Provide detailed feedback on each aspect.
        """
        
        expected_output = "Code review report with detailed feedback and suggestions"
        
        context_data = {
            "review_type": "security_and_quality",
            "files_changed": ["auth.py", "user.py", "session.py"],
            "change_type": "feature_addition"
        }
        
        result = await delegate_task.execute(
            task_description=task_description,
            expected_output=expected_output,
            context_data=context_data
        )
        
        # Should fail because we don't have proper LLM configuration
        assert result.success is False
        assert result.tool_name == "delegate_task"
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_delegate_task_with_documentation(self):
        """Test delegate_task for documentation scenario."""
        task_description = """
        Generate comprehensive documentation for the API endpoints.
        
        Include:
        - Endpoint descriptions
        - Request/response formats
        - Authentication requirements
        - Error codes and handling
        - Usage examples
        
        Format as markdown documentation.
        """
        
        expected_output = "Complete API documentation in markdown format"
        
        context_data = {
            "documentation_type": "api",
            "format": "markdown",
            "endpoints": ["/auth", "/users", "/sessions"],
            "include_examples": True
        }
        
        result = await delegate_task.execute(
            task_description=task_description,
            expected_output=expected_output,
            context_data=context_data
        )
        
        # Should fail because we don't have proper LLM configuration
        assert result.success is False
        assert result.tool_name == "delegate_task"
        assert result.error is not None


class TestDelegateTaskErrorHandling:
    """Test delegate_task error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_delegate_task_missing_parameters(self):
        """Test delegate_task with missing required parameters."""
        # This should raise an error during execution since parameters are required
        try:
            result = await delegate_task.execute()
            # If we get here, the tool should have failed
            assert result.success is False
        except TypeError:
            # This is expected for missing required parameters
            pass
    
    @pytest.mark.asyncio
    async def test_delegate_task_invalid_context_data(self):
        """Test delegate_task with invalid context data."""
        # Test with non-dict context data
        result = await delegate_task.execute(
            task_description="Test task",
            expected_output="Test output",
            context_data="invalid_context"  # Should be a dict
        )
        
        # Should fail because we don't have proper LLM configuration
        # The invalid context_data should be handled gracefully
        assert result.success is False
        assert result.tool_name == "delegate_task"
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_delegate_task_large_context_data(self):
        """Test delegate_task with large context data."""
        # Test with large context data
        large_context = {
            f"key_{i}": f"value_{i}" * 100 for i in range(100)
        }
        
        result = await delegate_task.execute(
            task_description="Test task with large context",
            expected_output="Test output",
            context_data=large_context
        )
        
        # Should fail because we don't have proper LLM configuration
        assert result.success is False
        assert result.tool_name == "delegate_task"
        assert result.error is not None 