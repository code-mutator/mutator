"""
Tests for enhanced error handling in the Mutator framework.

This module tests various error scenarios and the framework's ability to handle
them gracefully without silent failures.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from mutator.execution.executor import TaskExecutor, GracefulShutdown, _shutdown_handler
from mutator.execution.planner import TaskPlanner
from mutator.context.manager import ContextManager
from mutator.core.config import AgentConfig, LLMConfig, ExecutionConfig
from mutator.core.types import AgentEvent, ExecutionMode
from mutator.llm.client import LLMClient
from mutator.tools.manager import ToolManager


class TestErrorHandling:
    """Test enhanced error handling mechanisms."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AgentConfig(
            llm_config=LLMConfig(
                provider="openai",
                model="gpt-4",
                timeout=30
            ),
            execution_config=ExecutionConfig(
                max_iterations=5,
                timeout=60,
                task_timeout=120
            )
        )
    
    @pytest.fixture
    def executor(self, config):
        """Create TaskExecutor instance for testing."""
        llm_client = Mock(spec=LLMClient)
        tool_manager = Mock(spec=ToolManager)
        context_manager = Mock(spec=ContextManager)
        planner = Mock(spec=TaskPlanner)
        
        executor = TaskExecutor(llm_client, tool_manager, context_manager, planner, config)
        executor.workflow_app = Mock()
        return executor
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, executor):
        """Test that task timeouts are properly handled and reported."""
        # Mock workflow that takes too long
        async def slow_workflow(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            yield {"agent": {"messages": [Mock(content="response")]}}
        
        executor.workflow_app.astream = slow_workflow
        executor.config.execution_config.task_timeout = 1  # 1 second timeout
        
        events = []
        with pytest.raises(TimeoutError, match="Task execution timed out"):
            async for event in executor.execute_task("test task"):
                events.append(event)
        
        # Should have task_started and task_failed events
        assert len(events) >= 2
        assert events[0].event_type == "task_started"
        
        # Find the timeout event
        timeout_event = next((e for e in events if e.event_type == "task_failed"), None)
        assert timeout_event is not None
        assert "timed out" in timeout_event.data["error"].lower()
        assert timeout_event.data["timeout"] == 1
    
    @pytest.mark.asyncio
    async def test_recursion_limit_handling(self, executor):
        """Test that recursion limits are properly enforced."""
        iteration_count = 0
        
        async def infinite_workflow(*args, **kwargs):
            nonlocal iteration_count
            while True:
                iteration_count += 1
                yield {"agent": {"messages": [Mock(content="response", tool_calls=[])]}}
        
        executor.workflow_app.astream = infinite_workflow
        executor.config.execution_config.max_iterations = 3
        
        events = []
        with pytest.raises(RuntimeError, match="exceeded maximum iterations"):
            async for event in executor.execute_task("test task"):
                events.append(event)
                # Safety break to prevent actual infinite loop in test
                if len(events) > 20:
                    break
        
        # Should detect the iteration limit exceeded
        error_event = next((e for e in events if e.event_type == "task_failed"), None)
        assert error_event is not None
        assert "maximum iterations" in error_event.data["error"]
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_handling(self, executor):
        """Test graceful shutdown during task execution."""
        # Mock shutdown request
        original_check_shutdown = _shutdown_handler.check_shutdown
        _shutdown_handler.check_shutdown = Mock(return_value=True)
        
        try:
            async def workflow_with_shutdown_check(*args, **kwargs):
                yield {"agent": {"messages": [Mock(content="response")]}}
            
            executor.workflow_app.astream = workflow_with_shutdown_check
            
            events = []
            with pytest.raises(KeyboardInterrupt, match="interrupted by shutdown"):
                async for event in executor.execute_task("test task"):
                    events.append(event)
            
            # Should have shutdown event
            shutdown_event = next((e for e in events if e.event_type == "task_failed"), None)
            assert shutdown_event is not None
            assert "shutdown request" in shutdown_event.data["error"]
            assert shutdown_event.data.get("shutdown_requested") is True
            
        finally:
            # Restore original function
            _shutdown_handler.check_shutdown = original_check_shutdown
    
    @pytest.mark.asyncio
    async def test_workflow_exception_handling(self, executor):
        """Test handling of exceptions within workflow execution."""
        async def failing_workflow(*args, **kwargs):
            raise ValueError("Simulated workflow failure")
        
        executor.workflow_app.astream = failing_workflow
        
        events = []
        with pytest.raises(ValueError, match="Simulated workflow failure"):
            async for event in executor.execute_task("test task"):
                events.append(event)
        
        # Should have proper error event
        error_event = next((e for e in events if e.event_type == "task_failed"), None)
        assert error_event is not None
        assert "Workflow execution failed" in error_event.data["error"]
        assert error_event.data["error_type"] == "ValueError"
    
    @pytest.mark.asyncio
    async def test_interactive_chat_timeout(self, executor):
        """Test timeout handling in interactive chat."""
        async def slow_chat_workflow(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            yield {"agent": {"messages": [Mock(content="response")]}}
        
        executor.workflow_app.astream = slow_chat_workflow
        executor.config.execution_config.timeout = 1  # 1 second timeout
        
        events = []
        with pytest.raises(TimeoutError, match="Interactive chat timed out"):
            async for event in executor.execute_interactive_chat("test message"):
                events.append(event)
        
        # Should have timeout event
        timeout_event = next((e for e in events if e.event_type == "task_failed"), None)
        assert timeout_event is not None
        assert "timed out" in timeout_event.data["error"].lower()
    
    @pytest.mark.asyncio
    async def test_event_processing_error_handling(self, executor):
        """Test handling of errors during event processing."""
        async def workflow_with_bad_event(*args, **kwargs):
            # Yield an event that will cause processing errors
            yield {"agent": {"messages": [None]}}  # None message should cause issues
        
        executor.workflow_app.astream = workflow_with_bad_event
        
        events = []
        # Should not raise unhandled exception, should yield error events
        async for event in executor.execute_task("test task"):
            events.append(event)
            if len(events) > 5:  # Safety break
                break
        
        # Should have at least task_started event
        assert len(events) >= 1
        assert events[0].event_type == "task_started"
    
    def test_graceful_shutdown_signal_handling(self):
        """Test signal handler setup for graceful shutdown."""
        shutdown_handler = GracefulShutdown()
        
        # Test initial state
        assert not shutdown_handler.check_shutdown()
        
        # Simulate signal
        shutdown_handler._signal_handler(2, None)  # SIGINT
        assert shutdown_handler.check_shutdown()
    
    @pytest.mark.asyncio
    async def test_error_event_data_completeness(self, executor):
        """Test that error events contain comprehensive diagnostic information."""
        async def failing_workflow(*args, **kwargs):
            raise RuntimeError("Test error with context")
        
        executor.workflow_app.astream = failing_workflow
        
        events = []
        with pytest.raises(RuntimeError):
            async for event in executor.execute_task("test task"):
                events.append(event)
        
        # Find error event and check data completeness
        error_event = next((e for e in events if e.event_type == "task_failed"), None)
        assert error_event is not None
        
        error_data = error_event.data
        assert "error" in error_data
        assert "error_type" in error_data
        assert "iterations_completed" in error_data
        assert error_data["error_type"] == "RuntimeError"
        assert "Test error with context" in error_data["error"]
    
    @pytest.mark.asyncio
    async def test_multiple_error_scenarios(self, executor):
        """Test handling of multiple consecutive errors."""
        call_count = 0
        
        async def multi_error_workflow(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First error")
            elif call_count == 2:
                raise RuntimeError("Second error")
            else:
                yield {"agent": {"messages": [Mock(content="final response")]}}
        
        executor.workflow_app.astream = multi_error_workflow
        
        # First call should fail with ValueError
        events1 = []
        with pytest.raises(ValueError, match="First error"):
            async for event in executor.execute_task("test task 1"):
                events1.append(event)
        
        # Second call should fail with RuntimeError
        events2 = []
        with pytest.raises(RuntimeError, match="Second error"):
            async for event in executor.execute_task("test task 2"):
                events2.append(event)
        
        # Both should have proper error events
        error1 = next((e for e in events1 if e.event_type == "task_failed"), None)
        error2 = next((e for e in events2 if e.event_type == "task_failed"), None)
        
        assert error1 is not None and error2 is not None
        assert error1.data["error_type"] == "ValueError"
        assert error2.data["error_type"] == "RuntimeError"


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration with retry enabled."""
        return AgentConfig(
            execution_config=ExecutionConfig(
                retry_on_failure=True,
                max_retry_attempts=3,
                retry_delay=0.1
            )
        )
    
    @pytest.mark.asyncio
    async def test_tool_failure_recovery(self, config):
        """Test recovery from tool failures."""
        # This would test retry mechanisms for tool failures
        # Implementation depends on specific retry logic
        pass
    
    @pytest.mark.asyncio
    async def test_llm_failure_recovery(self, config):
        """Test recovery from LLM API failures."""
        # This would test retry mechanisms for LLM failures
        # Implementation depends on specific retry logic in LLMClient
        pass


if __name__ == "__main__":
    pytest.main([__file__]) 