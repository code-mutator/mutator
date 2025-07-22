"""
Unit tests for core types module.
"""

import pytest
from datetime import datetime
from mutator.core.types import (
    TaskType, ExecutionMode, TaskStatus, ConfirmationLevel,
    AgentEvent, ToolCall, ToolResult, LLMResponse, TaskPlan, PlanStep,
    ConversationTurn, AgentMemory, SafetyCheck, ContextItem, ContextType
)


class TestEnums:
    """Test enum types."""
    
    def test_task_type_enum(self):
        """Test TaskType enum."""
        assert TaskType.SIMPLE.value == "simple"
        assert TaskType.COMPLEX.value == "complex"
        assert len(TaskType) == 2
    
    def test_execution_mode_enum(self):
        """Test ExecutionMode enum."""
        assert ExecutionMode.CHAT.value == "chat"
        assert ExecutionMode.AGENT.value == "agent"
        assert len(ExecutionMode) == 2
    
    def test_task_status_enum(self):
        """Test TaskStatus enum."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert len(TaskStatus) == 4
    
    def test_confirmation_level_enum(self):
        """Test ConfirmationLevel enum."""
        assert ConfirmationLevel.NONE.value == "none"
        assert ConfirmationLevel.LOW.value == "low"
        assert ConfirmationLevel.MEDIUM.value == "medium"
        assert ConfirmationLevel.HIGH.value == "high"
        assert len(ConfirmationLevel) == 4


class TestAgentEvent:
    """Test AgentEvent model."""
    
    def test_agent_event_creation(self):
        """Test creating an AgentEvent."""
        event = AgentEvent(
            event_type="test_event",
            data={"key": "value"},
            message="Test message",
            level="info"
        )
        
        assert event.event_type == "test_event"
        assert event.data == {"key": "value"}
        assert event.message == "Test message"
        assert event.level == "info"
        assert isinstance(event.timestamp, datetime)
    
    def test_agent_event_default_timestamp(self):
        """Test AgentEvent with default timestamp."""
        event = AgentEvent(
            event_type="test_event"
        )
        
        assert event.event_type == "test_event"
        assert event.data == {}
        assert event.message is None
        assert event.level == "info"
        assert isinstance(event.timestamp, datetime)


class TestToolCall:
    """Test ToolCall model."""
    
    def test_tool_call_creation(self):
        """Test creating a ToolCall."""
        tool_call = ToolCall(
            id="call_123",
            name="test_tool",
            arguments={"param": "value"},
            description="Test tool call"
        )
        
        assert tool_call.id == "call_123"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments == {"param": "value"}
        assert tool_call.description == "Test tool call"
        assert tool_call.confirmation_level == ConfirmationLevel.NONE
    
    def test_tool_call_custom_id(self):
        """Test ToolCall with custom ID."""
        tool_call = ToolCall(
            id="custom_id",
            name="test_tool",
            arguments={"param": "value"}
        )
        
        assert tool_call.id == "custom_id"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments == {"param": "value"}
    
    def test_tool_call_backward_compatibility(self):
        """Test ToolCall backward compatibility with 'parameters' field."""
        tool_call = ToolCall(
            id="call_123",
            name="test_tool",
            parameters={"param": "value"}
        )
        
        assert tool_call.id == "call_123"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments == {"param": "value"}
    
    def test_tool_call_parameters_override(self):
        """Test that 'arguments' takes precedence over 'parameters'."""
        tool_call = ToolCall(
            id="call_123",
            name="test_tool",
            arguments={"arg": "value"},
            parameters={"param": "value"}
        )
        
        assert tool_call.arguments == {"arg": "value"}
    
    def test_tool_call_backward_compatibility_parameters(self):
        """Test ToolCall backward compatibility - parameters field should be converted to arguments."""
        # Test the __init__ method handles 'parameters' correctly
        tool_call = ToolCall(id="call_123", name="test_tool", parameters={"param": "value"})
        
        assert tool_call.id == "call_123"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments == {"param": "value"}


class TestToolResult:
    """Test ToolResult model."""
    
    def test_successful_tool_result(self):
        """Test successful ToolResult."""
        result = ToolResult(
            tool_name="test_tool",
            success=True,
            result={"output": "success"},
            execution_time=1.5
        )
        
        assert result.tool_name == "test_tool"
        assert result.success is True
        assert result.result == {"output": "success"}
        assert result.execution_time == 1.5
        assert result.error is None
        assert isinstance(result.timestamp, datetime)
    
    def test_failed_tool_result(self):
        """Test failed ToolResult."""
        result = ToolResult(
            tool_name="test_tool",
            success=False,
            error="Tool execution failed"
        )
        
        assert result.tool_name == "test_tool"
        assert result.success is False
        assert result.error == "Tool execution failed"
        assert result.result is None


class TestLLMResponse:
    """Test LLMResponse model."""
    
    def test_successful_llm_response(self):
        """Test successful LLMResponse."""
        response = LLMResponse(
            content="Hello, world!",
            finish_reason="stop",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5}
        )
        
        assert response.content == "Hello, world!"
        assert response.finish_reason == "stop"
        assert response.model == "gpt-4"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}
        assert response.success is True
        assert response.error is None
        assert isinstance(response.timestamp, datetime)
    
    def test_llm_response_with_tool_calls(self):
        """Test LLMResponse with tool calls."""
        tool_call = ToolCall(id="call_123", name="test_tool", parameters={"param": "value"})
        response = LLMResponse(
            content="I'll help you with that.",
            tool_calls=[tool_call]
        )
        
        assert response.content == "I'll help you with that."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "test_tool"
    
    def test_failed_llm_response(self):
        """Test failed LLMResponse."""
        response = LLMResponse(
            content="",
            success=False,
            error="API request failed"
        )
        
        assert response.content == ""
        assert response.success is False
        assert response.error == "API request failed"


class TestPlanStep:
    """Test PlanStep model."""
    
    def test_plan_step_creation(self):
        """Test creating a PlanStep."""
        step = PlanStep(
            id="step_1",
            description="First step",
            type=TaskType.SIMPLE,
            status=TaskStatus.PENDING,
            estimated_duration=30,
            tools_required=["tool1", "tool2"]
        )
        
        assert step.id == "step_1"
        assert step.description == "First step"
        assert step.type == TaskType.SIMPLE
        assert step.status == TaskStatus.PENDING
        assert step.estimated_duration == 30
        assert step.tools_required == ["tool1", "tool2"]
        assert isinstance(step.created_at, datetime)
    
    def test_plan_step_with_dependencies(self):
        """Test PlanStep with dependencies."""
        step = PlanStep(
            id="step_2",
            description="Second step",
            dependencies=["step_1"],
            status=TaskStatus.PENDING
        )
        
        assert step.id == "step_2"
        assert step.dependencies == ["step_1"]
        assert step.status == TaskStatus.PENDING
    
    def test_plan_step_list_processing(self):
        """Test PlanStep with list processing."""
        step = PlanStep(
            id="step_3",
            description="Process list",
            is_list_processing=True
        )
        
        assert step.id == "step_3"
        assert step.is_list_processing is True


class TestTaskPlan:
    """Test TaskPlan model."""
    
    def test_task_plan_creation(self):
        """Test creating a TaskPlan."""
        step1 = PlanStep(
            id="step_1",
            description="First step",
            status=TaskStatus.PENDING
        )
        
        step2 = PlanStep(
            id="step_2",
            description="Second step",
            status=TaskStatus.PENDING
        )
        
        plan = TaskPlan(
            id="plan_1",
            description="Test plan",
            steps=[step1, step2],
            status=TaskStatus.PENDING
        )
        
        assert plan.id == "plan_1"
        assert plan.description == "Test plan"
        assert len(plan.steps) == 2
        assert plan.status == TaskStatus.PENDING
        assert isinstance(plan.created_at, datetime)
    
    def test_empty_task_plan(self):
        """Test creating an empty TaskPlan."""
        plan = TaskPlan(
            id="empty_plan",
            description="Empty plan",
            steps=[]
        )
        
        assert plan.id == "empty_plan"
        assert plan.description == "Empty plan"
        assert len(plan.steps) == 0
        assert plan.status == TaskStatus.PENDING


class TestConversationTurn:
    """Test ConversationTurn model."""
    
    def test_conversation_turn_creation(self):
        """Test creating a ConversationTurn."""
        turn = ConversationTurn(
            id="turn_1",
            role="user",
            content="Hello, assistant!",
            metadata={"source": "test"}
        )
        
        assert turn.id == "turn_1"
        assert turn.role == "user"
        assert turn.content == "Hello, assistant!"
        assert turn.metadata == {"source": "test"}
        assert isinstance(turn.timestamp, datetime)
    
    def test_conversation_turn_with_context(self):
        """Test ConversationTurn with context."""
        turn = ConversationTurn(
            id="turn_2",
            role="assistant",
            content="Hello, user!",
            context={"task": "greeting"}
        )
        
        assert turn.id == "turn_2"
        assert turn.role == "assistant"
        assert turn.content == "Hello, user!"
        assert turn.context == {"task": "greeting"}


class TestSafetyCheck:
    """Test SafetyCheck model."""
    
    def test_passed_safety_check(self):
        """Test passed SafetyCheck."""
        check = SafetyCheck(
            check_type="file_access",
            passed=True,
            message="File access is safe",
            severity="info"
        )
        
        assert check.check_type == "file_access"
        assert check.passed is True
        assert check.message == "File access is safe"
        assert check.severity == "info"
    
    def test_failed_safety_check(self):
        """Test failed SafetyCheck."""
        check = SafetyCheck(
            check_type="shell_command",
            passed=False,
            message="Dangerous command detected",
            severity="error"
        )
        
        assert check.check_type == "shell_command"
        assert check.passed is False
        assert check.message == "Dangerous command detected"
        assert check.severity == "error"


class TestContextItem:
    """Test ContextItem model."""
    
    def test_context_item_creation(self):
        """Test creating a ContextItem."""
        item = ContextItem(
            type=ContextType.FILE,
            content="def hello(): pass",
            metadata={"language": "python"},
            relevance_score=0.8,
            source="test.py"
        )
        
        assert item.type == ContextType.FILE
        assert item.content == "def hello(): pass"
        assert item.metadata == {"language": "python"}
        assert item.relevance_score == 0.8
        assert item.source == "test.py"
    
    def test_context_item_with_relevance_score(self):
        """Test ContextItem with relevance score."""
        item = ContextItem(
            type=ContextType.FUNCTION,
            content="function implementation",
            relevance_score=0.95
        )
        
        assert item.type == ContextType.FUNCTION
        assert item.content == "function implementation"
        assert item.relevance_score == 0.95


class TestAgentMemory:
    """Test AgentMemory model."""
    
    def test_agent_memory_creation(self):
        """Test creating AgentMemory."""
        memory = AgentMemory(
            working_directory="/test/path",
            environment_variables={"TEST_VAR": "value"}
        )
        
        assert memory.working_directory == "/test/path"
        assert memory.environment_variables == {"TEST_VAR": "value"}
        assert len(memory.conversation_history) == 0
        # AgentMemory doesn't have last_updated field
    
    def test_agent_memory_with_conversation_data(self):
        """Test AgentMemory with conversation data."""
        event_dict = {
            "event_type": "test",
            "timestamp": "2023-01-01T00:00:00",
            "data": {}
        }
        
        memory = AgentMemory(
            execution_history=[event_dict]  # execution_history expects dict, not AgentEvent
        )
        
        assert len(memory.execution_history) == 1
        assert memory.execution_history[0]["event_type"] == "test" 