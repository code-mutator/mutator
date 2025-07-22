"""
Core types and data structures for the Coding Agent Framework.

This module defines all the fundamental types, enums, and data structures
used throughout the framework, including task types, execution modes,
and various event and result types.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Type
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
import json


class TaskType(str, Enum):
    """Types of tasks that can be handled by the coding agent."""
    SIMPLE = "simple"
    COMPLEX = "complex"


class ExecutionMode(str, Enum):
    """Execution modes for the coding agent."""
    CHAT = "chat"      # Read-only mode
    AGENT = "agent"    # Can modify codebase


class ConfirmationLevel(str, Enum):
    """Levels of confirmation required for operations."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskStatus(str, Enum):
    """Status of a task or plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ContextType(str, Enum):
    """Types of context items."""
    FILE = "file"
    DIRECTORY = "directory"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    DOCUMENTATION = "documentation"
    ERROR = "error"
    COMMIT = "commit"


# Type aliases for common function signatures
ConfirmationCallback = Callable[[str, str], bool]


class AgentEvent(BaseModel):
    """Event emitted by the agent during execution."""
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None
    level: str = "info"  # debug, info, warning, error, critical


class ToolCall(BaseModel):
    """Represents a tool call from the LLM."""
    id: str
    name: str = Field(..., description="The name of the function to call.")
    arguments: Dict[str, Any] = Field(..., description="The arguments to pass to the function.")
    call_id: Optional[str] = None
    confirmation_level: ConfirmationLevel = ConfirmationLevel.NONE
    description: Optional[str] = None
    
    def __init__(self, **data):
        # Handle both 'parameters' and 'arguments' for backward compatibility
        if 'parameters' in data and 'arguments' not in data:
            data['arguments'] = data.pop('parameters')
        super().__init__(**data)


class ContextItem(BaseModel):
    """Represents a piece of context for the agent."""
    type: ContextType  # Changed from 'item_type' to 'type' to match tests
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0
    source: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


class PlanStep(BaseModel):
    """Represents a step in a task plan."""
    id: str  # Changed from 'step_id' to 'id' to match tests
    description: str
    type: TaskType = TaskType.SIMPLE
    dependencies: List[str] = Field(default_factory=list)  # Changed to List[str] to match tests
    status: TaskStatus = TaskStatus.PENDING
    estimated_duration: Optional[int] = None  # in minutes
    tools_required: List[str] = Field(default_factory=list)
    context_required: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None  # Added to match tests
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)  # Added to match tests
    is_list_processing: bool = False  # Added to match tests
    estimated_tools: List[str] = Field(default_factory=list)  # Added to match executor usage
    notes: Optional[str] = None  # Added to match executor usage
    
    def __init__(self, **data):
        # Handle both 'step_id' and 'id' for backward compatibility
        if 'step_id' in data and 'id' not in data:
            data['id'] = str(data.pop('step_id'))
        super().__init__(**data)


class TaskPlan(BaseModel):
    """Represents a complete task plan."""
    id: str  # Changed from 'task_id' to 'id' to match tests
    description: str  # Changed from 'task_description' to 'description' to match tests
    steps: List[PlanStep]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None  # Added to match tests
    status: TaskStatus = TaskStatus.PENDING
    estimated_duration: Optional[int] = None
    current_step: Optional[str] = None
    context: List[ContextItem] = Field(default_factory=list)
    task_type: TaskType = TaskType.SIMPLE  # Added to match tests
    error: Optional[str] = None  # Added to match executor usage
    
    def __init__(self, **data):
        # Handle both 'task_description' and 'description' for backward compatibility
        if 'task_description' in data and 'description' not in data:
            data['description'] = data.pop('task_description')
        super().__init__(**data)
    
    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current step being executed."""
        if self.current_step:
            return next((s for s in self.steps if s.id == self.current_step), None)
        return None
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next step to execute."""
        for step in self.steps:
            if step.status == TaskStatus.PENDING:
                # Check if dependencies are met
                if all(
                    any(s.id == dep_id and s.status == TaskStatus.COMPLETED for s in self.steps)
                    for dep_id in step.dependencies
                ):
                    return step
        return None


class SafetyCheck(BaseModel):
    """Represents a safety check result."""
    check_type: str
    passed: bool
    message: str
    severity: str = "info"  # debug, info, warning, error, critical
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of a tool execution."""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    safety_checks: List[SafetyCheck] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)  # Added to match tests
    
    def __getitem__(self, key):
        """Allow dictionary-style access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self.result if isinstance(self.result, dict) else False:
            return self.result[key]
        else:
            raise KeyError(f"'{key}' not found in ToolResult")
    
    def __contains__(self, key):
        """Support 'in' operator for backward compatibility."""
        return hasattr(self, key) or (isinstance(self.result, dict) and key in self.result)
    
    def get(self, key, default=None):
        """Dictionary-style get method for backward compatibility."""
        try:
            return self[key]
        except KeyError:
            return default


class LLMResponse(BaseModel):
    """Response from an LLM."""
    content: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    # Added fields to match tests
    success: bool = True
    error: Optional[str] = None
    
    def extract_code_blocks(self) -> List[Tuple[str, str]]:
        """Extract code blocks from the response content."""
        import re
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, self.content, re.DOTALL)
        return [(lang or 'text', code.strip()) for lang, code in matches]
    
    def has_list_processing(self) -> bool:
        """Check if the response indicates list processing is needed."""
        indicators = [
            "for each", "one by one", "process each", "iterate through",
            "for every", "loop through", "process individually"
        ]
        return any(indicator in self.content.lower() for indicator in indicators)


class ConversationTurn(BaseModel):
    """Represents a turn in the conversation."""
    id: str
    role: str  # "user", "assistant", "system"
    content: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Added fields to match tests
    user_message: Optional[str] = None
    assistant_response: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    def __init__(self, **data):
        # Handle backward compatibility for field names
        if 'user_message' in data and 'content' not in data:
            data['content'] = data['user_message']
            data['role'] = 'user'
        if 'assistant_response' in data and 'content' not in data:
            data['content'] = data['assistant_response']
            data['role'] = 'assistant'
        if 'id' not in data:
            data['id'] = str(int(datetime.now().timestamp() * 1000000))
        super().__init__(**data)


class AgentMemory(BaseModel):
    """Represents the agent's memory state."""
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    current_context: List[ContextItem] = Field(default_factory=list)
    active_plan: Optional[TaskPlan] = None
    working_directory: str = "."
    file_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    git_state: Optional[Dict[str, Any]] = None
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    # Added field to match tests
    execution_history: List[Dict[str, Any]] = Field(default_factory=list)
    context_cache: Dict[str, Any] = Field(default_factory=dict)
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn to history."""
        self.conversation_history.append(turn)
    
    def get_recent_turns(self, limit: int = 10) -> List[ConversationTurn]:
        """Get recent conversation turns."""
        return self.conversation_history[-limit:]
    
    def compress_history(self, target_size: int = 5) -> None:
        """Compress conversation history to target size."""
        if len(self.conversation_history) > target_size:
            # Keep first and last turns, summarize middle
            self.conversation_history = (
                self.conversation_history[:2] + 
                self.conversation_history[-target_size+2:]
            )


class TaskResult(BaseModel):
    """Result of a task execution with optional Pydantic output support."""
    raw: str = Field(..., description="Raw output from the task")
    pydantic: Optional[BaseModel] = Field(default=None, description="Structured Pydantic output if output_pydantic was used")
    json_dict: Optional[Dict[str, Any]] = Field(default=None, description="JSON dictionary representation if applicable")
    success: bool = Field(default=True, description="Whether the task completed successfully")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
    execution_time: float = Field(default=0.0, description="Time taken to execute the task")
    events: List[AgentEvent] = Field(default_factory=list, description="Events generated during task execution")
    output_format: str = Field(default="raw", description="Format of the output (raw, pydantic, json)")
    
    class Config:
        arbitrary_types_allowed = True  # Allow Pydantic models as field values
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access for backward compatibility."""
        # First check if it's a direct attribute
        if hasattr(self, key):
            return getattr(self, key)
        # If pydantic output exists, try to get from there
        if self.pydantic and hasattr(self.pydantic, key):
            return getattr(self.pydantic, key)
        # If json_dict exists, try to get from there
        if self.json_dict and key in self.json_dict:
            return self.json_dict[key]
        raise KeyError(f"'{key}' not found in TaskResult")
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for backward compatibility."""
        return (
            hasattr(self, key) or 
            (self.pydantic and hasattr(self.pydantic, key)) or
            (self.json_dict and key in self.json_dict)
        )
    
    def get(self, key: str, default=None) -> Any:
        """Dictionary-style get method for backward compatibility."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        if self.pydantic:
            # Use model_dump for Pydantic V2 compatibility
            if hasattr(self.pydantic, 'model_dump'):
                return self.pydantic.model_dump()
            else:
                # Fallback for older Pydantic versions
                return self.pydantic.dict()
        elif self.json_dict:
            return self.json_dict
        else:
            return {"raw": self.raw}
    
    def __str__(self) -> str:
        """String representation prioritizing structured output."""
        if self.pydantic:
            return str(self.pydantic)
        elif self.json_dict:
            return json.dumps(self.json_dict, indent=2)
        else:
            return self.raw


# Export all types for easier imports
__all__ = [
    "TaskType", "ExecutionMode", "ConfirmationLevel", "TaskStatus", "ContextType",
    "AgentEvent", "ToolCall", "ContextItem", "PlanStep", "TaskPlan",
    "SafetyCheck", "ToolResult", "LLMResponse", "ConversationTurn", "AgentMemory",
    "TaskResult", "ConfirmationCallback"
] 