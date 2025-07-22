"""
Base tool classes and interfaces for the Coding Agent Framework.

This module provides the abstract base classes and interfaces that all tools
must implement, along with safety check functionality.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.types import ToolResult, SafetyCheck, ConfirmationLevel
from ..core.config import ToolConfig


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: Optional[str] = None, description: Optional[str] = None, config: Optional[ToolConfig] = None):
        """Initialize the tool."""
        # Handle both patterns: constructor args or class attributes
        self.name = name or getattr(self, 'name', None)
        self.description = description or getattr(self, 'description', None)
        
        if not self.name:
            raise ValueError("Tool name must be provided either as constructor argument or class attribute")
        if not self.description:
            raise ValueError("Tool description must be provided either as constructor argument or class attribute")
        
        # Extract short description if available
        self.short_description = self._extract_short_description(self.description)
        
        self.config = config or ToolConfig(name=self.name)
        self.enabled = getattr(self, 'enabled', True)  # Default to enabled
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.safety_checks: List[SafetyCheck] = []
    
    def _extract_short_description(self, description: str) -> str:
        """Extract short description from description text."""
        if not description:
            return ""
        
        # Try to extract from <short_description> tags
        import re
        short_match = re.search(r'<short_description>(.*?)</short_description>', 
                               description, re.DOTALL)
        if short_match:
            return short_match.group(1).strip()
        
        # Fallback: use first line or first sentence
        lines = description.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        if len(first_line) > 150:
            # Try to find first sentence
            for ending in ['. ', '! ', '? ']:
                pos = first_line.find(ending)
                if pos > 0 and pos < 150:
                    return first_line[:pos + 1]
            
            # Truncate at word boundary
            words = first_line[:150].split()
            if len(words) > 1:
                words.pop()
                return ' '.join(words) + "..."
            else:
                return first_line[:150] + "..."
        
        return first_line
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "short_description": self.short_description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    
    def get_short_schema(self) -> Dict[str, Any]:
        """Get the short schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.short_description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given parameters."""
        pass
    
    def perform_safety_checks(self, **kwargs) -> List[SafetyCheck]:
        """Perform safety checks before execution."""
        checks = []
        
        # Basic validation
        try:
            self.validate_parameters(**kwargs)
            checks.append(SafetyCheck(
                check_type="parameter_validation",
                passed=True,
                message="Parameters validated successfully"
            ))
        except Exception as e:
            checks.append(SafetyCheck(
                check_type="parameter_validation",
                passed=False,
                message=f"Parameter validation failed: {str(e)}",
                severity="error"
            ))
        
        return checks
    
    def validate_parameters(self, **kwargs) -> None:
        """Validate tool parameters against schema."""
        schema = self.get_schema()
        required = schema.get("function", {}).get("parameters", {}).get("required", [])
        
        # Check required parameters
        for param in required:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")
    
    def needs_confirmation(self, **kwargs) -> bool:
        """Check if this tool execution needs confirmation."""
        return self.config.confirmation_level != ConfirmationLevel.NONE
    
    def get_confirmation_message(self, **kwargs) -> str:
        """Get confirmation message for this tool execution."""
        return f"Execute {self.name} with parameters: {kwargs}"
    
    def add_safety_check(self, check: SafetyCheck) -> None:
        """Add a safety check to this tool."""
        self.safety_checks.append(check)
    
    async def run_safety_checks(self, parameters: Dict[str, Any]) -> List[str]:
        """Run all safety checks and return any warnings."""
        warnings = []
        for check in self.safety_checks:
            try:
                result = await check.check(self.name, parameters)
                if not result.passed:
                    warnings.append(result.message)
            except Exception as e:
                warnings.append(f"Safety check failed: {str(e)}")
        return warnings


class ToolSafetyChecker:
    """Utility class for performing safety checks on tools."""
    
    @staticmethod
    def check_shell_command(command: str, blocked_commands: List[str], allowed_commands: Optional[List[str]] = None) -> SafetyCheck:
        """Check if a shell command is safe to execute."""
        # Check blocked commands
        for blocked_cmd in blocked_commands:
            if blocked_cmd in command:
                return SafetyCheck(
                    check_type="shell",
                    passed=False,
                    message=f"Command contains blocked term: {blocked_cmd}",
                    severity="error"
                )
        
        # Check allowed commands
        if allowed_commands:
            command_parts = command.split()
            if command_parts and command_parts[0] not in allowed_commands:
                return SafetyCheck(
                    check_type="shell",
                    passed=False,
                    message=f"Command not in allowed list: {command_parts[0]}",
                    severity="warning"
                )
        
        return SafetyCheck(
            check_type="shell",
            passed=True,
            message="Command passed safety checks"
        )
    
    @staticmethod
    def check_file_access(file_path: str) -> SafetyCheck:
        """Check if file access is safe."""
        # Check for dangerous paths
        dangerous_paths = ["/etc/passwd", "/etc/shadow", "/root", "/sys", "/proc"]
        for dangerous_path in dangerous_paths:
            if dangerous_path in file_path:
                return SafetyCheck(
                    check_type="file_access",
                    passed=True,  # Allow but warn
                    message=f"Accessing potentially sensitive file: {file_path}",
                    severity="warning"
                )
        
        return SafetyCheck(
            check_type="file_access",
            passed=True,
            message="File access is safe"
        ) 