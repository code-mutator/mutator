"""
Unit tests for disabled tools functionality.
"""

import pytest
from mutator.tools.manager import ToolManager
from mutator.tools.registry import ToolRegistry
from mutator.tools.decorator import tool
from mutator.core.config import AgentConfig
from mutator.agent import Mutator


class TestDisabledTools:
    """Test disabled tools functionality."""
    
    def test_tool_manager_with_disabled_tools(self):
        """Test ToolManager with disabled tools."""
        disabled_tools = ["test_tool", "another_tool"]
        # Use a fresh registry to avoid global state issues
        registry = ToolRegistry()
        manager = ToolManager(disabled_tools=disabled_tools, registry=registry)
        
        assert manager.disabled_tools == {"test_tool", "another_tool"}
        assert manager.is_tool_disabled("test_tool")
        assert manager.is_tool_disabled("another_tool")
        assert not manager.is_tool_disabled("enabled_tool")
    
    def test_register_disabled_tool(self):
        """Test that disabled tools are not registered."""
        registry = ToolRegistry()
        manager = ToolManager(disabled_tools=["test_tool"], registry=registry)
        
        @tool
        def test_tool(param: str) -> str:
            """Test tool that should be disabled."""
            return f"Result: {param}"
        
        # Try to register the disabled tool
        manager.register_function(test_tool)
        
        # Tool should not be registered
        assert "test_tool" not in manager.tools
        assert "test_tool" not in manager.execution_stats
    
    def test_register_enabled_tool(self):
        """Test that enabled tools are registered normally."""
        registry = ToolRegistry()
        manager = ToolManager(disabled_tools=["other_tool"], registry=registry)
        
        @tool
        def test_tool(param: str) -> str:
            """Test tool that should be enabled."""
            return f"Result: {param}"
        
        # Register the enabled tool
        manager.register_function(test_tool)
        
        # Tool should be registered
        assert "test_tool" in manager.tools
        assert "test_tool" in manager.execution_stats
    
    def test_disable_tool_runtime(self):
        """Test disabling a tool at runtime."""
        registry = ToolRegistry()
        manager = ToolManager(registry=registry)
        
        @tool
        def test_tool(param: str) -> str:
            """Test tool."""
            return f"Result: {param}"
        
        # Register the tool
        manager.register_function(test_tool)
        assert "test_tool" in manager.tools
        
        # Disable the tool
        manager.disable_tool("test_tool")
        assert "test_tool" not in manager.tools
        assert manager.is_tool_disabled("test_tool")
    
    def test_enable_tool_runtime(self):
        """Test enabling a tool at runtime."""
        registry = ToolRegistry()
        manager = ToolManager(disabled_tools=["test_tool"], registry=registry)
        
        # Tool should be disabled
        assert manager.is_tool_disabled("test_tool")
        
        # Enable the tool
        manager.enable_tool("test_tool")
        assert not manager.is_tool_disabled("test_tool")
    
    def test_get_disabled_tools(self):
        """Test getting list of disabled tools."""
        disabled_tools = ["tool1", "tool2", "tool3"]
        registry = ToolRegistry()
        manager = ToolManager(disabled_tools=disabled_tools, registry=registry)
        
        result = manager.get_disabled_tools()
        assert set(result) == set(disabled_tools)
    
    def test_agent_config_with_disabled_tools(self):
        """Test AgentConfig with disabled tools."""
        config = AgentConfig(
            disabled_tools=["web_search", "fetch_url", "run_shell"]
        )
        
        assert config.disabled_tools == ["web_search", "fetch_url", "run_shell"]
    
    def test_builtin_tools_registration_with_disabled_tools(self):
        """Test that built-in tools respect disabled_tools configuration."""
        registry = ToolRegistry()
        manager = ToolManager(disabled_tools=["read_file"], registry=registry)
        
        # Register built-in tools
        manager.register_builtin_tools()
        
        # Disabled tools should not be registered
        assert "read_file" not in manager.tools
        
        # Other tools should be registered
        assert "search_files_by_name" in manager.tools
        assert "run_shell" in manager.tools
    
    def test_web_tools_disabled_scenario(self):
        """Test the specific use case of disabling web tools for custom integrations."""
        # This replaces the git tools test case
        disabled_web_tools = ["web_search", "fetch_url"]
        registry = ToolRegistry()
        manager = ToolManager(disabled_tools=disabled_web_tools, registry=registry)
        
        # Register built-in tools
        manager.register_builtin_tools()
        
        # Web tools should be disabled
        for tool_name in disabled_web_tools:
            assert tool_name not in manager.tools
            assert manager.is_tool_disabled(tool_name)
        
        # Other tools should still be available
        assert "read_file" in manager.tools
        assert "run_shell" in manager.tools
    
    @pytest.mark.asyncio
    async def test_agent_with_disabled_tools(self):
        """Test Mutator with disabled tools configuration."""
        config = AgentConfig(
            disabled_tools=["web_search", "fetch_url", "run_shell"]
        )
        
        agent = Mutator(config)
        await agent.initialize()
        
        # Check that disabled tools are not available
        available_tools = agent.tool_manager.list_tools()
        assert "web_search" not in available_tools
        assert "fetch_url" not in available_tools
        assert "run_shell" not in available_tools
        
        # Check that other tools are still available
        assert "read_file" in available_tools 