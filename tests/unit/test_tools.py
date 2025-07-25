"""
Unit tests for tools module.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from mutator.tools import BaseTool, ToolManager, MCPServer, tool, SimpleTool
from mutator.tools.categories.file_tools import (
    read_file, edit_file, create_file
)
from mutator.tools.categories.search_tools import (
    search_files_by_name, search_files_by_content, list_directory
)
from mutator.tools.categories.system_tools import (
    run_shell
)
from mutator.tools.categories.task_tools import (
    delegate_task
)

from mutator.core.types import ToolCall, ToolResult, SafetyCheck
from mutator.core.config import SafetyConfig, ConfirmationLevel


class TestBaseTool:
    """Test BaseTool abstract class."""
    
    def test_base_tool_cannot_be_instantiated(self):
        """Test that BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTool()
    
    def test_base_tool_subclass(self):
        """Test creating a subclass of BaseTool."""
        class TestTool(BaseTool):
            name = "test_tool"
            description = "A test tool"
            
            async def execute(self, **kwargs):
                return {"result": "test"}
        
        tool = TestTool()
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.enabled is True
    
    @pytest.mark.asyncio
    async def test_base_tool_execute_method(self):
        """Test that subclasses must implement execute method."""
        class TestTool(BaseTool):
            name = "test_tool"
            description = "A test tool"
            
            async def execute(self, **kwargs):
                return {"result": "test"}
        
        tool = TestTool()
        result = await tool.execute()
        assert result == {"result": "test"}


class TestSimpleTool:
    """Test SimpleTool class."""
    
    def test_simple_tool_creation(self):
        """Test creating a SimpleTool."""
        def test_func():
            return "test"
        
        tool = SimpleTool("test_tool", "A test tool", test_func)
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.function == test_func
    
    @pytest.mark.asyncio
    async def test_simple_tool_execute(self):
        """Test executing a SimpleTool."""
        def test_func(x, y):
            return x + y
        
        tool = SimpleTool("test_tool", "A test tool", test_func)
        result = await tool.execute(x=1, y=2)
        assert result == 3
    
    @pytest.mark.asyncio
    async def test_simple_tool_execute_async(self):
        """Test executing a SimpleTool with async function."""
        async def test_func(x, y):
            return x + y
        
        tool = SimpleTool("test_tool", "A test tool", test_func)
        result = await tool.execute(x=1, y=2)
        assert result == 3


class TestToolManager:
    """Test ToolManager class."""
    
    def test_tool_manager_creation(self):
        """Test creating a ToolManager."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        assert manager.safety_config == safety_config
        assert len(manager.tools) == 0
    
    def test_register_tool(self):
        """Test registering a tool."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        
        class TestTool(BaseTool):
            name = "test_tool"
            description = "A test tool"
            
            async def execute(self, **kwargs):
                return {"result": "test"}
        
        tool = TestTool()
        manager.register_tool(tool)
        assert "test_tool" in manager.tools
        assert manager.tools["test_tool"] == tool
    
    def test_register_function(self):
        """Test registering a function as a tool."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        
        def test_func():
            return "test"
        
        manager.register_function("test_func", "A test function", test_func)
        assert "test_func" in manager.tools
        assert isinstance(manager.tools["test_func"], SimpleTool)
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        
        def test_func():
            return "test"
        
        manager.register_function("test_func", "A test function", test_func)
        assert "test_func" in manager.tools
        
        manager.unregister_tool("test_func")
        assert "test_func" not in manager.tools
    
    def test_get_tool(self):
        """Test getting a tool."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        
        def test_func():
            return "test"
        
        manager.register_function("test_func", "A test function", test_func)
        tool = manager.get_tool("test_func")
        assert tool is not None
        assert tool.name == "test_func"
    
    def test_list_tools(self):
        """Test listing tools."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        
        def test_func1():
            return "test1"
        
        def test_func2():
            return "test2"
        
        manager.register_function("test_func1", "A test function", test_func1)
        manager.register_function("test_func2", "A test function", test_func2)
        
        tools = manager.list_tools()
        assert "test_func1" in tools
        assert "test_func2" in tools
        assert len(tools) == 2
    
    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a tool."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        
        def test_func(x, y):
            return x + y
        
        manager.register_function("test_func", "A test function", test_func)
        
        result = await manager.execute_tool("test_func", {"x": 1, "y": 2})
        assert result.success is True
        assert result.result == 3
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing a non-existent tool."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        
        result = await manager.execute_tool("nonexistent", {})
        assert result.success is False
        assert "Tool 'nonexistent' not found" in result.error
    
    def test_disable_tool(self):
        """Test disabling a tool."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        
        def test_func():
            return "test"
        
        manager.register_function("test_func", "A test function", test_func)
        manager.disable_tool("test_func")
        
        assert manager.is_tool_disabled("test_func") is True
    
    def test_enable_tool(self):
        """Test enabling a tool."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        
        def test_func():
            return "test"
        
        manager.register_function("test_func", "A test function", test_func)
        manager.disable_tool("test_func")
        manager.enable_tool("test_func")
        
        assert manager.is_tool_disabled("test_func") is False
    
    @pytest.mark.asyncio
    async def test_execute_disabled_tool(self):
        """Test executing a disabled tool."""
        safety_config = SafetyConfig()
        manager = ToolManager(safety_config)
        
        def test_func():
            return "test"
        
        manager.register_function("test_func", "A test function", test_func)
        manager.disable_tool("test_func")
        
        result = await manager.execute_tool("test_func", {})
        assert result.success is False
        assert "Tool 'test_func' is disabled" in result.error


class TestMCPServer:
    """Test MCPServer class."""
    
    def test_mcp_server_creation(self):
        """Test creating an MCPServer."""
        config = {
            "command": "test",
            "args": ["arg1", "arg2"],
            "env": {"TEST": "value"}
        }
        server = MCPServer("test_server", config)
        assert server.name == "test_server"
        assert server.config == config
        assert server.process is None
    
    @pytest.mark.asyncio
    async def test_mcp_server_start_stop(self):
        """Test starting and stopping an MCP server."""
        config = {
            "command": "echo",
            "args": ["test"],
            "env": {}
        }
        server = MCPServer("test_server", config)
        
        # Mock the process
        with patch('asyncio.create_subprocess_exec') as mock_create:
            mock_process = Mock()
            mock_process.pid = 12345
            mock_process.returncode = None
            mock_process.terminate = Mock()
            mock_process.wait = AsyncMock(return_value=0)
            mock_create.return_value = mock_process
            
            await server.start()
            assert server.process == mock_process
            assert server.is_running() is True
            
            await server.stop()
            mock_process.terminate.assert_called_once()
            mock_process.wait.assert_called_once()


class TestToolDecorator:
    """Test @tool decorator."""
    
    def test_tool_decorator_basic(self):
        """Test basic usage of @tool decorator."""
        @tool
        def test_func():
            return "test"
        
        assert hasattr(test_func, 'name')
        assert hasattr(test_func, 'description')
        assert hasattr(test_func, 'parameters')
        assert hasattr(test_func, 'execute')
    
    def test_tool_decorator_with_params(self):
        """Test @tool decorator with parameters."""
        @tool
        def test_func(x: int, y: str = "default"):
            return f"{x}: {y}"
        
        assert test_func.name == "test_func"
        assert "x" in test_func.parameters
        assert "y" in test_func.parameters
        assert test_func.parameters["y"]["default"] == "default"
    
    @pytest.mark.asyncio
    async def test_tool_decorator_execute(self):
        """Test executing a decorated tool."""
        @tool
        def test_func(x: int, y: int):
            return x + y
        
        result = await test_func.execute(x=1, y=2)
        assert result == 3
    
    @pytest.mark.asyncio
    async def test_tool_decorator_async_func(self):
        """Test @tool decorator with async function."""
        @tool
        async def test_func(x: int, y: int):
            return x + y
        
        result = await test_func.execute(x=1, y=2)
        assert result == 3


class TestFileTools:
    """Test file operation tools."""
    
    def test_read_file_tool_exists(self):
        """Test that read_file tool exists and has correct attributes."""
        assert hasattr(read_file, 'name')
        assert hasattr(read_file, 'description')
        assert hasattr(read_file, 'execute')
    
    def test_edit_file_tool_exists(self):
        """Test that edit_file tool exists and has correct attributes."""
        assert hasattr(edit_file, 'name')
        assert hasattr(edit_file, 'description')
        assert hasattr(edit_file, 'execute')
    
    def test_create_file_tool_exists(self):
        """Test that create_file tool exists and has correct attributes."""
        assert hasattr(create_file, 'name')
        assert hasattr(create_file, 'description')
        assert hasattr(create_file, 'execute')
    
    @pytest.mark.asyncio
    async def test_read_file_execution(self):
        """Test reading a file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            f.flush()
            
            result = await read_file.execute(file_path=f.name)
            # read_file now returns line-prefixed content
            assert result.success is True
            assert "1\t|\ttest content" in result.result["content"]
    
    @pytest.mark.asyncio
    async def test_create_file_execution(self):
        """Test creating a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            
            result = await create_file.execute(
                file_path=str(file_path),
                full_content="test content"
            )
            assert result.success is True
            assert file_path.exists()
            assert file_path.read_text() == "test content"


class TestSearchTools:
    """Test search tools."""
    
    def test_search_files_by_name_tool_exists(self):
        """Test that search_files_by_name tool exists."""
        assert hasattr(search_files_by_name, 'name')
        assert hasattr(search_files_by_name, 'description')
        assert hasattr(search_files_by_name, 'execute')
    
    def test_search_files_by_content_tool_exists(self):
        """Test that search_files_by_content tool exists."""
        assert hasattr(search_files_by_content, 'name')
        assert hasattr(search_files_by_content, 'description')
        assert hasattr(search_files_by_content, 'execute')
    
    def test_list_directory_tool_exists(self):
        """Test that list_directory tool exists."""
        assert hasattr(list_directory, 'name')
        assert hasattr(list_directory, 'description')
        assert hasattr(list_directory, 'execute')
    
    @pytest.mark.asyncio
    async def test_list_directory_execution(self):
        """Test listing directory contents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test")
            
            result = await list_directory.execute(directory_path=temp_dir)
            assert result["success"] is True
            assert len(result["items"]) >= 1
            assert any(item["name"] == "test.txt" for item in result["items"])


class TestSystemTools:
    """Test system tools."""
    
    def test_run_shell_tool_exists(self):
        """Test that run_shell tool exists."""
        assert hasattr(run_shell, 'name')
        assert hasattr(run_shell, 'description')
        assert hasattr(run_shell, 'execute')
    
    @pytest.mark.asyncio
    async def test_run_shell_execution(self):
        """Test running a shell command."""
        result = await run_shell.execute(command="echo 'test'")
        assert result["success"] is True
        assert "test" in result["output"]


class TestTaskTools:
    """Test task management tools."""
    
    def test_delegate_task_tool_exists(self):
        """Test that delegate_task tool exists."""
        assert hasattr(delegate_task, 'name')
        assert hasattr(delegate_task, 'description')
        assert hasattr(delegate_task, 'execute')
    
    @pytest.mark.asyncio
    async def test_delegate_task_execution(self):
        """Test delegating a task."""
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


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)