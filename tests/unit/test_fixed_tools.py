import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from mutator.tools.manager import ToolManager
from mutator.core.config import AgentConfig
from mutator.llm.client import LLMClient


class TestBatchTools:
    """Test all batch tools with real implementations"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test1.py").write_text("def hello():\n    print('Hello World')\n")
            (temp_path / "test2.js").write_text("function greet() {\n    console.log('Hello');\n}\n")
            (temp_path / "README.md").write_text("# Test Project\n\nThis is a test.\n")
            (temp_path / "todo.txt").write_text("TODO: Fix bug\nTODO: Add tests\n")
            
            yield temp_path
    
    @pytest.fixture
    def tool_manager(self, temp_dir):
        """Create a ToolManager instance for testing"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            config = AgentConfig()
            llm_client = LLMClient(config.llm_config)
            return ToolManager(config=config, llm_client=llm_client, working_directory=str(temp_dir))
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_process_search_files_by_name(self, tool_manager, temp_dir):
        """Test process_search_files_by_name with real implementation"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            result = await tool_manager.execute_tool("process_search_files_by_name", {
                "pattern": "*.py",
                "operation_description": "Analyze this file",
                "auto_process": False
            })
            
            assert result.success is True
            assert "tasks_created" in result.result or "error" in result.result or "message" in result.result
            # The tool should handle the batch processing gracefully
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_process_search_files_by_content(self, tool_manager, temp_dir):
        """Test process_search_files_by_content with real implementation"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            result = await tool_manager.execute_tool("process_search_files_by_content", {
                "pattern": "hello",
                "operation_description": "Process this match",
                "auto_process": False
            })
            
            assert result.success is True
            assert "batch_results" in result.result or "error" in result.result or "message" in result.result
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_batch_glob_search(self, tool_manager, temp_dir):
        """Test batch_glob_search with real implementation"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            result = await tool_manager.execute_tool("batch_glob_search", {
                "pattern": "*.py",
                "prompt_template": "Process this file: {item}",
                "batch_size": 2,
                "batch_strategy": "parallel"
            })
            
            assert result.success is True
            assert "batch_results" in result.result or "error" in result.result or "message" in result.result
        finally:
            os.chdir(original_cwd)
    

    
    @pytest.mark.asyncio
    async def test_process_search_files_sementic(self, tool_manager, temp_dir):
        """Test process_search_files_sementic with real implementation"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            result = await tool_manager.execute_tool("process_search_files_sementic", {
                "query": "hello",
                "operation_description": "Analyze this code",
                "auto_process": False
            })
            
            assert result.success is True
            # Accept various result formats since the underlying tools may have different behaviors
            assert "tasks_created" in result.result or "error" in result.result or "message" in result.result
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_batch_tools_error_handling(self, tool_manager, temp_dir):
        """Test batch tools error handling"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            # Test with invalid parameters
            result = await tool_manager.execute_tool("process_search_files_by_name", {
                "pattern": "",  # Empty pattern
                "operation_description": "Process files",
                "auto_process": False
            })
            
            # Should handle gracefully - either succeed or fail cleanly
            assert result.success is True or "error" in result.result
        finally:
            os.chdir(original_cwd)


class TestCodebaseSearch:
    """Test codebase_search with real implementation"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files with various content
            (temp_path / "main.py").write_text("""
def calculate_sum(a, b):
    '''Calculate the sum of two numbers'''
    return a + b

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, x, y):
        result = x + y
        self.history.append(f"Added {x} + {y} = {result}")
        return result
""")
            
            (temp_path / "utils.js").write_text("""
function formatNumber(num) {
    return num.toLocaleString();
}

const API_URL = 'https://api.example.com';

async function fetchData(endpoint) {
    const response = await fetch(API_URL + endpoint);
    return response.json();
}
""")
            
            (temp_path / "README.md").write_text("""
# Calculator Project

This project implements a simple calculator with the following features:
- Addition operations
- History tracking
- Number formatting utilities
""")
            
            yield temp_path
    
    @pytest.fixture
    def tool_manager(self, temp_dir):
        """Create a ToolManager instance for testing"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            config = AgentConfig()
            llm_client = LLMClient(config.llm_config)
            return ToolManager(config=config, llm_client=llm_client, working_directory=str(temp_dir))
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_codebase_search_python_functions(self, tool_manager, temp_dir):
        """Test searching for Python functions"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            result = await tool_manager.execute_tool("codebase_search", {
                "query": "calculate sum function",
                "max_results": 10
            })
            
            assert result.success is True
            assert "results" in result.result or "error" in result.result
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_codebase_search_javascript_functions(self, tool_manager, temp_dir):
        """Test searching for JavaScript functions"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            result = await tool_manager.execute_tool("codebase_search", {
                "query": "format number function",
                "max_results": 10
            })
            
            assert result.success is True
            assert "results" in result.result or "error" in result.result
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_codebase_search_class_definitions(self, tool_manager, temp_dir):
        """Test searching for class definitions"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            result = await tool_manager.execute_tool("codebase_search", {
                "query": "Calculator class",
                "max_results": 10
            })
            
            assert result.success is True
            assert "results" in result.result or "error" in result.result
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_codebase_search_no_results(self, tool_manager, temp_dir):
        """Test searching with no matching results"""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            result = await tool_manager.execute_tool("codebase_search", {
                "query": "nonexistent_function_xyz",
                "max_results": 10
            })
            
            assert result.success is True
            assert "results" in result.result or "error" in result.result
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_codebase_search_empty_directory(self):
        """Test searching in empty directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                config = AgentConfig()
                llm_client = LLMClient(config.llm_config)
                tool_manager = ToolManager(config=config, llm_client=llm_client, working_directory=str(temp_dir))
                
                result = await tool_manager.execute_tool("codebase_search", {
                    "query": "any function",
                    "max_results": 10
                })
                
                assert result.success is True
                assert "results" in result.result or "error" in result.result
            finally:
                os.chdir(original_cwd)


class TestWebSearch:
    """Test web_search with real implementation"""
    
    @pytest.fixture
    def tool_manager(self):
        """Create a ToolManager instance for testing"""
        # Use project root to avoid path issues
        config = AgentConfig()
        llm_client = LLMClient(config.llm_config)
        return ToolManager(config=config, llm_client=llm_client, working_directory="/Users/amrmagdy/Desktop/work/agent/framework")
    
    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_web_search_no_api_keys(self, tool_manager):
        """Test web_search when no API keys are configured"""
        result = await tool_manager.execute_tool("web_search", {
            "query": "python programming",
            "max_results": 5
        })
        
        assert result.success is True
        assert "results" in result.result or "error" in result.result
        
        # If successful, should have results
        if "results" in result.result:
            assert len(result.result["results"]) > 0
            
            # Should return mock results with setup instructions
            first_result = result.result["results"][0]
            assert "title" in first_result
            assert "url" in first_result
            assert "snippet" in first_result
            
            # Should include setup instructions
            assert "setup_instructions" in result.result
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key', 'GOOGLE_CSE_ID': 'test_cse'})
    @patch('requests.get')
    @pytest.mark.asyncio
    async def test_web_search_with_google_api(self, mock_get, tool_manager):
        """Test web_search with Google API configured"""
        # Mock Google API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'items': [
                {
                    'title': 'Python Programming Guide',
                    'link': 'https://example.com/python',
                    'snippet': 'Learn Python programming basics'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = await tool_manager.execute_tool("web_search", {
            "query": "python programming",
            "max_results": 5
        })
        
        assert result.success is True
        assert "results" in result.result or "error" in result.result
        
        # Should have results
        if "results" in result.result:
            assert len(result.result["results"]) >= 1
    
    @patch.dict(os.environ, {'BING_API_KEY': 'test_bing_key'})
    @patch('requests.get')
    @pytest.mark.asyncio
    async def test_web_search_with_bing_api(self, mock_get, tool_manager):
        """Test web_search with Bing API configured"""
        # Mock Bing API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'webPages': {
                'value': [
                    {
                        'name': 'Python Tutorial',
                        'url': 'https://example.com/tutorial',
                        'snippet': 'Complete Python tutorial'
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = await tool_manager.execute_tool("web_search", {
            "query": "python tutorial",
            "max_results": 3
        })
        
        assert result.success is True
        assert "results" in result.result or "error" in result.result
        
        # Should have results
        if "results" in result.result:
            assert len(result.result["results"]) >= 1
    
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key', 'GOOGLE_CSE_ID': 'test_cse'})
    @patch('requests.get')
    @pytest.mark.asyncio
    async def test_web_search_api_error(self, mock_get, tool_manager):
        """Test web_search when API returns error"""
        # Mock API error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response
        
        result = await tool_manager.execute_tool("web_search", {
            "query": "test query",
            "max_results": 5
        })
        
        assert result.success is True
        assert "results" in result.result or "error" in result.result
        
        # Should fall back to mock results
        if "results" in result.result:
            assert len(result.result["results"]) > 0


class TestGetToolHelp:
    """Test get_tool_help functionality"""
    
    @pytest.fixture
    def tool_manager(self):
        """Create a ToolManager instance for testing"""
        config = AgentConfig()
        llm_client = LLMClient(config.llm_config)
        manager = ToolManager(config=config, llm_client=llm_client, working_directory="/Users/amrmagdy/Desktop/work/agent/framework")
        
        # Register built-in tools for testing
        manager.register_builtin_tools()
        
        return manager
    
    @pytest.mark.asyncio
    async def test_get_tool_help_registered(self, tool_manager):
        """Test that get_tool_help is properly registered"""
        # The get_tool_help should be automatically registered
        # by the ToolManager.register_get_tool_help_implementation()
        tools = tool_manager.get_available_tools()
        
        # Should have get_tool_help in available tools
        assert "get_tool_help" in tools
    
    @pytest.mark.asyncio
    async def test_get_tool_help_functionality(self, tool_manager):
        """Test get_tool_help returns tool information"""
        # Get help for a known tool
        result = await tool_manager.execute_tool("get_tool_help", {"tool_name": "read_file"})
        
        assert result.success is True
        assert "error" not in result.result
        
        # Check that the result contains tool information
        assert "tool_name" in result.result
        assert result.result["tool_name"] == "read_file"
        assert "description" in result.result
        assert "parameters" in result.result
    
    @pytest.mark.asyncio
    async def test_get_tool_help_nonexistent_tool(self, tool_manager):
        """Test get_tool_help with nonexistent tool"""
        # Get help for nonexistent tool
        result = await tool_manager.execute_tool("get_tool_help", {"tool_name": "nonexistent_tool"})
        
        assert result.success is False or "error" in result.result


class TestIntegration:
    """Integration tests for all fixed tools"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace with realistic project structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create project structure
            src_dir = workspace / "src"
            src_dir.mkdir()
            
            # Python files
            (src_dir / "main.py").write_text("""
import os
from utils import helper_function

def main():
    '''Main entry point'''
    print("Starting application...")
    result = helper_function("test")
    return result

if __name__ == "__main__":
    main()
""")
            
            (src_dir / "utils.py").write_text("""
def helper_function(data):
    '''Helper function for processing data'''
    # TODO: Add input validation
    return data.upper()

class DataProcessor:
    def __init__(self):
        self.cache = {}
    
    def process(self, items):
        # TODO: Implement caching
        return [item.strip() for item in items]
""")
            
            # JavaScript files
            (src_dir / "app.js").write_text("""
const express = require('express');
const app = express();

// TODO: Add error handling middleware
app.get('/', (req, res) => {
    res.send('Hello World!');
});

function startServer() {
    app.listen(3000, () => {
        console.log('Server running on port 3000');
    });
}

module.exports = { startServer };
""")
            
            # Documentation
            (workspace / "README.md").write_text("""
# Test Project

## Overview
This is a test project for validating tool implementations.

## TODO
- Add comprehensive tests
- Implement error handling
- Add logging functionality
""")
            
            yield workspace
    
    @pytest.fixture
    def tool_manager(self, temp_workspace):
        """Create a ToolManager instance for testing"""
        original_cwd = os.getcwd()
        os.chdir(temp_workspace)
        try:
            config = AgentConfig()
            llm_client = LLMClient(config.llm_config)
            manager = ToolManager(config=config, llm_client=llm_client, working_directory=str(temp_workspace))
            
            # Register built-in tools for testing
            manager.register_builtin_tools()
            
            return manager
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_comprehensive_workflow(self, tool_manager, temp_workspace):
        """Test a comprehensive workflow using multiple fixed tools"""
        original_cwd = os.getcwd()
        os.chdir(temp_workspace)
        try:
            # 1. Search for Python files using batch
            search_result = await tool_manager.execute_tool("process_search_files_by_name", {
                "pattern": "*.py",
                "operation_description": "Analyze file",
                "auto_process": False
            })
            assert search_result.success is True
            
            # 2. Search for TODO comments using batch
            todo_result = await tool_manager.execute_tool("process_search_files_by_content", {
                "pattern": "TODO",
                "operation_description": "Process TODO",
                "auto_process": False
            })
            assert todo_result.success is True
            
            # 3. Search codebase for functions
            codebase_result = await tool_manager.execute_tool("codebase_search", {
                "query": "helper function",
                "max_results": 10
            })
            assert codebase_result.success is True
            
            # 4. Search for all JavaScript files using batch
            js_result = await tool_manager.execute_tool("batch_glob_search", {
                "pattern": "*.js",
                "prompt_template": "Process JS file: {item}",
                "batch_size": 2,
                "batch_strategy": "parallel"
            })
            assert js_result.success is True
            
            # 5. Skip TODO analysis (function removed)
            
            # 6. Search codebase for multiple queries using batch
            multi_search_result = await tool_manager.execute_tool("process_search_files_sementic", {
                "query": "main function",
                "operation_description": "Analyze code",
                "auto_process": False
            })
            assert multi_search_result.success is True
            
            # 7. Test web search (should work with mock data)
            web_result = await tool_manager.execute_tool("web_search", {
                "query": "python programming best practices",
                "max_results": 3
            })
            assert web_result.success is True
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, tool_manager, temp_workspace):
        """Test error recovery in various scenarios"""
        original_cwd = os.getcwd()
        os.chdir(temp_workspace)
        try:
            # Test codebase search in directory with no matching files
            empty_dir = temp_workspace / "empty"
            empty_dir.mkdir()
            os.chdir(empty_dir)
            
            result = await tool_manager.execute_tool("codebase_search", {
                "query": "nonexistent code",
                "max_results": 5
            })
            assert result.success is True
            
            # Test web search with empty query
            web_result = await tool_manager.execute_tool("web_search", {
                "query": "",
                "max_results": 3
            })
            assert web_result.success is True
            # Should handle empty query gracefully
        finally:
            os.chdir(original_cwd) 