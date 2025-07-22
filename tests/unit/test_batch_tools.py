"""
Tests for batch tools functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestProcessSearchFilesByName:
    """Test process_search_files_by_name function."""
    
    @pytest.mark.asyncio
    async def test_process_search_files_by_name_success(self):
        """Test successful process search files by name operation."""
        from mutator.tools.batch_tools import process_search_files_by_name
        
        # Mock the search tool
        mock_search_result = Mock()
        mock_search_result.success = True
        mock_search_result.result = {
            "matches": [
                {"file": "test1.py", "size": 1000},
                {"file": "test2.py", "size": 2000}
            ]
        }
        
        # Mock the delegate_task tool
        mock_delegate_result = Mock()
        mock_delegate_result.success = True
        mock_delegate_result.result = {
            "success": True,
            "summary": "Processed 2 files successfully",
            "tool_calls_made": 5,
            "successful_tool_calls": 5,
            "failed_tool_calls": 0
        }
        
        with patch('mutator.tools.categories.search_tools.search_files_by_name') as mock_search, \
             patch('mutator.tools.categories.task_tools.delegate_task') as mock_delegate:
            
            mock_search.execute = AsyncMock(return_value=mock_search_result)
            mock_delegate.execute = AsyncMock(return_value=mock_delegate_result)
            
            result = await process_search_files_by_name.execute(
                pattern="test.*\\.py",
                operation_description="Add type hints"
            )
            
            assert result["success"] is True
            assert result["total_files_found"] == 2
            assert result["total_groups"] == 1
            assert result["successful_groups"] == 1
            assert result["failed_groups"] == 0
            assert "Processed 2 files in 1 groups" in result["summary"]
            
            # Verify delegate_task was called
            mock_delegate.execute.assert_called_once()
            call_args = mock_delegate.execute.call_args
            assert "Add type hints" in call_args[1]["task_description"]
            assert call_args[1]["expected_output"] == "Summary of processing 2 files with details of what was accomplished for each file"
    
    @pytest.mark.asyncio
    async def test_process_search_files_by_name_no_matches(self):
        """Test process search files by name with no matches."""
        from mutator.tools.batch_tools import process_search_files_by_name
        
        # Mock the search tool with no matches
        mock_search_result = Mock()
        mock_search_result.success = True
        mock_search_result.result = {"matches": []}
        
        with patch('mutator.tools.categories.search_tools.search_files_by_name') as mock_search:
            mock_search.execute = AsyncMock(return_value=mock_search_result)
            
            result = await process_search_files_by_name.execute(
                pattern="nonexistent.*\\.py",
                operation_description="Add type hints"
            )
            
            assert result["success"] is True  # Tool executed successfully
            assert result.result["success"] is False  # But found no matches
            assert "No files found matching the pattern" in result.result["message"]
    
    @pytest.mark.asyncio
    async def test_process_search_files_by_name_max_results(self):
        """Test process search files by name with max_results limit."""
        from mutator.tools.batch_tools import process_search_files_by_name
        
        # Mock the search tool with many matches
        mock_search_result = Mock()
        mock_search_result.success = True
        mock_search_result.result = {
            "matches": [
                {"file": f"test{i}.py", "size": 1000} for i in range(10)
            ]
        }
        
        # Mock the delegate_task tool
        mock_delegate_result = Mock()
        mock_delegate_result.success = True
        mock_delegate_result.result = {
            "success": True,
            "summary": "Processed 5 files successfully",
            "tool_calls_made": 10,
            "successful_tool_calls": 10,
            "failed_tool_calls": 0
        }
        
        with patch('mutator.tools.categories.search_tools.search_files_by_name') as mock_search, \
             patch('mutator.tools.categories.task_tools.delegate_task') as mock_delegate:
            
            mock_search.execute = AsyncMock(return_value=mock_search_result)
            mock_delegate.execute = AsyncMock(return_value=mock_delegate_result)
            
            result = await process_search_files_by_name.execute(
                pattern="test.*\\.py",
                operation_description="Add type hints",
                max_results=5
            )
            
            assert result["success"] is True
            assert result["total_files_found"] == 5  # Limited by max_results
            assert result["total_groups"] == 1
            assert result["successful_groups"] == 1
            assert result["failed_groups"] == 0
    
    @pytest.mark.asyncio
    async def test_process_search_files_by_name_delegate_processing_results(self):
        """Test process search files by name with delegate_processing_results=False."""
        from mutator.tools.batch_tools import process_search_files_by_name
        
        # Mock the search tool
        mock_search_result = Mock()
        mock_search_result.success = True
        mock_search_result.result = {
            "matches": [
                {"file": "test1.py", "size": 1000},
                {"file": "test2.py", "size": 2000}
            ]
        }
        
        with patch('mutator.tools.categories.search_tools.search_files_by_name') as mock_search:
            mock_search.execute = AsyncMock(return_value=mock_search_result)
            
            result = await process_search_files_by_name.execute(
                pattern="test.*\\.py",
                operation_description="Add type hints",
                delegate_processing_results=False
            )
            
            assert result["success"] is True
            assert result["files_found"] == 2
            assert "matches" in result


class TestProcessSearchFilesByContent:
    """Test process_search_files_by_content function."""
    
    @pytest.mark.asyncio
    async def test_process_search_files_by_content_success(self):
        """Test successful process search files by content operation."""
        from mutator.tools.batch_tools import process_search_files_by_content
        
        # Mock the search tool
        mock_search_result = Mock()
        mock_search_result.success = True
        mock_search_result.result = {
            "matches": [
                {"file": "test1.py", "line_number": 10, "line_content": "old_function()"},
                {"file": "test2.py", "line_number": 25, "line_content": "old_function()"}
            ]
        }
        
        # Mock the delegate_task tool
        mock_delegate_result = Mock()
        mock_delegate_result.success = True
        mock_delegate_result.result = {
            "success": True,
            "summary": "Processed 2 matches successfully",
            "tool_calls_made": 4,
            "successful_tool_calls": 4,
            "failed_tool_calls": 0
        }
        
        with patch('mutator.tools.categories.search_tools.search_files_by_content') as mock_search, \
             patch('mutator.tools.categories.task_tools.delegate_task') as mock_delegate:
            
            mock_search.execute = AsyncMock(return_value=mock_search_result)
            mock_delegate.execute = AsyncMock(return_value=mock_delegate_result)
            
            result = await process_search_files_by_content.execute(
                pattern="old_function",
                operation_description="Replace with new_function"
            )
            
            assert result["success"] is True
            assert result["total_matches_found"] == 2
            assert result["total_groups"] == 1
            assert result["successful_groups"] == 1
            assert result["failed_groups"] == 0
            assert "Processed 2 matches in 1 groups" in result["summary"]
            
            # Verify delegate_task was called
            mock_delegate.execute.assert_called_once()
            call_args = mock_delegate.execute.call_args
            assert "Replace with new_function" in call_args[1]["task_description"]
    
    @pytest.mark.asyncio
    async def test_process_search_files_by_content_delegate_processing_results(self):
        """Test process search files by content with delegate_processing_results=False."""
        from mutator.tools.batch_tools import process_search_files_by_content
        
        # Mock the search tool
        mock_search_result = Mock()
        mock_search_result.success = True
        mock_search_result.result = {
            "matches": [
                {"file": "test1.py", "line_number": 10, "line_content": "old_function()"},
                {"file": "test2.py", "line_number": 25, "line_content": "old_function()"}
            ]
        }
        
        with patch('mutator.tools.categories.search_tools.search_files_by_content') as mock_search:
            mock_search.execute = AsyncMock(return_value=mock_search_result)
            
            result = await process_search_files_by_content.execute(
                pattern="old_function",
                operation_description="Replace with new_function",
                delegate_processing_results=False
            )
            
            assert result["success"] is True
            assert result["matches_found"] == 2
            assert "matches" in result


class TestProcessSearchFilesSementic:
    """Test process_search_files_sementic function."""
    
    @pytest.mark.asyncio
    async def test_process_search_files_sementic_success(self):
        """Test successful process search files sementic operation."""
        from mutator.tools.batch_tools import process_search_files_sementic
        
        # Mock the semantic_search function
        mock_search_result = Mock()
        mock_search_result.success = True
        mock_search_result.result = {
            "results": [
                {"file": "auth.py", "score": 0.9, "context": "authentication logic"},
                {"file": "login.py", "score": 0.8, "context": "login handling"}
            ]
        }
        
        # Mock the delegate_task tool
        mock_delegate_result = Mock()
        mock_delegate_result.success = True
        mock_delegate_result.result = {
            "success": True,
            "summary": "Processed 2 semantic results successfully",
            "tool_calls_made": 6,
            "successful_tool_calls": 6,
            "failed_tool_calls": 0
        }
        
        with patch('mutator.tools.categories.ai_tools.search_files_sementic') as mock_search, \
             patch('mutator.tools.categories.task_tools.delegate_task') as mock_delegate:
            
            mock_search.execute = AsyncMock(return_value=mock_search_result)
            mock_delegate.execute = AsyncMock(return_value=mock_delegate_result)
            
            result = await process_search_files_sementic.execute(
                query="authentication logic",
                operation_description="Refactor authentication system"
            )
            
            assert result["success"] is True
            assert result["total_results_found"] == 2
            assert result["total_groups"] == 1
            assert result["successful_groups"] == 1
            assert result["failed_groups"] == 0
            assert "Processed 2 semantic search results in 1 groups" in result["summary"]
            
            # Verify delegate_task was called
            mock_delegate.execute.assert_called_once()
            call_args = mock_delegate.execute.call_args
            assert "Refactor authentication system" in call_args[1]["task_description"]
    
    @pytest.mark.asyncio
    async def test_process_search_files_sementic_delegate_processing_results(self):
        """Test process search files sementic with delegate_processing_results=False."""
        from mutator.tools.batch_tools import process_search_files_sementic
        
        # Mock the semantic_search function
        mock_search_result = Mock()
        mock_search_result.success = True
        mock_search_result.result = {
            "results": [
                {"file": "auth.py", "score": 0.9, "context": "authentication logic"},
                {"file": "login.py", "score": 0.8, "context": "login handling"}
            ]
        }
        
        with patch('mutator.tools.categories.ai_tools.search_files_sementic') as mock_search:
            mock_search.execute = AsyncMock(return_value=mock_search_result)
            
            result = await process_search_files_sementic.execute(
                query="authentication logic",
                operation_description="Refactor authentication system",
                delegate_processing_results=False
            )
            
            assert result["success"] is True
            assert result["results_found"] == 2
            assert "results" in result 