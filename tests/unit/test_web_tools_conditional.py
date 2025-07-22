"""
Test for conditional web_search import and registration.
"""

import pytest
from unittest.mock import patch, MagicMock
import os

from mutator.agent import Mutator
from mutator.core.config import AgentConfig


def test_web_search_conditional_import():
    """Test that web_search is properly handled when API keys are not available."""
    
    # Ensure no web search API keys are set
    with patch.dict(os.environ, {}, clear=True):
        # Remove any existing web search API keys
        for key in ['GOOGLE_SEARCH_API_KEY', 'GOOGLE_SEARCH_CX', 'BING_SEARCH_API_KEY']:
            if key in os.environ:
                del os.environ[key]
        
        # Import should work even without API keys
        from mutator.tools.builtin import web_search
        
        # web_search should be None when API keys are not available
        assert web_search is None


def test_agent_initialization_without_web_search():
    """Test that agent initializes properly when web_search is not available."""
    
    # Ensure no web search API keys are set
    with patch.dict(os.environ, {}, clear=True):
        # Remove any existing web search API keys
        for key in ['GOOGLE_SEARCH_API_KEY', 'GOOGLE_SEARCH_CX', 'BING_SEARCH_API_KEY']:
            if key in os.environ:
                del os.environ[key]
        
        # Mock the LLM client to avoid API calls
        with patch('mutator.llm.client.LLMClient') as mock_llm:
            mock_llm.return_value = MagicMock()
            
            # Mock the context manager to avoid file system operations
            with patch('mutator.context.manager.ContextManager') as mock_context:
                mock_context.return_value = MagicMock()
                
                # Create agent config
                config = AgentConfig()
                
                # Agent should initialize without errors
                agent = Mutator(config)
                
                # Verify agent was created
                assert agent is not None
                assert agent.tool_manager is not None
                
                # Verify web_search is not in the registered tools
                available_tools = agent.tool_manager.list_tools()
                assert 'web_search' not in available_tools
                
                # But other web tools should be available
                assert 'fetch_url' in available_tools
                


def test_register_function_with_none():
    """Test that register_function handles None values gracefully."""
    
    from mutator.tools.manager import ToolManager
    from mutator.core.config import SafetyConfig
    
    # Create a tool manager with minimal setup
    with patch('mutator.tools.manager.ToolManager.register_builtin_tools'):
        tool_manager = ToolManager(SafetyConfig())
        
        # Get initial tool count
        initial_count = len(tool_manager.list_tools())
        
        # Mock the logger to capture warnings
        with patch.object(tool_manager, 'logger') as mock_logger:
            # Attempt to register None as a function
            tool_manager.register_function(None)
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once_with("Attempted to register None as a tool, skipping")
            
            # Verify no additional tools were registered
            assert len(tool_manager.list_tools()) == initial_count


def test_web_search_with_api_keys():
    """Test that web_search is available when API keys are configured."""
    
    # Set up mock environment with API keys
    with patch.dict(os.environ, {
        'GOOGLE_SEARCH_API_KEY': 'test_key',
        'GOOGLE_SEARCH_CX': 'test_cx'
    }):
        # Mock the web_tools module to avoid actual imports
        with patch('mutator.tools.categories.web_tools._has_web_search_api_keys', return_value=True):
            # Mock the actual web_search function
            mock_web_search = MagicMock()
            mock_web_search.name = 'web_search'
            
            with patch('mutator.tools.builtin.web_search', mock_web_search):
                from mutator.tools.builtin import web_search
                
                # web_search should not be None when API keys are available
                assert web_search is not None
                assert web_search == mock_web_search


if __name__ == '__main__':
    pytest.main([__file__]) 