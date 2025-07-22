"""
Tests for Pydantic output functionality.
"""

import pytest
import asyncio
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from mutator.core.types import TaskResult, AgentEvent
from mutator.core.path_utils import (
    parse_pydantic_output, 
    format_pydantic_for_llm,
    extract_json_from_text,
    clean_json_string,
    create_example_from_schema
)
from mutator.agent import Mutator
from mutator.core.config import AgentConfig


# Test Pydantic models
class BlogPost(BaseModel):
    """Simple blog post model for testing."""
    title: str = Field(..., description="Title of the blog post")
    content: str = Field(..., description="Content of the blog post")
    author: str = Field(..., description="Author of the blog post")
    published: bool = Field(default=False, description="Whether the blog post is published")


class CodeAnalysis(BaseModel):
    """Code analysis model for testing."""
    language: str = Field(..., description="Programming language")
    complexity: int = Field(..., ge=1, le=10, description="Complexity score from 1-10")
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    suggestions: Optional[str] = Field(None, description="Suggestions for improvement")


class NestedModel(BaseModel):
    """Nested model for testing complex structures."""
    id: str
    metadata: dict = Field(default_factory=dict)
    items: List[BlogPost] = Field(default_factory=list)


class TestPydanticUtilities:
    """Test utility functions for Pydantic parsing."""
    
    def test_extract_json_from_text_simple_object(self):
        """Test extracting JSON object from text."""
        text = '''
        Here is the result:
        {"title": "Test", "content": "Test content", "author": "John"}
        That's all.
        '''
        result = extract_json_from_text(text)
        assert result == '{"title": "Test", "content": "Test content", "author": "John"}'
    
    def test_extract_json_from_text_code_block(self):
        """Test extracting JSON from code block."""
        text = '''
        Here is the result:
        ```json
        {
            "title": "Test",
            "content": "Test content",
            "author": "John"
        }
        ```
        '''
        result = extract_json_from_text(text)
        assert result is not None
        assert '"title": "Test"' in result
        assert '"content": "Test content"' in result
        assert '"author": "John"' in result
    
    def test_extract_json_from_text_array(self):
        """Test extracting JSON array from text."""
        text = '''
        Here are the results:
        [{"title": "Test1"}, {"title": "Test2"}]
        '''
        result = extract_json_from_text(text)
        assert result is not None
        # The new implementation prioritizes objects over arrays
        # but should still extract valid JSON structures
        assert '"title": "Test1"' in result or '"title": "Test2"' in result
    
    def test_extract_json_from_text_no_json(self):
        """Test handling text with no JSON."""
        text = "This is just plain text with no JSON"
        result = extract_json_from_text(text)
        assert result is None
    
    def test_clean_json_string_trailing_commas(self):
        """Test cleaning JSON with trailing commas."""
        json_str = '{"title": "Test", "content": "Test content",}'
        result = clean_json_string(json_str)
        assert result == '{"title": "Test", "content": "Test content"}'
    
    def test_clean_json_string_single_quotes(self):
        """Test cleaning JSON with single quotes."""
        json_str = "{'title': 'Test', 'content': 'Test content'}"
        result = clean_json_string(json_str)
        assert result == '{"title": "Test", "content": "Test content"}'
    
    def test_clean_json_string_comments(self):
        """Test cleaning JSON with comments."""
        json_str = '''
        {
            "title": "Test", // This is a comment
            "content": "Test content"
        }
        '''
        result = clean_json_string(json_str)
        assert '// This is a comment' not in result
        assert '"title": "Test"' in result
    
    def test_parse_pydantic_output_success(self):
        """Test successful Pydantic parsing."""
        raw_output = '''
        Here is the blog post:
        {
            "title": "Test Blog Post",
            "content": "This is test content.",
            "author": "John Doe",
            "published": true
        }
        '''
        result = parse_pydantic_output(raw_output, BlogPost)
        assert result is not None
        assert isinstance(result, BlogPost)
        assert result.title == "Test Blog Post"
        assert result.content == "This is test content."
        assert result.author == "John Doe"
        assert result.published is True
    
    def test_parse_pydantic_output_failure(self):
        """Test Pydantic parsing failure."""
        raw_output = "This is just plain text with no JSON"
        result = parse_pydantic_output(raw_output, BlogPost)
        assert result is None
    
    def test_parse_pydantic_output_invalid_json(self):
        """Test handling invalid JSON."""
        raw_output = '{"title": "Test", "missing_required_field": true}'
        result = parse_pydantic_output(raw_output, BlogPost)
        assert result is None
    
    def test_format_pydantic_for_llm(self):
        """Test formatting Pydantic model for LLM."""
        result = format_pydantic_for_llm(BlogPost)
        assert "BlogPost" in result
        assert "title" in result
        assert "content" in result
        assert "author" in result
        assert "published" in result
        assert "JSON" in result
    
    def test_create_example_from_schema(self):
        """Test creating example from Pydantic schema."""
        # Use model_json_schema for Pydantic V2 compatibility
        if hasattr(BlogPost, 'model_json_schema'):
            schema = BlogPost.model_json_schema()
        else:
            schema = BlogPost.schema()
        example = create_example_from_schema(schema)
        assert "title" in example
        assert "content" in example
        assert "author" in example
        assert "published" in example
        assert isinstance(example["published"], bool)


class TestTaskResult:
    """Test TaskResult class functionality."""
    
    def test_task_result_creation_raw_only(self):
        """Test creating TaskResult with raw output only."""
        events = [
            AgentEvent(event_type="task_started", data={"task": "test"}),
            AgentEvent(event_type="task_completed", data={"task": "test"})
        ]
        
        result = TaskResult(
            raw="This is the raw output",
            events=events
        )
        
        assert result.raw == "This is the raw output"
        assert result.pydantic is None
        assert result.json_dict is None
        assert result.success is True
        assert result.output_format == "raw"
    
    def test_task_result_with_pydantic(self):
        """Test TaskResult with Pydantic output."""
        blog_post = BlogPost(
            title="Test",
            content="Test content",
            author="John Doe"
        )
        
        # Use model_dump for Pydantic V2 compatibility
        if hasattr(blog_post, 'model_dump'):
            json_dict = blog_post.model_dump()
        else:
            json_dict = blog_post.dict()
        
        result = TaskResult(
            raw='{"title": "Test", "content": "Test content", "author": "John Doe", "published": false}',
            pydantic=blog_post,
            json_dict=json_dict,
            output_format="pydantic"
        )
        
        assert result.pydantic == blog_post
        assert result.json_dict == json_dict
        assert result.output_format == "pydantic"
    
    def test_task_result_dictionary_access(self):
        """Test dictionary-style access to TaskResult."""
        blog_post = BlogPost(
            title="Test",
            content="Test content", 
            author="John Doe"
        )
        
        # Use model_dump for Pydantic V2 compatibility
        if hasattr(blog_post, 'model_dump'):
            json_dict = blog_post.model_dump()
        else:
            json_dict = blog_post.dict()
        
        result = TaskResult(
            raw="raw content",
            pydantic=blog_post,
            json_dict=json_dict,
            output_format="pydantic"
        )
        
        # Test direct attribute access
        assert result["raw"] == "raw content"
        assert result["success"] is True
        
        # Test Pydantic attribute access
        assert result["title"] == "Test"
        assert result["content"] == "Test content"
        assert result["author"] == "John Doe"
        
        # Test 'in' operator
        assert "title" in result
        assert "nonexistent" not in result
        
        # Test get method
        assert result.get("title") == "Test"
        assert result.get("nonexistent", "default") == "default"
    
    def test_task_result_to_dict(self):
        """Test converting TaskResult to dictionary."""
        blog_post = BlogPost(
            title="Test",
            content="Test content",
            author="John Doe"
        )
        
        # Use model_dump for Pydantic V2 compatibility
        if hasattr(blog_post, 'model_dump'):
            json_dict = blog_post.model_dump()
        else:
            json_dict = blog_post.dict()
        
        result = TaskResult(
            raw="raw content",
            pydantic=blog_post,
            json_dict=json_dict,
            output_format="pydantic"
        )
        
        dict_result = result.to_dict()
        assert dict_result["title"] == "Test"
        assert dict_result["content"] == "Test content"
        assert dict_result["author"] == "John Doe"
    
    def test_task_result_string_representation(self):
        """Test string representation of TaskResult."""
        blog_post = BlogPost(
            title="Test",
            content="Test content",
            author="John Doe"
        )
        
        result = TaskResult(
            raw="raw content",
            pydantic=blog_post,
            output_format="pydantic"
        )
        
        str_result = str(result)
        assert "Test" in str_result
        assert "Test content" in str_result
        assert "John Doe" in str_result


class TestPydanticIntegration:
    """Test integration with the Mutator."""
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration for testing."""
        return AgentConfig(
            working_directory=".",
            llm_config={
                "model": "gpt-3.5-turbo",
                "temperature": 0.1
            }
        )
    
    @pytest.mark.asyncio
    async def test_agent_with_pydantic_output(self, agent_config):
        """Test agent execution with Pydantic output."""
        # Skip if no OpenAI key available
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")
        
        agent = Mutator(agent_config)
        await agent.initialize()
        
        # Simple task that should produce structured output
        task = "Create a simple blog post about Python programming"
        
        events = []
        async for event in agent.execute_task(
            task,
            output_pydantic=BlogPost
        ):
            events.append(event)
        
        # Check that we got the expected events
        assert len(events) > 0
        
        # Look for task completion event
        completion_events = [e for e in events if e.event_type == "task_completed"]
        assert len(completion_events) > 0
        
        # Check that the result has the expected structure
        completion_event = completion_events[0]
        assert "result" in completion_event.data
        
        # The result should be a TaskResult with Pydantic output
        result_data = completion_event.data["result"]
        assert "output_format" in result_data
        assert result_data["raw"] is not None
    
    @pytest.mark.asyncio
    async def test_agent_without_pydantic_output(self, agent_config):
        """Test agent execution without Pydantic output."""
        # Skip if no OpenAI key available
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")
        
        agent = Mutator(agent_config)
        await agent.initialize()
        
        # Simple task without Pydantic output
        task = "Say hello world"
        
        events = []
        async for event in agent.execute_task(task):
            events.append(event)
        
        # Check that we got the expected events
        assert len(events) > 0
        
        # Look for task completion event
        completion_events = [e for e in events if e.event_type == "task_completed"]
        assert len(completion_events) > 0
        
        # Check that the result has raw output
        completion_event = completion_events[0]
        assert "result" in completion_event.data
        
        result_data = completion_event.data["result"]
        assert "output_format" in result_data
        assert result_data["output_format"] == "raw"
        assert result_data["pydantic"] is None
    
    def test_complex_pydantic_model(self):
        """Test with complex nested Pydantic model."""
        raw_output = '''
        {
            "id": "test-123",
            "metadata": {"version": "1.0", "tags": ["test", "example"]},
            "items": [
                {
                    "title": "First Post",
                    "content": "Content of first post",
                    "author": "Author 1",
                    "published": true
                },
                {
                    "title": "Second Post", 
                    "content": "Content of second post",
                    "author": "Author 2",
                    "published": false
                }
            ]
        }
        '''
        
        result = parse_pydantic_output(raw_output, NestedModel)
        assert result is not None
        assert isinstance(result, NestedModel)
        assert result.id == "test-123"
        assert result.metadata == {"version": "1.0", "tags": ["test", "example"]}
        assert len(result.items) == 2
        assert result.items[0].title == "First Post"
        assert result.items[1].title == "Second Post"


if __name__ == "__main__":
    pytest.main([__file__]) 