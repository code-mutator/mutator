# output_pydantic Guide - mutator Coding Agent Framework

## Overview

The `output_pydantic` feature allows you to get structured, validated output from the mutator Coding Agent Framework using Pydantic models. 
output_pydantic` functionality, enabling you to define the exact structure and validation rules for your task outputs.

## Key Benefits

- **Structured Output**: Get well-defined, typed results instead of raw text
- **Data Validation**: Automatic validation of output against your Pydantic model
- **Type Safety**: Full type hints and IDE autocompletion support
- **Integration Ready**: Easy to integrate with other systems and APIs
- **Backwards Compatible**: Works alongside existing raw output functionality

## Basic Usage

### 1. Define Your Pydantic Model

First, create a Pydantic model that defines the structure of your expected output:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class BlogPost(BaseModel):
    title: str = Field(..., description="The title of the blog post")
    content: str = Field(..., description="The main content of the blog post")
    author: str = Field(..., description="The author of the blog post")
    published: bool = Field(default=False, description="Whether the blog post is published")
    tags: List[str] = Field(default_factory=list, description="List of tags for the blog post")
```

### 2. Execute Task with Pydantic Output

Pass your Pydantic model to the `execute_task` method:

```python
from mutator.agent import Mutator
from mutator.core.config import AgentConfig
from mutator.core.types import ExecutionMode

async def main():
    # Create agent
    config = AgentConfig(working_directory=".")
    agent = Mutator(config)
    await agent.initialize()
    
    # Execute task with Pydantic output
    task = "Create a blog post about Python programming"
    
    events = []
    async for event in agent.execute_task(
        task,
        execution_mode=ExecutionMode.AGENT,
        output_pydantic=BlogPost  # Pass your Pydantic model
    ):
        events.append(event)
    
    # Access the structured result
    completion_events = [e for e in events if e.event_type == "task_completed"]
    if completion_events:
        result_data = completion_events[0].data["result"]
        
        # result_data is a TaskResult object with:
        # - raw: Original text output
        # - pydantic: Validated Pydantic model instance
        # - json_dict: Dictionary representation
        # - output_format: Format indicator
        
        if result_data["output_format"] == "pydantic":
            print("✅ Successfully parsed as Pydantic model!")
            print(f"Title: {result_data['title']}")
            print(f"Author: {result_data['author']}")
            print(f"Tags: {result_data['tags']}")
```

## Advanced Examples

### Complex Nested Models

```python
from typing import List
from pydantic import BaseModel, Field

class FileInfo(BaseModel):
    path: str = Field(..., description="Relative path to the file")
    type: str = Field(..., description="Type of file")
    size: int = Field(..., description="Size in bytes")
    description: str = Field(..., description="File description")

class ProjectStructure(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    description: str = Field(..., description="Project description")
    main_language: str = Field(..., description="Primary programming language")
    files: List[FileInfo] = Field(default_factory=list, description="Important files")
    dependencies: List[str] = Field(default_factory=list, description="Main dependencies")

# Use the nested model
async for event in agent.execute_task(
    "Analyze the project structure",
    output_pydantic=ProjectStructure
):
    # Handle events...
```

### Data Validation with Constraints

```python
from pydantic import BaseModel, Field, validator

class CodeAnalysis(BaseModel):
    language: str = Field(..., description="Programming language")
    complexity: int = Field(..., ge=1, le=10, description="Complexity score 1-10")
    issues: List[str] = Field(default_factory=list, description="Issues found")
    maintainability: str = Field(..., description="Maintainability rating")
    
    @validator('maintainability')
    def validate_maintainability(cls, v):
        if v.lower() not in ['high', 'medium', 'low']:
            raise ValueError('Maintainability must be High, Medium, or Low')
        return v.title()
    
    @validator('language')
    def validate_language(cls, v):
        return v.lower()
```

## TaskResult Object

When using `output_pydantic`, the framework returns a `TaskResult` object with the following properties:

```python
class TaskResult(BaseModel):
    raw: str                                    # Original text output
    pydantic: Optional[BaseModel]               # Validated Pydantic instance
    json_dict: Optional[Dict[str, Any]]         # JSON dictionary representation
    success: bool                               # Whether task completed successfully
    error: Optional[str]                        # Error message if failed
    execution_time: float                       # Time taken to execute
    events: List[AgentEvent]                    # Events during execution
    output_format: str                          # "raw", "pydantic", or "json"
```

### Accessing TaskResult Data

The `TaskResult` object provides multiple ways to access the structured data:

```python
# Dictionary-style access (recommended)
title = result["title"]
content = result["content"]

# Direct attribute access (if pydantic model was successfully parsed)
if result.pydantic:
    title = result.pydantic.title
    content = result.pydantic.content

# JSON dictionary access
if result.json_dict:
    title = result.json_dict["title"]
    content = result.json_dict["content"]

# Convert to dictionary
data_dict = result.to_dict()

# String representation (prioritizes structured output)
structured_output = str(result)
```

## Error Handling

The framework gracefully handles parsing failures:

```python
async for event in agent.execute_task(
    "Some task",
    output_pydantic=BlogPost
):
    if event.event_type == "task_completed":
        result = event.data["result"]
        
        if result["output_format"] == "pydantic":
            # Successfully parsed as Pydantic model
            print("✅ Structured output available")
            blog_post = result["pydantic"]
        elif result["output_format"] == "json":
            # Parsed as JSON but couldn't create Pydantic model
            print("⚠️  JSON output available, but Pydantic validation failed")
            json_data = result["json_dict"]
        else:
            # Fallback to raw output
            print("❌ Could not parse structured output, using raw")
            raw_text = result["raw"]
```

## Best Practices

### 1. Provide Clear Field Descriptions

```python
class BlogPost(BaseModel):
    title: str = Field(..., description="A catchy, engaging title for the blog post")
    content: str = Field(..., description="The main body content, should be informative and well-structured")
    author: str = Field(..., description="Full name of the blog post author")
```

### 2. Use Appropriate Default Values

```python
class BlogPost(BaseModel):
    title: str = Field(..., description="Blog post title")
    content: str = Field(..., description="Blog post content")
    published: bool = Field(default=False, description="Publication status")
    tags: List[str] = Field(default_factory=list, description="Category tags")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
```

### 3. Add Validation for Critical Fields

```python
class EmailContent(BaseModel):
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    recipient: str = Field(..., description="Recipient email address")
    
    @validator('recipient')
    def validate_email(cls, v):
        import re
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', v):
            raise ValueError('Invalid email address format')
        return v
```

### 4. Handle Optional Fields Gracefully

```python
class APIResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation succeeded")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
```

## Integration Patterns

### 1. With Existing Systems

```python
# Easy integration with APIs
async def create_blog_post_api(task_description: str):
    async for event in agent.execute_task(
        task_description,
        output_pydantic=BlogPost
    ):
        if event.event_type == "task_completed":
            result = event.data["result"]
            if result["output_format"] == "pydantic":
                # Convert to API payload
                api_payload = {
                    "title": result["title"],
                    "content": result["content"],
                    "author": result["author"],
                    "tags": result["tags"]
                }
                # Send to API
                await send_to_blog_api(api_payload)
```

### 2. With Database Models

```python
# Convert to database model
from sqlalchemy import Column, String, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BlogPostDB(Base):
    __tablename__ = 'blog_posts'
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    author = Column(String, nullable=False)
    published = Column(Boolean, default=False)

# Convert Pydantic to SQLAlchemy
async def save_blog_post(task_description: str):
    async for event in agent.execute_task(
        task_description,
        output_pydantic=BlogPost
    ):
        if event.event_type == "task_completed":
            result = event.data["result"]
            if result.pydantic:
                blog_post = result.pydantic
                
                # Create database model
                db_post = BlogPostDB(
                    title=blog_post.title,
                    content=blog_post.content,
                    author=blog_post.author,
                    published=blog_post.published
                )
                
                # Save to database
                session.add(db_post)
                session.commit()
```

## Troubleshooting

### Common Issues

1. **Pydantic Validation Fails**
   - Check that your model's required fields match the expected output
   - Ensure field types are compatible with the LLM output
   - Review field descriptions for clarity

2. **JSON Parsing Fails**
   - The LLM might not be generating valid JSON
   - Try simplifying your Pydantic model
   - Check the raw output to see what the LLM actually generated

3. **Partial Success**
   - The framework will fall back to JSON or raw output if Pydantic parsing fails
   - Check the `output_format` field to understand what parsing succeeded

### Debug Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check raw output
async for event in agent.execute_task(task, output_pydantic=BlogPost):
    if event.event_type == "task_completed":
        result = event.data["result"]
        print(f"Raw output: {result['raw']}")
        print(f"Output format: {result['output_format']}")
        
        if result.get("error"):
            print(f"Parsing error: {result['error']}")
```

## Limitations

1. **LLM Dependency**: The quality of structured output depends on the LLM's ability to generate valid JSON
2. **Complex Models**: Very complex nested models may be challenging for the LLM to generate correctly
3. **Validation Strictness**: Overly strict validation rules may cause parsing to fail frequently

## Migration from Raw Output

If you're migrating from raw output to Pydantic output:

1. **Gradual Migration**: Start with simple models and gradually add complexity
2. **Fallback Handling**: Always handle cases where Pydantic parsing fails
3. **Testing**: Test with various task types to ensure reliability
4. **Monitoring**: Monitor success rates and adjust models as needed

## Example Templates

### Basic Information Extraction

```python
class ContactInfo(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    company: Optional[str] = Field(None, description="Company name")
```

### Code Analysis

```python
class CodeReview(BaseModel):
    language: str = Field(..., description="Programming language")
    issues: List[str] = Field(default_factory=list, description="Code issues found")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    complexity_score: int = Field(..., ge=1, le=10, description="Complexity rating")
    security_concerns: List[str] = Field(default_factory=list, description="Security issues")
```

### Task Summary

```python
class TaskSummary(BaseModel):
    task_description: str = Field(..., description="What was accomplished")
    steps_taken: List[str] = Field(default_factory=list, description="Steps completed")
    files_modified: List[str] = Field(default_factory=list, description="Files changed")
    outcome: str = Field(..., description="Final result")
    next_steps: Optional[List[str]] = Field(None, description="Recommended next actions")
```

## Support

For questions or issues with the `output_pydantic` feature:

1. Check the examples in `examples/output_pydantic_example.py`
2. Review the test cases in `tests/unit/test_pydantic_output.py`
3. Consult the main framework documentation

The `output_pydantic` feature makes the mutator Coding Agent Framework more powerful and integration-ready by providing structured, validated output that can be easily consumed by other systems and applications. 