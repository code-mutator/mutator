"""
Pytest configuration and shared fixtures for the Coding Agent Framework tests.
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator

from mutator import Mutator, AgentConfig
from mutator.core.config import (
    LLMConfig, ContextConfig, SafetyConfig, VectorStoreConfig, ExecutionConfig
)
from mutator.core.types import ExecutionMode


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_project_dir() -> Generator[Path, None, None]:
    """Create a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create a basic Python project structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "docs").mkdir()
        
        # Create some sample files
        (project_path / "README.md").write_text("""
# Test Project

This is a test project for the coding agent framework.

## Features
- Authentication system
- User management
- API endpoints

## Installation
```bash
pip install -r requirements.txt
```
        """)
        
        (project_path / "requirements.txt").write_text("""
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
pytest==7.4.3
        """)
        
        (project_path / "setup.py").write_text("""
from setuptools import setup, find_packages

setup(
    name="test-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "sqlalchemy",
    ],
)
        """)
        
        # Create main application file
        (project_path / "src" / "main.py").write_text("""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Test API", version="0.1.0")

class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool = True

# In-memory storage for demo
users_db = []

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/users", response_model=List[User])
async def get_users():
    return users_db

@app.post("/users", response_model=User)
async def create_user(user: User):
    users_db.append(user)
    return user

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    for user in users_db:
        if user.id == user_id:
            return user
    raise HTTPException(status_code=404, detail="User not found")

@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    for i, user in enumerate(users_db):
        if user.id == user_id:
            del users_db[i]
            return {"message": "User deleted"}
    raise HTTPException(status_code=404, detail="User not found")
        """)
        
        # Create auth module
        (project_path / "src" / "auth.py").write_text("""
import hashlib
import secrets
from typing import Optional

def hash_password(password: str) -> str:
    \"\"\"Hash a password with salt.\"\"\"
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{password_hash.hex()}"

def verify_password(password: str, hashed: str) -> bool:
    \"\"\"Verify a password against its hash.\"\"\"
    try:
        salt, password_hash = hashed.split(':')
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex() == password_hash
    except ValueError:
        return False

class AuthService:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, user_id: int) -> str:
        \"\"\"Create a new session for a user.\"\"\"
        session_token = secrets.token_urlsafe(32)
        self.sessions[session_token] = user_id
        return session_token
    
    def validate_session(self, token: str) -> Optional[int]:
        \"\"\"Validate a session token and return user ID.\"\"\"
        return self.sessions.get(token)
    
    def invalidate_session(self, token: str) -> bool:
        \"\"\"Invalidate a session token.\"\"\"
        if token in self.sessions:
            del self.sessions[token]
            return True
        return False
        """)
        
        # Create test file
        (project_path / "tests" / "test_main.py").write_text("""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_user():
    user_data = {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com",
        "is_active": True
    }
    response = client.post("/users", json=user_data)
    assert response.status_code == 200
    assert response.json() == user_data

def test_get_users():
    response = client.get("/users")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
        """)
        
        yield project_path


@pytest.fixture
def test_config() -> AgentConfig:
    """Create a test configuration."""
    # Use real API key if available, otherwise use test key
    api_key = os.getenv("SONNET_KEY")
    if not api_key or api_key == "test-api-key-for-mocking":
        api_key = "test-key"
    
    return AgentConfig(
        llm=LLMConfig(
            model="claude-3-haiku-20240307",  # Fast model for tests
            api_key=api_key,
            max_tokens=1000,
            temperature=0.1
        ),
        context=ContextConfig(
            max_context_files=5,  # Limit files for faster testing
            max_file_size=50000,  # Limit file size
            ignore_patterns=["*.pyc", "__pycache__", ".git", "*.log", ".pytest_cache", "vector_store/", "*.egg-info/"]
        ),
        vector_store=VectorStoreConfig(
            type="chromadb",
            path="./test_vector_store",
            collection_name="test_collection",
            embedding_model="all-MiniLM-L6-v2",  # Fast, small model
            chunk_size=500,  # Smaller chunks for faster processing
            chunk_overlap=50
        ),
        safety=SafetyConfig(
            confirmation_level="none",  # No confirmations in tests
            allowed_shell_commands=["ls", "cat", "find", "git", "python", "pytest"],
            blocked_shell_commands=["rm", "sudo", "wget", "curl", "dd"]
        ),
        execution_config=ExecutionConfig(
            default_mode=ExecutionMode.CHAT,  # Safe default for tests
            max_iterations=5,  # Fewer iterations for faster tests
            timeout=30,  # Shorter timeout
            max_parallel_tasks=2  # Limit parallel tasks
        ),
        logging_level="WARNING"  # Reduce noise in tests
    )


@pytest.fixture
async def test_agent(test_config: AgentConfig, temp_project_dir: Path) -> AsyncGenerator[Mutator, None]:
    """Create a test agent instance."""
    # Update config with temp project path
    test_config.working_directory = str(temp_project_dir)
    
    agent = Mutator(test_config)
    await agent.initialize()
    
    yield agent
    
    # Cleanup
    try:
        if hasattr(agent.context_manager, 'cleanup'):
            await agent.context_manager.cleanup()
        if hasattr(agent.llm_client, 'cleanup'):
            await agent.llm_client.cleanup()
        if hasattr(agent.tool_manager, 'cleanup'):
            await agent.tool_manager.cleanup()
    except Exception:
        pass  # Ignore cleanup errors in tests


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    from mutator.core.types import LLMResponse, ToolCall
    
    def create_response(content: str = "Test response", tool_calls: list = None):
        return LLMResponse(
            content=content,
            success=True,
            tool_calls=tool_calls or [],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="test-model"
        )
    
    return create_response


@pytest.fixture
def sample_files(temp_project_dir: Path):
    """Create additional sample files for testing."""
    # Create a config file
    (temp_project_dir / "config.json").write_text("""
{
    "database_url": "sqlite:///test.db",
    "secret_key": "test-secret",
    "debug": true,
    "api_settings": {
        "rate_limit": 100,
        "timeout": 30
    }
}
    """)
    
    # Create a utility module
    (temp_project_dir / "src" / "utils.py").write_text("""
import json
import logging
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str) -> Dict[str, Any]:
    \"\"\"Load configuration from JSON file.\"\"\"
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(level: str = "INFO"):
    \"\"\"Setup logging configuration.\"\"\"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class FileManager:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
    
    def read_file(self, filename: str) -> str:
        \"\"\"Read file contents.\"\"\"
        return (self.base_path / filename).read_text()
    
    def write_file(self, filename: str, content: str):
        \"\"\"Write content to file.\"\"\"
        (self.base_path / filename).write_text(content)
    
    def list_files(self, pattern: str = "*") -> list:
        \"\"\"List files matching pattern.\"\"\"
        return list(self.base_path.glob(pattern))
    """)
    
    return temp_project_dir


# Skip e2e tests if no API key is available
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test requiring API key"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to configure e2e tests."""
    # Don't set mock API key for e2e tests - let them handle their own API key validation
    pass 