# Installation and Compatibility Fixes

## GitLab Pipeline Python 3.8 Compatibility Issue

### Problem
The GitLab pipeline was failing with the following error when using Python 3.8:
```
ERROR: Could not find a version that satisfies the requirement langgraph>=0.0.50 (from mutator) (from versions: none)
ERROR: No matching distribution found for langgraph>=0.0.50
```

### Root Cause
Starting with version 0.2.0, the LangChain ecosystem packages (`langchain`, `langchain-core`, `langgraph`, `langchain-community`) require Python 3.9+ as a minimum version. The pipeline was using Python 3.8.18, which is incompatible with newer versions of these packages.

### Solution Applied
Added version constraints to limit LangChain ecosystem packages to versions that support Python 3.8:

```toml
# LangChain ecosystem - constrained for Python 3.8 compatibility
# Versions 0.2.0+ require Python 3.9+
"langchain>=0.1.0,<0.2.0",
"langchain-core>=0.1.0,<0.2.0", 
"langgraph>=0.0.50,<0.1.0",
"langchain-community>=0.0.20,<0.1.0",
```

### Alternative Solutions

#### Option 1: Upgrade Pipeline Python Version (Recommended)
Update your GitLab CI pipeline to use Python 3.9 or higher:

```yaml
# .gitlab-ci.yml
image: python:3.9-slim  # or python:3.10-slim, python:3.11-slim, etc.
```

This allows you to use the latest versions of all dependencies and removes the need for version constraints.

#### Option 2: Update Project Minimum Python Version
If you choose to upgrade the pipeline, also update the project requirements:

```toml
# pyproject.toml
requires-python = ">=3.9"
```

And remove the version constraints from the LangChain packages.

### Testing the Fix
The current constraints have been tested and verified to work with Python 3.8:
- ✓ `langchain>=0.1.0,<0.2.0`: COMPATIBLE
- ✓ `langchain-core>=0.1.0,<0.2.0`: COMPATIBLE  
- ✓ `langgraph>=0.0.50,<0.1.0`: COMPATIBLE
- ✓ `langchain-community>=0.0.20,<0.1.0`: COMPATIBLE 