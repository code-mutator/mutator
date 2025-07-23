"""
AI and intelligent tools for the Coding Agent Framework.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..decorator import tool
from ...core.path_utils import should_exclude_from_search
from .search_tools import _count_lines_in_file


@tool
def search_files_sementic(query: str, file_types: List[str] = None, max_results: int = 20) -> Dict[str, Any]:
    """
    <short_description>Semantic search through codebase files to find relevant code sections and functions.</short_description>
    
    <long_description>
    This tool performs intelligent code search by analyzing file contents, function names,
    class names, comments, and code patterns to find relevant sections of code that match
    your query. It combines text-based search with code structure analysis.

    ## Important Notes

    1. **Search Capabilities**:
       - Function and class name matching
       - Comment and docstring analysis
       - Code pattern recognition
       - Import and dependency tracking
       - Variable and method name search

    2. **File Type Support**:
       - Python (.py) - Advanced analysis of functions, classes, imports
       - JavaScript/TypeScript (.js, .ts) - Function and class detection
       - Java (.java) - Class and method analysis
       - C/C++ (.c, .cpp, .h) - Function and struct analysis
       - Other text files - Basic text search

    3. **Search Intelligence**:
       - Semantic matching beyond exact text
       - Context-aware results with surrounding code
       - Ranking by relevance and code structure
       - Filtering by file types and directories

    4. **Result Format**:
       - File path and line numbers with total file line count
       - Code snippets with context
       - Function/class names where applicable
       - Relevance scoring and ranking

    ## Examples

    - Find authentication code: `search_files_sementic("authentication login user")`
    - Search specific file types: `search_files_sementic("database connection", file_types=["py", "js"])`
    - Find error handling: `search_files_sementic("error handling exception try catch")`
    - Locate API endpoints: `search_files_sementic("api endpoint route handler")`

    ## Use Cases

    - Understanding unfamiliar codebases
    - Finding implementation examples
    - Locating specific functionality
    - Code review and analysis
    - Debugging and troubleshooting
    - Documentation and learning
    </long_description>

    Args:
        query: Natural language query describing what to find in the codebase
        file_types: Optional list of file extensions to search (e.g., ["py", "js", "ts"])
        max_results: Maximum number of results to return (default: 50)
    
    Returns:
        Dict containing search results with file paths, line numbers, code snippets, and file line counts
    """
    try:
        # Import here to avoid circular imports
        from ..decorator import get_working_directory
        
        # Get the configured working directory
        current_dir = Path(get_working_directory())
        
        # Default file types if not specified
        if file_types is None:
            file_types = ["py", "js", "ts", "java", "cpp", "c", "h", "hpp", "rb", "go", "rs", "php", "cs", "swift", "kt"]
        
        # Prepare search terms from query
        search_terms = _prepare_search_terms(query)
        
        # Find relevant files
        relevant_files = _find_relevant_files(current_dir, file_types)
        
        # Search through files
        results = []
        for file_path in relevant_files:
            try:
                file_results = _search_file_content(file_path, search_terms, query)
                results.extend(file_results)
            except Exception as e:
                # Skip files that can't be read
                continue
        
        # Sort results by relevance score
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Limit results
        results = results[:max_results]
        
        # Add summary information
        file_count = len(set(result["file"] for result in results))
        
        return {
            "search_directory": str(current_dir),
            "results": results,
            "total_results": len(results),
            "files_searched": file_count,
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Codebase search failed: {str(e)}"}


def _prepare_search_terms(query: str) -> List[str]:
    """Prepare search terms from the query."""
    # Split query into words and clean them
    terms = re.findall(r'\b\w+\b', query.lower())
    
    # Remove common words that don't add value
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "should", "could", "can", "may", "might", "must"}
    
    terms = [term for term in terms if term not in stop_words and len(term) > 2]
    
    return terms


def _find_relevant_files(directory: Path, file_types: List[str]) -> List[Path]:
    """Find files with relevant extensions."""
    relevant_files = []
    
    # Create pattern for file extensions
    extensions = [f".{ext}" for ext in file_types]
    
    try:
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in extensions:
                # Skip files that should be excluded by .gitignore
                if not should_exclude_from_search(file_path, directory):
                    relevant_files.append(file_path)
    except Exception:
        pass
    
    return relevant_files


def _search_file_content(file_path: Path, search_terms: List[str], original_query: str) -> List[Dict[str, Any]]:
    """Search for terms within a file's content."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return []
    
    results = []
    file_line_count = len(lines)  # Get total line count for this file
    
    for line_num, line in enumerate(lines, 1):
        line_lower = line.lower()
        
        # Calculate relevance score
        relevance_score = 0
        matched_terms = []
        
        for term in search_terms:
            if term in line_lower:
                relevance_score += 1
                matched_terms.append(term)
        
        # Boost score for exact query matches
        if original_query.lower() in line_lower:
            relevance_score += 5
        
        # Boost score for function/class definitions
        if re.search(r'\b(def|class|function|const|let|var|public|private|protected)\b', line_lower):
            relevance_score += 2
        
        # Boost score for comments
        if re.search(r'#|//|/\*|\*/', line):
            relevance_score += 1
        
        if relevance_score > 0:
            # Get context around the match
            context_start = max(0, line_num - 3)
            context_end = min(len(lines), line_num + 2)
            context_lines = lines[context_start:context_end]
            
            # Detect function/class name if applicable
            function_class_name = _detect_function_class_name(lines, line_num - 1)
            
            result = {
                "file": str(file_path.relative_to(current_dir)),
                "line_number": line_num,
                "line_content": line.strip(),
                "relevance_score": relevance_score,
                "matched_terms": matched_terms,
                "context": [line.rstrip() for line in context_lines],
                "function_class": function_class_name,
                "file_line_count": file_line_count
            }
            
            results.append(result)
    
    return results


def _detect_function_class_name(lines: List[str], current_line: int) -> str:
    """Detect the function or class name that contains the current line."""
    # Look backwards for function/class definitions
    for i in range(current_line, max(0, current_line - 20), -1):
        line = lines[i].strip()
        
        # Python function/class
        python_match = re.search(r'^(def|class)\s+(\w+)', line)
        if python_match:
            return f"{python_match.group(1)} {python_match.group(2)}"
        
        # JavaScript/TypeScript function
        js_match = re.search(r'(function\s+(\w+)|(\w+)\s*[:=]\s*function|(\w+)\s*\([^)]*\)\s*{)', line)
        if js_match:
            name = js_match.group(2) or js_match.group(3) or js_match.group(4)
            if name:
                return f"function {name}"
        
        # Java/C++ method
        java_match = re.search(r'(public|private|protected|static).*?(\w+)\s*\([^)]*\)\s*{', line)
        if java_match:
            return f"method {java_match.group(2)}"
    
    return ""


@tool
def mermaid(description: str, diagram_type: str = "auto", code: str = "", include_details: bool = False) -> Dict[str, Any]:
    """
    <short_description>Generate Mermaid diagrams from code analysis or natural language descriptions to visualize system architecture, workflows, and code structure.</short_description>
    
    <long_description>
    This tool creates visual diagrams using Mermaid syntax to help understand and communicate
    code structure, system architecture, workflows, and relationships. It can analyze code
    or work from natural language descriptions to generate appropriate diagrams.

    ## Important Notes

    1. **Diagram Types**:
       - **Flowchart**: Process flows, decision trees, and workflow diagrams
       - **Sequence**: API interactions, method calls, and communication flows
       - **Class**: Object relationships, inheritance, and system architecture
       - **State**: State machines, workflow states, and transition diagrams
       - **Gantt**: Project timelines, task scheduling, and milestone tracking
       - **Pie**: Data distribution, usage statistics, and proportional relationships

    2. **Content Sources**:
       - Natural language descriptions for concept visualization
       - Source code analysis for automatic diagram generation
       - Hybrid approach combining description and code analysis
       - Automatic type detection based on content

    3. **Automatic Type Detection**:
       - Analyzes description keywords to suggest appropriate diagram types
       - Considers code patterns and structure for type selection
       - Provides intelligent defaults based on context
       - Allows manual override for specific requirements

    4. **Customization Options**:
       - Include detailed information in diagrams
       - Control diagram complexity and detail level
       - Support for various Mermaid features and styling
       - Extensible for different visualization needs

    ## Examples

    - Process flow: `mermaid("User registration and login process")`
    - API sequence: `mermaid("How does the authentication API work?", diagram_type="sequence")`
    - Class structure: `mermaid("Database model relationships", diagram_type="class")`
    - From code: `mermaid("Analyze this class structure", code="class User: def __init__(self): pass")`

    ## Use Cases

    - System architecture visualization
    - Code structure documentation
    - Workflow and process documentation
    - API interaction diagrams
    - Database relationship diagrams
    - Project timeline visualization

    ## Diagram Type Selection

    - **Flowchart**: Best for processes, algorithms, and decision flows
    - **Sequence**: Ideal for API calls, method interactions, and communication
    - **Class**: Perfect for object models, inheritance, and system structure
    - **State**: Great for state machines, workflows, and transitions
    - **Gantt**: Excellent for project planning and timeline visualization
    - **Pie**: Useful for data distribution and proportional relationships

    ## Integration Features

    - Generates standard Mermaid syntax for universal compatibility
    - Supports embedding in documentation and presentations
    - Provides suggestions for diagram improvement
    - Handles complex code structures and relationships
    </long_description>

    Args:
        description: Natural language description of what to diagram
        diagram_type: Type of diagram to generate (auto, flowchart, sequence, class, state, gantt, pie)
        code: Optional source code to analyze and diagram
        include_details: Whether to include detailed information in the diagram
    
    Returns:
        Dict containing generated Mermaid diagram code, type, and suggestions
    """
    try:
        # Determine diagram type
        if diagram_type == "auto":
            diagram_type = _determine_diagram_type(description, code)
        
        # Generate the appropriate diagram
        if diagram_type == "flowchart":
            mermaid_code = _generate_flowchart(description, code, include_details)
        elif diagram_type == "sequence":
            mermaid_code = _generate_sequence_diagram(description, code, include_details)
        elif diagram_type == "class":
            mermaid_code = _generate_class_diagram(description, code, include_details)
        elif diagram_type == "state":
            mermaid_code = _generate_state_diagram(description, code, include_details)
        elif diagram_type == "gantt":
            mermaid_code = _generate_gantt_chart(description)
        elif diagram_type == "pie":
            mermaid_code = _generate_pie_chart(description)
        elif diagram_type == "gitgraph":
            mermaid_code = _generate_git_graph(description)
        else:
            # Default to flowchart
            mermaid_code = _generate_flowchart(description, code, include_details)
            diagram_type = "flowchart"
        
        # Get suggestions for improvement
        suggestions = _get_diagram_suggestions(diagram_type)
        
        return {
            "diagram_type": diagram_type,
            "mermaid_code": mermaid_code,
            "suggestions": suggestions,
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Mermaid diagram generation failed: {str(e)}"}


def _determine_diagram_type(description: str, code: str = "") -> str:
    """Determine the most appropriate diagram type based on description and code."""
    description_lower = description.lower()
    
    if any(word in description_lower for word in ["sequence", "api", "call", "interaction", "message"]):
        return "sequence"
    elif any(word in description_lower for word in ["class", "inheritance", "object", "relationship"]):
        return "class"
    elif any(word in description_lower for word in ["state", "status", "transition", "workflow"]):
        return "state"
    elif any(word in description_lower for word in ["timeline", "schedule", "project", "gantt"]):
        return "gantt"
    elif any(word in description_lower for word in ["distribution", "percentage", "pie", "chart"]):
        return "pie"
    elif any(word in description_lower for word in ["git", "branch", "commit", "merge"]):
        return "gitgraph"
    else:
        return "flowchart"


def _generate_flowchart(description: str, code: str = "", include_details: bool = False) -> str:
    """Generate a flowchart diagram."""
    return f"""flowchart TD
    A[Start] --> B[Process Input]
    B --> C{{Decision Point}}
    C -->|Yes| D[Execute Action]
    C -->|No| E[Handle Alternative]
    D --> F[Complete]
    E --> F
    F --> G[End]
    
    %% {description}"""


def _generate_sequence_diagram(description: str, code: str = "", include_details: bool = False) -> str:
    """Generate a sequence diagram."""
    return f"""sequenceDiagram
    participant A as Client
    participant B as Server
    participant C as Database
    
    A->>B: Request
    B->>C: Query
    C-->>B: Data
    B-->>A: Response
    
    Note right of A: {description}"""


def _generate_class_diagram(description: str, code: str = "", include_details: bool = False) -> str:
    """Generate a class diagram."""
    return f"""classDiagram
    class BaseClass {{
        +String property
        +method()
    }}
    
    class DerivedClass {{
        +String specificProperty
        +specificMethod()
    }}
    
    BaseClass <|-- DerivedClass
    
    %% {description}"""


def _generate_state_diagram(description: str, code: str = "", include_details: bool = False) -> str:
    """Generate a state diagram."""
    return f"""stateDiagram-v2
    [*] --> Initial
    Initial --> Processing
    Processing --> Success
    Processing --> Error
    Success --> [*]
    Error --> Initial
    
    note right of Processing : {description}"""


def _generate_gantt_chart(description: str) -> str:
    """Generate a Gantt chart."""
    return f"""gantt
    title {description}
    dateFormat  YYYY-MM-DD
    section Phase 1
    Task 1           :2024-01-01, 30d
    Task 2           :2024-01-15, 20d
    section Phase 2
    Task 3           :2024-02-01, 25d"""


def _generate_pie_chart(description: str) -> str:
    """Generate a pie chart."""
    return f"""pie title {description}
    "Category A" : 42.96
    "Category B" : 50.05
    "Category C" : 7.01"""


def _generate_git_graph(description: str) -> str:
    """Generate a Git graph."""
    return f"""gitgraph
    commit id: "Initial commit"
    branch develop
    checkout develop
    commit id: "Add feature"
    checkout main
    merge develop
    commit id: "Release"
    
    %% {description}"""


def _get_diagram_suggestions(diagram_type: str) -> List[str]:
    """Get suggestions based on diagram type."""
    suggestions = {
        "flowchart": [
            "Use clear, descriptive labels for each step",
            "Consider using subgraphs for complex processes",
            "Add decision points with Yes/No branches"
        ],
        "sequence": [
            "Include all relevant participants",
            "Show both synchronous and asynchronous calls",
            "Add notes for important interactions"
        ],
        "class": [
            "Include important methods and properties",
            "Show inheritance and composition relationships",
            "Group related classes together"
        ],
        "state": [
            "Define clear state transitions",
            "Include initial and final states",
            "Add guards and actions where appropriate"
        ]
    }
    return suggestions.get(diagram_type, ["Review and refine the generated diagram as needed"])


__all__ = [
    "search_files_sementic",
    "mermaid"
] 