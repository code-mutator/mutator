"""
CLI history management for the Coding Agent Framework.

This module provides persistent command history storage for the CLI,
allowing users to navigate through previous commands using arrow keys.
"""

import os
from pathlib import Path
from typing import List, Optional
import logging


class CLIHistory:
    """Manages CLI command history with persistent storage."""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
        # Determine history file location
        self.history_file = self._get_history_file_path()
        
        # Load existing history
        self.history: List[str] = self._load_history()
        
        self.logger.debug(f"CLI history initialized with {len(self.history)} entries")
    
    def _get_history_file_path(self) -> Path:
        """Get the path to the history file."""
        # Try to use XDG_DATA_HOME or fall back to ~/.local/share
        if os.environ.get('XDG_DATA_HOME'):
            data_dir = Path(os.environ['XDG_DATA_HOME'])
        else:
            data_dir = Path.home() / '.local' / 'share'
        
        # Create mutator directory
        mutator_dir = data_dir / 'mutator'
        mutator_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a simple text file for prompt_toolkit compatibility
        return mutator_dir / 'cli_history.txt'
    
    def _load_history(self) -> List[str]:
        """Load history from file."""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Clean up lines and remove empty ones
            history = [line.strip() for line in lines if line.strip()]
            
            # Return only the last max_history entries
            return history[-self.max_history:]
                
        except IOError as e:
            self.logger.warning(f"Failed to load CLI history: {e}")
            return []
    
    def _save_history(self) -> None:
        """Save history to file."""
        try:
            # Create directory if it doesn't exist
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write history as simple text file (prompt_toolkit format)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                for entry in self.history[-self.max_history:]:
                    f.write(f"{entry}\n")
                
        except IOError as e:
            self.logger.error(f"Failed to save CLI history: {e}")
    
    def add_command(self, command: str) -> None:
        """Add a command to history."""
        if not command or not command.strip():
            return
        
        command = command.strip()
        
        # Remove duplicate if it exists
        if command in self.history:
            self.history.remove(command)
        
        # Add to end of history
        self.history.append(command)
        
        # Trim history if it's too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Save to file
        self._save_history()
        
        self.logger.debug(f"Added command to history: {command[:50]}...")
    
    def get_history(self) -> List[str]:
        """Get the current history."""
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear all history."""
        self.history.clear()
        self._save_history()
        self.logger.debug("Cleared CLI history")
    
    def search_history(self, query: str) -> List[str]:
        """Search history for commands containing the query."""
        if not query:
            return []
        
        query = query.lower()
        return [cmd for cmd in self.history if query in cmd.lower()]
    
    def get_last_command(self) -> Optional[str]:
        """Get the last command from history."""
        return self.history[-1] if self.history else None
    
    def get_history_file_path(self) -> Path:
        """Get the path to the history file (for external access)."""
        return self.history_file


# Global history instance
_cli_history: Optional[CLIHistory] = None


def get_cli_history() -> CLIHistory:
    """Get the global CLI history instance."""
    global _cli_history
    if _cli_history is None:
        _cli_history = CLIHistory()
    return _cli_history


def add_to_history(command: str) -> None:
    """Add a command to the global history."""
    get_cli_history().add_command(command)


def get_history_list() -> List[str]:
    """Get the current history list."""
    return get_cli_history().get_history()


def clear_history() -> None:
    """Clear the global history."""
    get_cli_history().clear_history() 