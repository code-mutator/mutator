"""
Git integration for the Coding Agent Framework.

This module handles Git repository operations and provides Git-related context
for the codebase analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from git import Repo, InvalidGitRepositoryError

from ..core.types import ContextItem, ContextType


class GitIntegration:
    """Handles Git repository operations and context."""
    
    def __init__(self, working_directory: Path):
        """Initialize Git integration."""
        self.working_directory = working_directory
        self.logger = logging.getLogger(__name__)
        self._git_repo: Optional[Repo] = None
        self._setup_git_repo()
    
    def _setup_git_repo(self) -> None:
        """Setup Git repository if available."""
        try:
            self._git_repo = Repo(self.working_directory)
            self.logger.debug("Git repository detected")
        except InvalidGitRepositoryError:
            self.logger.debug("No Git repository found")
            self._git_repo = None
    
    def has_git_repo(self) -> bool:
        """Check if a Git repository is available."""
        return self._git_repo is not None
    
    def is_dirty(self) -> bool:
        """Check if the working directory has uncommitted changes."""
        if not self._git_repo:
            return False
        
        try:
            return self._git_repo.is_dirty()
        except Exception as e:
            self.logger.warning(f"Failed to check git status: {str(e)}")
            return False
    
    def get_current_branch(self) -> Optional[str]:
        """Get the current branch name."""
        if not self._git_repo:
            return None
        
        try:
            return self._git_repo.active_branch.name
        except Exception as e:
            self.logger.warning(f"Failed to get current branch: {str(e)}")
            return None
    
    def get_recent_commits(self, max_count: int = 10) -> List[Dict[str, Any]]:
        """Get recent commits from the repository."""
        if not self._git_repo:
            return []
        
        try:
            commits = list(self._git_repo.iter_commits(max_count=max_count))
            return [
                {
                    'sha': commit.hexsha[:8],
                    'message': commit.summary,
                    'author': commit.author.name,
                    'date': datetime.fromtimestamp(commit.committed_date).isoformat(),
                    'full_sha': commit.hexsha
                }
                for commit in commits
            ]
        except Exception as e:
            self.logger.warning(f"Failed to get recent commits: {str(e)}")
            return []
    
    def get_git_context(self) -> List[ContextItem]:
        """Get Git repository context."""
        if not self._git_repo:
            return []
        
        context_items = []
        
        try:
            # Get current branch and status
            current_branch = self.get_current_branch()
            is_dirty = self.is_dirty()
            
            # Get recent commits
            recent_commits = self.get_recent_commits()
            
            # Build git context
            git_content = self._build_git_content(current_branch, is_dirty, recent_commits)
            
            context_items.append(ContextItem(
                type=ContextType.COMMIT,
                content=git_content,
                metadata={
                    'current_branch': current_branch,
                    'is_dirty': is_dirty,
                    'recent_commits': recent_commits
                },
                relevance_score=0.8,
                source='git'
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to get git context: {str(e)}")
        
        return context_items
    
    def _build_git_content(self, current_branch: Optional[str], is_dirty: bool, 
                          recent_commits: List[Dict[str, Any]]) -> str:
        """Build Git context content string."""
        content = "Git Repository Status\n\n"
        
        if current_branch:
            content += f"Current Branch: {current_branch}\n"
        
        content += f"Working Directory Clean: {'No' if is_dirty else 'Yes'}\n\n"
        
        if recent_commits:
            content += "Recent Commits:\n"
            for commit in recent_commits:
                content += f"  {commit['sha']} - {commit['message']}\n"
                content += f"    Author: {commit['author']}\n"
                content += f"    Date: {commit['date']}\n\n"
        
        return content
    
    def get_file_history(self, file_path: str, max_count: int = 5) -> List[Dict[str, Any]]:
        """Get commit history for a specific file."""
        if not self._git_repo:
            return []
        
        try:
            commits = list(self._git_repo.iter_commits(paths=file_path, max_count=max_count))
            return [
                {
                    'sha': commit.hexsha[:8],
                    'message': commit.summary,
                    'author': commit.author.name,
                    'date': datetime.fromtimestamp(commit.committed_date).isoformat()
                }
                for commit in commits
            ]
        except Exception as e:
            self.logger.warning(f"Failed to get file history for {file_path}: {str(e)}")
            return []
    
    def get_changed_files(self) -> List[str]:
        """Get list of changed files in the working directory."""
        if not self._git_repo:
            return []
        
        try:
            # Get untracked files
            untracked = self._git_repo.untracked_files
            
            # Get modified files
            modified = [item.a_path for item in self._git_repo.index.diff(None)]
            
            # Get staged files
            staged = [item.a_path for item in self._git_repo.index.diff("HEAD")]
            
            # Combine all changed files
            changed_files = list(set(untracked + modified + staged))
            return changed_files
            
        except Exception as e:
            self.logger.warning(f"Failed to get changed files: {str(e)}")
            return []
    
    def has_recent_changes(self, since_time: datetime) -> bool:
        """Check if there are recent changes since the given time."""
        if not self._git_repo:
            return True  # Assume changes if no git repo
        
        try:
            # Check if there are uncommitted changes
            if self.is_dirty():
                return True
            
            # Check recent commits
            recent_commits = self.get_recent_commits(max_count=5)
            
            for commit in recent_commits:
                commit_time = datetime.fromisoformat(commit['date'])
                if commit_time > since_time:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to check recent changes: {str(e)}")
            return True
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        if not self._git_repo:
            return {}
        
        try:
            current_branch = self.get_current_branch()
            is_dirty = self.is_dirty()
            recent_commits = self.get_recent_commits(max_count=1)
            changed_files = self.get_changed_files()
            
            return {
                'has_git': True,
                'current_branch': current_branch,
                'is_dirty': is_dirty,
                'last_commit': recent_commits[0] if recent_commits else None,
                'changed_files_count': len(changed_files),
                'changed_files': changed_files[:10]  # Limit to first 10
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get repository stats: {str(e)}")
            return {'has_git': False, 'error': str(e)} 