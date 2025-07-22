"""
MCP (Model Context Protocol) server integration for the Coding Agent Framework.

This module provides the MCPServer class for integrating with external MCP servers
and managing their lifecycle.
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Any, Dict, List, Optional

from ..core.config import MCPServerConfig
from ..core.types import ToolResult


class MCPServer:
    """Represents an MCP (Model Context Protocol) server."""
    
    def __init__(self, name: str, command: List[str], env: Optional[Dict[str, str]] = None, **kwargs):
        """Initialize the MCP server."""
        # Handle both old and new initialization patterns
        if isinstance(command, MCPServerConfig):
            # Old pattern: MCPServer(config)
            config = command
            self.name = config.name
            self.command = config.command
            self.env = config.env
            self.config = config
        else:
            # New pattern: MCPServer(name, command, env)
            self.name = name
            self.command = command if isinstance(command, list) else [command]
            self.env = env or {}
            self.config = MCPServerConfig(name=name, command=self.command, env=self.env, **kwargs)
        
        self.process: Optional[subprocess.Popen] = None
        self.logger = logging.getLogger(f"{__name__}.mcp.{self.name}")
        self._tools_cache: Dict[str, Dict[str, Any]] = {}
    
    async def start(self) -> bool:
        """Start the MCP server process."""
        if self.process and self.process.poll() is None:
            return True  # Already running
        
        try:
            self.process = subprocess.Popen(
                self.command,
                env={**self.env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for the process to start
            await asyncio.sleep(1)
            
            if self.process.poll() is None:
                self.logger.debug(f"MCP server '{self.name}' started")
                return True
            else:
                self.logger.error(f"MCP server '{self.name}' failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start MCP server '{self.name}': {str(e)}")
            return False
    
    async def stop(self) -> None:
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(1)
                
                if self.process.poll() is None:
                    self.process.kill()
                
                self.logger.debug(f"MCP server '{self.name}' stopped")
            except Exception as e:
                self.logger.error(f"Failed to stop MCP server '{self.name}': {str(e)}")
            finally:
                self.process = None
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Call a tool on this MCP server."""
        if not self.process or self.process.poll() is not None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="MCP server not running"
            )
        
        try:
            # This is a simplified implementation
            # In a real implementation, you would use the MCP protocol
            request = {
                "jsonrpc": "2.0",
                "id": int(time.time()),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            # Send request to MCP server
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # Read response (simplified)
            response_line = self.process.stdout.readline()
            if response_line:
                response = json.loads(response_line.strip())
                
                if "result" in response:
                    return ToolResult(
                        tool_name=tool_name,
                        success=True,
                        result=response["result"]
                    )
                elif "error" in response:
                    return ToolResult(
                        tool_name=tool_name,
                        success=False,
                        error=response["error"]["message"]
                    )
            
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="No response from MCP server"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to call tool '{tool_name}' on MCP server: {str(e)}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e)
            )
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from this MCP server."""
        if not self.process or self.process.poll() is not None:
            return []
        
        try:
            # This is a simplified implementation
            request = {
                "jsonrpc": "2.0",
                "id": int(time.time()),
                "method": "tools/list"
            }
            
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            response_line = self.process.stdout.readline()
            if response_line:
                response = json.loads(response_line.strip())
                if "result" in response:
                    return response["result"].get("tools", [])
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to list tools from MCP server: {str(e)}")
            return []
    
    def is_running(self) -> bool:
        """Check if the MCP server is running."""
        return self.process is not None and self.process.poll() is None


class MCPServerManager:
    """Manager for multiple MCP servers."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.logger = logging.getLogger(__name__)
    
    async def add_server(self, config: MCPServerConfig) -> bool:
        """Add and start an MCP server."""
        if config.name in self.servers:
            self.logger.warning(f"MCP server '{config.name}' is already registered")
            return False
        
        server = MCPServer(config.name, config.command, config.env)
        if await server.start():
            self.servers[config.name] = server
            
            # Cache available tools from this server
            try:
                tools = await server.list_tools()
                for tool_info in tools:
                    tool_name = f"{config.name}.{tool_info['name']}"
                    self.logger.debug(f"Available MCP tool: {tool_name}")
            except Exception as e:
                self.logger.error(f"Failed to list tools from MCP server '{config.name}': {str(e)}")
            
            return True
        else:
            return False
    
    async def remove_server(self, server_name: str) -> bool:
        """Remove and stop an MCP server."""
        if server_name in self.servers:
            server = self.servers[server_name]
            await server.stop()
            del self.servers[server_name]
            return True
        return False
    
    def get_server(self, server_name: str) -> Optional[MCPServer]:
        """Get an MCP server by name."""
        return self.servers.get(server_name)
    
    def list_servers(self) -> List[str]:
        """List all registered MCP server names."""
        return list(self.servers.keys())
    
    async def start_all(self) -> Dict[str, bool]:
        """Start all MCP servers."""
        results = {}
        for name, server in self.servers.items():
            results[name] = await server.start()
        return results
    
    async def stop_all(self) -> None:
        """Stop all MCP servers."""
        for server in self.servers.values():
            await server.stop()
    
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all MCP servers."""
        health = {}
        for name, server in self.servers.items():
            try:
                is_running = server.is_running()
                health[name] = {
                    "status": "healthy" if is_running else "stopped",
                    "process_running": is_running
                }
            except Exception as e:
                health[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        return health 