"""
Command-line interface for the Coding Agent Framework.

This module provides a CLI interface for interacting with the coding agent,
including task execution, chat mode, and configuration management.
"""

import asyncio
import json
import sys
import os
import logging
from pathlib import Path
from typing import Optional, List
from io import StringIO
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

from .agent import Mutator
from .core.types import TaskType, ExecutionMode, AgentEvent
from .core.config import AgentConfig, ConfigManager
from . import create_agent

# Create CLI application
app = typer.Typer(
    name="mutator",
    help="Mutator Framework - AI-powered coding assistant",
    add_completion=False
)

# Console for rich output
console = Console()


async def _check_and_process_pending_todos(agent: Mutator, verbose: bool = False) -> None:
    """
    Check for pending todos and automatically trigger processing if any exist.
    
    Args:
        agent: The coding agent instance
        verbose: Whether to show verbose output
    """
    try:
        # Import the todo functions
        from .tools.categories.task_tools import todo_read, todo_process
        
        # Check for pending todos
        todo_result = await todo_read.execute()
        
        if not todo_result.success:
            if verbose:
                console.print(f"[dim]Could not check todos: {todo_result.error}[/dim]")
            return
        
        todo_data = todo_result.result
        aggregated_stats = todo_data.get("aggregated_statistics", {})
        not_started_tasks = aggregated_stats.get("not_started", 0)
        
        if not_started_tasks > 0:
            if verbose:
                console.print(f"[dim]Found {not_started_tasks} pending tasks, processing automatically...[/dim]")
            
            console.print(f"\n[bold blue]📋 Processing {not_started_tasks} pending tasks...[/bold blue]")
            
            # Process all pending tasks
            process_result = await todo_process.execute(process_all=True)
            
            if process_result.success:
                result_data = process_result.result
                successful_tasks = result_data.get("successful_tasks", 0)
                failed_tasks = result_data.get("failed_tasks", 0)
                
                if successful_tasks > 0:
                    console.print(f"[green]✅ Successfully processed {successful_tasks} tasks[/green]")
                
                if failed_tasks > 0:
                    console.print(f"[red]❌ {failed_tasks} tasks failed[/red]")
                
                # Show summary if available
                summary = result_data.get("summary", "")
                if summary:
                    console.print(f"[dim]Summary: {summary}[/dim]")
                    
            else:
                console.print(f"[red]❌ Failed to process todos: {process_result.error}[/red]")
        else:
            if verbose:
                console.print("[dim]No pending tasks found[/dim]")
                
    except Exception as e:
        if verbose:
            console.print(f"[dim]Error checking/processing todos: {str(e)}[/dim]")
        # Don't raise - this is a best-effort feature


# Configuration commands
config_app = typer.Typer(name="config", help="Configuration management")
app.add_typer(config_app)


def setup_cli_logging():
    """Setup logging configuration for CLI usage."""
    # Suppress noisy loggers during interactive sessions
    logging.getLogger("mutator_framework.context.indexer").setLevel(logging.WARNING)
    logging.getLogger("mutator_framework.context.vector_store").setLevel(logging.WARNING)
    logging.getLogger("mutator_framework.tools.manager").setLevel(logging.WARNING)
    logging.getLogger("mutator_framework.agent").setLevel(logging.WARNING)
    logging.getLogger("mutator_framework.context.git_integration").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


def _update_config_with_overrides(config: Optional[AgentConfig], model: Optional[str], provider: Optional[str]) -> AgentConfig:
    """Update configuration with model and provider overrides, ensuring API key is correctly set."""
    if config is None:
        config = AgentConfig()
    
    # Override model and provider if provided
    if model:
        config.llm_config.model = model
    if provider:
        from .core.config import LLMProvider
        config.llm_config.provider = LLMProvider(provider.lower())
        
        # Important: Clear the API key so it gets re-resolved for the new provider
        config.llm_config.api_key = None
        
        # Trigger API key resolution for the new provider
        config.llm_config = config.llm_config.model_validate(config.llm_config.model_dump())
    
    return config


@app.command()
def agent(
    message: Optional[str] = typer.Argument(None, help="Message to send (interactive mode if not provided)"),
    project_path: Optional[str] = typer.Option(None, "--project", "-p", help="Path to project directory"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name to use (e.g., gpt-4, claude-3-sonnet-20240229)"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider to use (openai, anthropic, azure, google, huggingface, ollama, custom)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose debugging output"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Force interactive mode"),
):
    """Chat with the coding agent with full read-write access."""
    
    # Always use agent mode for this command (read-write)
    execution_mode = ExecutionMode.AGENT
    
    # Validate provider if provided
    if provider:
        try:
            from .core.config import LLMProvider
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            valid_providers = [p.value for p in LLMProvider]
            console.print(f"[red]Invalid provider: {provider}. Valid options: {', '.join(valid_providers)}[/red]")
            raise typer.Exit(1)
    
    # Determine mode
    if message is None or interactive:
        # Interactive mode
        console.print(Panel(f"[bold blue]Interactive Agent Mode[/bold blue]\nType 'exit', 'quit', or press Ctrl+C to exit", title="Coding Agent (Read-Write)"))
        
        def _run_async_safely(coro):
            """Run async coroutine safely handling event loop issues."""
            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in a nested event loop situation
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(coro)
                else:
                    return asyncio.run(coro)
            except RuntimeError:
                # No event loop running, create a new one
                return asyncio.run(coro)
            except Exception as e:
                # Handle other async-related issues
                console.print(f"[red]Error in async execution: {str(e)}[/red]")
                # Try running in a thread as fallback
                import threading
                import concurrent.futures
                
                result = None
                exception = None
                
                def run_in_thread():
                    nonlocal result, exception
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(coro)
                        new_loop.close()
                    except Exception as e:
                        exception = e
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()
                
                if exception:
                    raise exception
                return result
        
        _run_async_safely(_chat_interactive_async(project_path, config_file, model, provider, verbose, execution_mode))
    else:
        # Single message mode
        console.print(Panel(f"[bold blue]Agent Mode Message[/bold blue]\n{message}", title="Coding Agent (Read-Write)"))
        asyncio.run(_chat_single_async(message, project_path, config_file, model, provider, verbose, execution_mode))


@app.command()
def chat(
    message: Optional[str] = typer.Argument(None, help="Message to send (interactive mode if not provided)"),
    project_path: Optional[str] = typer.Option(None, "--project", "-p", help="Path to project directory"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name to use (e.g., gpt-4, claude-3-sonnet-20240229)"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider to use (openai, anthropic, azure, google, huggingface, ollama, custom)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose debugging output"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Force interactive mode"),
):
    """Chat with the coding agent in read-only mode."""
    
    # Always use chat mode for this command (read-only)
    execution_mode = ExecutionMode.CHAT
    
    # Validate provider if provided
    if provider:
        try:
            from .core.config import LLMProvider
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            valid_providers = [p.value for p in LLMProvider]
            console.print(f"[red]Invalid provider: {provider}. Valid options: {', '.join(valid_providers)}[/red]")
            raise typer.Exit(1)
    
    # Determine mode
    if message is None or interactive:
        # Interactive mode
        console.print(Panel(f"[bold blue]Interactive Chat Mode[/bold blue]\nType 'exit', 'quit', or press Ctrl+C to exit", title="Coding Agent (Read-Only)"))
        
        def _run_async_safely(coro):
            """Run async coroutine safely handling event loop issues."""
            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in a nested event loop situation
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(coro)
                else:
                    return asyncio.run(coro)
            except RuntimeError:
                # No event loop running, create a new one
                return asyncio.run(coro)
            except Exception as e:
                # Handle other async-related issues
                console.print(f"[red]Error in async execution: {str(e)}[/red]")
                # Try running in a thread as fallback
                import threading
                import concurrent.futures
                
                result = None
                exception = None
                
                def run_in_thread():
                    nonlocal result, exception
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(coro)
                        new_loop.close()
                    except Exception as e:
                        exception = e
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()
                
                if exception:
                    raise exception
                return result
        
        _run_async_safely(_chat_interactive_async(project_path, config_file, model, provider, verbose, execution_mode))
    else:
        # Single message mode
        console.print(Panel(f"[bold blue]Chat Mode Message[/bold blue]\n{message}", title="Coding Agent (Read-Only)"))
        asyncio.run(_chat_single_async(message, project_path, config_file, model, provider, verbose, execution_mode))


async def _chat_single_async(message: str, project_path: Optional[str], config_file: Optional[str], model: Optional[str], provider: Optional[str], verbose: bool, execution_mode: ExecutionMode):
    """Handle single chat message."""
    
    try:
        # Create agent
        config = None
        if config_file:
            config = ConfigManager.load_config(config_file)
        
        # Override model and provider if provided
        config = _update_config_with_overrides(config, model, provider)
        
        # Enable debug mode if verbose
        if verbose:
            if not config:
                config = AgentConfig()
            config.debug = True
            config.logging_level = "DEBUG"
            # Set up debug logging
            import logging
            logging.getLogger("mutator_framework").setLevel(logging.DEBUG)
            console.print(f"[dim]Debug mode enabled - Model: {config.llm_config.model}, Provider: {config.llm_config.provider}[/dim]")
        
        agent = await create_agent(project_path=project_path, config=config)
        
        # Show additional debug info
        if verbose:
            tools = agent.get_available_tools()
            console.print(f"[dim]Available tools: {len(tools)}[/dim]")
            console.print(f"[dim]Function calling enabled: {config.llm_config.function_calling}[/dim]")
            
            # Show system prompt in verbose mode
            try:
                # Get the system prompt from the executor
                system_prompt = agent.executor._create_system_message()
                console.print(f"[dim]System prompt length: {len(system_prompt)} characters[/dim]")
                console.print("[dim]System prompt preview:[/dim]")
                # Show first 200 characters of system prompt
                preview = system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt
                console.print(Panel(f"[dim]{preview}[/dim]", title="System Prompt Preview", border_style="dim"))
            except Exception as e:
                console.print(f"[dim]Could not display system prompt: {e}[/dim]")
        
        # Get response from agent using interactive chat (read-only mode)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Thinking..."),
            console=console,
            transient=False
        ) as progress:
            task = progress.add_task("Processing", total=None)
            
            final_response = ""
            tool_calls_made = False
            warnings_shown = []
            
            # Always use interactive_chat for chat command (read-only)
            async for event in agent.interactive_chat(message):
                # Show debug events if verbose
                if verbose and event.event_type not in ["tool_call_started", "tool_call_completed", "llm_response", "warning", "task_failed"]:
                    console.print(f"[dim]DEBUG: {event.event_type} - {event.data}[/dim]")
                
                if event.event_type == "tool_call_started":
                    # Stop thinking and show tool execution
                    progress.stop()
                    tool_calls_made = True
                    tool_name = event.data.get("tool_name", "unknown")
                    
                    # Show tool execution clearly
                    console.print(f"\n[bold blue]🔧 Using tool: {tool_name}[/bold blue]")
                    
                    # Show tool parameters if available
                    if "parameters" in event.data:
                        params = event.data["parameters"]
                        if params:
                            console.print(f"[dim]   Parameters: {params}[/dim]")
                
                elif event.event_type == "tool_call_completed":
                    tool_name = event.data.get("tool_name", "unknown")
                    success = event.data.get("success", False)
                    execution_time = event.data.get("execution_time", 0)
                    
                    if success:
                        console.print(f"[green]✅ Tool {tool_name} completed ({execution_time:.2f}s)[/green]")
                    else:
                        error = event.data.get("error", "Unknown error")
                        console.print(f"[red]❌ Tool {tool_name} failed: {error}[/red]")
                
                elif event.event_type == "warning":
                    # Handle new warning events
                    progress.stop()
                    warning_msg = event.data.get("message", "Unknown warning")
                    response_content = event.data.get("response_content", "")
                    model_name = event.data.get("model", "unknown")
                    function_calling_enabled = event.data.get("function_calling_enabled", False)
                    available_tools = event.data.get("available_tools", [])
                    
                    console.print(f"\n[bold yellow]⚠️  Configuration Warning[/bold yellow]")
                    console.print(Panel(
                        f"[yellow]{warning_msg}[/yellow]",
                        title="Function Calling Issue",
                        border_style="yellow"
                    ))
                    
                    # Show diagnostic information
                    console.print("\n[yellow]🔍 Diagnostic Information:[/yellow]")
                    console.print(f"• Model: {model_name}")
                    console.print(f"• Function calling enabled: {function_calling_enabled}")
                    console.print(f"• Available tools: {len(available_tools)}")
                    
                    if available_tools:
                        console.print(f"• Tool names: {', '.join(available_tools[:5])}{'...' if len(available_tools) > 5 else ''}")
                    
                    # Show LLM response preview
                    if response_content:
                        preview = response_content[:200] + "..." if len(response_content) > 200 else response_content
                        console.print(f"• LLM response preview: \"{preview}\"")
                    
                    # Provide troubleshooting tips
                    console.print("\n[yellow]💡 Troubleshooting Tips:[/yellow]")
                    if not function_calling_enabled:
                        console.print("• Function calling is disabled - enable it in your configuration")
                    elif not available_tools:
                        console.print("• No tools are available - check your tool registration")
                    elif "gpt-3.5" in model_name.lower():
                        console.print("• Consider using GPT-4 or Claude for better function calling support")
                        console.print("• GPT-3.5 has limited function calling capabilities")
                    elif "claude" in model_name.lower():
                        console.print("• Ensure you're using Claude 3 (Sonnet/Opus) for function calling")
                        console.print("• Claude 2 has limited function calling support")
                    else:
                        console.print("• Verify your model supports function calling")
                        console.print("• Check if your API key has the necessary permissions")
                        console.print("• Try enabling debug mode with --verbose for more details")
                    
                    warnings_shown.append(warning_msg)
                
                elif event.event_type == "llm_response":
                    content = event.data.get("content", "")
                    has_tool_calls = event.data.get("has_tool_calls", False)
                    is_follow_up = event.data.get("is_follow_up", False)
                    tool_call_count = event.data.get("tool_call_count", 0)
                    
                    if verbose:
                        console.print(f"[dim]DEBUG: LLM response - has_tool_calls: {has_tool_calls}, tool_count: {tool_call_count}, is_follow_up: {is_follow_up}[/dim]")
                    
                    if content and not has_tool_calls:
                        if not is_follow_up:
                            progress.stop()
                        final_response = content
                
                elif event.event_type == "task_failed":
                    progress.stop()
                    error = event.data.get("error", "Unknown error")
                    
                    # Enhanced error display with better formatting
                    if "LLM API Error:" in error:
                        # Extract the API error part
                        api_error = error.replace("LLM API Error: ", "")
                        console.print(f"\n[bold red]🚨 API Error[/bold red]")
                        console.print(Panel(
                            f"[red]{api_error}[/red]",
                            title="API Error Details",
                            border_style="red"
                        ))
                        
                        # Provide helpful guidance based on error type
                        if "authentication" in error.lower() or "unauthorized" in error.lower():
                            console.print("\n[yellow]💡 Troubleshooting Tips:[/yellow]")
                            console.print("• Check that your API key is correctly set")
                            console.print("• Verify the API key has the necessary permissions")
                            console.print("• Ensure you're using the correct provider")
                        elif "rate limit" in error.lower() or "quota" in error.lower():
                            console.print("\n[yellow]💡 Troubleshooting Tips:[/yellow]")
                            # Check if it's specifically a quota issue
                            if "exceeded your current quota" in error.lower() or "billing details" in error.lower():
                                console.print("• Your OpenAI quota has been exceeded")
                                console.print("• Consider upgrading your OpenAI plan")
                                console.print("• [bold cyan]Alternative: Use Anthropic Claude instead[/bold cyan]")
                                console.print("  Try: --provider anthropic --model claude-3-haiku-20240307")
                            else:
                                console.print("• Wait a few minutes before trying again")
                                console.print("• Check your API usage limits")
                                console.print("• Consider upgrading your API plan")
                        elif "model" in error.lower() and ("not found" in error.lower() or "invalid" in error.lower()):
                            console.print("\n[yellow]💡 Troubleshooting Tips:[/yellow]")
                            console.print("• Verify the model name is correct")
                            console.print("• Check if the model is available for your provider")
                            console.print("• Try using a different model")
                        elif "connection" in error.lower() or "timeout" in error.lower():
                            console.print("\n[yellow]💡 Troubleshooting Tips:[/yellow]")
                            console.print("• Check your internet connection")
                            console.print("• Try again in a few moments")
                            console.print("• Verify firewall settings")
                    else:
                        console.print(f"\n[bold red]Agent Task Failed[/bold red]: {error}")
                    
                    # Don't display final response on error
                    final_response = ""
                    break
        
        # Display response
        if final_response:
            console.print(Panel(f"[bold green]Agent Response[/bold green]\n{final_response}", title="Agent"))
        elif not warnings_shown:
            # Only show "no response" if we haven't already shown warnings
            console.print(Panel(f"[bold red]Error[/bold red]\nNo response generated", title="Agent"))
        
        # Check for pending todos and process them automatically
        await _check_and_process_pending_todos(agent, verbose)
        
    except Exception as e:
        console.print(f"[red]Error in chat: {str(e)}[/red]")
        if verbose:
            import traceback
            console.print(f"[dim]Debug traceback:\n{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


async def _chat_interactive_async(project_path: Optional[str], config_file: Optional[str], model: Optional[str], provider: Optional[str], verbose: bool, execution_mode: ExecutionMode):
    """Handle interactive chat session."""
    
    try:
        console.print(Panel(
            "[bold blue]Starting Coding Agent...[/bold blue]\n"
            "Please wait while the system initializes.",
            title="Initialization"
        ))
        
        # Create agent (let initialization logs show normally)
        config = None
        if config_file:
            config = ConfigManager.load_config(config_file)
        
        # Override model and provider if provided
        config = _update_config_with_overrides(config, model, provider)
        
        # Enable debug mode if verbose
        if verbose:
            if not config:
                config = AgentConfig()
            config.debug = True
            config.logging_level = "DEBUG"
            # Set up debug logging
            import logging
            logging.getLogger("mutator_framework").setLevel(logging.DEBUG)
        
        agent = await create_agent(project_path=project_path, config=config)
        
        # Wait a moment for any background processes to complete their initial logging
        await asyncio.sleep(1.0)
        
        # Now suppress logs for the interactive session if not in verbose mode
        if not verbose:
            setup_cli_logging()
        
        # Clear screen and show clean interface
        console.clear()
        
        console.print(Panel(
            "[bold green]✓ Initialization Complete[/bold green]\n"
            "Agent is ready for interaction.",
            title="System Ready"
        ))
        
        if verbose:
            tools = await agent.get_available_tools()
            console.print(Panel(
                f"[dim]Debug Mode Enabled[/dim]\n"
                f"Model: {config.llm_config.model}\n"
                f"Provider: {config.llm_config.provider}\n"
                f"Function calling: {config.llm_config.function_calling}\n"
                f"Available tools: {len(tools)}",
                title="Debug Information",
                border_style="dim"
            ))
            
            # Show system prompt in interactive verbose mode
            try:
                system_prompt = agent.executor._create_system_message()
                console.print(f"[dim]System prompt: {len(system_prompt)} characters[/dim]")
                preview = system_prompt[:150] + "..." if len(system_prompt) > 150 else system_prompt
                console.print(Panel(f"[dim]{preview}[/dim]", title="System Prompt Preview", border_style="dim"))
            except Exception as e:
                console.print(f"[dim]Could not display system prompt: {e}[/dim]")
        
        console.print(Panel(
            "[bold blue]Interactive Chat Mode[/bold blue]\n"
            "Type your messages and press Enter. Type 'quit' or 'exit' to end the session.",
            title="Coding Agent Chat"
        ))
        
        # Set up prompt session with limited autocomplete
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.formatted_text import HTML
        
        # Create a completer with just common commands
        completer = WordCompleter([], ignore_case=True)
        session = PromptSession(completer=completer, complete_style='column')
        
        if project_path:
            console.print(f"[dim]Project: {project_path}[/dim]")
        
        console.print()
        
        # Chat loop
        while True:
            # Get user input with history support
            try:
                prompt_text = HTML('<ansiblue><b>You</b></ansiblue>: ')
                user_input = await session.prompt_async(prompt_text)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("[dim]Goodbye![/dim]")
                break
            
            if not user_input.strip():
                continue
            
            # Get agent response using interactive chat (read-only mode)
            try:
                # Start with thinking spinner immediately
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Thinking..."),
                    console=console,
                    transient=False
                ) as progress:
                    task = progress.add_task("Processing", total=None)
                    
                    tool_calls_made = False
                    final_response_printed = False
                    
                    # Always use interactive_chat for chat command (read-only)
                    async for event in agent.interactive_chat(user_input):
                        # Show debug events if verbose
                        if verbose and event.event_type not in ["tool_call_started", "tool_call_completed", "llm_response", "warning", "task_failed"]:
                            console.print(f"[dim]DEBUG: {event.event_type} - {event.data}[/dim]")
                        
                        if event.event_type == "tool_call_started":
                            progress.stop()
                            tool_calls_made = True
                            tool_name = event.data.get("tool_name", "unknown")
                            console.print(f"\n[bold blue]🔧 Using tool: {tool_name}[/bold blue]")
                            params = event.data.get("parameters", {})
                            if params:
                                console.print(f"   Parameters: {params}")
                        
                        elif event.event_type == "tool_call_completed":
                            tool_name = event.data.get("tool_name", "unknown")
                            success = event.data.get("success", False)
                            execution_time = event.data.get("execution_time", 0)
                            if success:
                                console.print(f"[green]✅ Tool {tool_name} completed ({execution_time:.2f}s)[/green]")
                            else:
                                error = event.data.get("error", "Unknown error")
                                console.print(f"[red]❌ Tool {tool_name} failed: {error}[/red]")
                        
                        elif event.event_type == "warning":
                            # Handle warning events in interactive mode
                            progress.stop()
                            warning_msg = event.data.get("message", "Unknown warning")
                            model_name = event.data.get("model", "unknown")
                            function_calling_enabled = event.data.get("function_calling_enabled", False)
                            available_tools = event.data.get("available_tools", [])
                            
                            console.print(f"\n[bold yellow]⚠️  Function Calling Issue Detected[/bold yellow]")
                            console.print(Panel(
                                f"[yellow]{warning_msg}[/yellow]",
                                title="Configuration Warning",
                                border_style="yellow"
                            ))
                            
                            # Show quick diagnostic info
                            console.print(f"\n[dim]Model: {model_name} | Function calling: {function_calling_enabled} | Tools: {len(available_tools)}[/dim]")
                            
                            if not function_calling_enabled:
                                console.print("[yellow]💡 Try: Enable function calling in your config[/yellow]")
                            elif "gpt-3.5" in model_name.lower():
                                console.print("[yellow]💡 Try: Use --model gpt-4-turbo-preview for better tool support[/yellow]")
                            else:
                                console.print("[yellow]💡 Try: Check your model and API key permissions[/yellow]")
                        
                        elif event.event_type == "llm_response":
                            content = event.data.get("content", "")
                            has_tool_calls = event.data.get("has_tool_calls", False)
                            is_follow_up = event.data.get("is_follow_up", False)
                            tool_call_count = event.data.get("tool_call_count", 0)
                            
                            if verbose:
                                console.print(f"[dim]DEBUG: LLM response - has_tool_calls: {has_tool_calls}, tool_count: {tool_call_count}, is_follow_up: {is_follow_up}[/dim]")
                            
                            if content and not has_tool_calls:
                                if not is_follow_up:
                                    progress.stop()
                                console.print(Panel(content, title="Agent Response", border_style="green"))
                                final_response_printed = True

                        elif event.event_type == "task_failed":
                            progress.stop()
                            error = event.data.get("error", "Unknown error")
                            
                            # Enhanced error display with better formatting
                            if "LLM API Error:" in error:
                                # Extract the API error part
                                api_error = error.replace("LLM API Error: ", "")
                                console.print(f"\n[bold red]🚨 API Error[/bold red]")
                                console.print(Panel(
                                    f"[red]{api_error}[/red]",
                                    title="API Error Details",
                                    border_style="red"
                                ))
                                
                                # Provide helpful guidance based on error type
                                if "authentication" in error.lower() or "unauthorized" in error.lower():
                                    console.print("\n[yellow]💡 Troubleshooting Tips:[/yellow]")
                                    console.print("• Check that your API key is correctly set")
                                    console.print("• Verify the API key has the necessary permissions")
                                    console.print("• Ensure you're using the correct provider")
                                elif "rate limit" in error.lower() or "quota" in error.lower():
                                    console.print("\n[yellow]💡 Troubleshooting Tips:[/yellow]")
                                    # Check if it's specifically a quota issue
                                    if "exceeded your current quota" in error.lower() or "billing details" in error.lower():
                                        console.print("• Your OpenAI quota has been exceeded")
                                        console.print("• Consider upgrading your OpenAI plan")
                                        console.print("• [bold cyan]Alternative: Use Anthropic Claude instead[/bold cyan]")
                                        console.print("  Try: --provider anthropic --model claude-3-haiku-20240307")
                                    else:
                                        console.print("• Wait a few minutes before trying again")
                                        console.print("• Check your API usage limits")
                                        console.print("• Consider upgrading your API plan")
                                elif "model" in error.lower() and ("not found" in error.lower() or "invalid" in error.lower()):
                                    console.print("\n[yellow]💡 Troubleshooting Tips:[/yellow]")
                                    console.print("• Verify the model name is correct")
                                    console.print("• Check if the model is available for your provider")
                                    console.print("• Try using a different model")
                                elif "connection" in error.lower() or "timeout" in error.lower():
                                    console.print("\n[yellow]💡 Troubleshooting Tips:[/yellow]")
                                    console.print("• Check your internet connection")
                                    console.print("• Try again in a few moments")
                                    console.print("• Verify firewall settings")
                            else:
                                console.print(f"\n[bold red]Agent Task Failed[/bold red]: {error}")
                            
                            final_response_printed = True # Avoid "No response" message
                            break

                        elif event.event_type == "chat_completed":
                            break
                
                if tool_calls_made and not final_response_printed:
                    console.print(Panel("[dim]No final response generated after tool execution.[/dim]", title="Agent", border_style="yellow"))

                # Check for pending todos and process them automatically after each interaction
                await _check_and_process_pending_todos(agent, verbose)

            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                if verbose:
                    import traceback
                    console.print(f"[dim]Debug traceback:\n{traceback.format_exc()}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error in interactive chat: {str(e)}[/red]")
        if verbose:
            import traceback
            console.print(f"[dim]Debug traceback:\n{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def status(
    project_path: Optional[str] = typer.Option(None, "--project", "-p", help="Path to project directory"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name to use (e.g., gpt-4, claude-3-sonnet-20240229)"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider to use (openai, anthropic, azure, google, huggingface, ollama, custom)"),
    format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)"),
):
    """Show status of the coding agent and project."""
    
    setup_cli_logging()
    
    # Validate provider if provided
    if provider:
        try:
            from .core.config import LLMProvider
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            valid_providers = [p.value for p in LLMProvider]
            console.print(f"[red]Invalid provider: {provider}. Valid options: {', '.join(valid_providers)}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_status_async(project_path, config_file, model, provider, format))


async def _status_async(project_path: Optional[str], config_file: Optional[str], model: Optional[str], provider: Optional[str], format: str):
    """Check agent status asynchronously."""
    
    try:
        # Create agent
        config = None
        if config_file:
            config = ConfigManager.load_config(config_file)
        
        # Override model and provider if provided
        config = _update_config_with_overrides(config, model, provider)
        
        agent = await create_agent(project_path=project_path, config=config)
        
        # Get status
        status = await agent.health_check()
        
        if format == "json":
            # JSON output
            console.print(json.dumps(status, indent=2))
        else:
            # Table output (default)
            console.print(Panel("[bold blue]Agent Status[/bold blue]", title="Coding Agent"))
            
            # Create status table
            table = Table(title="System Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="dim")
            
            # Add rows
            overall_status = status.get("status", "unknown")
            table.add_row("Overall", overall_status, "")
            
            # LLM Status
            llm_status = "✓ Ready" if status.get("llm_ready", False) else "✗ Not Ready"
            table.add_row("LLM", llm_status, f"Model: {config.llm_config.model if config else 'default'}")
            
            # Context Status
            context_status = "✓ Ready" if status.get("context_ready", False) else "✗ Not Ready"
            indexed_files = status.get("indexed_files", 0)
            table.add_row("Context", context_status, f"Indexed files: {indexed_files}")
            
            # Tools Status
            tool_count = status.get("tool_count", 0)
            table.add_row("Tools", f"✓ {tool_count} tools", "")
            
            console.print(table)
        
        await agent.cleanup()
        
    except Exception as e:
        if format == "json":
            console.print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(f"[red]Error getting status: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def tools(
    project_path: Optional[str] = typer.Option(None, "--project", "-p", help="Path to project directory"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name to use (e.g., gpt-4, claude-3-sonnet-20240229)"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider to use (openai, anthropic, azure, google, huggingface, ollama, custom)"),
):
    """List available tools."""
    
    # Validate provider if provided
    if provider:
        try:
            from .core.config import LLMProvider
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            valid_providers = [p.value for p in LLMProvider]
            console.print(f"[red]Invalid provider: {provider}. Valid options: {', '.join(valid_providers)}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_tools_async(project_path, config_file, model, provider))


async def _tools_async(project_path: Optional[str], config_file: Optional[str], model: Optional[str], provider: Optional[str]):
    """List tools asynchronously."""
    
    try:
        # Create agent
        config = None
        if config_file:
            config = ConfigManager.load_config(config_file)
        
        # Override model and provider if provided
        config = _update_config_with_overrides(config, model, provider)
        
        agent = await create_agent(project_path=project_path, config=config)
        
        # Get available tools with their schemas
        tool_names = agent.get_available_tools()
        tool_schemas = agent.tool_manager.get_tool_schemas()
        
        # Display tools
        console.print(Panel("[bold blue]Available Tools[/bold blue]", title="Tools"))
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Tool Name", style="bold")
        table.add_column("Description", style="dim")
        
        for tool_name in tool_names:
            # Get tool schema and description
            schema = tool_schemas.get(tool_name, {})
            if "function" in schema:
                # OpenAI function format
                func_info = schema["function"]
                description = func_info.get("description", "No description")
            else:
                # Try to get description from tool directly
                try:
                    tool_info = agent.get_tool_info(tool_name)
                    description = tool_info.get("description", "No description")
                except:
                    description = "No description"
            
            table.add_row(tool_name, description)
        
        console.print(table)
        
        await agent.cleanup()
        
    except Exception as e:
        console.print(f"[red]Error listing tools: {str(e)}[/red]")
        raise typer.Exit(1)


# Configuration commands
@config_app.command("create")
def create_config(
    output: str = typer.Option("agent_config.json", "--output", "-o", help="Output file path"),
    template: str = typer.Option("default", "--template", "-t", help="Configuration template"),
):
    """Create a new configuration file."""
    
    try:
        # Create default configuration
        config = AgentConfig()
        
        # Save to file
        ConfigManager.save_config(config, output)
        
        console.print(f"[green]Configuration created: {output}[/green]")
        console.print(f"[dim]You can now edit this file to customize your agent settings.[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error creating configuration: {str(e)}[/red]")
        raise typer.Exit(1)


@config_app.command("validate")
def validate_config(
    config_file: str = typer.Argument(..., help="Path to configuration file"),
):
    """Validate a configuration file."""
    
    try:
        # Load and validate configuration
        config = ConfigManager.load_config(config_file)
        
        console.print(f"[green]Configuration is valid: {config_file}[/green]")
        
        # Show summary
        console.print(f"\n[dim]LLM Model: {config.llm_config.model}[/dim]")
        console.print(f"[dim]Project Path: {config.context_config.project_path}[/dim]")
        console.print(f"[dim]Execution Mode: {config.execution_config.default_mode.value}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Configuration validation failed: {str(e)}[/red]")
        raise typer.Exit(1)


@config_app.command("show")
def show_config(
    config_file: Optional[str] = typer.Argument(None, help="Path to configuration file (shows default if not provided)"),
):
    """Display configuration details."""
    
    try:
        if config_file:
            # Load specified config file
            config = ConfigManager.load_config(config_file)
            console.print(f"[dim]Configuration from: {config_file}[/dim]\n")
        else:
            # Show default configuration
            config = AgentConfig()
            console.print("[dim]Default configuration:[/dim]\n")
        
        # Display configuration
        config_dict = config.to_dict()
        
        # Pretty print the configuration
        console.print(Panel(
            Syntax(json.dumps(config_dict, indent=2), "json"),
            title="Configuration",
            border_style="blue"
        ))
        
    except Exception as e:
        console.print(f"[red]Error loading configuration: {str(e)}[/red]")
        raise typer.Exit(1)


# Helper functions
def _print_event(event: AgentEvent):
    """Print a single event."""
    
    timestamp = event.timestamp.strftime("%H:%M:%S")
    
    if event.event_type == "task_started":
        console.print(f"[dim]{timestamp}[/dim] [blue]Task started[/blue]")
    elif event.event_type == "plan_created":
        console.print(f"[dim]{timestamp}[/dim] [blue]Plan created[/blue]")
    elif event.event_type == "step_started":
        step_desc = event.data.get("description", "")[:50]
        console.print(f"[dim]{timestamp}[/dim] [yellow]Step: {step_desc}[/yellow]")
    elif event.event_type == "tool_call_started":
        tool_name = event.data.get("tool_name", "")
        console.print(f"[dim]{timestamp}[/dim] [cyan]Tool: {tool_name}[/cyan]")
    elif event.event_type == "tool_call_completed":
        tool_name = event.data.get("tool_name", "")
        console.print(f"[dim]{timestamp}[/dim] [green]Tool completed: {tool_name}[/green]")
    elif event.event_type == "tool_call_failed":
        tool_name = event.data.get("tool_name", "")
        error = event.data.get("error", "")
        console.print(f"[dim]{timestamp}[/dim] [red]Tool failed: {tool_name} - {error}[/red]")
    elif event.event_type == "task_completed":
        console.print(f"[dim]{timestamp}[/dim] [green]Task completed[/green]")
    elif event.event_type == "task_failed":
        error = event.data.get("error", "")
        console.print(f"[dim]{timestamp}[/dim] [red]Task failed: {error}[/red]")


def _print_execution_summary(events: List[AgentEvent]):
    """Print execution summary."""
    
    # Count events by type
    event_counts = {}
    for event in events:
        event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
    
    # Create summary table
    table = Table(title="Execution Summary")
    table.add_column("Event Type", style="bold")
    table.add_column("Count", style="bold")
    
    for event_type, count in event_counts.items():
        table.add_row(event_type, str(count))
    
    console.print(table)
    
    # Show final status
    last_event = events[-1] if events else None
    if last_event:
        if last_event.event_type == "task_completed":
            console.print("\n[green]✓ Task completed successfully[/green]")
        elif last_event.event_type == "task_failed":
            console.print("\n[red]✗ Task failed[/red]")


def _confirm_tool_execution(tool_name: str, parameters: dict) -> bool:
    """Confirm tool execution with user."""
    
    console.print(f"\n[yellow]About to execute tool: {tool_name}[/yellow]")
    console.print(f"[dim]Parameters: {json.dumps(parameters, indent=2)}[/dim]")
    
    return Confirm.ask("Continue?")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main() 