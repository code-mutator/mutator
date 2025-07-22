#!/usr/bin/env python3
"""
Test runner script for the Coding Agent Framework.
This script provides convenient ways to run different types of tests.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description or ' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    else:
        print(f"✅ Command succeeded")
        return True


def check_environment():
    """Check if the environment is set up correctly."""
    print("🔍 Checking environment...")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} is installed")
    except ImportError:
        print("❌ pytest is not installed. Run: pip install -r requirements-test.txt")
        return False
    
    # Check if the framework is installed
    try:
        import mutator
        print(f"✅ mutator_framework is installed")
    except ImportError:
        print("❌ mutator_framework is not installed. Run: pip install -e .")
        return False
    
    # Check for API key if running e2e tests
    if os.getenv("SONNET_KEY"):
        print("✅ SONNET_KEY environment variable is set")
    else:
        print("⚠️  SONNET_KEY environment variable is not set (e2e tests will be skipped)")
    
    return True


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=mutator_framework", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, "Unit Tests")


def run_e2e_tests(verbose=False, slow=False):
    """Run end-to-end tests."""
    cmd = ["python", "-m", "pytest", "tests/e2e/", "-m", "e2e"]
    
    if verbose:
        cmd.append("-v")
    
    if not slow:
        cmd.extend(["-m", "not slow"])
    
    return run_command(cmd, "End-to-End Tests")


def run_all_tests(verbose=False, coverage=False, slow=False):
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=mutator_framework", "--cov-report=html", "--cov-report=term"])
    
    if not slow:
        cmd.extend(["-m", "not slow"])
    
    return run_command(cmd, "All Tests")


def run_linting():
    """Run linting checks."""
    success = True
    
    # Run black
    if not run_command(["black", "--check", "mutator_framework/", "tests/"], "Black formatting check"):
        success = False
    
    # Run isort
    if not run_command(["isort", "--check-only", "mutator_framework/", "tests/"], "Import sorting check"):
        success = False
    
    # Run flake8
    if not run_command(["flake8", "mutator_framework/", "tests/"], "Flake8 linting"):
        success = False
    
    return success


def run_type_checking():
    """Run type checking."""
    return run_command(["mypy", "mutator_framework/"], "Type checking")


def run_security_checks():
    """Run security checks."""
    try:
        # Try to run bandit if available
        return run_command(["bandit", "-r", "mutator_framework/"], "Security checks")
    except FileNotFoundError:
        print("⚠️  bandit not installed, skipping security checks")
        return True


def run_performance_tests():
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "benchmark", "--benchmark-only"]
    return run_command(cmd, "Performance Tests")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test runner for Coding Agent Framework")
    parser.add_argument("--type", choices=["unit", "e2e", "all", "lint", "type", "security", "perf"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--no-env-check", action="store_true", help="Skip environment check")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    
    args = parser.parse_args()
    
    # Check environment
    if not args.no_env_check and not check_environment():
        print("❌ Environment check failed")
        sys.exit(1)
    
    # Set up parallel execution if requested
    if args.parallel:
        os.environ["PYTEST_XDIST_WORKER_COUNT"] = str(os.cpu_count() or 4)
    
    success = True
    
    if args.type == "unit":
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.type == "e2e":
        success = run_e2e_tests(args.verbose, args.slow)
    elif args.type == "all":
        success = run_all_tests(args.verbose, args.coverage, args.slow)
    elif args.type == "lint":
        success = run_linting()
    elif args.type == "type":
        success = run_type_checking()
    elif args.type == "security":
        success = run_security_checks()
    elif args.type == "perf":
        success = run_performance_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 