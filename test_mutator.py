#!/usr/bin/env python3
"""
Test script to verify mutator installation and functionality.
"""

import sys
import platform
import importlib
import subprocess

def test_python_architecture():
    """Test if Python is running on correct architecture."""
    print("🔍 Testing Python architecture...")
    arch = platform.machine()
    print(f"   Architecture: {arch}")
    
    if arch == 'arm64':
        print("   ✅ Running on native ARM64")
        return True
    elif arch == 'x86_64':
        print("   ⚠️  Running on x86_64 (may be through Rosetta)")
        return False
    else:
        print(f"   ❓ Unknown architecture: {arch}")
        return False

def test_package_import(package_name):
    """Test if a package can be imported."""
    try:
        importlib.import_module(package_name)
        print(f"   ✅ {package_name}")
        return True
    except ImportError as e:
        print(f"   ❌ {package_name}: {e}")
        return False

def test_critical_packages():
    """Test critical packages that commonly have architecture issues."""
    print("\n📦 Testing critical packages...")
    packages = [
        'numpy',
        'regex', 
        'tiktoken',
        'litellm',
        'chromadb',
        'mutator'
    ]
    
    success_count = 0
    for package in packages:
        if test_package_import(package):
            success_count += 1
    
    print(f"\n   {success_count}/{len(packages)} packages imported successfully")
    return success_count == len(packages)

def test_mutator_cli():
    """Test if mutator CLI command works."""
    print("\n🚀 Testing mutator CLI...")
    try:
        result = subprocess.run(['mutator', '--help'], 
                              capture_output=True, 
                              text=True, 
                              timeout=30)
        
        if result.returncode == 0:
            print("   ✅ mutator --help works")
            return True
        else:
            print(f"   ❌ mutator --help failed with code {result.returncode}")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ mutator --help timed out")
        return False
    except FileNotFoundError:
        print("   ❌ mutator command not found")
        return False
    except Exception as e:
        print(f"   ❌ mutator test failed: {e}")
        return False

def test_package_architectures():
    """Test if packages are compiled for the correct architecture."""
    print("\n🏗️  Testing package architectures...")
    
    try:
        import numpy
        numpy_file = numpy.__file__.replace('__init__.py', 'core/_multiarray_umath.cpython-312-darwin.so')
        
        result = subprocess.run(['file', numpy_file], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            if 'arm64' in result.stdout:
                print("   ✅ numpy compiled for ARM64")
                return True
            elif 'x86_64' in result.stdout:
                print("   ⚠️  numpy compiled for x86_64")
                return False
            else:
                print(f"   ❓ numpy architecture unclear: {result.stdout}")
                return False
        else:
            print("   ❓ Could not check numpy architecture")
            return False
            
    except Exception as e:
        print(f"   ❓ Architecture check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Mutator Installation Test")
    print("=" * 40)
    
    tests = [
        ("Python Architecture", test_python_architecture),
        ("Package Imports", test_critical_packages),
        ("Package Architectures", test_package_architectures),
        ("Mutator CLI", test_mutator_cli),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Mutator should work correctly.")
        print("🚀 You can now run: mutator chat")
    else:
        print("⚠️  Some tests failed. Please check the installation.")
        print("💡 Try running the fix_architecture.sh script or follow INSTALL_FIX.md")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 