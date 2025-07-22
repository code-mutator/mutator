"""
Comprehensive tests for the indentation fix functionality.
Tests cover various programming languages and corner cases.
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add the mutator to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mutator.tools.categories.indentation_fixer import (
    fix_indentation, 
    _detect_language, 
    _remove_comments_and_strings,
    _is_opening_line,
    _is_closing_line,
    _detect_indentation_pattern,
    _analyze_file_indentation,
    _find_common_indent_size
)


class TestLanguageDetection:
    """Test language detection from file extensions."""
    
    def test_python_detection(self):
        """Test Python file detection."""
        assert _detect_language("test.py") == "python"
        assert _detect_language("/path/to/file.py") == "python"
    
    def test_javascript_detection(self):
        """Test JavaScript file detection."""
        assert _detect_language("test.js") == "javascript"
        assert _detect_language("test.jsx") == "javascript"
        assert _detect_language("test.ts") == "typescript"
        assert _detect_language("test.tsx") == "typescript"
    
    def test_c_cpp_detection(self):
        """Test C/C++ file detection."""
        assert _detect_language("test.c") == "c"
        assert _detect_language("test.cpp") == "cpp"
        assert _detect_language("test.cc") == "cpp"
        assert _detect_language("test.cxx") == "cpp"
        assert _detect_language("test.h") == "c"
        assert _detect_language("test.hpp") == "cpp"
    
    def test_unknown_extension(self):
        """Test unknown file extension."""
        assert _detect_language("test.unknown") == "unknown"
        assert _detect_language("") == "unknown"
        assert _detect_language("noextension") == "unknown"


class TestCommentAndStringRemoval:
    """Test comment and string removal functionality."""
    
    def test_python_comments(self):
        """Test Python comment removal."""
        # Basic comment
        assert _remove_comments_and_strings("print('hello')  # comment", "python") == "print('hello')  "
        
        # Comment at start
        assert _remove_comments_and_strings("# This is a comment", "python") == ""
        
        # Hash in string should be preserved
        assert _remove_comments_and_strings("print('hello # world')", "python") == "print('hello # world')"
        
        # Mixed quotes
        assert _remove_comments_and_strings('print("hello # world")  # comment', "python") == 'print("hello # world")  '
    
    def test_javascript_comments(self):
        """Test JavaScript comment removal."""
        # Single-line comment
        assert _remove_comments_and_strings("console.log('hello');  // comment", "javascript") == "console.log('hello');  "
        
        # Multi-line comment
        assert _remove_comments_and_strings("/* comment */ console.log('hello');", "javascript") == " console.log('hello');"
        
        # Comment markers in strings
        assert _remove_comments_and_strings("console.log('hello // world');", "javascript") == "console.log('hello // world');"
        
        # Multi-line comment in middle
        assert _remove_comments_and_strings("var x = /* comment */ 5;", "javascript") == "var x =  5;"
    
    def test_c_comments(self):
        """Test C-style comment removal."""
        # Single-line comment
        assert _remove_comments_and_strings("printf('hello');  // comment", "c") == "printf('hello');  "
        
        # Multi-line comment
        assert _remove_comments_and_strings("/* comment */ printf('hello');", "c") == " printf('hello');"
    
    def test_escaped_quotes(self):
        """Test handling of escaped quotes."""
        # Python
        assert _remove_comments_and_strings("print('hello \\'world\\' test')  # comment", "python") == "print('hello \\'world\\' test')  "
        
        # JavaScript
        assert _remove_comments_and_strings("console.log('hello \\'world\\' test');  // comment", "javascript") == "console.log('hello \\'world\\' test');  "


class TestOpeningClosingLines:
    """Test opening and closing line detection."""
    
    def test_python_opening_lines(self):
        """Test Python opening line detection."""
        # Basic if statement
        assert _is_opening_line("if condition:", "python") == True
        
        # Function definition
        assert _is_opening_line("def function():", "python") == True
        
        # Class definition
        assert _is_opening_line("class MyClass:", "python") == True
        
        # For loop
        assert _is_opening_line("for i in range(10):", "python") == True
        
        # While loop
        assert _is_opening_line("while True:", "python") == True
        
        # Try block
        assert _is_opening_line("try:", "python") == True
        
        # With statement
        assert _is_opening_line("with open('file') as f:", "python") == True
        
        # Not opening lines
        assert _is_opening_line("print('hello')", "python") == False
        assert _is_opening_line("x = 5", "python") == False
        assert _is_opening_line("return value", "python") == False
    
    def test_python_opening_lines_with_comments(self):
        """Test Python opening lines with comments."""
        # Comment after colon
        assert _is_opening_line("if condition:  # comment", "python") == True
        
        # Comment before colon
        assert _is_opening_line("if condition  # comment\n:", "python") == False  # Invalid syntax
        
        # Colon in comment should not trigger
        assert _is_opening_line("# if condition:", "python") == False
        
        # Colon in string should not trigger
        assert _is_opening_line("print('if condition:')", "python") == False
    
    def test_javascript_opening_lines(self):
        """Test JavaScript opening line detection."""
        # Basic if statement
        assert _is_opening_line("if (condition) {", "javascript") == True
        
        # Function definition
        assert _is_opening_line("function test() {", "javascript") == True
        
        # For loop
        assert _is_opening_line("for (let i = 0; i < 10; i++) {", "javascript") == True
        
        # While loop
        assert _is_opening_line("while (true) {", "javascript") == True
        
        # Object/array literals
        assert _is_opening_line("const obj = {", "javascript") == True
        assert _is_opening_line("const arr = [", "javascript") == True
        
        # Not opening lines
        assert _is_opening_line("console.log('hello');", "javascript") == False
        assert _is_opening_line("const x = 5;", "javascript") == False
        assert _is_opening_line("return value;", "javascript") == False
    
    def test_javascript_opening_lines_with_comments(self):
        """Test JavaScript opening lines with comments."""
        # Comment after brace
        assert _is_opening_line("if (condition) {  // comment", "javascript") == True
        
        # Brace in comment should not trigger
        assert _is_opening_line("// if (condition) {", "javascript") == False
        
        # Brace in string should not trigger
        assert _is_opening_line("console.log('if (condition) {');", "javascript") == False
        
        # Multi-line comment
        assert _is_opening_line("if (condition) { /* comment */", "javascript") == True
    
    def test_python_closing_lines(self):
        """Test Python closing line detection."""
        # Else statement
        assert _is_closing_line("else:", "python") == True
        
        # Elif statement
        assert _is_closing_line("elif condition:", "python") == True
        
        # Except block
        assert _is_closing_line("except Exception:", "python") == True
        
        # Finally block
        assert _is_closing_line("finally:", "python") == True
        
        # Not closing lines
        assert _is_closing_line("print('hello')", "python") == False
        assert _is_closing_line("if condition:", "python") == False
    
    def test_javascript_closing_lines(self):
        """Test JavaScript closing line detection."""
        # Closing braces
        assert _is_closing_line("}", "javascript") == True
        assert _is_closing_line("})", "javascript") == True
        assert _is_closing_line("}]", "javascript") == True
        
        # Not closing lines
        assert _is_closing_line("console.log('hello');", "javascript") == False
        assert _is_closing_line("if (condition) {", "javascript") == False
    
    def test_closing_lines_with_comments(self):
        """Test closing lines with comments."""
        # Python
        assert _is_closing_line("else:  # comment", "python") == True
        assert _is_closing_line("# else:", "python") == False
        
        # JavaScript
        assert _is_closing_line("}  // comment", "javascript") == True
        assert _is_closing_line("// }", "javascript") == False


class TestIndentationDetection:
    """Test indentation pattern detection."""
    
    def test_space_indentation_detection(self):
        """Test space indentation detection."""
        content = """def main():
    if True:
        print('hello')
        for i in range(3):
            print(i)
"""
        style, size = _analyze_file_indentation(content)
        assert style == ' '
        assert size == 4
    
    def test_tab_indentation_detection(self):
        """Test tab indentation detection."""
        content = """def main():
\tif True:
\t\tprint('hello')
\t\tfor i in range(3):
\t\t\tprint(i)
"""
        style, size = _analyze_file_indentation(content)
        assert style == '\t'
        assert size == 1
    
    def test_mixed_indentation_detection(self):
        """Test mixed indentation detection (should prefer most common)."""
        content = """def main():
    if True:  # 4 spaces
        print('hello')  # 8 spaces
\telse:  # 1 tab
\t\tprint('world')  # 2 tabs
"""
        style, size = _analyze_file_indentation(content)
        # Should prefer spaces since they're more common
        assert style == ' '
        assert size == 4
    
    def test_two_space_indentation(self):
        """Test 2-space indentation detection."""
        content = """function main() {
  if (condition) {
    console.log('hello');
    for (let i = 0; i < 3; i++) {
      console.log(i);
    }
  }
}
"""
        style, size = _analyze_file_indentation(content)
        assert style == ' '
        assert size == 2
    
    def test_base_indentation_detection(self):
        """Test base indentation detection from old content."""
        old_content = """    if condition:
        print('hello')
        for i in range(3):
            print(i)"""
        
        full_content = """def main():
    if condition:
        print('hello')
        for i in range(3):
            print(i)
    else:
        print('world')
"""
        
        base_indent, style, size = _detect_indentation_pattern(old_content, full_content)
        assert base_indent == "    "
        assert style == " "
        assert size == 4


class TestIndentationFix:
    """Test the main indentation fix functionality."""
    
    def test_python_basic_indentation_fix(self):
        """Test basic Python indentation fix."""
        old_content = """    if condition:
        print('hello')"""
        
        full_content = """def main():
    if condition:
        print('hello')
    else:
        print('world')
"""
        
        new_content = """if new_condition:
print('new hello')
for i in range(3):
print(i)"""
        
        expected = """    if new_condition:
        print('new hello')
    for i in range(3):
        print(i)"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.py")
        assert result == expected
    
    def test_javascript_basic_indentation_fix(self):
        """Test basic JavaScript indentation fix."""
        old_content = """    if (condition) {
        console.log('hello');
    }"""
        
        full_content = """function main() {
    if (condition) {
        console.log('hello');
    } else {
        console.log('world');
    }
}
"""
        
        new_content = """if (newCondition) {
console.log('new hello');
}
for (let i = 0; i < 3; i++) {
console.log(i);
}"""
        
        expected = """    if (newCondition) {
        console.log('new hello');
    }
    for (let i = 0; i < 3; i++) {
        console.log(i);
    }"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.js")
        assert result == expected
    
    def test_nested_blocks_indentation(self):
        """Test nested blocks indentation."""
        old_content = """    if condition:
        for i in range(3):
            if i > 0:
                print(i)"""
        
        full_content = """def main():
    if condition:
        for i in range(3):
            if i > 0:
                print(i)
"""
        
        new_content = """if new_condition:
for i in range(5):
if i > 2:
print(i * 2)
else:
print(i)"""
        
        expected = """    if new_condition:
        for i in range(5):
            if i > 2:
                print(i * 2)
            else:
                print(i)"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.py")
        assert result == expected
    
    def test_comments_in_indentation_fix(self):
        """Test indentation fix with comments."""
        old_content = """    if condition:
        print('hello')  # comment"""
        
        full_content = """def main():
    if condition:
        print('hello')  # comment
    else:
        print('world')
"""
        
        new_content = """if new_condition:  # new comment
print('new hello')
# standalone comment
for i in range(3):
print(i)  # loop comment"""
        
        expected = """    if new_condition:  # new comment
        print('new hello')
        # standalone comment
    for i in range(3):
        print(i)  # loop comment"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.py")
        assert result == expected
    
    def test_strings_with_brackets_indentation(self):
        """Test indentation fix with strings containing brackets."""
        old_content = """    if condition:
        print('hello')"""
        
        full_content = """def main():
    if condition:
        print('hello')
"""
        
        new_content = """if new_condition:
print('hello { world }')
print("another { string }")
for item in items:
print(f'item: {item}')"""
        
        expected = """    if new_condition:
        print('hello { world }')
        print("another { string }")
    for item in items:
        print(f'item: {item}')"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.py")
        assert result == expected
    
    def test_javascript_comments_with_brackets(self):
        """Test JavaScript indentation fix with comments containing brackets."""
        old_content = """    if (condition) {
        console.log('hello');
    }"""
        
        full_content = """function main() {
    if (condition) {
        console.log('hello');
    }
}
"""
        
        new_content = """if (newCondition) {
console.log('hello { world }');  // comment with {
/* multi-line comment with {
   more content } */
console.log('test');
}"""
        
        expected = """    if (newCondition) {
        console.log('hello { world }');  // comment with {
        /* multi-line comment with {
           more content } */
        console.log('test');
    }"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.js")
        assert result == expected
    
    def test_tab_indentation_fix(self):
        """Test indentation fix with tabs."""
        old_content = """\tif condition:
\t\tprint('hello')"""
        
        full_content = """def main():
\tif condition:
\t\tprint('hello')
\telse:
\t\tprint('world')
"""
        
        new_content = """if new_condition:
print('new hello')
for i in range(3):
print(i)"""
        
        expected = """\tif new_condition:
\t\tprint('new hello')
\tfor i in range(3):
\t\tprint(i)"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.py")
        assert result == expected
    
    def test_empty_lines_preservation(self):
        """Test that empty lines are preserved."""
        old_content = """    if condition:
        print('hello')"""
        
        full_content = """def main():
    if condition:
        print('hello')
"""
        
        new_content = """if new_condition:
    print('new hello')

    for i in range(3):
        print(i)

print('done')"""
        
        expected = """    if new_condition:
        print('new hello')

        for i in range(3):
            print(i)

    print('done')"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.py")
        assert result == expected
    
    def test_no_indentation_needed(self):
        """Test when no indentation fix is needed."""
        old_content = """if condition:
    print('hello')"""
        
        full_content = """if condition:
    print('hello')
else:
    print('world')
"""
        
        new_content = """if new_condition:
    print('new hello')
for i in range(3):
    print(i)"""
        
        expected = """if new_condition:
    print('new hello')
for i in range(3):
    print(i)"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.py")
        assert result == expected
    
    def test_complex_javascript_indentation(self):
        """Test complex JavaScript indentation scenarios."""
        old_content = """    function test() {
        if (condition) {
            console.log('hello');
        }
    }"""
        
        full_content = """class MyClass {
    function test() {
        if (condition) {
            console.log('hello');
        }
    }
}
"""
        
        new_content = """function newTest() {
const items = [
1, 2, 3
];
for (const item of items) {
if (item > 1) {
console.log(item);
}
}
}"""
        
        expected = """    function newTest() {
        const items = [
            1, 2, 3
        ];
        for (const item of items) {
            if (item > 1) {
                console.log(item);
            }
        }
    }"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.js")
        assert result == expected


class TestEdgeCases:
    """Test edge cases and corner cases."""
    
    def test_empty_content(self):
        """Test with empty content."""
        result = fix_indentation("", "test", "test", "test.py")
        assert result == "test"
        
        result = fix_indentation("test", "", "test", "test.py")
        assert result == ""
    
    def test_only_whitespace(self):
        """Test with only whitespace content."""
        result = fix_indentation("   ", "test", "test", "test.py")
        assert result == "test"
        
        result = fix_indentation("test", "   ", "test", "test.py")
        assert result == "   "
    
    def test_single_line_content(self):
        """Test with single line content."""
        old_content = "    print('hello')"
        new_content = "print('world')"
        full_content = "def main():\n    print('hello')\n"
        
        result = fix_indentation(old_content, new_content, full_content, "test.py")
        assert result == "    print('world')"
    
    def test_unknown_language(self):
        """Test with unknown file extension."""
        old_content = "    some code"
        new_content = "new code"
        full_content = "def main():\n    some code\n"
        
        result = fix_indentation(old_content, new_content, full_content, "test.unknown")
        assert result == "    new code"
    
    def test_no_file_extension(self):
        """Test with no file extension."""
        old_content = "    some code"
        new_content = "new code"
        full_content = "def main():\n    some code\n"
        
        result = fix_indentation(old_content, new_content, full_content, "")
        assert result == "    new code"
    
    def test_malformed_brackets(self):
        """Test with malformed brackets in comments."""
        old_content = """    if condition:
        print('hello')"""
        
        full_content = """def main():
    if condition:
        print('hello')
"""
        
        new_content = """if new_condition:
print('hello { world')  # missing closing }
print('another } world')  # missing opening {
for item in items:
print(item)"""
        
        expected = """    if new_condition:
        print('hello { world')  # missing closing }
        print('another } world')  # missing opening {
    for item in items:
        print(item)"""
        
        result = fix_indentation(old_content, new_content, full_content, "test.py")
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 