#!/usr/bin/env python3
"""
Temporary script to convert all print statements to _write_stream calls
"""

import re

# Read the file
with open(r"c:\Users\ethan\OneDrive\Desktop\Aero-\src\aero\research_planner\research_planning_nodes.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace all print statements with _write_stream calls
# Pattern: print(f"...") -> _write_stream(writer, f"...\n")
# Pattern: print("...") -> _write_stream(writer, "...\n")

def replace_print(match):
    inner_content = match.group(1)
    # If it's an f-string, keep as is
    if inner_content.startswith('f"') or inner_content.startswith("f'"):
        return f"_write_stream(writer, {inner_content[:-1]}\\n\")"
    # If it's a regular string
    elif inner_content.startswith('"') or inner_content.startswith("'"):
        return f"_write_stream(writer, {inner_content[:-1]}\\n\")"
    # If it's a variable or expression
    else:
        return f"_write_stream(writer, str({inner_content}) + \"\\n\")"

# Replace all print statements
content = re.sub(r'print\(([^)]+)\)', replace_print, content)

# Write back to file
with open(r"c:\Users\ethan\OneDrive\Desktop\Aero-\src\aero\research_planner\research_planning_nodes.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ All print statements have been converted to _write_stream calls!")