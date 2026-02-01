"""
Quick diagnostic script to check tool loading
"""
from app.config import settings
from app.tools import get_tool_definitions

print("=== Tool Loading Diagnostic ===")
print(f"enable_tool_calling: {settings.enable_tool_calling}")

tools = get_tool_definitions() if settings.enable_tool_calling else []
print(f"Number of tools loaded: {len(tools)}")

if tools:
    print("\nTool names:")
    for tool in tools:
        print(f"  - {tool['function']['name']}")
else:
    print("\n⚠️ WARNING: No tools loaded!")

print(f"\ntools variable evaluation:")
print(f"  bool(tools): {bool(tools)}")
print(f"  tools if tools else None: {tools if tools else None}")
print(f"  'auto' if tools else None: {'auto' if tools else None}")
