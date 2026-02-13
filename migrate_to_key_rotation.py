"""
Helper script to update all agent files to use key rotation.
Run this once to migrate all agents to the new rotation system.

Usage:
    python migrate_to_key_rotation.py
"""

from pathlib import Path

def update_agent_file(filepath: Path) -> bool:
    """Update a single agent file to use key rotation."""
    content = filepath.read_text(encoding='utf-8')
    
    # Check if already using rotator
    if 'get_groq_client' in content:
        print(f"✓ {filepath.name} already using key rotation")
        return False
    
    # Check if needs update
    if 'AsyncGroq(api_key=settings.groq_api_key)' not in content:
        print(f"• {filepath.name} doesn't use standard pattern")
        return False
    
    # Add import at top if not present
    if 'from ..config.key_rotator import get_groq_client' not in content:
        # Find the imports section
        lines = content.split('\n')
        import_end_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('from ..config'):
                import_end_idx = i + 1
        
        # Insert import
        lines.insert(import_end_idx, 'from ..config.key_rotator import get_groq_client')
        content = '\n'.join(lines)
    
    # Replace the client initialization
    content = content.replace(
        'self.client = AsyncGroq(api_key=settings.groq_api_key)',
        'self.client = get_groq_client()'
    )
    
    # Write back
    filepath.write_text(content, encoding='utf-8')
    
    print(f"✅ Updated {filepath.name}")
    return True

def main():
    """Update all agent files."""
    print("🔄 Updating agents to use API key rotation...\n")
    
    # Get agents directory relative to this script
    script_dir = Path(__file__).parent
    agents_dir = script_dir / 'app' / 'agents'
    
    if not agents_dir.exists():
        print(f"❌ Agents directory not found: {agents_dir}")
        print("Make sure you're running this from the backend directory")
        return
    
    updated_count = 0
    for filepath in agents_dir.glob('*.py'):
        if filepath.name != '__init__.py':
            if update_agent_file(filepath):
                updated_count += 1
    
    print(f"\n✅ Updated {updated_count} agent files to use key rotation")
    print("🔑 All agents will now rotate through 5 API keys automatically!")
    print("\n💡 Next steps:")
    print("   1. Restart your backend server to apply changes")
    print("   2. Check logs for '✅ API key rotation initialized with 5 keys'")

if __name__ == "__main__":
    main()
