"""
Quick test script to verify Groq API keys
"""
import asyncio
from groq import AsyncGroq

async def test_api_key(api_key: str) -> bool:
    """Test if a Groq API key is valid."""
    try:
        client = AsyncGroq(api_key=api_key)
        # Try a simple completion
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        print(f"✓ API Key {api_key[:20]}... is VALID")
        return True
    except Exception as e:
        error_msg = str(e)
        if "organization_restricted" in error_msg:
            print(f"✗ API Key {api_key[:20]}... - ORGANIZATION RESTRICTED")
        elif "invalid" in error_msg.lower():
            print(f"✗ API Key {api_key[:20]}... - INVALID KEY")
        else:
            print(f"✗ API Key {api_key[:20]}... - ERROR: {error_msg[:100]}")
        return False

async def main():
    """Test all API keys from .env"""
    print("Testing Groq API Keys...\n")
    
    # Read keys from .env
    keys = []
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('GROQ_API_KEY'):
                    key = line.split('=')[1].strip()
                    if key and not key.startswith('your_'):
                        keys.append(key)
    except Exception as e:
        print(f"Error reading .env: {e}")
        return
    
    print(f"Found {len(keys)} API keys to test\n")
    
    valid_keys = 0
    for i, key in enumerate(keys, 1):
        print(f"Testing key {i}/{len(keys)}...")
        if await test_api_key(key):
            valid_keys += 1
        print()
    
    print(f"\nResult: {valid_keys}/{len(keys)} keys are valid")
    
    if valid_keys == 0:
        print("\n⚠️  ALL KEYS ARE INVALID OR RESTRICTED")
        print("You need to get new API keys from https://console.groq.com/keys")

if __name__ == "__main__":
    asyncio.run(main())
