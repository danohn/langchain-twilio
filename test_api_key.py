#!/usr/bin/env python3
"""
Simple script to test if your OpenAI API key is working correctly.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if the OpenAI API key is valid and working"""
    print("\n" + "="*60)
    print("Testing OpenAI API Key")
    print("="*60)

    # Check if key exists
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY found in environment or .env file")
        return False

    print(f"✓ API Key found: {api_key[:20]}...{api_key[-4:]}")
    print(f"✓ Key length: {len(api_key)} characters")

    # Test the API key
    print("\nTesting API connection...")
    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'user', 'content': 'Reply with just the word "success"'}
            ],
            max_tokens=10
        )

        result = response.choices[0].message.content.strip()
        print(f"✓ API call successful!")
        print(f"✓ Model responded: {result}")
        print("\n" + "="*60)
        print("✅ Your API key is working correctly!")
        print("="*60 + "\n")
        return True

    except Exception as e:
        print(f"❌ API call failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        print("\n" + "="*60)
        print("Possible issues:")
        print("="*60)
        print("1. Invalid or expired API key")
        print("2. Insufficient permissions on the API key")
        print("3. Billing/quota issues with your OpenAI account")
        print("4. Network connectivity issues")
        print("\nPlease:")
        print("- Check your API key at: https://platform.openai.com/api-keys")
        print("- Verify your account has credits: https://platform.openai.com/usage")
        print("- Try generating a new API key if needed")
        print("="*60 + "\n")
        return False

if __name__ == "__main__":
    test_api_key()
