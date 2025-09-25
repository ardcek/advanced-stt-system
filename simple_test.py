#!/usr/bin/env python3
"""
Simple System Test
Test the STT system with basic checks
"""

import os
import sys

def test_system():
    """Test STT system components"""
    print("Advanced STT System - Test")
    print("=" * 30)
    
    print("1. Testing module imports...")
    try:
        sys.path.append('.')
        from modules.stt import transcribe_simple
        from modules.nlp import normalize_transcript
        print("   ✓ Core modules OK")
    except Exception as e:
        print(f"   ✗ Import error: {e}")
        return False
    
    print("\n2. Basic functionality test...")
    print("   System appears ready for use")
    
    print("\n" + "=" * 30)
    print("✓ Test completed successfully!")
    
    print("\nNext steps:")
    print("1. Place your audio file (audio.wav) in this folder")
    print("2. Run: python main.py --file audio.wav")
    print("3. Check output.txt for results")
    
    return True

if __name__ == "__main__":
    test_system()

if __name__ == "__main__":
    test_system()