#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple System Test - No Heavy Dependencies
Made by Mehmet Arda Çekiç © 2025
"""

import os
import sys
import subprocess

def test_basic_main():
    """Test main.py functionality"""
    print("🌟 ULTRA-ADVANCED STT SYSTEM - SIMPLE TEST")
    print("=" * 50)
    
    print("🧪 Testing main.py import...")
    
    try:
        # Test if main.py can be imported
        sys.path.append('.')
        import main
        print("✅ main.py import successful")
        
        # Test basic modules
        print("\n📦 Testing basic modules:")
        
        modules = ['audio', 'stt', 'nlp', 'report', 'diarize']
        success_count = 0
        
        for module in modules:
            try:
                exec(f"from modules import {module}")
                print(f"✅ {module}")
                success_count += 1
            except Exception as e:
                print(f"⚠️ {module}: {str(e)[:50]}...")
        
        print(f"\n📊 Basic modules: {success_count}/{len(modules)} loaded")
        
        if success_count >= 3:
            print("\n🎯 Core system is functional!")
            print("\n📝 You can test with:")
            print("   python main.py --help")
            print("   python main.py --file audio.wav --quality fastest")
            
            # Test help command
            print("\n🔧 Testing help command...")
            try:
                result = subprocess.run([sys.executable, "main.py", "--help"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("✅ Help command works")
                    print("📖 Help output available")
                else:
                    print("⚠️ Help command had issues")
            except Exception as e:
                print(f"⚠️ Help command test: {e}")
        
        else:
            print("\n⚠️ Some modules need attention")
            
    except Exception as e:
        print(f"❌ Main import failed: {e}")
        return False
    
    return True

def create_test_instructions():
    """Create test instructions"""
    print(f"\n" + "=" * 60)
    print("📋 MANUAL TESTING INSTRUCTIONS")
    print("=" * 60)
    
    print("""
🎯 How to test the system manually:

1. 📁 GET AUDIO FILE:
   - Download any audio file (.wav, .mp3, .mp4)
   - Or record a short voice memo on your phone
   - Place it in this directory

2. 🚀 BASIC TEST:
   python main.py --file your_audio.wav --quality fastest

3. 🎓 STUDENT MODE TEST:
   python main.py --file lecture.wav --academic --format student

4. 🏥 MEDICAL MODE TEST (requires OpenAI API):
   python main.py --file medical.wav --medical --format medical

5. 🎭 MEETING MODE TEST:
   python main.py --file meeting.wav --mode meeting --diarization advanced

📊 Expected Results:
- output.txt: Basic transcript
- output.md: Markdown formatted
- Various specialized outputs based on mode

⚠️ Notes:
- First run will download ML models (takes time)
- Medical AI requires OpenAI API key
- Ultra quality is very slow but accurate
- Use "balanced" or "fastest" for quick testing

🎯 Success Indicators:
✅ No Python import errors
✅ Audio file processed successfully  
✅ Output files created
✅ Reasonable transcript quality

❌ Common Issues:
- Missing audio file: Specify correct path
- Import errors: Install requirements
- API errors: Set OpenAI key for medical features
- Timeout: Use shorter audio or faster quality
    """)

def main():
    """Run simple test"""
    
    # Basic functionality test
    basic_ok = test_basic_main()
    
    # Instructions for manual testing
    create_test_instructions()
    
    print(f"\n" + "=" * 60)
    print("📊 SIMPLE TEST COMPLETE")
    print("=" * 60)
    
    if basic_ok:
        print("✅ Core system appears functional")
        print("📝 Ready for manual testing with audio files")
    else:
        print("⚠️ Some issues detected - check dependencies")
        print("📦 Try: pip install -r requirements.txt")
    
    print("\n🚀 Next Steps:")
    print("1. Place an audio file in this directory")
    print("2. Run: python main.py --file your_audio.wav --quality fastest")
    print("3. Check output.txt for results")
    print("4. Try different modes and formats")

if __name__ == "__main__":
    main()