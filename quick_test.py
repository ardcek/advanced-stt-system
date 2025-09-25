#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test Script for Ultra-Advanced STT System
Made by Mehmet Arda Çekiç © 2025

Fast and simple system testing
"""

import os
import sys
import subprocess
import tempfile
import wave
import numpy as np

def create_simple_test_audio(filename="test.wav", duration=5.0):
    """Create a simple test audio file"""
    
    # Create a simple sine wave audio for testing
    sample_rate = 16000
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    
    # Generate simple audio signal
    audio_data = 0.3 * np.sin(2 * np.pi * 440 * t)  # A note
    audio_data = np.clip(audio_data, -1, 1)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✅ Test audio created: {filename}")
    return filename

def test_basic_import():
    """Test basic module imports"""
    print("\n🧪 TESTING MODULE IMPORTS")
    print("=" * 40)
    
    modules_to_test = [
        ("modules.audio", "Audio processing"),
        ("modules.stt", "STT engine"),
        ("modules.nlp", "NLP processing"),
        ("modules.report", "Report generation"),
        ("modules.diarize", "Speaker diarization")
    ]
    
    success_count = 0
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {description}: {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {description}: {module_name} - {e}")
    
    print(f"\n📊 Basic imports: {success_count}/{len(modules_to_test)} successful")
    return success_count == len(modules_to_test)

def test_new_modules():
    """Test new practical modules"""
    print("\n🎓 TESTING NEW MODULES")
    print("=" * 40)
    
    new_modules = [
        ("modules.smart_academic_processor", "Smart Academic Processor"),
        ("modules.advanced_meeting_diarization", "Advanced Meeting Diarization"),
        ("modules.long_form_audio_processor", "Long Form Audio Processor"),
        ("modules.academic_meeting_intelligence", "Academic Meeting Intelligence"),
        ("modules.student_friendly_formats", "Student Friendly Formats")
    ]
    
    success_count = 0
    
    for module_name, description in new_modules:
        try:
            __import__(module_name)
            print(f"✅ {description}")
            success_count += 1
        except ImportError as e:
            print(f"⚠️ {description}: Not available ({str(e)[:50]}...)")
    
    print(f"\n📊 New modules: {success_count}/{len(new_modules)} available")
    return success_count > 0

def test_dependencies():
    """Test critical dependencies"""
    print("\n📦 TESTING DEPENDENCIES")
    print("=" * 40)
    
    critical_deps = [
        ("numpy", "NumPy"),
        ("librosa", "Librosa"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("openai", "OpenAI"),
        ("whisper", "Whisper")
    ]
    
    success_count = 0
    
    for dep, name in critical_deps:
        try:
            __import__(dep)
            print(f"✅ {name}")
            success_count += 1
        except ImportError:
            print(f"❌ {name} - Install with: pip install {dep}")
    
    print(f"\n📊 Dependencies: {success_count}/{len(critical_deps)} available")
    return success_count >= len(critical_deps) // 2

def run_basic_test():
    """Run basic functionality test"""
    print("\n🚀 RUNNING BASIC TEST")
    print("=" * 40)
    
    try:
        # Create test audio
        test_file = create_simple_test_audio("quick_test.wav", 3.0)
        
        # Run simplest possible test
        cmd = [
            sys.executable, "main.py",
            "--file", test_file,
            "--quality", "fastest",
            "--duration", "3"
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        print("⏳ Processing (this may take a moment)...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Basic processing test PASSED!")
            if result.stdout:
                print("📄 Sample output:")
                print(result.stdout[-200:])  # Show last 200 chars
        else:
            print(f"⚠️ Basic test completed with warnings:")
            if result.stderr:
                print(result.stderr[-200:])
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏱️ Test timed out (this is normal for first run)")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Run quick system test"""
    print("🌟 ULTRA-ADVANCED STT SYSTEM - QUICK TEST")
    print("Made by Mehmet Arda Çekiç © 2025")
    print("=" * 60)
    
    # Run tests
    basic_imports = test_basic_import()
    new_modules = test_new_modules() 
    deps_ok = test_dependencies()
    
    if basic_imports and deps_ok:
        basic_test = run_basic_test()
    else:
        print("\n⚠️ Skipping basic test due to import/dependency issues")
        basic_test = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", basic_imports),
        ("New Modules", new_modules),
        ("Dependencies", deps_ok),
        ("Basic Processing", basic_test)
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print("-" * 30)
    print(f"TOTAL: {passed}/{total} ({passed/total*100:.0f}%)")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS:")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n📝 Try these commands:")
        print("   python main.py --file audio.wav --quality ultra")
        print("   python main.py --file lecture.wav --academic --format student")
        print("   python main.py --file meeting.wav --mode meeting --diarization advanced")
    
    elif passed >= 3:
        print("✅ Core system working. Some advanced features may need setup.")
        if not new_modules:
            print("   ℹ️ New practical modules available but may need dependencies")
        print("\n📝 Try basic command:")
        print("   python main.py --file audio.wav --quality balanced")
    
    else:
        print("⚠️ System needs setup. Check these:")
        if not deps_ok:
            print("   📦 Install dependencies: pip install -r requirements.txt")
        if not basic_imports:
            print("   🔧 Check module structure and imports")
        print("\n📖 See USAGE.md for detailed setup instructions")

if __name__ == "__main__":
    main()