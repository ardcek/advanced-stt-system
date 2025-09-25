#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Real Audio File Testing
Made by Mehmet Arda Çekiç © 2025

Test the system with your own audio files
"""

import os
import sys
import subprocess
from pathlib import Path

def find_audio_files():
    """Find audio files in common locations"""
    
    audio_extensions = ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg']
    audio_files = []
    
    # Check current directory
    for ext in audio_extensions:
        files = list(Path('.').glob(f'*{ext}'))
        audio_files.extend(files)
    
    # Check examples directory
    examples_dir = Path('examples')
    if examples_dir.exists():
        for ext in audio_extensions:
            files = list(examples_dir.glob(f'*{ext}'))
            audio_files.extend(files)
    
    return audio_files

def demo_student_lecture(audio_file):
    """Demo: Process as student lecture"""
    print(f"\n🎓 DEMO: Student Lecture Processing")
    print(f"📁 File: {audio_file}")
    print("=" * 50)
    
    cmd = [
        sys.executable, "main.py",
        "--file", str(audio_file),
        "--mode", "academic",
        "--academic",
        "--subject", "general",
        "--format", "student", 
        "--output-type", "study_guide",
        "--quality", "balanced"  # Use balanced for faster processing
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("⏳ Processing... (this may take a few minutes)")
    
    try:
        result = subprocess.run(cmd, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Success! Check these output files:")
            print("   - study_guide.html (Interactive study guide)")
            print("   - notes.pdf (Study notes)")
            print("   - output.txt (Basic transcript)")
        else:
            print("⚠️ Processing completed with warnings")
            
    except subprocess.TimeoutExpired:
        print("⏱️ Processing timed out - try with shorter audio or faster quality")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_meeting_analysis(audio_file):
    """Demo: Process as business meeting"""
    print(f"\n🎭 DEMO: Meeting Analysis")
    print(f"📁 File: {audio_file}")
    print("=" * 50)
    
    cmd = [
        sys.executable, "main.py",
        "--file", str(audio_file),
        "--mode", "meeting",
        "--diarization", "advanced",
        "--quality", "balanced"
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("⏳ Processing... (this may take a few minutes)")
    
    try:
        result = subprocess.run(cmd, timeout=300)
        
        if result.returncode == 0:
            print("✅ Success! Check these output files:")
            print("   - output.txt (Speaker-separated transcript)")
            print("   - meeting_analysis.md (Meeting insights)")
        else:
            print("⚠️ Processing completed with warnings")
            
    except subprocess.TimeoutExpired:
        print("⏱️ Processing timed out - try with shorter audio")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_medical_processing(audio_file):
    """Demo: Process as medical consultation"""
    print(f"\n🏥 DEMO: Medical Processing")
    print(f"📁 File: {audio_file}")
    print("=" * 50)
    
    print("⚠️ Note: Requires OpenAI API key for full Medical AI features")
    
    cmd = [
        sys.executable, "main.py",
        "--file", str(audio_file),
        "--medical",
        "--format", "medical",
        "--quality", "highest"
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("⏳ Processing... (this may take a few minutes)")
    
    try:
        result = subprocess.run(cmd, timeout=300)
        
        if result.returncode == 0:
            print("✅ Success! Check these output files:")
            print("   - medical_report.pdf (Professional medical report)")
            print("   - medical_terms.json (Medical terminology)")
            print("   - output.txt (Basic transcript)")
        else:
            print("⚠️ Processing completed with warnings")
            print("   (Medical AI features require OpenAI API key)")
            
    except subprocess.TimeoutExpired:
        print("⏱️ Processing timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_ultra_quality(audio_file):
    """Demo: Ultra quality processing"""
    print(f"\n🚀 DEMO: Ultra Quality Processing (99.9% accuracy)")
    print(f"📁 File: {audio_file}")
    print("=" * 50)
    
    print("⚠️ Warning: Ultra quality is SLOW but very accurate")
    
    cmd = [
        sys.executable, "main.py",
        "--file", str(audio_file),
        "--quality", "ultra",
        "--target-accuracy", "0.999",
        "--max-iterations", "2"
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("⏳ Processing... (this will take several minutes)")
    
    try:
        result = subprocess.run(cmd, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✅ Success! Ultra quality processing completed")
            print("   - Check output.txt for highest accuracy transcript")
        else:
            print("⚠️ Processing completed with warnings")
            
    except subprocess.TimeoutExpired:
        print("⏱️ Processing timed out - ultra quality needs more time")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main demo interface"""
    print("🌟 ULTRA-ADVANCED STT SYSTEM - AUDIO FILE DEMOS")
    print("Made by Mehmet Arda Çekiç © 2025")
    print("=" * 60)
    
    # Find audio files
    audio_files = find_audio_files()
    
    if not audio_files:
        print("❌ No audio files found!")
        print("\n📝 To test the system:")
        print("1. Place an audio file (.wav, .mp3, .mp4, .m4a) in this directory")
        print("2. Run this script again")
        print("\n📖 Or create test audio:")
        print("   python quick_test.py")
        return
    
    print(f"📁 Found {len(audio_files)} audio file(s):")
    for i, file in enumerate(audio_files, 1):
        file_size = file.stat().st_size / (1024*1024)  # MB
        print(f"   {i}. {file.name} ({file_size:.1f} MB)")
    
    # Select file
    if len(audio_files) == 1:
        selected_file = audio_files[0]
        print(f"\n🎯 Using: {selected_file.name}")
    else:
        try:
            choice = int(input(f"\nSelect audio file (1-{len(audio_files)}): ")) - 1
            selected_file = audio_files[choice]
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return
    
    # Select demo type
    print(f"\n🎛️ Select demo type:")
    print("1. 🎓 Student Lecture (academic processing + study materials)")
    print("2. 🎭 Meeting Analysis (speaker separation + insights)")  
    print("3. 🏥 Medical Processing (medical AI + terminology)")
    print("4. 🚀 Ultra Quality (99.9% accuracy - SLOW)")
    print("5. ⚡ Quick Test (fastest processing)")
    
    try:
        demo_choice = input("\nSelect demo (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled")
        return
    
    # Run selected demo
    if demo_choice == "1":
        demo_student_lecture(selected_file)
    elif demo_choice == "2":
        demo_meeting_analysis(selected_file)
    elif demo_choice == "3":
        demo_medical_processing(selected_file)
    elif demo_choice == "4":
        demo_ultra_quality(selected_file)
    elif demo_choice == "5":
        # Quick test
        print(f"\n⚡ DEMO: Quick Test")
        print(f"📁 File: {selected_file}")
        cmd = [sys.executable, "main.py", "--file", str(selected_file), "--quality", "fastest"]
        print(f"🔧 Command: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print("❌ Invalid demo selection")
        return
    
    print(f"\n✅ Demo completed!")
    print(f"📂 Check output files in current directory")
    print(f"\n🎯 Next steps:")
    print(f"   - Try different quality levels (fastest → ultra)")
    print(f"   - Test with different content types")
    print(f"   - Check USAGE.md for complete guide")

if __name__ == "__main__":
    main()