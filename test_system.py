# Simple Test Script

import os
import sys

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def test_imports():
    """Test if all modules can be imported"""
    try:
        import stt
        import nlp
        import audio
        print("✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic system functionality"""
    try:
        from modules import stt, nlp
        
        # Test STT initialization
        stt_engine = stt.STTEngine()
        print("✅ STT engine created")
        
        # Test NLP basic function
        test_text = "This is a test sentence."
        normalized = nlp.normalize_text_basic(test_text)
        print(f"✅ Text normalization: {normalized[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 BASIC SYSTEM TEST")
    print("=" * 50)
    
    print("\n📦 Testing imports...")
    import_success = test_imports()
    
    if import_success:
        print("\n🔧 Testing basic functionality...")
        func_success = test_basic_functionality()
        
        if func_success:
            print("\n✅ ALL TESTS PASSED!")
            print("🚀 System is ready to use!")
        else:
            print("\n❌ Functionality tests failed")
    else:
        print("\n❌ Import tests failed")
        print("💡 Try: pip install -r requirements.txt")