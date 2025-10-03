#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kalite modları test scripti
"""

import os
import sys

def test_quality_modes():
    """Kalite modlarını test et"""
    
    print("🧪 KALİTE MODLARI TESTİ")
    print("="*50)
    
    # Test audio dosyası
    test_audio = r"c:\Users\Arda\Desktop\test\recordings\kayit_20251003_230007.wav"
    
    if not os.path.exists(test_audio):
        print(f"❌ Test audio dosyası bulunamadı: {test_audio}")
        return
    
    print(f"📁 Test dosyası: {test_audio}")
    
    # Ultra STT interface'i test et
    try:
        from ultra_stt_interface import UltraSTTInterface
        
        interface = UltraSTTInterface()
        
        # Her kalite modunu test et
        modes = ["fastest", "balanced", "highest", "ultra"]
        
        for mode in modes:
            print(f"\n🔧 {mode.upper()} Modu Test Ediliyor...")
            
            params = {
                'audio_file': test_audio,
                'quality_mode': mode,
                'language': 'tr',
                'use_medical': False,
                'use_diarization': False,
                'use_ai_summary': False,
                'ai_provider': 'local'
            }
            
            try:
                result = interface.process_with_mode(params)
                
                if result and 'transcription' in result:
                    transcript = result['transcription']
                    confidence = result.get('confidence', 0.0)
                    print(f"✅ {mode}: Başarılı ({len(transcript)} karakter, güven: {confidence:.2f})")
                else:
                    print(f"❌ {mode}: Başarısız - {result}")
                    
            except Exception as e:
                print(f"❌ {mode}: Hata - {e}")
                
    except ImportError as e:
        print(f"❌ Import hatası: {e}")
    except Exception as e:
        print(f"❌ Genel hata: {e}")

if __name__ == "__main__":
    test_quality_modes()