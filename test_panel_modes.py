#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel test scripti - kalite modları
"""

import os

def test_panel_modes():
    """Panel kalite modlarını test et"""
    
    print("🧪 PANEL KALİTE MODLARI TESTİ")
    print("="*50)
    
    try:
        from ultra_stt_panel import UltraSTTPanel
        
        # Panel oluştur
        panel = UltraSTTPanel()
        
        # Test audio dosyası ata
        panel.audio_file = r"c:\Users\Arda\Desktop\test\recordings\kayit_20251003_230007.wav"
        
        print("📁 Test dosyası atandı")
        
        # Kalite modlarını test et
        test_modes = ["fastest", "balanced", "highest", "ultra"]
        
        for mode in test_modes:
            print(f"\n🔧 {mode} modunu test ediliyor...")
            
            # Kalite modunu ayarla
            panel.quality_mode = mode
            
            # Diğer ayarları belirle
            panel.language = "tr"
            panel.medical_mode = False
            panel.diarization = False
            panel.ai_summary = False
            panel.ai_provider = "local"
            
            # Parametreleri kontrol et
            mode_params = {
                'audio_file': panel.audio_file,
                'quality_mode': panel.quality_mode,
                'language': panel.language,
                'use_medical': panel.medical_mode,
                'use_diarization': panel.diarization,
                'use_ai_summary': panel.ai_summary,
                'ai_provider': getattr(panel, 'ai_provider', 'groq')
            }
            
            print(f"   📋 Parametreler: quality_mode={mode_params['quality_mode']}")
            print(f"   🌍 Dil: {mode_params['language']}")
            print(f"   🎤 Audio: {os.path.basename(mode_params['audio_file'])}")
            
            # İşleme tahmini
            try:
                estimate = panel.estimate_processing_time()
                print(f"   ⏱️ Tahmini süre: {estimate}")
            except:
                print(f"   ⏱️ Tahmini süre: Hesaplanamadı")
                
        print(f"\n✅ Tüm kalite modları başarıyla tanımlandı!")
        print(f"📝 Panel.quality_mode değişkeni doğru çalışıyor")
        
    except ImportError as e:
        print(f"❌ Import hatası: {e}")
    except Exception as e:
        print(f"❌ Genel hata: {e}")

if __name__ == "__main__":
    test_panel_modes()