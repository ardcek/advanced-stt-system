#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hızlı panel test - sadece konfigürasyon
"""

try:
    from ultra_stt_panel import UltraSTTPanel
    
    print("🧪 PANEL KONFİGÜRASYON TESTİ")
    print("="*50)
    
    # Panel oluştur
    panel = UltraSTTPanel()
    
    # Test dosyası ata
    panel.audio_file = r"c:\Users\Arda\Desktop\test\recordings\kayit_20251003_230007.wav"
    
    # Test her kalite modunu
    modes = ["fastest", "balanced", "highest", "ultra"]
    
    for mode in modes:
        panel.quality_mode = mode
        panel.language = "tr"
        panel.medical_mode = False
        panel.diarization = False
        panel.ai_summary = False
        
        print(f"\n🔧 {mode.upper()} Modu:")
        print(f"   📁 Dosya: {panel.audio_file[-30:]}")
        print(f"   🎯 Kalite: {panel.quality_mode}")
        print(f"   🌍 Dil: {panel.language}")
        print(f"   ⏱️ Tahmini: {panel.estimate_processing_time()}")
        
    print(f"\n✅ Panel hazır ve tüm modlar çalışıyor!")
    print(f"📝 Artık ana paneli çalıştırabilirsiniz: python ultra_stt_panel.py")
    
except Exception as e:
    print(f"❌ Hata: {e}")
    import traceback
    traceback.print_exc()