#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geliştirilmiş STT Modülü Test ve Örnek Kullanım
===============================================

Bu dosya yeni stt.py modülünün tüm özelliklerini gösterir ve test eder.
"""

import os
import sys
import time
from pathlib import Path

# Modül yolunu ekle
sys.path.insert(0, str(Path(__file__).parent))

from modules import stt
from modules.stt import transcribe_advanced, transcribe_simple, transcribe_with_speakers, transcribe_for_meeting

def test_basic_functionality():
    """Temel fonksiyonalite testleri"""
    print("=" * 60)
    print("TEMEL FONKSİYONALİTE TESTLERİ")
    print("=" * 60)
    
    # Mevcut ses dosyalarını kontrol et
    test_files = []
    for ext in ['.wav', '.mp3', '.m4a']:
        for file in Path('.').glob(f'*{ext}'):
            test_files.append(str(file))
    
    if not test_files:
        print("❌ Test için ses dosyası bulunamadı!")
        print("   Lütfen test dizinine .wav, .mp3 veya .m4a dosyası ekleyin.")
        return False
    
    test_file = test_files[0]
    print(f"📁 Test dosyası: {test_file}")
    
    # 1. En basit kullanım
    print("\n1️⃣  Basit Transkripsiyon:")
    try:
        start_time = time.time()
        simple_result = transcribe_simple(test_file)
        duration = time.time() - start_time
        
        print(f"✅ Başarılı! ({duration:.1f}s)")
        print(f"📝 Metin: {simple_result[:100]}...")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False
    
    # 2. Gelişmiş transkripsiyon
    print("\n2️⃣  Gelişmiş Transkripsiyon:")
    try:
        start_time = time.time()
        advanced_result = transcribe_advanced(
            test_file,
            quality="balanced",
            preprocess=True
        )
        duration = time.time() - start_time
        
        print(f"✅ Başarılı! ({duration:.1f}s)")
        print(f"📊 Model: {advanced_result.model_used}")
        print(f"🎯 Güven: {advanced_result.confidence:.2f}")
        print(f"⏱️  Süre: {advanced_result.duration:.1f}s")
        print(f"📝 Metin: {advanced_result.text[:150]}...")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False
    
    return True

def test_quality_levels():
    """Farklı kalite seviyelerini test et"""
    print("\n" + "=" * 60)
    print("KALİTE SEVİYELERİ TESTİ")
    print("=" * 60)
    
    test_files = list(Path('.').glob('*.wav'))[:1]  # İlk dosyayı al
    if not test_files:
        print("❌ Test dosyası bulunamadı!")
        return
    
    test_file = str(test_files[0])
    
    qualities = ["fastest", "balanced", "highest"]
    
    for quality in qualities:
        print(f"\n🔧 Kalite: {quality.upper()}")
        try:
            start_time = time.time()
            result = transcribe_advanced(
                test_file,
                quality=quality,
                preprocess=(quality != "fastest")
            )
            duration = time.time() - start_time
            
            print(f"✅ Tamamlandı ({duration:.1f}s)")
            print(f"📊 Model: {result.model_used}")
            print(f"🎯 Güven: {result.confidence:.2f}")
            print(f"📏 Uzunluk: {len(result.text)} karakter")
            
            # Kalite metrikleri
            if result.quality_metrics:
                qm = result.quality_metrics
                print(f"📈 Kalite Skoru: {qm.get('overall_score', 0):.2f}")
                if 'processing_speed_rtf' in qm:
                    print(f"⚡ RTF: {qm['processing_speed_rtf']:.2f}")
            
        except Exception as e:
            print(f"❌ Hata: {e}")

def test_meeting_features():
    """Toplantı özelliklerini test et"""
    print("\n" + "=" * 60)
    print("TOPLANTI ÖZELLİKLERİ TESTİ")
    print("=" * 60)
    
    test_files = list(Path('.').glob('*.wav'))[:1]
    if not test_files:
        print("❌ Test dosyası bulunamadı!")
        return
    
    test_file = str(test_files[0])
    
    print(f"📁 Dosya: {test_file}")
    try:
        start_time = time.time()
        meeting_result = transcribe_for_meeting(test_file)
        duration = time.time() - start_time
        
        print(f"✅ Başarılı! ({duration:.1f}s)")
        
        # Sonuçları göster
        print(f"\n📝 Transkripsiyon:")
        print(f"   {meeting_result['transcript'][:200]}...")
        
        print(f"\n👥 Konuşmacılar: {len(meeting_result.get('speakers', []))}")
        for speaker in meeting_result.get('speakers', []):
            print(f"   - {speaker}")
        
        print(f"\n📋 Görevler: {len(meeting_result.get('tasks', []))}")
        for i, task in enumerate(meeting_result.get('tasks', [])[:3], 1):
            print(f"   {i}. {task}")
        
        print(f"\n⚖️  Kararlar: {len(meeting_result.get('decisions', []))}")
        for i, decision in enumerate(meeting_result.get('decisions', [])[:3], 1):
            print(f"   {i}. {decision}")
        
        print(f"\n📊 Performans:")
        print(f"   🎯 Güven: {meeting_result['confidence']:.2f}")
        print(f"   📈 Kalite: {meeting_result['quality_score']:.2f}")
        print(f"   ⏱️  İşlem: {meeting_result['processing_time']:.1f}s")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

def test_audio_preprocessing():
    """Ses ön işleme testleri"""
    print("\n" + "=" * 60)
    print("SES ÖN İŞLEME TESTİ")
    print("=" * 60)
    
    # Test dosyasını kontrol et
    test_files = list(Path('.').glob('*.wav'))[:1]
    if not test_files:
        print("❌ Test dosyası bulunamadı!")
        return
    
    test_file = str(test_files[0])
    
    print(f"📁 Orijinal dosya: {test_file}")
    
    # Ses bilgilerini al
    try:
        audio_info = stt._get_audio_info(test_file)
        print(f"📊 Ses Bilgileri:")
        print(f"   ⏱️  Süre: {audio_info.duration:.1f}s")
        print(f"   🎵 Sample Rate: {audio_info.sample_rate}Hz")
        print(f"   📻 Kanal: {audio_info.channels}")
        print(f"   💿 Bit Derinliği: {audio_info.bit_depth}bit")
        print(f"   📦 Dosya Boyutu: {audio_info.file_size / 1024:.1f}KB")
        if audio_info.snr_estimate:
            print(f"   📈 SNR Tahmini: {audio_info.snr_estimate:.1f}dB")
        
    except Exception as e:
        print(f"❌ Ses bilgisi alınamadı: {e}")
    
    # Ön işleme ile ve olmadan karşılaştır
    print(f"\n🔬 Ön İşleme Karşılaştırması:")
    
    # Ön işlemesiz
    try:
        start_time = time.time()
        result_no_prep = transcribe_advanced(
            test_file,
            preprocess=False,
            quality="balanced"
        )
        time_no_prep = time.time() - start_time
        
        print(f"❌ Ön işlemesiz:")
        print(f"   ⏱️  Süre: {time_no_prep:.1f}s")
        print(f"   🎯 Güven: {result_no_prep.confidence:.2f}")
        print(f"   📏 Uzunluk: {len(result_no_prep.text)} karakter")
        
    except Exception as e:
        print(f"❌ Ön işlemesiz hata: {e}")
        result_no_prep = None
    
    # Ön işlemeli
    try:
        start_time = time.time()
        result_with_prep = transcribe_advanced(
            test_file,
            preprocess=True,
            quality="balanced"
        )
        time_with_prep = time.time() - start_time
        
        print(f"✅ Ön işlemeli:")
        print(f"   ⏱️  Süre: {time_with_prep:.1f}s")
        print(f"   🎯 Güven: {result_with_prep.confidence:.2f}")
        print(f"   📏 Uzunluk: {len(result_with_prep.text)} karakter")
        
        # Karşılaştırma
        if result_no_prep:
            conf_diff = result_with_prep.confidence - result_no_prep.confidence
            len_diff = len(result_with_prep.text) - len(result_no_prep.text)
            print(f"\n📊 İyileştirme:")
            print(f"   🎯 Güven farkı: {conf_diff:+.3f}")
            print(f"   📏 Uzunluk farkı: {len_diff:+d} karakter")
        
    except Exception as e:
        print(f"❌ Ön işlemeli hata: {e}")

def show_usage_examples():
    """Kullanım örneklerini göster"""
    print("\n" + "=" * 60)
    print("KULLANIM ÖRNEKLERİ")
    print("=" * 60)
    
    examples = [
        {
            "title": "En Basit Kullanım",
            "code": """
from modules.stt import transcribe_simple

# Sadece metin al
text = transcribe_simple("audio.wav")
print(text)
"""
        },
        {
            "title": "Yüksek Kalite Transkripsiyon",
            "code": """
from modules.stt import transcribe_advanced

# Tüm özelliklerle
result = transcribe_advanced(
    "audio.wav",
    quality="highest",
    preprocess=True,
    engine="auto"
)

print(f"Metin: {result.text}")
print(f"Güven: {result.confidence:.2f}")
print(f"Model: {result.model_used}")
"""
        },
        {
            "title": "Toplantı Kaydı Analizi",
            "code": """
from modules.stt import transcribe_for_meeting

# Toplantı için optimize
result = transcribe_for_meeting("meeting.wav")

print(f"Transkripsiyon: {result['transcript']}")
print(f"Görevler: {len(result['tasks'])}")
print(f"Kararlar: {len(result['decisions'])}")
print(f"Konuşmacılar: {len(result['speakers'])}")
"""
        },
        {
            "title": "Konuşmacı Ayrımı",
            "code": """
from modules.stt import transcribe_with_speakers

# Konuşmacı bilgileri ile
result = transcribe_with_speakers("audio.wav")

for segment in result['segments']:
    print(f"{segment['speaker']}: {segment['text']}")
"""
        }
    ]
    
    for example in examples:
        print(f"\n📌 {example['title']}:")
        print(example['code'])

def main():
    """Ana test fonksiyonu"""
    print("🚀 Geliştirilmiş STT Modülü Test ve Demo")
    print("=" * 60)
    
    # Bağımlılık kontrolü
    missing_deps = []
    
    try:
        import librosa
    except ImportError:
        missing_deps.append("librosa")
    
    try:
        import noisereduce
    except ImportError:
        missing_deps.append("noisereduce")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        missing_deps.append("faster-whisper")
    
    if missing_deps:
        print(f"⚠️  Eksik bağımlılıklar: {', '.join(missing_deps)}")
        print("   Kurulum: pip install -r requirements.txt")
        print()
    
    # Testleri çalıştır
    success = test_basic_functionality()
    
    if success:
        test_quality_levels()
        test_meeting_features() 
        test_audio_preprocessing()
    
    show_usage_examples()
    
    print("\n" + "=" * 60)
    print("✅ Test tamamlandı!")
    print("📚 Daha fazla bilgi için modül dokümantasyonunu inceleyin.")

if __name__ == "__main__":
    main()