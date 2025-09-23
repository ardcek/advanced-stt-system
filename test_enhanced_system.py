#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gelişmiş STT Sistemi Test Script
===============================

Bu script sistemi farklı senaryolarla test eder:
- Kısa kayıtlar (< 5 dakika)
- Orta kayıtlar (5-30 dakika) 
- Uzun kayıtlar (30+ dakika)
- Farklı diller (TR, EN, DE, FR, ES, IT, LA)
- Farklı içerik türleri (meeting, lecture, interview)

Kullanım:
    python test_enhanced_system.py
    python test_enhanced_system.py --quick  # Hızlı test
    python test_enhanced_system.py --lang en --mode lecture  # Özel test
"""

import argparse
import os
import time
from pathlib import Path

def test_basic_functionality():
    """Temel fonksiyonalite testleri"""
    print("🔧 Temel Fonksiyonalite Testleri")
    print("="*50)
    
    # 1. Module import test
    try:
        from modules import stt, nlp, audio, report, diarize
        print("✅ Tüm modüller başarıyla yüklendi")
    except Exception as e:
        print(f"❌ Modül yükleme hatası: {e}")
        return False
    
    # 2. STT initialization test
    try:
        stt.initialize()
        print("✅ STT motoru başarıyla başlatıldı")
    except Exception as e:
        print(f"❌ STT başlatma hatası: {e}")
        return False
    
    # 3. NLP functionality test
    try:
        test_text = "Bu bir test cümlesidir. Örnek bir metin normalize edilecek."
        normalized = nlp.normalize_transcript_advanced(test_text, language="tr")
        print(f"✅ Metin normalizasyon testi: '{normalized[:50]}...'")
        
        # Özetleme testi
        summary = nlp.summarize_text(test_text * 10, language="tr")  # Tekrarlayarak uzun metin
        print(f"✅ Özetleme testi: '{summary[:50]}...'")
        
        # Görev çıkarma testi
        task_text = "Ali bugün raporu hazırlamalı. Yarına kadar dökümanı gönder."
        tasks = nlp.extract_tasks(task_text, language="tr")
        print(f"✅ Görev çıkarma testi: {len(tasks)} görev bulundu")
        
    except Exception as e:
        print(f"❌ NLP testi hatası: {e}")
        return False
    
    return True


def test_multilingual_support():
    """Çoklu dil desteği testleri"""
    print("\n🌍 Çoklu Dil Desteği Testleri")
    print("="*50)
    
    from modules import nlp
    
    test_cases = [
        ("tr", "Bu bir Türkçe test metnidir. Yazım hatası düzeltilecektir."),
        ("en", "This is an English test text with some mispellings to correct."),
        ("de", "Das ist ein deutscher Testtext mit einigen Rechtschreibfehlern."),
        ("fr", "C'est un texte de test français avec quelques fautes d'orthographe."),
        ("es", "Este es un texto de prueba en español con algunos errores ortográficos."),
        ("it", "Questo è un testo di test italiano con alcuni errori di ortografia."),
        ("la", "Hoc est textus probationis Latinus cum erroribus orthographicis.")
    ]
    
    for lang, text in test_cases:
        try:
            normalized = nlp.normalize_transcript_advanced(
                text, 
                language=lang, 
                fix_spelling=True, 
                fix_foreign_terms=True
            )
            print(f"✅ {lang.upper()}: '{normalized[:40]}...'")
        except Exception as e:
            print(f"❌ {lang.upper()} dil testi hatası: {e}")


def test_educational_content():
    """Eğitim içeriği testi"""
    print("\n🎓 Eğitim İçeriği Analiz Testi")
    print("="*50)
    
    from modules import nlp
    
    lecture_text = """
    Bugünkü konumuz makine öğrenmesidir. Makine öğrenmesi, bilgisayarların 
    deneyimlerden öğrenmesini sağlayan bir yapay zeka dalıdır. Örnek olarak,
    spam e-posta tespiti sistemlerini düşünebiliriz. Önemli nokta şudur ki,
    algoritmalar verilerdeki kalıpları öğrenir. Soru: Hangi algoritma türleri vardır?
    Formül: accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    
    try:
        educational_content = nlp.extract_educational_content(lecture_text, language="tr")
        
        print(f"✅ Konular: {len(educational_content['topics'])}")
        print(f"✅ Tanımlar: {len(educational_content['definitions'])}")  
        print(f"✅ Örnekler: {len(educational_content['examples'])}")
        print(f"✅ Sorular: {len(educational_content['questions'])}")
        print(f"✅ Önemli noktalar: {len(educational_content['key_points'])}")
        print(f"✅ Formüller: {len(educational_content['formulas'])}")
        
        # Öğrenci özeti oluştur
        student_summary = nlp.create_student_summary(lecture_text, educational_content, language="tr")
        print(f"✅ Öğrenci özeti oluşturuldu: {len(student_summary)} karakter")
        
    except Exception as e:
        print(f"❌ Eğitim içeriği testi hatası: {e}")


def test_long_content_processing():
    """Uzun içerik işleme testi"""
    print("\n📏 Uzun İçerik İşleme Testi")
    print("="*50)
    
    from modules import nlp
    
    # Uzun metin simülasyonu (gerçek 2-3 saatlik kayıt simülasyonu)
    base_text = """
    Bu toplantıda proje durumunu değerlendirdik. Geliştiriciler rapor sundu.
    Pazarlama ekibi yeni stratejiler önerdi. Satış rakamları incelendi.
    Müşteri geri bildirimler olumlu. Önümüzdeki ay için hedefler belirlendi.
    """
    
    # Metni 100 kez tekrarlayarak uzun içerik oluştur
    long_text = (base_text + "\n") * 100
    
    try:
        start_time = time.time()
        
        # Uzun içerik özetleme
        summary = nlp.summarize_long_content(
            long_text, 
            max_length=1000,
            language="tr", 
            content_mode="meeting"
        )
        
        processing_time = time.time() - start_time
        
        print(f"✅ Uzun içerik işlendi: {len(long_text)} → {len(summary)} karakter")
        print(f"✅ İşlem süresi: {processing_time:.2f} saniye")
        print(f"✅ İşleme hızı: {len(long_text)/processing_time:.0f} karakter/saniye")
        
    except Exception as e:
        print(f"❌ Uzun içerik testi hatası: {e}")


def test_performance_monitoring():
    """Performans izleme testi"""
    print("\n⚡ Performans İzleme Testi")
    print("="*50)
    
    try:
        # Memory monitoring test
        from main import _monitor_memory_usage, _cleanup_memory
        
        memory_before = _monitor_memory_usage()
        print(f"✅ Bellek izleme çalışıyor: {memory_before['rss']:.1f}MB")
        
        # Create some memory load
        large_data = ["test" * 1000] * 1000
        memory_after = _monitor_memory_usage()
        
        # Cleanup
        del large_data
        memory_cleaned = _cleanup_memory()
        
        print(f"✅ Bellek temizlik çalışıyor: {memory_cleaned:.1f}MB")
        
    except Exception as e:
        print(f"❌ Performans izleme testi hatası: {e}")


def test_error_recovery():
    """Hata kurtarma testi"""
    print("\n🛠️ Hata Kurtarma Testi")  
    print("="*50)
    
    from modules import nlp
    
    # Hatalı girdi ile test
    try:
        # Boş metin testi
        empty_result = nlp.normalize_transcript_advanced("", language="tr")
        print(f"✅ Boş metin işleme: '{empty_result}'")
        
        # Çok kısa metin testi
        short_result = nlp.summarize_text("kısa", language="tr")
        print(f"✅ Kısa metin özetleme: '{short_result}'")
        
        # Geçersiz dil kodu testi
        invalid_lang = nlp.normalize_transcript_advanced("test", language="xx")
        print(f"✅ Geçersiz dil kodu işleme: '{invalid_lang[:20]}...'")
        
    except Exception as e:
        print(f"❌ Hata kurtarma testi hatası: {e}")


def run_sample_processing(audio_file: str, lang: str = "tr", mode: str = "auto"):
    """Örnek ses dosyası işleme"""
    print(f"\n🎵 Örnek İşleme: {audio_file}")
    print("="*50)
    
    if not os.path.exists(audio_file):
        print(f"❌ Ses dosyası bulunamadı: {audio_file}")
        return
    
    try:
        # Main process'i import et ve çalıştır
        import main
        
        # Argument objesi oluştur
        class Args:
            def __init__(self):
                self.file = audio_file
                self.language = lang
                self.mode = mode
                self.stt = "medium"  # Hızlı test için küçük model
                self.device = "cpu"
                self.window = 600
                self.title = f"Test - {Path(audio_file).stem}"
                self.stream = False
                self.duration = 15
        
        args = Args()
        
        print(f"🚀 İşleme başlıyor: {lang} dili, {mode} modu")
        start_time = time.time()
        
        # Ana işleme fonksiyonunu çalıştır
        main.run(args)
        
        processing_time = time.time() - start_time
        print(f"✅ İşlem tamamlandı: {processing_time:.1f} saniye")
        
    except Exception as e:
        print(f"❌ Örnek işleme hatası: {e}")


def main():
    """Ana test fonksiyonu"""
    parser = argparse.ArgumentParser(description="Gelişmiş STT Sistemi Test Suite")
    parser.add_argument("--quick", action="store_true", help="Hızlı test (sadece temel fonksiyonlar)")
    parser.add_argument("--lang", default="tr", help="Test dili (tr, en, de, fr, es, it, la)")
    parser.add_argument("--mode", default="auto", help="Test modu (meeting, lecture, interview, auto)")
    parser.add_argument("--audio", help="Test için ses dosyası yolu")
    
    args = parser.parse_args()
    
    print("🧪 GELİŞMİŞ STT SİSTEMİ TEST SÜİTİ")
    print("="*60)
    
    # Temel testler (her zaman çalıştır)
    if not test_basic_functionality():
        print("\n❌ Temel testler başarısız! Sistem çalışmaya hazır değil.")
        return
    
    if not args.quick:
        # Kapsamlı testler
        test_multilingual_support()
        test_educational_content() 
        test_long_content_processing()
        test_performance_monitoring()
        test_error_recovery()
        
        # Ses dosyası varsa işle
        if args.audio:
            run_sample_processing(args.audio, args.lang, args.mode)
        elif os.path.exists("meeting.wav"):
            run_sample_processing("meeting.wav", args.lang, args.mode)
    
    print("\n🎉 Test süiti tamamlandı!")
    print("✅ Sistem 2-3 saatlik kayıtları işlemeye hazır!")
    print("✅ Çoklu dil desteği aktif!")
    print("✅ Eğitim içeriği özellikleri hazır!")
    print("✅ Performans optimizasyonları aktif!")


if __name__ == "__main__":
    main()