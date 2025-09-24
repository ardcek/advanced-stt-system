#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Advanced STT Sistemi - %99.9 Doğruluk Hedefi
Made by Mehmet Arda Çekiç © 2025

🚀 ÖZELLİKLER:
- Ultra-advanced audio preprocessing (SpectralNoiseReducer, EchoCanceller)
- Multi-model STT ensemble (Whisper + Azure + Google + IBM Watson)
- AI-powered post-processing (GPT-based correction)
- Advanced VAD & speaker diarization
- Adaptive learning system
- Ultra quality mode with 99.9% accuracy target
"""

import argparse
import os
import gc
import time
from modules import audio, stt, nlp, report, diarize

# Ultra Quality System import
try:
    from modules.ultra_quality_mode import UltraQualitySTT, transcribe_with_ultra_quality
    _HAS_ULTRA_MODE = True
    print("✅ Ultra Quality Mode available")
except ImportError:
    _HAS_ULTRA_MODE = False
    print("⚠️ Ultra Quality Mode not available")

# Performans izleme için opsiyonel import
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _monitor_memory_usage():
    """Bellek kullanımını izle (opsiyonel psutil ile)"""
    if _HAS_PSUTIL:
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    else:
        return {'rss': 0, 'vms': 0, 'percent': 0}


def _cleanup_memory():
    """Bellek temizliği yap"""
    gc.collect()
    if _HAS_PSUTIL:
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0
    return 0


def _log_performance_metrics(step_name: str, start_time: float, memory_before: dict):
    """Performans metriklerini logla"""
    duration = time.time() - start_time
    memory_after = _monitor_memory_usage()
    
    print(f"   ⏱️ {step_name}: {duration:.1f}s")
    if _HAS_PSUTIL and memory_before:
        memory_delta = memory_after['rss'] - memory_before['rss']
        print(f"   🧠 Bellek kullanımı: {memory_after['rss']:.1f}MB ({memory_delta:+.1f}MB)")
    
    return memory_after


def run(args):
    """Ana uygulama mantığı - Ultra Quality Mode ile %99.9 doğruluk"""
    
    # Performans izleme başlatma
    total_start_time = time.time()
    initial_memory = _monitor_memory_usage()
    
    print("🌟 ULTRA-ADVANCED STT SİSTEMİ BAŞLATILIYOR...")
    print(f"🎯 Hedef Doğruluk: {args.target_accuracy:.1%}")
    print(f"🔥 Quality Mode: {args.quality}")
    
    if _HAS_PSUTIL:
        print(f"   💾 Başlangıç bellek kullanımı: {initial_memory['rss']:.1f}MB")
    
    # Ultra Quality Mode kontrolü
    use_ultra_mode = (args.quality == "ultra" and _HAS_ULTRA_MODE)
    
    if use_ultra_mode:
        print("🚀 ULTRA QUALITY MODE AKTIF!")
        print("   • Advanced Audio Preprocessing")
        print("   • Multi-Model STT Ensemble") 
        print("   • AI-Powered Post-Processing")
        print("   • Advanced VAD & Diarization")
        print("   • Adaptive Learning System")
    
    # Dosya boyutu kontrolü için uzun kayıt optimizasyonu
    is_long_recording = False
    file_size_mb = 0
    
    if args.file and os.path.exists(args.file):
        file_size_mb = os.path.getsize(args.file) / (1024 * 1024)
        is_long_recording = file_size_mb > 50  # 50MB üzeri uzun kayıt sayılır
        
        if is_long_recording:
            print(f"🔍 Büyük dosya tespit edildi ({file_size_mb:.1f} MB)")
            print("⚡ Uzun kayıt optimizasyonu devreye giriyor...")
            
            # Uzun kayıtlar için optimized parametreler
            if args.window == 600:  # Default değerse
                args.window = 1800  # 30 dakikalık pencereler
            print(f"   • Pencere süresi: {args.window} saniye ({args.window//60} dakika)")

    # STT motor başlatma (sadece ultra mode değilse)
    if not use_ultra_mode:
        print("🚀 Standart STT motoru başlatılıyor...")
        stt.initialize()

    # 1) Ses kaynağını belirleme
    if args.file:
        wav_path = args.file
        title = args.title or os.path.splitext(os.path.basename(args.file))[0]
        print(f"🎵 Dosya işleniyor: '{title}' ({file_size_mb:.1f} MB)")
    elif args.stream:
        title = args.title or "Canlı Kayıt"
        print("🎙️ Canlı kayıt başlıyor (ENTER = durdur)...")
        wav_path = audio.record_stream(filename="meeting.wav")
    else:
        title = f"{args.title} ({args.duration}s)"
        print(f"⏱️ {args.duration} saniyelik kayıt başlıyor...")
        wav_path = audio.record_audio(duration=args.duration, filename="meeting.wav")

    # 2) Ultra Quality Transkripsiyon
    print(f"🎤 Transkripsiyon başlıyor...")
    
    # Performans ölçümü başlat
    transcription_start = time.time()
    memory_before_transcription = _monitor_memory_usage()
    
    # Ultra Quality Mode ile transkripsiyon
    if use_ultra_mode:
        print("🌟 ULTRA QUALITY MODE ile transkripsiyon...")
        try:
            ultra_stt = UltraQualitySTT()
            ultra_result = ultra_stt.transcribe_ultra_quality(
                audio_path=wav_path,
                user_id=args.user_id,
                context_type=args.mode,
                target_accuracy=args.target_accuracy,
                max_iterations=args.max_iterations
            )
            
            raw_text = ultra_result.text
            confidence_score = ultra_result.confidence
            audio_quality = f"Ultra ({ultra_result.quality_metrics.overall_score:.3f})"
            
            # Ultra mode'da segments oluştur (basitleştirilmiş)
            segments = []
            if ultra_result.speakers:
                for speaker in ultra_result.speakers:
                    segments.append({
                        'start': speaker['start_time'],
                        'end': speaker['end_time'],
                        'text': raw_text,  # Basitleştirilmiş
                        'speaker': speaker['speaker_id']
                    })
            else:
                # Tek segment
                segments = [{
                    'start': 0.0,
                    'end': 10.0,  # Placeholder
                    'text': raw_text,
                    'speaker': 'Speaker_0'
                }]
            
            print(f"✅ ULTRA QUALITY transkripsiyon tamamlandı!")
            print(f"   🎯 Final Confidence: {confidence_score:.3f}")
            print(f"   🏆 Quality Score: {ultra_result.quality_metrics.overall_score:.3f}")
            print(f"   🔄 Iterations Used: {ultra_result.processing_stats.get('iterations_used', 1)}")
            print(f"   ✅ Target Achieved: {ultra_result.processing_stats.get('target_achieved', False)}")
            
        except Exception as e:
            print(f"❌ Ultra Quality transkripsiyon hatası: {e}")
            print("   🔄 Standart transkripsiyon modu ile devam ediliyor...")
            use_ultra_mode = False  # Fallback to standard mode
            
    # Standart transkripsiyon (fallback veya ultra mode mevcut değilse)
    if not use_ultra_mode:
        print(f"🎤 Standart transkripsiyon ({args.stt} model, {args.language} dil)...")
        try:
            # Gelişmiş transkripsiyon çağrısı
            result = stt.transcribe_advanced(
                wav_path,
                model_name=args.stt,
                device=args.device,
                language=args.language,
                content_type=args.mode,
                quality=args.quality,
                long_form=is_long_recording,
                beam_size=10 if args.quality == "ultra" else (5 if is_long_recording else 1),
                vad_threshold=0.2 if args.quality == "ultra" else (0.3 if is_long_recording else 0.5)
            )
            
            segments = result['segments']
            raw_text = result['text']
            confidence_score = result.get('confidence', 0.0)
            audio_quality = result.get('audio_quality', 'Bilinmiyor')
            
            print(f"✅ Standart transkripsiyon tamamlandı! Güvenilirlik: {confidence_score:.1%}")
            
        except Exception as e:
            print(f"❌ Transkripsiyon hatası: {e}")
            # Fallback: basit transkripsiyon
            print("   🔄 Basit transkripsiyon modu ile yeniden deneniyor...")
            try:
                r = stt.transcribe(wav_path, language=args.language, model_size=args.stt, device=args.device)
                segments = r.get("segments", [])
                raw_text = r.get("text", "")
                confidence_score = 0.0
                audio_quality = "Bilinmiyor"
                print("   ✅ Basit transkripsiyon başarılı!")
            except Exception as e2:
                print(f"   ❌ Basit transkripsiyon da başarısız: {e2}")
                # En son fallback: boş sonuç
                segments, raw_text = [], ""
                confidence_score, audio_quality = 0.0, "Hata"
    
    # Transkripsiyon performansını logla
    memory_after_transcription = _log_performance_metrics(
        "Transkripsiyon", transcription_start, memory_before_transcription
    )

    # 3) Gelişmiş metin düzeltme
    print("✏️ Gelişmiş metin düzeltme ve normalizasyon...")
    
    # Performans ölçümü
    normalization_start = time.time()
    memory_before_norm = _monitor_memory_usage()
    
    try:
        text = nlp.normalize_transcript_advanced(
            raw_text,
            language=args.language,
            fix_spelling=True,
            fix_foreign_terms=True
        )
    except:
        # Fallback: basit normalizasyon
        print("   🔄 Basit normalizasyon ile devam ediliyor...")
        try:
            text = nlp.normalize_transcript_advanced(raw_text, language=args.language)
        except Exception as e2:
            print(f"   ⚠️ Basit normalizasyon da başarısız: {e2}")
            text = raw_text  # En basit fallback
    
    report.save_transcript(text, "transcript.txt")
    
    # Normalizasyon performansını logla
    _log_performance_metrics("Normalizasyon", normalization_start, memory_before_norm)
    
    # Bellek temizliği (uzun kayıtlarda önemli)
    if is_long_recording:
        before_cleanup = _cleanup_memory()
        print(f"   🧹 Bellek temizliği yapıldı")

    # 4) Gelişmiş özetleme sistemi
    print("📝 Gelişmiş özetleme başlıyor...")
    
    # Performans ölçümü
    summarization_start = time.time()
    memory_before_summ = _monitor_memory_usage()
    
    try:
        if is_long_recording:
            # Uzun kayıtlar için chunk-based summarization
            general = nlp.summarize_long_content(
                text,
                max_length=2000,  # Daha uzun özetler
                language=args.language,
                content_mode=args.mode
            )
        else:
            general = nlp.summarize_text(text, language=args.language)
            
        windows = nlp.summarize_by_windows(segments, window_sec=args.window, language=args.language)
        
    except Exception as e:
        print(f"⚠️ Özetleme hatası, basit özetleme kullanılıyor: {e}")
        # Fallback
        general = nlp.summarize_text(text)
        windows = nlp.summarize_by_windows(segments, window_sec=args.window)
    
    report.save_summary(general, "summary.txt")
    
    # Özetleme performansını logla
    _log_performance_metrics("Özetleme", summarization_start, memory_before_summ)

    # 4.5) Eğitim içeriği analizi (lecture mode için)
    educational_content = None
    student_summary = None
    
    if args.mode == "lecture":
        print("🎓 Eğitim içeriği analizi yapılıyor...")
        try:
            educational_content = nlp.extract_educational_content(text, language=args.language)
            student_summary = nlp.create_student_summary(text, educational_content, language=args.language)
            
            # Öğrenci notlarını kaydet
            with open("student_notes.md", "w", encoding="utf-8") as f:
                f.write(student_summary)
            print("   ✅ Öğrenci notları oluşturuldu: student_notes.md")
            
        except Exception as e:
            print(f"   ⚠️ Eğitim içeriği analizi hatası: {e}")
            educational_content = {}

    # 5) Aksiyon ve karar çıkarımı
    print("🎯 Aksiyon ve kararlar çıkarılıyor...")
    
    try:
        tasks = nlp.extract_tasks(text, language=args.language)
        decisions = nlp.extract_decisions(text, language=args.language)
    except:
        # Fallback
        tasks = nlp.extract_tasks(text)
        decisions = nlp.extract_decisions(text)
    
    if tasks:
        report.save_list(tasks, "tasks.txt")

    # 6) Çıktı dosyaları
    print("📄 Raporlar oluşturuluyor...")
    diar = diarize.assign_speakers(segments)
    report.export_srt(segments, "meeting.srt")
    report.export_notes_md(title, general, windows, "notes.md")
    report.build_docx(title, general, tasks, decisions, diar, window_summaries=windows)

    # 7) Sonuç raporu
    total_duration = time.time() - total_start_time
    final_memory = _monitor_memory_usage()
    
    print("\n" + "="*60)
    print("✅ İŞLEM TAMAMLANDI!")
    print("="*60)
    print(f"⏱️ Toplam İşlem Süresi: {total_duration:.1f} saniye")
    if _HAS_PSUTIL:
        memory_used = final_memory['rss'] - initial_memory['rss']
        print(f"💾 Toplam Bellek Kullanımı: {memory_used:+.1f}MB")
        print(f"🚀 İşleme Hızı: {file_size_mb/total_duration:.1f} MB/s")
    print(f"📊 Kalite Değerlendirmesi:")
    print(f"   • Güvenilirlik Skoru: {confidence_score:.1%}")
    print(f"   • Ses Kalitesi: {audio_quality}")
    print(f"⏱️ İşleme İstatistikleri:")
    print(f"   • Segment Sayısı: {len(segments)} adet")
    print(f"   • Metin Uzunluğu: {len(text.split())} kelime")
    print(f"   • Dosya Boyutu: {file_size_mb:.1f} MB")
    
    if is_long_recording:
        print(f"🔥 Uzun Kayıt Optimizasyonu:")
        print(f"   • Pencere Sayısı: {len(windows)} adet")
        print(f"   • Pencere Süresi: {args.window//60} dakika")
        if file_size_mb > 0 and total_duration > 0:
            print(f"   • Gerçek zamanlı işleme oranı: {total_duration/(file_size_mb*0.1):.1f}x")
        
    print(f"\n📁 Oluşturulan Dosyalar:")
    print(f"   • 📝 notes.md - Ana rapor")
    print(f"   • 🎬 meeting.srt - Altyazı dosyası") 
    print(f"   • 📖 meeting_minutes_*.docx - Word dökümanı")
    print(f"   • 🔤 transcript.txt - Ham metin")
    print(f"   • 📋 summary.txt - Özet")
    if tasks:
        print(f"   • ✅ tasks.txt - Görevler")
    
    # Eğitim modu özel çıktıları
    if args.mode == "lecture" and student_summary:
        print(f"   • 🎓 student_notes.md - Öğrenci notları")
        
        # Eğitim istatistikleri
        if educational_content:
            print(f"\n📚 Eğitim İçeriği İstatistikleri:")
            print(f"   • Konu sayısı: {len(educational_content.get('topics', []))}")
            print(f"   • Tanım sayısı: {len(educational_content.get('definitions', []))}")
            print(f"   • Örnek sayısı: {len(educational_content.get('examples', []))}")
            print(f"   • Soru sayısı: {len(educational_content.get('questions', []))}")
            print(f"   • Önemli nokta sayısı: {len(educational_content.get('key_points', []))}")
            if educational_content.get('formulas'):
                print(f"   • Formül sayısı: {len(educational_content['formulas'])}")
    
    print("="*60)

def parse_args():
    p = argparse.ArgumentParser(description="Ultra-Advanced STT System – %99.9 Accuracy Target")
    p.add_argument("--file", help="Kayıt dosyası (mp3/mp4/m4a/wav)")
    p.add_argument("--stream", action="store_true", help="Sınırsız canlı kayıt (ENTER ile durdur)")
    p.add_argument("--duration", type=int, default=15, help="Süreli kayıt (sn)")
    p.add_argument("--window", type=int, default=600, help="Pencere özeti süresi (sn, varsayılan 600=10 dk)")
    p.add_argument("--stt", default="large-v3", choices=["tiny","base","small","medium","large-v2","large-v3"], help="Whisper modeli (standart modda)")
    p.add_argument("--device", default="cpu", choices=["cpu","cuda"], help="STT cihazı")
    p.add_argument("--language", default="tr", choices=["tr","en","de","fr","es","it","la"], help="Kayıt dili")
    p.add_argument("--mode", default="auto", choices=["meeting","lecture","interview","auto"], help="İçerik türü")
    p.add_argument("--quality", default="ultra", choices=["fastest","balanced","highest","ultra"], help="Doğruluk seviyesi (ultra = %99.9 hedef)")
    p.add_argument("--target-accuracy", type=float, default=0.999, help="Hedef doğruluk oranı (0.999 = %99.9)")
    p.add_argument("--max-iterations", type=int, default=3, help="Ultra modda maksimum iterasyon sayısı")
    p.add_argument("--user-id", default="default", help="Adaptive learning için kullanıcı ID")
    p.add_argument("--title", default="Ultra-Advanced STT – Notlar")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())