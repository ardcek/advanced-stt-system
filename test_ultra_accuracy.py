# Ultra High Accuracy STT Test

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules import stt, nlp, utils
import time

def test_ultra_accuracy():
    """Test ultra accuracy mode with different settings"""
    
    print("🎯 ULTRA ACCURACY TEST")
    print("=" * 60)
    
    # Test audio file (create a simple one if not exists)
    test_audio = "test_audio.wav"
    if not os.path.exists(test_audio):
        print("⚠️  Test audio dosyası bulunamadı")
        print("💡 Test için bir ses dosyası ekleyin: test_audio.wav")
        return False
    
    print(f"🎵 Test dosyası: {test_audio}")
    print()
    
    # Test different quality modes
    quality_modes = ["fastest", "balanced", "highest", "ultra"]
    
    results = {}
    
    for quality in quality_modes:
        print(f"📊 {quality.upper()} mode testi...")
        start_time = time.time()
        
        try:
            # Initialize STT engine
            stt.initialize()
            
            # Run transcription with different quality settings
            result = stt.transcribe_advanced(
                test_audio,
                language="tr",
                quality=quality,
                model_name="large-v3",
                device="cpu",
                content_type="meeting",
                long_form=False
            )
            
            duration = time.time() - start_time
            confidence = getattr(result, 'confidence', 0.0) * 100
            text_length = len(getattr(result, 'text', ''))
            
            results[quality] = {
                'duration': duration,
                'confidence': confidence,
                'text_length': text_length,
                'text': getattr(result, 'text', '')[:100] + "..." if text_length > 100 else getattr(result, 'text', '')
            }
            
            print(f"   ✅ Süre: {duration:.1f}s")
            print(f"   📈 Güvenilirlik: {confidence:.1f}%")
            print(f"   📝 Metin uzunluğu: {text_length} karakter")
            print()
            
        except Exception as e:
            print(f"   ❌ Hata: {e}")
            results[quality] = {'error': str(e)}
            print()
    
    # Results summary
    print("🏆 SONUÇ KARŞILAŞTIRMASI")
    print("=" * 60)
    
    for quality, result in results.items():
        if 'error' not in result:
            print(f"{quality.upper():>10}: {result['confidence']:>5.1f}% güven | {result['duration']:>5.1f}s süre")
        else:
            print(f"{quality.upper():>10}: HATA - {result['error']}")
    
    print("\n🎯 ULTRA MODE ÖZELLİKLERİ:")
    print("• Temperature sampling: 5 farklı sıcaklık değeri")  
    print("• Beam size: 10 (maksimum arama genişliği)")
    print("• Best of 5: En iyi 5 denemeden seçim")
    print("• Word timestamps: Kelime bazlı zaman damgaları")
    print("• Enhanced punctuation: Gelişmiş noktalama")
    print("• Patience: 2.0 (daha sabırlı decode)")
    
    return True

def test_accuracy_tips():
    """Display accuracy improvement tips"""
    print("\n💡 %100'E YAKIN DOĞRULUK İÇİN İPUÇLARI:")
    print("=" * 60)
    
    tips = [
        "🎤 Ses Kalitesi:",
        "   • Sessiz ortamda kayıt yapın",
        "   • Mikrofonu ağza 15-20cm yakın tutun", 
        "   • Rüzgar ve yankıyı engelleyin",
        "",
        "⚙️ Teknik Ayarlar:",
        "   • --quality ultra modunu kullanın",
        "   • --stt large-v3 modelini tercih edin",
        "   • Ses dosyasını WAV formatında kaydedin",
        "",
        "🗣️ Konuşma Teknikleri:",
        "   • Net ve yavaş konuşun",
        "   • Kelimeleri tam telaffuz edin",
        "   • Çok hızlı konuşmaktan kaçının",
        "",
        "📝 İçerik Optimizasyonu:",
        "   • Teknik terimleri custom_terms.txt'e ekleyin",
        "   • Kişi isimlerini önceden tanımlayın",
        "   • Yabancı kelimeleri belirtin",
        "",
        "🔧 Sistem Optimizasyonu:",
        "   • GPU kullanın (--device cuda)",
        "   • Yeterli RAM'e sahip olun (4GB+)",
        "   • Arka plan uygulamalarını kapatın"
    ]
    
    for tip in tips:
        print(tip)
    
    print(f"\n🚀 Kullanım örneği:")
    print("python main.py --file audio.wav --quality ultra --stt large-v3 --device cuda --language tr")

if __name__ == "__main__":
    success = test_ultra_accuracy()
    test_accuracy_tips()
    
    if success:
        print(f"\n✅ Test tamamlandı!")
    else:
        print(f"\n⚠️  Test dosyası gerekli!")