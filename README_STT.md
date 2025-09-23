# Geliştirilmiş Speech-to-Text (STT) Modülü

Bu modül, ses kayıtlarından mükemmel transkripsiyon elde etmek için tasarlanmış gelişmiş bir ses-metin dönüştürme sistemidir. Özellikle Türkçe toplantı kayıtları, eğitim videoları ve profesyonel ses kayıtları için optimize edilmiştir.

## 🌟 Özellikler

### Çoklu Model Desteği
- **Whisper Ailesi**: OpenAI Whisper (GPU/CPU), Faster-Whisper
- **Cloud API'ler**: Azure Cognitive Services, Google Cloud Speech
- **Hibrit Yaklaşım**: Birden fazla modeli paralel çalıştırıp en iyi sonucu seçme

### Gelişmiş Ses Ön İşleme
- Gürültü azaltma (Noise Reduction)
- Ses normalizasyonu ve temizleme
- Gelişmiş Voice Activity Detection (VAD)
- Otomatik ses kalitesi analizi ve SNR tahmini

### Akıllı Post-Processing
- Bağlam farkında metin düzeltme
- Teknik terim ve özel isim optimizasyonu
- Gelişmiş tekrar temizleme algoritması
- Cümle yapısı düzeltme

### Kalite Kontrolü
- Otomatik güven skoru hesaplama
- Transkripsiyon kalitesi değerlendirmesi
- Realtime performans metrikleri

### Konuşmacı Analizi
- Temel konuşmacı segmentasyonu
- Sessizlik bazlı konuşmacı değişimi tespiti
- Segment düzeyinde konuşmacı bilgileri

## 📦 Kurulum

```bash
# Temel bağımlılıkları yükle
pip install -r requirements.txt

# GPU desteği için (CUDA gerekli)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Opsiyonel: Cloud API'ler için
pip install azure-cognitiveservices-speech google-cloud-speech
```

### Gerekli Kütüphaneler
```
faster-whisper>=1.0.0
librosa>=0.10.0
noisereduce>=3.0.0
webrtcvad>=2.0.10
pydub>=0.25.1
rapidfuzz>=3.0.0
numpy>=1.24.0
torch>=2.0.0
transformers>=4.30.0
```

## 🚀 Hızlı Başlangıç

### En Basit Kullanım
```python
from modules.stt import transcribe_simple

# Sadece metin al
text = transcribe_simple("meeting.wav")
print(text)
```

### Gelişmiş Transkripsiyon
```python
from modules.stt import transcribe_advanced

# Tüm özelliklerle
result = transcribe_advanced(
    "meeting.wav",
    language="tr",           # Dil kodu
    quality="highest",       # Kalite seviyesi
    preprocess=True,         # Ses ön işleme
    engine="auto"           # Model seçimi
)

print(f"📝 Transkripsiyon: {result.text}")
print(f"🎯 Güven Skoru: {result.confidence:.2f}")
print(f"⚡ Model: {result.model_used}")
print(f"⏱️ Süre: {result.processing_time:.1f}s")
```

### Toplantı Kaydı Analizi
```python
from modules.stt import transcribe_for_meeting

# Toplantı için optimize edilmiş analiz
result = transcribe_for_meeting("meeting.wav")

print(f"📝 Transkripsiyon: {result['transcript']}")
print(f"👥 Konuşmacı Sayısı: {len(result['speakers'])}")
print(f"📋 Tespit Edilen Görevler: {len(result['tasks'])}")
print(f"⚖️ Alınan Kararlar: {len(result['decisions'])}")

# Görevleri listele
for i, task in enumerate(result['tasks'], 1):
    print(f"  {i}. {task}")
```

### Konuşmacı Ayrımı
```python
from modules.stt import transcribe_with_speakers

result = transcribe_with_speakers("meeting.wav")

# Konuşmacı bazında segment gösterimi
for segment in result['segments']:
    speaker = segment['speaker']
    text = segment['text']
    start = segment['start']
    end = segment['end']
    
    print(f"[{start:.1f}s-{end:.1f}s] {speaker}: {text}")
```

## ⚙️ Yapılandırma Seçenekleri

### Kalite Seviyeleri
- **`fastest`**: En hızlı (tiny model, minimal işlem)
- **`balanced`**: Dengeli (medium model, ön işleme)
- **`highest`**: En yüksek (large model, tüm özellikler)
- **`hybrid`**: Çoklu model karşılaştırma

### Model/Engine Seçenekleri
- **`auto`**: Ses özelliklerine göre otomatik seçim
- **`whisper`**: OpenAI Whisper modelleri
- **`azure`**: Azure Cognitive Services
- **`google`**: Google Cloud Speech
- **`hybrid`**: Birden fazla model paralel

### Dil Kodları
- `tr`: Türkçe (varsayılan)
- `en`: İngilizce
- `de`: Almanca
- `fr`: Fransızca
- `es`: İspanyolca
- Desteklenen diğer diller...

## 🔧 Gelişmiş Kullanım

### Cloud API Yapılandırması

#### Azure Cognitive Services
```python
import os
os.environ["AZURE_SPEECH_KEY"] = "your-api-key"
os.environ["AZURE_SPEECH_REGION"] = "eastus"

result = transcribe_advanced("audio.wav", engine="azure")
```

#### Google Cloud Speech
```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/credentials.json"

result = transcribe_advanced("audio.wav", engine="google")
```

### Özel Terimler Ekleme
```python
# custom_terms.txt dosyası oluştur
with open("custom_terms.txt", "w", encoding="utf-8") as f:
    f.write("""
# Şirket özel terimleri
KurumsalApp
VeriTabanıSistemi
ProjeAlfa
MüşteriPortalı

# Teknik terimler
Kubernetes
PostgreSQL
Redis
""")
```

### Metin Düzeltme Kuralları
```python
# corrections.txt dosyası oluştur
with open("corrections.txt", "w", encoding="utf-8") as f:
    f.write("""
# Yaygın STT hataları
payton => Python
cubernets => Kubernetes
veri tabanı => veritabanı
deil => değil
""")
```

## 📊 Performans ve Kalite

### Kalite Metrikleri
```python
result = transcribe_advanced("audio.wav", quality="highest")

if result.quality_metrics:
    metrics = result.quality_metrics
    print(f"📈 Genel Kalite: {metrics['overall_score']:.2f}")
    print(f"📝 Metin Kalitesi: {metrics['text_quality']:.2f}")
    print(f"⏱️ Zamanlama Doğruluğu: {metrics['timing_accuracy']:.2f}")
    print(f"📏 Tamlık: {metrics['completeness']:.2f}")
```

### Performans Optimizasyonu
```python
# Ses özelliklerine göre otomatik parametre ayarlama
from modules.stt import _get_audio_info, _adaptive_parameters

audio_info = _get_audio_info("audio.wav")
params = _adaptive_parameters(audio_info)

print(f"📊 Ses Bilgileri:")
print(f"  ⏱️ Süre: {audio_info.duration:.1f}s")
print(f"  📈 SNR: {audio_info.snr_estimate:.1f}dB")
print(f"🔧 Önerilen Parametreler:")
print(f"  beam_size: {params['beam_size']}")
print(f"  temperature: {params['temperature']}")
```

## 🎯 En İyi Uygulamalar

### Ses Dosyası Hazırlığı
1. **Format**: WAV, 16kHz, mono tercih edilir
2. **Kalite**: SNR > 20dB için en iyi sonuçlar
3. **Uzunluk**: Segment başına 30 saniye - 10 dakika ideal
4. **Gürültü**: Mümkün olduğunca temiz kayıt

### Model Seçimi
- **Kısa kayıtlar** (< 5 dk): `quality="balanced"`
- **Uzun kayıtlar** (> 1 saat): `engine="azure"` veya `"google"`
- **Düşük kalite ses**: `quality="highest"` + `preprocess=True`
- **Yüksek hız gerekli**: `quality="fastest"`

### Toplu İşlem
```python
import os
from pathlib import Path

# Klasördeki tüm ses dosyalarını işle
audio_dir = Path("audio_files")
results = []

for audio_file in audio_dir.glob("*.wav"):
    print(f"İşleniyor: {audio_file.name}")
    
    try:
        result = transcribe_for_meeting(str(audio_file))
        results.append({
            "file": audio_file.name,
            "transcript": result["transcript"],
            "confidence": result["confidence"],
            "tasks": result["tasks"]
        })
    except Exception as e:
        print(f"Hata: {audio_file.name} - {e}")

# Sonuçları kaydet
import json
with open("batch_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

## 🐛 Sorun Giderme

### Yaygın Hatalar

**1. Import Hataları**
```bash
# Eksik kütüphaneler için
pip install librosa noisereduce webrtcvad pydub
```

**2. CUDA/GPU Sorunları**
```python
# CPU'ya zorla
result = transcribe_advanced("audio.wav", device="cpu")
```

**3. Bellek Sorunları**
```python
# Uzun kayıtlar için chunk'lara böl
result = transcribe_advanced(
    "long_audio.wav", 
    quality="balanced",  # highest yerine
    preprocess=False     # Bellek tasarrufu için
)
```

**4. Düşük Kalite Sonuçlar**
```python
# Daha agresif ön işleme
result = transcribe_advanced(
    "noisy_audio.wav",
    quality="highest",
    preprocess=True,
    engine="hybrid"  # Birden fazla model dene
)
```

## 📈 Geliştirme Planı

### Gelecek Özellikler
- [ ] Real-time transkripsiyon desteği
- [ ] Gelişmiş speaker diarization (pyannote.audio)
- [ ] Emotion detection entegrasyonu
- [ ] WebRTC tabanlı noise cancellation
- [ ] Custom model fine-tuning desteği
- [ ] REST API interface

### Katkıda Bulunma
1. Repository'yi fork edin
2. Feature branch oluşturun
3. Değişiklikleri commit edin
4. Pull request gönderin

## 📄 Lisans

Bu proje MIT lisansı altında yayınlanmıştır.

## 🆘 Destek

Sorunlarınız için:
1. GitHub Issues kullanın
2. Dokumentasyonu kontrol edin
3. Test dosyasını çalıştırın: `python test_enhanced_stt.py`

---

**Not**: Bu modül sürekli geliştirilmektedir. En son sürüm için repository'yi takip edin.

---

**Made by Mehmet Arda Çekiç** © 2025