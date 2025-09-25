# 🎤 Advanced Speech-to-Text (STT) System

**Made by Mehmet Arda Çekiç** © 2025

Bu sistem, ses dosyalarını yüksek doğrulukla metne çeviren gelişmiş bir STT sistemidir. Öğrenciler, profesyoneller ve araştırmacılar için tasarlanmıştır.

## ✨ Özellikler

- 🎯 **Yüksek Doğruluk**: Gelişmiş AI algoritmaları
- 🎓 **Ders Kaydı Desteği**: Uzun ders kaydlarını transkript eder
- 👥 **Konuşmacı Ayrıştırma**: Farklı konuşmacıları ayırır
- 🏥 **Medikal Terimler**: Tıbbi terminolojiyi destekler
- 🌐 **Çok Dilli**: Türkçe ve İngilizce desteği
- ⚡ **Hızlı İşlem**: Gerçek zamanlı transkripsiyon

## 🚀 Kurulum

1. **Repoyu klonlayın:**
```bash
git clone https://github.com/ardcek/advanced-stt-system.git
cd advanced-stt-system
```

2. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

3. **Test edin:**
```bash
python simple_test.py
```

## 📖 Kullanım

### Basit Kullanım
```python
from modules.stt import transcribe_simple
from modules.nlp import normalize_transcript

# Ses dosyasını metne çevir
result = transcribe_simple("audio.wav")
print(f"Sonuç: {result}")

# Metni düzelt ve iyileştir
enhanced = normalize_transcript(result)
print(f"İyileştirilmiş: {enhanced}")
```

### Ana Program İle
```bash
# Basit transkripsiyon
python main.py --file audio.wav

# Yüksek kalite
python main.py --file audio.wav --quality highest

# Medikal içerik
python main.py --file medical.wav --medical
```

## 📂 Proje Yapısı

```
advanced-stt-system/
├── main.py              # Ana program
├── modules/             # STT sistem modülleri
│   ├── stt.py           # Temel STT fonksiyonları
│   ├── nlp.py           # Metin işleme
│   ├── audio.py         # Ses kayıt fonksiyonları
│   └── ...              # Diğer modüller
├── simple_test.py       # Test scripti
├── requirements.txt     # Gerekli paketler
└── README.md           # Bu dosya
```

## 🔧 Teknik Detaylar

- **Python 3.8+** gereklidir
- **Torch/TorchAudio** ses işleme için
- **Transformers** AI modelleri için
- **OpenAI Whisper** STT motoru olarak
- **16+ modül** ile kapsamlı işleme

## ⚠️ Notlar

- İlk çalıştırmada modeller indirilecektir
- GPU kullanımı performansı artırır
- Uzun ses dosyaları zaman alabilir
- İnternet bağlantısı gerekebilir

## 📞 Destek

Sorular veya sorunlar için GitHub Issues kullanın.

## 📄 Lisans

MIT License - Detaylar için `LICENSE` dosyasına bakın.