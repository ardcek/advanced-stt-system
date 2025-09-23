# Advanced STT System 🎤

**Made by Mehmet Arda Çekiç** © 2025

Advanced Speech-to-Text system with multi-language support, long recording processing (2-3 hours), and educational content analysis.

## 🌟 Features

### � Core Capabilities
- **Multi-engine STT**: Whisper (tiny to large-v3), Azure Cognitive Services, Google Cloud Speech
- **Long recordings**: 2-3 hour recordings with chunk-based processing
- **Real-time progress**: Live progress tracking and performance monitoring
- **Quality assessment**: Audio quality analysis and reliability scoring

### 🌍 Multi-Language Support
- **7 Languages**: Turkish, English, German, French, Spanish, Italian, Latin
- **Perfect spelling**: Advanced spell correction for technical terms
- **Custom terms**: Specialized terminology dictionary
- **Foreign words**: Accurate transcription of international content

### � Advanced Processing
- **Memory management**: Efficient processing with memory optimization
- **Error recovery**: Robust error handling and automatic fallbacks
- **Multiple formats**: TXT, MD, SRT, DOCX output formats
- **Performance monitoring**: Real-time system metrics and reporting
├── 📁 modules/              # Uygulama modülleri
├── 📁 venv/                 # Python sanal ortamı
└── 📄 requirements.txt      # Gerekli kütüphaneler
```

## ⚙️ Özellikler

✅ **Çoklu Model**: Whisper, Azure, Google Cloud  
✅ **Yüksek Kalite**: Large-v3 model ile %95+ doğruluk  
✅ **Akıllı Analiz**: Görev/karar çıkarımı  
✅ **Çoklu Format**: WAV, MP3, MP4, M4A  
✅ **Canlı Kayıt**: Real-time transkripsiyon  
✅ **Çoklu Çıktı**: TXT, MD, SRT, DOCX  

## 📊 Çıktı Dosyaları

Her işlem sonrası otomatik oluşur:

- 📄 **transcript.txt** - Ham transkripsiyon
- 📋 **summary.txt** - Genel özet  
- ✅ **tasks.txt** - Tespit edilen görevler
- 📝 **notes.md** - Yapılandırılmış notlar
- 🎬 **meeting.srt** - Alt yazı dosyası
- 📊 **meeting_minutes.docx** - Resmi tutanak

## 📚 Dokümantasyon

| Dosya | Seviye | İçerik |
|-------|--------|---------|
| `README_STT.md` | 🔬 İleri | Tam teknik dokümantasyon |
| `KULLANIM_KILAVUZU.md` | 📖 Orta | Adım adım kılavuz |
| `CHEAT_SHEET.md` | ⚡ Temel | Hızlı komut referansı |

## 🎯 Kullanım Senaryoları

### 🏢 İş Toplantısı (Standard)
```bash
python main.py --file toplanti.wav --quality highest --stt large-v3 --title "Haftalık Toplantı"
```

### 🎯 Ultra Accuracy Mode (%100'e yakın)
```bash
python main.py --file toplanti.wav --quality ultra --stt large-v3 --device cuda --title "Kritik Toplantı"
```

### 🎓 Eğitim/Ders
```bash  
python main.py --file ders.mp4 --quality ultra --stt large-v3 --title "Python Eğitimi"
```

### ⚡ Hızlı Test
```bash
python main.py --file test.wav --quality fastest --stt small --title "Hızlı Test"
```

### 🎙️ Podcast/Röportaj
```bash
python main.py --file podcast.mp3 --stt medium --title "Teknoloji Sohbeti"
```

## 🎯 Ultra Accuracy Mode

**%100'e yakın doğruluk** için özel olarak optimize edilmiş mod:

```bash
# Ultra mode - Maximum accuracy
python main.py --file audio.wav --quality ultra --stt large-v3 --device cuda
```

### Ultra Mode Özellikleri:
- 🎯 **5x Temperature Sampling**: Farklı sıcaklık değerleriyle çoklu analiz
- 🔍 **Beam Size 10**: Maksimum arama genişliği  
- 🏆 **Best of 5**: En iyi 5 denemeden otomatik seçim
- ⏱️ **Word Timestamps**: Kelime bazlı zaman damgaları
- 📝 **Enhanced Punctuation**: Gelişmiş noktalama düzeltmesi
- 🧠 **Patience 2.0**: Daha sabırlı decode algoritması

### Doğruluk Seviyeleri:
- `fastest`: ~60-70% (Hızlı test için)
- `balanced`: ~70-80% (Genel kullanım)  
- `highest`: ~80-85% (Yüksek kalite)
- `ultra`: ~90-95% (Maximum doğruluk)

## ⚡ Hızlı Test

```bash
# Ultra accuracy test
python test_ultra_accuracy.py
```

## 🆘 Destek

1. **Hızlı Çözümler**: `CHEAT_SHEET.md`
2. **Detaylı Rehber**: `KULLANIM_KILAVUZU.md` 
3. **Tam Dokümantasyon**: `README_STT.md`

---

## 📄 Lisans

Bu proje MIT lisansı altında yayınlanmıştır.

**Made by Mehmet Arda Çekiç** © 2025

> 🚀 **Profesyonel ses transkripsiyon sistemi - Maksimum doğruluk, minimum çaba**