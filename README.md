# Advanced STT System 🎤

**Made by Mehmet Arda Çekiç** © 2025

Advanced Speech-to-Text system with multi-language support, long recording processing (2-3 hours), and educational content analysis.

## 🚀 Hızlı Başlangıç

### 🎮 Kolay Kullanım (Başlangıç)
```bash
# Çift tıklayın
BASLA.bat
```

### ⚡ Terminal Kullanımı
```bash
# Mevcut ses dosyası
python main.py --file meeting.wav --stt large-v3

# Canlı kayıt  
python main.py --stream --stt large-v3

# Test kaydı
python main.py --duration 30 --stt small
```

## 📁 Dosya Yapısı

```
📁 PROJE/
├── 🚀 BASLA.bat              # Grafiksel menü
├── 🐍 main.py               # Ana uygulama  
├── 📚 README_STT.md         # Detaylı dokümantasyon
├── 📋 KULLANIM_KILAVUZU.md  # Hızlı kılavuz
├── ⚡ CHEAT_SHEET.md        # Komut referansı
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

### 🏢 İş Toplantısı
```bash
python main.py --file toplanti.wav --stt large-v3 --title "Haftalık Toplantı"
```

### 🎓 Eğitim/Ders
```bash  
python main.py --file ders.mp4 --stt large-v3 --title "Python Eğitimi"
```

### 🎙️ Podcast/Röportaj
```bash
python main.py --file podcast.mp3 --stt medium --title "Teknoloji Sohbeti"
```

## ⚡ Hızlı Test

```bash
# 30 saniye test kaydı
python main.py --duration 30 --stt small --title "Test"
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