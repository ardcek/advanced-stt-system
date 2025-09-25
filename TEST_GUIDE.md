# 🧪 Test Rehberi - Ultra-Advanced STT System

**Made by Mehmet Arda Çekiç** © 2025

Bu rehber sistemi nasıl test edeceğinizi adım adım açıklar.

## 🚀 Hızlı Test (5 Dakikada)

### 1️⃣ Basit Ses Dosyası Testi
```bash
# Herhangi bir ses dosyası edinin (.wav, .mp3, .mp4, .m4a)
# Dosyayı proje klasörüne koyun
# Örnek: test.wav

# Basit test
python main.py --file test.wav --quality fastest

# Sonuç: output.txt dosyasını kontrol edin
```

### 2️⃣ Dependency Kontrolü
```bash
# Python environment check
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import numpy; print('NumPy OK')"
python -c "import librosa; print('Librosa OK')"
python -c "import openai; print('OpenAI OK')"

# Eksikse yükleyin
pip install torch torchaudio librosa openai transformers
```

## 🎯 Gerçek Kullanım Testleri

### 🎓 Test 1: Öğrenci Ders Notu
**Amaç:** Ders kaydından study guide oluşturma

```bash
# Ders kaydı (örnek: physics_lecture.wav)
python main.py --file physics_lecture.wav --academic --subject physics --format student --output-type study_guide

# Beklenen çıktılar:
# - study_guide.html (İnteraktif study guide)
# - notes.pdf (PDF ders notu)
# - flashcards.json (Çalışma kartları)
# - concept_map.png (Kavram haritası)
```

**✅ Başarı Kriterleri:**
- HTML dosyası oluştu ve tarayıcıda açılabiliyor
- PDF temiz ve okunabilir
- Konular mantıklı bölümlere ayrılmış
- Flashcard'lar anlamlı

### 🎭 Test 2: Toplantı Analizi  
**Amaç:** Çok konuşmacılı toplantı analizi

```bash
# Toplantı kaydı (örnek: team_meeting.wav)
python main.py --file team_meeting.wav --mode meeting --diarization advanced

# Beklenen çıktılar:
# - output.txt (Konuşmacı ayrımlı transkript)
# - meeting_analysis.md (Toplantı analizi)
# - action_items.txt (Yapılacaklar listesi)
```

**✅ Başarı Kriterleri:**
- Farklı konuşmacılar ayrılmış (Speaker 1, Speaker 2, etc.)
- Action item'lar tanımlanmış
- Toplantı özeti mantıklı

### 🏥 Test 3: Medical AI (OpenAI API Gerekli)
**Amaç:** Medical terminoloji ve AI analizi

```bash
# OpenAI API key ayarlayın
export OPENAI_API_KEY="your-api-key"

# Medical konsültasyon (örnek: consultation.wav)
python main.py --file consultation.wav --medical --format medical --quality ultra

# Beklenen çıktılar:
# - medical_report.pdf (SOAP formatında rapor)
# - medical_terms.json (Tıbbi terimler sözlüğü)
# - clinical_insights.md (AI görüşleri)
```

**✅ Başarı Kriterleri:**
- Medical terimler açıklanmış
- SOAP format'ında professional rapor
- AI insights mantıklı

### 🚀 Test 4: Ultra Quality (%99.9 Doğruluk)
**Amaç:** Maksimum doğruluk testi

```bash
# Kaliteli, net ses dosyası gerekli
python main.py --file clear_audio.wav --quality ultra --target-accuracy 0.999

# ⚠️ Uyarı: ÇOK YAVAS (10x normal süre)
# Kısa dosyalarla test edin (1-2 dakika)
```

**✅ Başarı Kriterleri:**
- Çok yüksek doğruluk
- Punctuation doğru
- Teknik terimler accurate

## 🔧 Troubleshooting Guide

### ❌ "ImportError: No module named..."
```bash
# Solution: Dependency yükleme
pip install -r requirements.txt

# Veya tek tek:
pip install torch librosa openai transformers whisper
```

### ❌ "CUDA out of memory" 
```bash
# Solution: CPU kullan
python main.py --file audio.wav --device cpu --quality balanced
```

### ❌ "OpenAI API error"
```bash
# Solution: API key kontrol
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY%  # Windows

# Set edilmemişse:
export OPENAI_API_KEY="sk-your-key-here"
```

### ❌ "Audio file not found"
```bash
# Solution: Dosya yolunu kontrol
ls -la *.wav  # Linux/Mac
dir *.wav     # Windows

# Absolute path kullan
python main.py --file "C:/path/to/audio.wav"
```

### ❌ "Processing too slow"
```bash
# Solution: Hızlı mode kullan
python main.py --file audio.wav --quality fastest

# Daha kısa dosyalarla test edin
```

## 📊 Performance Benchmarks

### Kalite vs Hız Tablosu

| Mode | Doğruluk | Hız | RAM | Use Case |
|------|----------|-----|-----|-----------|
| `fastest` | %85-90 | 10x hızlı | 2-4 GB | Hızlı test |
| `balanced` | %92-95 | 3x hızlı | 4-6 GB | Günlük kullanım |  
| `highest` | %96-98 | 1x normal | 6-8 GB | Professional |
| `ultra` | %99-99.9 | 0.3x (yavaş) | 8-12 GB | Kritik kayıtlar |

### Dosya Boyutu Önerileri

- **Test:** 30s - 2 dakika
- **Development:** 5 - 15 dakika  
- **Production:** 30 dakika - 3 saat
- **Long-form Mode:** 2+ saat

## 🎯 Başarı Metrikleri

### ✅ Sistem Çalışıyor Demek:
1. **No Python Errors:** Import hatası yok
2. **File Processing:** Audio dosyası işlendi  
3. **Output Generation:** Çıktı dosyaları oluştu
4. **Reasonable Quality:** Transcript okunabilir
5. **Expected Features:** Seçilen modda özellikler çalışıyor

### 📈 Kalite Kontrolü:
- **Word Error Rate (WER):** < %10 (iyi), < %5 (mükemmel)
- **Punctuation:** Noktalama işaretleri doğru
- **Speaker Separation:** Konuşmacılar ayrımlı
- **Technical Terms:** Uzman terimler doğru
- **Formatting:** Çıktı formatı professional

## 🔄 Test Workflow

### 📝 Step-by-Step Test Süreci:

1. **Setup Check**
   ```bash
   python simple_test.py
   ```

2. **Basic Function Test**
   ```bash
   python main.py --file test.wav --quality fastest
   ```

3. **Feature-Specific Tests**
   ```bash
   # Academic
   python main.py --file lecture.wav --academic --format student
   
   # Medical  
   python main.py --file medical.wav --medical
   
   # Meeting
   python main.py --file meeting.wav --mode meeting --diarization advanced
   ```

4. **Quality Tests**  
   ```bash
   # Different quality levels
   python main.py --file audio.wav --quality balanced
   python main.py --file audio.wav --quality highest  
   python main.py --file audio.wav --quality ultra
   ```

5. **Edge Case Tests**
   ```bash
   # Long audio
   python main.py --file 2hour_lecture.wav --mode longform
   
   # Multiple speakers
   python main.py --file panel_discussion.wav --diarization advanced
   
   # Poor audio quality
   python main.py --file noisy_recording.wav --quality ultra
   ```

## 🎨 Test Data Önerileri

### 🎯 İdeal Test Audio'ları:

1. **Clear Speech (5 min):** Tek kişi, net konuşma
2. **Lecture Sample (10-15 min):** Eğitim içeriği
3. **Meeting Recording (8-12 min):** 2-3 kişi konuşma
4. **Medical Consultation (5-10 min):** Tıbbi terminoloji
5. **Technical Discussion (10 min):** Uzman terminoloji
6. **Long-form (30+ min):** Uzun ders/konferans

### 📁 Test Audio Kaynakları:
- Kendi ses kaydınız (phone, laptop mic)
- YouTube audio download (fair use)
- Podcast segments
- University lecture recordings
- Professional meeting recordings

---

## 🎉 Sonuç

Bu rehberi takip ederek sisteminizin:
- ✅ Temel funktionalları çalışıyor mu?
- ✅ Kalitesi beklentileri karşılıyor mu?  
- ✅ Özel özellikler (medical, academic) aktif mi?
- ✅ Performance kabul edilebilir mi?

sorularını cevaplayabilirsiniz.

**🚀 Happy Testing!** 

---

**Made by Mehmet Arda Çekiç** © 2025