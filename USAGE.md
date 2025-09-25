# 📖 Kullanım Kılavuzu

## Kurulum

1. Python 3.8+ yükleyin
2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Temel Kullanım

### 1. Basit Transkripsiyon
```bash
python main.py --file audio.wav
```

### 2. Kalite Seçenekleri
```bash
# Hızlı (düşük kalite)
python main.py --file audio.wav --quality fastest

# Dengeli (orta kalite)  
python main.py --file audio.wav --quality balanced

# Yüksek kalite (yavaş)
python main.py --file audio.wav --quality highest
```

### 3. Özel Modlar
```bash
# Medikal içerik
python main.py --file medical.wav --medical

# Akademik ders
python main.py --file lecture.wav --academic

# Toplantı (konuşmacı ayrıştırma)
python main.py --file meeting.wav --mode meeting
```

## Programlama ile Kullanım

```python
from modules.stt import transcribe_simple
from modules.nlp import normalize_transcript

# Ses dosyasını transkript et
result = transcribe_simple("audio.wav")
print(result)

# Metni iyileştir
enhanced = normalize_transcript(result)
print(enhanced)
```

## Çıktı Formatları

- `output.txt` - Düz metin
- `output.md` - Markdown format
- `output.json` - Yapılandırılmış veri

## Sık Karşılaşılan Sorunlar

**Problem**: Import hatası
**Çözüm**: `pip install -r requirements.txt`

**Problem**: Ses dosyası bulunamadı
**Çözüm**: Dosya yolunu kontrol edin

**Problem**: Çok yavaş
**Çözüm**: `--quality fastest` kullanın

## Desteklenen Formatlar

- WAV, MP3, MP4, M4A
- 16kHz önerilir
- Mono veya stereo