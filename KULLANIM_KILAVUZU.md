# 🎯 Hızlı Kullanım Kılavuzu

> **📚 Tam teknik dokümantasyon**: `README_STT.md`

## 🚀 3 Adımda Kullanım

### 1️⃣ Hazırlık
```bash
# Terminal/PowerShell açın ve proje klasörüne gidin
cd "C:\Users\Arda\Desktop\test"

# Sanal ortamı aktif edin (otomatik olmalı)
# Eğer aktif değilse: .\venv\Scripts\activate
```

### 2️⃣ Kullanım Seçenekleri

#### 📁 **MEVCUT SES DOSYASI ANALİZİ**
```bash
# En basit kullanım
python main.py --file ses_dosyam.wav

# Yüksek kalite (önerilen)
python main.py --file ses_dosyam.wav --stt large-v3 --title "Toplantı Başlığı"

# Farklı ses formatları desteklenir
python main.py --file kayit.mp3 --stt large-v3
python main.py --file video.mp4 --stt medium
python main.py --file meeting.m4a --stt large-v3
```

#### 🎙️ **CANLI KAYIT**
```bash
# Sınırsız canlı kayıt (ENTER'a basana kadar)
python main.py --stream --stt large-v3 --title "Canlı Toplantı"

# Belirli süre kayıt (saniye cinsinden)
python main.py --duration 300 --stt large-v3 --title "5 Dakika Kayıt"
python main.py --duration 1800 --stt medium --title "30 Dakika Ders"
```

### 3️⃣ Sonuçları Al
Program bitince şu dosyalar **otomatik** oluşacak:
- 📄 `transcript.txt` - Ham transkripsiyon
- 📄 `summary.txt` - Özet
- 📄 `tasks.txt` - Tespit edilen görevler
- 📄 `notes.md` - Yapılandırılmış notlar
- 📄 `meeting.srt` - Alt yazı dosyası
- 📄 `meeting_minutes.docx` - Word belgesi

---

## ⚙️ Detaylı Parametreler

### 🎛️ **Model Seçenekleri** (`--stt`)
| Model | Hız | Kalite | Kullanım |
|-------|-----|--------|----------|
| `tiny` | ⚡⚡⚡ | ⭐ | Test/Demo |
| `base` | ⚡⚡ | ⭐⭐ | Hızlı transkripsiyon |
| `small` | ⚡ | ⭐⭐⭐ | Dengeli |
| `medium` | 🐌 | ⭐⭐⭐⭐ | Önerilen (varsayılan) |
| `large-v2` | 🐌🐌 | ⭐⭐⭐⭐⭐ | Yüksek kalite |
| `large-v3` | 🐌🐌 | ⭐⭐⭐⭐⭐ | En yüksek kalite |

### 🖥️ **Cihaz Seçenekleri** (`--device`)
```bash
--device cpu    # CPU kullan (varsayılan, güvenli)
--device cuda   # GPU kullan (hızlı, NVIDIA gerekli)
```

### 📋 **Diğer Seçenekler**
```bash
--window 300    # Bölüm özeti süresi (saniye, varsayılan: 600)
--title "Başlık" # Çıktı dosyalarında görülecek başlık
```

---

## 💡 Kullanım Örnekleri

### 📞 **Toplantı Kaydı**
```bash
python main.py --file toplanti.wav --stt large-v3 --title "Haftalık Ekip Toplantısı"
```
**Çıktı:** Görevler, kararlar, konuşmacı bilgileri ile detaylı toplantı tutanağı

### 🎓 **Ders/Eğitim Kaydı**
```bash
python main.py --file ders.mp4 --stt large-v3 --title "Python Eğitimi - Bölüm 1" --window 900
```
**Çıktı:** 15 dakikalık bölümler halinde özetlenmiş ders notları

### 🎙️ **Podcast/Röportaj**
```bash
python main.py --file podcast.mp3 --stt medium --title "Teknoloji Röportajı"
```
**Çıktı:** Ana konular ve önemli noktalar özetli

### ⚡ **Hızlı Test**
```bash
python main.py --duration 30 --stt small --title "Test Kaydı"
```
**Çıktı:** 30 saniyelik test kaydı ve analizi

### 🔴 **Canlı Sunum**
```bash
python main.py --stream --stt large-v3 --title "Canlı Demo Sunumu"
```
**Kullanım:** Kaydı başlatın, sunumu yapın, ENTER'a basarak durdurun

---

## 🔧 Özelleştirme Ayarları

### 📝 **Özel Terimler Ekleme**
`custom_terms.txt` dosyasını düzenleyin:
```
# Şirket terimleri
ProjeAlfa
MüşteriPortalı
VeriTabanıSistemi

# Teknik terimler
PostgreSQL
Redis
Kubernetes
```

### ✏️ **Metin Düzeltme Kuralları**
`corrections.txt` dosyasını düzenleyin:
```
# Yaygın hatalar
payton => Python
cubernets => Kubernetes
veri tabanı => veritabanı
deil => değil
```

---

## 📊 Kalite İpuçları

### ✅ **En İyi Sonuç İçin**
- 🎤 **Ses kalitesi**: Temiz, gürültüsüz kayıt
- 📏 **Süre**: 5 dakika - 2 saat arası ideal
- 🔊 **Hacim**: Çok yüksek/alçak olmasın
- 🗣️ **Konuşma**: Net telaffuz, hızlı değil

### 🎯 **Model Seçimi Tavsiyeleri**
| Durum | Önerilen Model | Sebep |
|-------|----------------|--------|
| Toplantı kaydı | `large-v3` | Yüksek doğruluk gerekli |
| Ders kaydı | `large-v3` | Teknik terimler için |
| Podcast | `medium` | Dengeli hız/kalite |
| Test/Demo | `small` | Hızlı sonuç |
| Gürültülü kayıt | `large-v3` | Gelişmiş filtreleme |

### ⚠️ **Dikkat Edilecekler**
- 🔥 `large-v3` modeli daha yavaş ama çok daha kaliteli
- 💻 GPU varsa `--device cuda` ile hızlandırın
- 📁 Büyük dosyalarda (>1 saat) sabırlı olun
- 🔄 İlk çalıştırmada model indirme olabilir

---

## 🆘 Sorun Giderme

### ❌ **Yaygın Hatalar**

#### "No module named 'faster_whisper'"
```bash
pip install -r requirements.txt
```

#### "CUDA not available"
```bash
# CPU'ya geçin
python main.py --file dosya.wav --device cpu
```

#### "File not found"
```bash
# Dosya yolunu kontrol edin
python main.py --file "C:\tam\yol\ses.wav"
```

#### Ses kayıt problemi
```bash
# Mevcut dosya kullanın
python main.py --file mevcut_kayit.wav
```

### 💡 **Performans İyileştirme**
```bash
# Küçük model ile hızlı test
python main.py --file test.wav --stt tiny

# Orta kalite, hızlı
python main.py --file kayit.wav --stt medium

# GPU varsa
python main.py --file kayit.wav --stt large-v3 --device cuda
```

---

## 📞 Hızlı Başvuru Komutları

```bash
# 🔥 EN POPÜLER - Yüksek kaliteli toplantı analizi
python main.py --file meeting.wav --stt large-v3 --title "Toplantı"

# ⚡ HIZLI TEST - 30 saniyelik kayıt
python main.py --duration 30 --stt small

# 🎙️ CANLI KAYIT - Sınırsız
python main.py --stream --stt medium --title "Canlı"

# 📁 MEVCUT DOSYA - Orta kalite
python main.py --file ses.mp3 --stt medium

# 🎯 MAKSIMUM KALİTE - En iyi sonuç
python main.py --file important.wav --stt large-v3 --device cuda
```

---

## 📁 Çıktı Dosyaları Açıklaması

| Dosya | İçerik | Kullanım |
|-------|--------|----------|
| `transcript.txt` | Ham transkripsiyon | Tam metin |
| `summary.txt` | Genel özet | Hızlı bakış |
| `tasks.txt` | Tespit edilen görevler | Yapılacaklar listesi |
| `notes.md` | Yapılandırılmış notlar | GitHub/dokümantasyon |
| `meeting.srt` | Alt yazı | Video editörleri |
| `meeting_minutes.docx` | Resmi tutanak | Paylaşım/arşiv |

**🎉 Artık hazırsınız! Herhangi bir ses dosyasını analiz edebilirsiniz.**

---

**Made by Mehmet Arda Çekiç** © 2025

> 📖 **Tam özellikler ve detaylı dokümantasyon**: `README_STT.md`