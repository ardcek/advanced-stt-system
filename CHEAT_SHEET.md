# 🚀 HIZLI BAŞVURU KARTLARI

## 📱 TEK KOMUT İLE KULLANIM

### 🔥 POPÜLER KOMUTLAR
```bash
# Toplantı kaydı analizi (ÖNERİLEN)
python main.py --file meeting.wav --stt large-v3 --title "Toplantı"

# Hızlı test (30 saniye)
python main.py --duration 30 --stt small

# Canlı kayıt (ENTER ile dur)
python main.py --stream --stt medium

# Mevcut dosya (orta kalite)
python main.py --file ses.mp3 --stt medium
```

## ⚙️ MODEL TABLOSU

| Model | Süre | Kalite | Ne Zaman Kullan |
|-------|------|--------|-----------------|
| `tiny` | 5 sn | ⭐ | Hızlı test |
| `small` | 30 sn | ⭐⭐⭐ | Demo/deneme |
| `medium` | 2 dk | ⭐⭐⭐⭐ | Normal kullanım |
| `large-v3` | 8 dk | ⭐⭐⭐⭐⭐ | Önemli kayıtlar |

## 🎯 SENARYOLARa GÖRE KULLANIM

### 🏢 **İş Toplantısı**
```bash
python main.py --file toplanti.wav --stt large-v3 --title "Haftalık Toplantı"
```
**→ Çıktı:** Görevler, kararlar, katılımcılar

### 🎓 **Ders/Eğitim**
```bash
python main.py --file ders.mp4 --stt large-v3 --title "Python Dersi"
```
**→ Çıktı:** Bölüm özetleri, anahtar konular

### 🎙️ **Röportaj/Podcast**
```bash
python main.py --file podcast.mp3 --stt medium --title "Teknoloji Sohbeti"
```
**→ Çıktı:** Ana konular, önemli noktalar

### ⚡ **Test/Demo**
```bash
python main.py --duration 60 --stt small --title "Test"
```
**→ Çıktı:** 1 dakika kayıt + analiz

## 📁 ÇIKTI DOSYALARI

```
transcript.txt     📝 Tam metin
summary.txt        📋 Özet  
tasks.txt          ✅ Görevler
notes.md           📄 Yapılandırılmış notlar
meeting.srt        🎬 Alt yazı
meeting_minutes.docx 📊 Word belgesi
```

## 🆘 HIZLI ÇÖZÜMLER

### Problem: Yavaş çalışıyor
```bash
--stt small    # Küçük model kullan
--device cpu   # CPU'ya geç
```

### Problem: Kalite düşük
```bash
--stt large-v3    # En iyi model
--device cuda     # GPU kullan (varsa)
```

### Problem: Dosya bulunamıyor
```bash
--file "C:\tam\yol\ses.wav"    # Tam yol ver
```

## 🔧 ÖZELLEŞTİRME

### `custom_terms.txt` - Özel terimler
```
ProjeAlfa
PostgreSQL  
Kubernetes
```

### `corrections.txt` - Düzeltmeler
```
payton => Python
deil => değil
```

---
**💡 İPUCU:** İlk defa kullanıyorsanız `python main.py --duration 30 --stt small` ile başlayın!

---

**Made by Mehmet Arda Çekiç** © 2025  
📖 **Detay**: `README_STT.md` | 📋 **Tam Kılavuz**: `KULLANIM_KILAVUZU.md`