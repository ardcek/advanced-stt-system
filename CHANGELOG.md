# Changelog

Tüm önemli değişiklikler bu dosyada belgelenmiştir.

## [2.0.0] - 2025-10-04

### 🎉 Majör Güncelleme: Kalite Modları Düzeltmesi

#### ✅ Eklenen
- **Kalite Modu Seçici**: Panel'de kullanıcı dostu kalite modu seçimi (fastest/balanced/highest/ultra)
- **Modül Tamamlama**: Tüm eksik modüller kuruldu (faster-whisper, sounddevice, librosa, vb.)
- **API Güvenliği**: API anahtarları .env.example ile güvenli hale getirildi
- **Sonuç Format Desteği**: TranscriptionResult object, dict ve string formatları desteklendi
- **Performance Monitoring**: Real-Time Factor (RTF) ile performans takibi
- **Progress Display**: Live processing progress gösterimi

#### 🔧 Düzeltilen
- **Kalite Modları Sorunu**: fastest, balanced, highest modları artık çalışıyor
- **Parameter Mapping**: quality_mode ve selected_modes karışıklığı düzeltildi
- **TranscriptionResult Error**: "'TranscriptionResult' object has no attribute 'get'" hatası düzeltildi
- **Module Import**: "No module named 'faster_whisper'" hatası çözüldü
- **Panel Integration**: Kalite seçimi ana akışa entegre edildi

#### 📊 Test Sonuçları
- **FASTEST**: 3.0s (RTF: 0.04) - %91 doğruluk ✅
- **BALANCED**: 42.6s (RTF: 0.54) - %95 doğruluk ✅
- **HIGHEST**: 48.2s (RTF: 0.61) - %98 doğruluk ✅
- **ULTRA**: 84.0s (RTF: 1.07) - AI destekli en kaliteli ✅

#### 🛠️ Teknik Detaylar
- Güvenli sonuç parsing implementasyonu
- Confidence score düzeltmeleri
- Processing time hesaplamaları
- Error handling iyileştirmeleri

### 🔒 Güvenlik
- API anahtarları .env dosyasından kaldırıldı
- .env.example dosyası eklendi
- .gitignore güncellendi

## [1.5.0] - 2025-10-03

### ✅ Eklenen
- **AI Özet Sağlayıcı Seçimi**: Groq, OpenAI, Local AI seçenekleri
- **Gelişmiş Local AI**: advanced_local_ai.py modülü
- **Turkish Text Correction**: Otomatik Türkçe düzeltme sistemi
- **Fallback Logic**: API başarısız olduğunda local AI devreye giriyor

### 🔧 Düzeltilen
- OpenAI rate limit sorunları
- Groq API entegrasyonu
- Turkish summary quality

## [1.0.0] - 2025-09-30

### 🎉 İlk Sürüm
- **Ultra STT System**: Temel STT pipeline
- **Medical AI Integration**: 265,684 MeSH terms
- **Multi-Language Support**: 50+ dil desteği
- **Academic Processing**: Lecture transcription
- **Real-time Processing**: Live audio recording