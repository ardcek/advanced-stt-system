# modules/stt.py - Enhanced Speech-to-Text Module
"""
Gelişmiş Ses-Metin Dönüştürme Modülü
====================================

Özellikler:
- Çoklu model desteği (Whisper, Azure, Google)
- Gelişmiş ses ön işleme (gürültü azaltma, normalize)
- Adaptif parametre optimizasyonu 
- Kalite değerlendirme ve güven skoru
- İleri düzey post-processing
- Konuşmacı diarization desteği
- Hibrit model yaklaşımı

Kullanım:
    result = transcribe_advanced("audio.wav", quality="highest")
    print(f"Transkripsiyon: {result['text']}")
    print(f"Güven Skoru: {result['confidence']:.2f}")
"""

import os
import re
import time
import json
import logging
import warnings
import numpy as np
from pathlib import Path
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

# ---- Opsiyonel normalizasyon (proje içi) ----
try:
    from . import nlp  # normalize_transcript
except Exception:
    nlp = None  # type: ignore

# ---- Audio Processing Imports ----
try:
    import librosa
    import noisereduce as nr
    import webrtcvad
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    _HAS_AUDIO_PROCESSING = True
except ImportError:
    _HAS_AUDIO_PROCESSING = False

# ---- STT Engines Imports ----
try:
    import azure.cognitiveservices.speech as speechsdk
    _HAS_AZURE = True
except ImportError:
    _HAS_AZURE = False

try:
    from google.cloud import speech
    _HAS_GOOGLE = True
except ImportError:
    _HAS_GOOGLE = False

try:
    from rapidfuzz import fuzz
    _HAS_FUZZ = True
except ImportError:
    _HAS_FUZZ = False

# ---- Data Classes ----
@dataclass
class AudioInfo:
    """Ses dosyası bilgileri"""
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int
    file_size: int
    snr_estimate: Optional[float] = None
    is_speech: bool = True

@dataclass 
class TranscriptionResult:
    """Gelişmiş transkripsiyon sonucu"""
    text: str
    segments: List[Dict]
    duration: float
    confidence: float
    audio_info: AudioInfo
    model_used: str
    processing_time: float
    word_level: Optional[List[Dict]] = None
    speaker_info: Optional[List[Dict]] = None
    quality_metrics: Optional[Dict] = None

# ---- Yardımcı Fonksiyonlar ----
def _enhance_meeting_segments(segments: List[Dict]) -> List[Dict]:
    """Toplantı segmentlerine speaker bilgisi ekleme"""
    if not segments:
        return segments
        
    # Basit speaker assignment (gerçek diarization için diarize.py kullanılacak)
    enhanced = []
    for i, segment in enumerate(segments):
        enhanced_segment = segment.copy()
        # Placeholder speaker assignment
        enhanced_segment['speaker'] = f"Speaker_{(i // 3) % 4 + 1}"  # Basit döngü
        enhanced.append(enhanced_segment)
    
    return enhanced


def _enhance_lecture_segments(segments: List[Dict]) -> List[Dict]:
    """Ders segmentlerine konu bilgisi ekleme"""
    if not segments:
        return segments
    
    # Ders içeriği için segment enhancements
    enhanced = []
    for segment in segments:
        enhanced_segment = segment.copy()
        
        # Konu başlığı tespiti (basit heuristic)
        text = segment.get('text', '').lower()
        if any(keyword in text for keyword in ['konu', 'başlık', 'bölüm', 'chapter']):
            enhanced_segment['content_type'] = 'topic_header'
        elif any(keyword in text for keyword in ['örnek', 'example', 'demo']):
            enhanced_segment['content_type'] = 'example'
        elif any(keyword in text for keyword in ['soru', 'question', '?']):
            enhanced_segment['content_type'] = 'question'
        else:
            enhanced_segment['content_type'] = 'content'
            
        enhanced.append(enhanced_segment)
    
    return enhanced


def _assess_audio_quality(audio_info: AudioInfo) -> str:
    """Ses kalitesi değerlendirmesi"""
    
    # Örnekleme hızı kontrolü
    if audio_info.sample_rate < 16000:
        quality = "Düşük"
    elif audio_info.sample_rate < 44100:
        quality = "Orta" 
    else:
        quality = "Yüksek"
    
    # SNR kontrolü (eğer hesaplandıysa)
    if audio_info.snr_estimate:
        if audio_info.snr_estimate < 10:
            quality = "Düşük (Gürültülü)"
        elif audio_info.snr_estimate < 20:
            if quality == "Yüksek":
                quality = "Orta"
        # Yüksek SNR değerinde quality değişmez
    
    # Süre faktörü
    if audio_info.duration > 7200:  # 2 saat+
        quality += " (Uzun Kayıt)"
    
    return quality


# ==============================
#  INITIALIZATION & SETUP
# ==============================

def initialize():
    """STT motoru başlatma - modelleri yükle ve hazırlık yap"""
    print("🚀 STT motoru başlatılıyor...")
    
    # Whisper modeli lazy loading için hazırlık
    global _WHISPER_MODEL_CACHE
    _WHISPER_MODEL_CACHE = {}
    
    # Audio processing kütüphanelerini kontrol et
    if not _HAS_AUDIO_PROCESSING:
        print("⚠️  Gelişmiş ses işleme kütüphaneleri bulunamadı (librosa, noisereduce)")
        print("   Basit işleme modu kullanılacak")
    else:
        print("✅ Ses işleme kütüphaneleri hazır")
    
    # Cloud servis kontrolü
    if _HAS_AZURE and os.getenv("AZURE_SPEECH_KEY"):
        print("✅ Azure Speech servis anahtarı bulundu")
    
    if _HAS_GOOGLE and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("✅ Google Cloud Speech kimlik bilgileri bulundu")
    
    print("✅ STT motoru hazır!")


# Model cache
_WHISPER_MODEL_CACHE = {}


def _canon(s: str) -> str:
    """Metni kanonik forma dönüştür (karşılaştırma için)"""
    return re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]+", "", (s or "").lower())

def _norm_txt(txt: str) -> str:
    """Metni normalize et"""
    if not txt:
        return ""
    if nlp and hasattr(nlp, "normalize_transcript"):
        try:
            return nlp.normalize_transcript(txt)
        except Exception:
            pass
    return " ".join((txt or "").split())

def _is_near_duplicate(prev_key: str, cur_key: str, thresh: int = 92) -> bool:
    """İki metnin benzer olup olmadığını kontrol et"""
    if not prev_key or not cur_key:
        return False
    if prev_key == cur_key:
        return True
    if _HAS_FUZZ:
        return fuzz.ratio(prev_key, cur_key) >= thresh
    return (cur_key in prev_key) or (prev_key in cur_key)

def _estimate_snr(audio_path: str) -> Optional[float]:
    """Ses dosyasının sinyal-gürültü oranını tahmin et"""
    if not _HAS_AUDIO_PROCESSING:
        return None
    try:
        y, sr = librosa.load(audio_path, sr=None)
        # Basit SNR tahmini: RMS oranı kullanarak
        rms = librosa.feature.rms(y=y)[0]
        signal_power = np.mean(rms**2)
        noise_power = np.min(rms**2) 
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            return max(0, min(60, snr_db))  # 0-60 dB arasında sınırla
    except Exception:
        pass
    return None

def _get_audio_info(audio_path: str) -> AudioInfo:
    """Ses dosyası hakkında detaylı bilgi al"""
    try:
        if _HAS_AUDIO_PROCESSING:
            # Pydub ile temel bilgiler
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0  # ms to seconds
            sample_rate = audio.frame_rate
            channels = audio.channels
            bit_depth = audio.sample_width * 8
            file_size = os.path.getsize(audio_path)
            
            # SNR tahmini
            snr = _estimate_snr(audio_path)
            
            return AudioInfo(
                duration=duration,
                sample_rate=sample_rate, 
                channels=channels,
                bit_depth=bit_depth,
                file_size=file_size,
                snr_estimate=snr,
                is_speech=True  # VAD ile belirlenebilir
            )
        else:
            # Fallback: dosya boyutu ve temel tahmin
            file_size = os.path.getsize(audio_path)
            return AudioInfo(
                duration=0.0,
                sample_rate=16000,
                channels=1,
                bit_depth=16,
                file_size=file_size
            )
    except Exception:
        return AudioInfo(
            duration=0.0,
            sample_rate=16000, 
            channels=1,
            bit_depth=16,
            file_size=0
        )

def _preprocess_audio(input_path: str, output_path: Optional[str] = None) -> str:
    """Ses dosyasını transkripsiyon için optimize et"""
    if not _HAS_AUDIO_PROCESSING:
        return input_path
        
    if output_path is None:
        output_path = input_path.replace('.wav', '_processed.wav')
    
    try:
        # Ses dosyasını yükle
        y, sr = librosa.load(input_path, sr=None)
        
        # 1. Gürültü azaltma
        y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.8)
        
        # 2. Normalize etme
        y_normalized = librosa.util.normalize(y_denoised)
        
        # 3. Mono'ya çevir (eğer stereo ise)
        if len(y_normalized.shape) > 1:
            y_normalized = librosa.to_mono(y_normalized)
        
        # 4. 16kHz'e resample (Whisper için optimum)
        if sr != 16000:
            y_resampled = librosa.resample(y_normalized, orig_sr=sr, target_sr=16000)
            sr = 16000
        else:
            y_resampled = y_normalized
        
        # 5. Sessizlikleri kırp
        y_trimmed, _ = librosa.effects.trim(y_resampled, top_db=20)
        
        # İşlenmiş ses dosyasını kaydet
        import soundfile as sf
        sf.write(output_path, y_trimmed, sr)
        
        print(f"[STT] Ses ön işleme tamamlandı: {input_path} -> {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[STT] Ses ön işleme başarısız: {e}")
        return input_path

def _adaptive_parameters(audio_info: AudioInfo) -> Dict[str, Any]:
    """Ses dosyası özelliklerine göre optimum parametreleri belirle"""
    params = {
        'beam_size': 2,
        'temperature': 0.0,
        'compression_ratio_threshold': 2.4,
        'no_speech_threshold': 0.5,
        'vad_min_silence_ms': 300
    }
    
    # SNR'a göre ayarlama
    if audio_info.snr_estimate:
        if audio_info.snr_estimate < 10:  # Düşük kalite
            params['beam_size'] = 5  # Daha dikkatli
            params['temperature'] = [0.0, 0.2]  # Fallback ile
            params['no_speech_threshold'] = 0.4
        elif audio_info.snr_estimate > 30:  # Yüksek kalite
            params['beam_size'] = 1  # Hızlı
            params['no_speech_threshold'] = 0.6
    
    # Süreye göre ayarlama
    if audio_info.duration > 3600:  # 1 saattan uzun
        params['vad_min_silence_ms'] = 500  # Daha agresif VAD
    elif audio_info.duration < 60:  # 1 dakikadan kısa
        params['vad_min_silence_ms'] = 100  # Hassas VAD
    
    return params

# ---- Gelişmiş Terim ve Prompt Yönetimi ----
_DEFAULT_TERMS = [
    "Python", "NumPy", "Pandas", "PostgreSQL", "Docker", "Kubernetes",
    "toplantı", "özet", "aksiyon", "karar", "görev",
    "önbellek", "bildirim", "oturum", "kullanıcı", "sistem",
    "güncelleme", "doküman", "metrik", "performans", "optimizasyon",
    "veritabanı", "API", "framework", "library", "deployment"
]

def _load_custom_terms(path: str = "custom_terms.txt") -> List[str]:
    """Kullanıcı özel terimlerini yükle"""
    terms = list(_DEFAULT_TERMS)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t and not t.startswith("#") and t not in terms:
                    terms.append(t)
    return terms

def _generate_smart_prompt(language: str = "tr", audio_info: Optional[AudioInfo] = None) -> str:
    """Ses özelliklerine göre akıllı prompt üret"""
    terms = ", ".join(_load_custom_terms())
    
    base_prompt = ""
    if language.lower().startswith("tr"):
        base_prompt = (
            "Bu kayıt Türkçe bir profesyonel toplantı/eğitimdir. "
            "Noktalama işaretlerini doğru kullan, özel isimleri ve teknik terimleri "
            "tam olarak yaz. Her kelimenin en doğru halini tercih et. "
        )
    else:
        base_prompt = (
            "This is a professional meeting/training recording in Turkish. "
            "Use correct punctuation, write proper nouns and technical terms accurately. "
            "Prefer the most accurate form of each word. "
        )
    
    # Ses kalitesine göre ek talimatlar
    if audio_info and audio_info.snr_estimate:
        if audio_info.snr_estimate < 15:
            base_prompt += "Ses kalitesi düşük, belirsiz sesleri tahmin etme. "
        elif audio_info.snr_estimate > 35:
            base_prompt += "Ses kalitesi yüksek, detayları kaçırma. "
    
    return base_prompt + f"Domain terimleri: {terms}."

def _calculate_confidence(segments: List[Dict], model_used: str) -> float:
    """Transkripsiyon güven skorunu hesapla"""
    if not segments:
        return 0.0
    
    confidences = []
    for seg in segments:
        # Segment bazında güven skoru faktörleri
        text = seg.get('text', '')
        
        # 1. Metin uzunluğu faktörü
        length_factor = min(1.0, len(text.split()) / 10)
        
        # 2. Özel terim varlığı
        terms_found = sum(1 for term in _load_custom_terms() 
                         if term.lower() in text.lower())
        term_factor = min(1.0, terms_found / 3)
        
        # 3. Model bazlı temel güven
        base_confidence = 0.85 if 'whisper' in model_used.lower() else 0.75
        
        # 4. Tekrar / dublicate yokluğu
        no_repeat_factor = 1.0 if len(set(text.split())) == len(text.split()) else 0.8
        
        segment_conf = base_confidence * (0.4 + 0.3 * length_factor + 
                                        0.2 * term_factor + 0.1 * no_repeat_factor)
        confidences.append(segment_conf)
    
    return sum(confidences) / len(confidences) if confidences else 0.0

# =============================================================================
#                           GPU (PyTorch / openai-whisper)
# =============================================================================
def _transcribe_whisper_gpu(
    path: str,
    language: str,
    model_size: str,
    progress: bool,
) -> Dict:
    import torch
    import whisper  # openai-whisper

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for PyTorch")

    print("[STT][GPU] openai-whisper başlatılıyor…")
    t0 = time.time()
    model = whisper.load_model(model_size, device="cuda")

    result = model.transcribe(
        audio=path,
        language=language,
        task="transcribe",
        initial_prompt=_generate_smart_prompt(language),
        fp16=True,                       # RTX 4060 için ideal
        temperature=0.0,                 # tek deneme, hızlı
        condition_on_previous_text=True,
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        word_timestamps=False,           # hız için kapalı
        verbose=False,
    )

    duration = float(result.get("duration") or 0.0)
    segs_in: List[dict] = result.get("segments", []) or []
    out_segments: List[Dict] = []
    full: List[str] = []

    last_key = ""
    for s in segs_in:
        txt_raw = (s.get("text") or "").strip()
        if not txt_raw:
            continue
        txt = _norm_txt(txt_raw)
        key = _canon(txt)
        if _is_near_duplicate(last_key, key):
            continue
        last_key = key

        st = float(s.get("start") or 0.0)
        en = float(s.get("end") or st)

        if progress and duration > 0:
            pct = min(100, int((en / duration) * 100))
            print(f"[STT] {en:6.1f}s / {duration:6.1f}s  ({pct:3d}%)")

        out_segments.append({"start": st, "end": en, "text": txt, "words": []})
        full.append(txt)

    t1 = time.time()
    if duration > 0:
        rtf = (t1 - t0) / duration
        print(f"[STT][GPU] tamamlandı — süre: {t1 - t0:.1f}s, kayıt uzunluğu: {duration:.1f}s, RTF: {rtf:.2f}")

    return {"text": " ".join(full).strip(), "segments": out_segments, "duration": duration}

# =============================================================================
#                         CPU (faster-whisper / CTranslate2)
# =============================================================================
_FW_MODEL: Optional[Any] = None  # WhisperModel instance
_FW_CFG: Optional[Tuple[str, str, str]] = None  # (size, device, compute_type)

def _get_fw_model(size: str, device: str, compute: Optional[str]):
    from faster_whisper import WhisperModel
    global _FW_MODEL, _FW_CFG
    if compute is None:
        compute = "int8" if device == "cpu" else "float16"
    cfg = (size, device, compute)
    if _FW_MODEL is None or _FW_CFG != cfg:
        _FW_MODEL = WhisperModel(size, device=device, compute_type=compute)
        _FW_CFG = cfg
    return _FW_MODEL

def _transcribe_fw_cpu(
    path: str,
    language: str,
    model_size: str,
    device: str,
    beam_size: int,
    vad_min_silence_ms: int,
    progress: bool,
) -> Dict:
    model = _get_fw_model(model_size, device=device, compute=None)

    t0 = time.time()
    segments_gen, info = model.transcribe(
        path,
        language=language,
        beam_size=beam_size,                          # 2 hızlı, 5 daha isabetli
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
        condition_on_previous_text=True,
        initial_prompt=_generate_smart_prompt(language),
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        temperature=[0.0],                            # fallback kapalı → stabil hız
        word_timestamps=False,                        # hız
    )

    total = float(getattr(info, "duration", 0.0) or 0.0)
    out_segments: List[Dict] = []
    full_text_parts: List[str] = []
    last_keys = deque(maxlen=6)
    last_end = 0.0

    for seg in segments_gen:
        raw = (seg.text or "").strip()
        if not raw:
            continue

        norm = _norm_txt(raw)
        key = _canon(norm)
        if key and any(_is_near_duplicate(prev, key) for prev in last_keys):
            if progress and total > 0 and seg.end:
                pct = min(100, int((float(seg.end) / total) * 100))
                print(f"[STT] skip {seg.end:6.1f}s / {total:6.1f}s  ({pct:3d}%)  — dedupe")
            continue
        last_keys.append(key)

        s_start = float(seg.start) if seg.start is not None else last_end
        s_end   = float(seg.end)   if seg.end   is not None else s_start
        last_end = s_end

        if progress and total > 0:
            pct = min(100, int((s_end / total) * 100))
            print(f"[STT] {s_end:6.1f}s / {total:6.1f}s  ({pct:3d}%)")

        out_segments.append({"start": s_start, "end": s_end, "text": norm, "words": []})
        full_text_parts.append(norm)

    dt = time.time() - t0
    if progress and total > 0:
        rtf = dt / total
        print(f"[STT][CPU] tamamlandı — süre: {dt:.1f}s, kayıt uzunluğu: {total:.1f}s, RTF: {rtf:.2f}")

    # tek tur daha nazik dedupe
    cleaned_segments: List[Dict] = []
    prev_key = ""
    for sd in out_segments:
        k = _canon(sd["text"])
        if k and _is_near_duplicate(prev_key, k):
            continue
        cleaned_segments.append(sd)
        prev_key = k

    text = " ".join(s["text"] for s in cleaned_segments).strip()
    return {"text": text, "segments": cleaned_segments, "duration": total}

# =============================================================================
#                              ORTAK API
# =============================================================================
def transcribe(
    path: str,
    language: str = "tr",
    model_size: str = "medium",
    device: str = "cpu",
    beam_size: int = 2,
    vad_min_silence_ms: int = 300,
    progress: bool = True,
    on_segment: Optional[Callable[[Dict], None]] = None,  # (CPU yolu kullanır)
) -> Dict:
    """
    device='cuda' + PyTorch CUDA True → openai-whisper (GPU)
    aksi halde → faster-whisper (CPU)
    """
    if device == "cuda":
        try:
            import torch  # gecikmeli import
            if torch.cuda.is_available():
                return _transcribe_whisper_gpu(path, language, model_size, progress)
        except Exception as e:
            print(f"[STT] GPU yolu başarısız: {e}. CPU yoluna düşülüyor…")

    # CPU yolu
    return _transcribe_fw_cpu(
        path=path,
        language=language,
        model_size=model_size,
        device="cpu",
        beam_size=beam_size,
        vad_min_silence_ms=vad_min_silence_ms,
        progress=progress,
    )

# =============================================================================
#                           AZURE SPEECH API
# =============================================================================
def _transcribe_azure(
    path: str,
    language: str = "tr-TR", 
    speech_key: Optional[str] = None,
    region: str = "eastus"
) -> Dict:
    """Azure Cognitive Services Speech-to-Text"""
    if not _HAS_AZURE:
        raise ImportError("Azure Speech SDK not available")
    
    # API key kontrolü
    if not speech_key:
        speech_key = os.getenv("AZURE_SPEECH_KEY")
    if not speech_key:
        raise ValueError("Azure Speech key required")
    
    try:
        t0 = time.time()
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
        speech_config.speech_recognition_language = language
        
        # Gelişmiş ayarlar
        speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "300")
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "300")
        
        audio_config = speechsdk.audio.AudioConfig(filename=path)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )
        
        segments = []
        full_text = []
        current_time = 0.0
        
        def recognized_handler(evt):
            nonlocal current_time
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = evt.result.text.strip()
                if text:
                    # Azure zamanlaması tam değil, tahmini hesapla
                    duration = len(text.split()) * 0.5  # ~0.5 saniye/kelime
                    segments.append({
                        "start": current_time,
                        "end": current_time + duration,
                        "text": text,
                        "words": []
                    })
                    full_text.append(text)
                    current_time += duration
        
        speech_recognizer.recognized.connect(recognized_handler)
        
        # Sürekli tanıma başlat
        speech_recognizer.start_continuous_recognition()
        
        # Ses dosyasının süresini bekle (basit yaklaşım)
        import time
        time.sleep(10)  # TODO: Gerçek ses uzunluğuna göre ayarla
        
        speech_recognizer.stop_continuous_recognition()
        
        duration = time.time() - t0
        print(f"[STT][Azure] tamamlandı — süre: {duration:.1f}s")
        
        return {
            "text": " ".join(full_text).strip(),
            "segments": segments,
            "duration": current_time
        }
        
    except Exception as e:
        print(f"[STT][Azure] hata: {e}")
        raise

# =============================================================================
#                           GOOGLE CLOUD SPEECH API
# =============================================================================
def _transcribe_google(
    path: str,
    language: str = "tr-TR",
    credentials_path: Optional[str] = None
) -> Dict:
    """Google Cloud Speech-to-Text"""
    if not _HAS_GOOGLE:
        raise ImportError("Google Cloud Speech not available")
    
    try:
        t0 = time.time()
        
        # Credentials ayarlama
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        client = speech.SpeechClient()
        
        # Ses dosyasını oku
        with open(path, "rb") as audio_file:
            content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            enable_speaker_diarization=False,
            model="latest_long",
            use_enhanced=True
        )
        
        # Uzun ses için async işlem
        if os.path.getsize(path) > 10 * 1024 * 1024:  # 10MB+
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=300)  # 5 dakika timeout
        else:
            response = client.recognize(config=config, audio=audio)
        
        segments = []
        full_text = []
        
        for result in response.results:
            alternative = result.alternatives[0]
            text = alternative.transcript.strip()
            
            if text:
                # Word timing bilgilerini kullan
                start_time = 0.0
                end_time = 0.0
                
                if hasattr(alternative, 'words') and alternative.words:
                    start_time = alternative.words[0].start_time.total_seconds()
                    end_time = alternative.words[-1].end_time.total_seconds()
                
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start_time.total_seconds(),
                            "end": w.end_time.total_seconds(),
                            "confidence": getattr(w, 'confidence', 0.9)
                        }
                        for w in getattr(alternative, 'words', [])
                    ]
                })
                full_text.append(text)
        
        duration = segments[-1]["end"] if segments else 0.0
        processing_time = time.time() - t0
        
        print(f"[STT][Google] tamamlandı — süre: {processing_time:.1f}s, kayıt uzunluğu: {duration:.1f}s")
        
        return {
            "text": " ".join(full_text).strip(),
            "segments": segments,
            "duration": duration
        }
        
    except Exception as e:
        print(f"[STT][Google] hata: {e}")
        raise

# =============================================================================
#                        HİBRİT MODEL YAKLAŞIMI
# =============================================================================
def _transcribe_hybrid(
    path: str,
    language: str = "tr",
    models: List[str] = ["whisper", "azure"],
    **kwargs
) -> Dict:
    """Çoklu model kullanarak hibrit transkripsiyon"""
    results = []
    
    for model_name in models:
        try:
            print(f"[STT][Hibrit] {model_name} modeli deneniyor...")
            
            if model_name == "whisper":
                result = transcribe(path, language=language, **kwargs)
            elif model_name == "azure" and _HAS_AZURE:
                result = _transcribe_azure(path, f"{language}-TR" if language == "tr" else language)
            elif model_name == "google" and _HAS_GOOGLE:
                result = _transcribe_google(path, f"{language}-TR" if language == "tr" else language)
            else:
                continue
            
            result["model"] = model_name
            results.append(result)
            
        except Exception as e:
            print(f"[STT][Hibrit] {model_name} başarısız: {e}")
            continue
    
    if not results:
        raise RuntimeError("Hiçbir model başarılı olamadı")
    
    # En iyi sonucu seç (uzunluk ve güven bazında)
    best_result = max(results, key=lambda r: len(r.get("text", "")))
    
    print(f"[STT][Hibrit] En iyi model: {best_result.get('model', 'unknown')}")
    return best_result

# =============================================================================
#                          GELİŞMİŞ ANA API  
# =============================================================================
def transcribe_advanced(
    path: str,
    language: str = "tr",
    quality: str = "ultra",  # "fastest", "balanced", "highest", "ultra", "hybrid"
    preprocess: bool = True,
    engine: str = "auto",  # "whisper", "azure", "google", "hybrid", "auto"
    model_name: Optional[str] = None,  # Model adı (large-v3, medium, vs.)
    device: str = "cpu",  # cpu/cuda
    content_type: str = "auto",  # meeting, lecture, interview, auto
    long_form: bool = False,  # Uzun kayıt optimizasyonu
    beam_size: Optional[int] = None,  # Beam search boyutu
    vad_threshold: Optional[float] = None,  # VAD eşiği
    **kwargs
) -> TranscriptionResult:
    """
    Gelişmiş transkripsiyon API
    
    Args:
        path: Ses dosyası yolu
        language: Dil kodu (tr, en, vs.)
        quality: Kalite seviyesi
            - fastest: En hızlı (small model, minimal işlem)
            - balanced: Dengeli (medium model, ön işleme)
            - highest: En yüksek (large model, tüm özellikler)
            - hybrid: Çoklu model karşılaştırma
        preprocess: Ses ön işleme aktif mi
        engine: Kullanılacak motor
        
    Returns:
        TranscriptionResult: Detaylı sonuç objesi
    """
    start_time = time.time()
    
    # Ses dosyası bilgilerini al
    audio_info = _get_audio_info(path)
    processed_path = path
    
    # Ön işleme
    if preprocess and quality != "fastest":
        processed_path = _preprocess_audio(path)
        # İşlenmiş dosya için bilgileri güncelle
        audio_info = _get_audio_info(processed_path)
    
    # Model parametrelerini belirle (önce manual değerler, sonra quality ayarları)
    if model_name:
        model_size = model_name
    elif quality == "fastest":
        model_size = "tiny"
    elif quality == "balanced":
        model_size = "medium"
    elif quality == "highest":
        model_size = "large-v3"
    elif quality == "ultra":
        model_size = "large-v3"  # Ultra mode with enhanced parameters
    else:  # hybrid
        return _transcribe_hybrid_advanced(processed_path, language, audio_info, **kwargs)
    
    # Beam size belirleme
    if beam_size is None:
        if quality == "fastest":
            beam_size = 1
        elif quality == "balanced":
            beam_size = 2
        elif quality == "highest":
            beam_size = 5
        elif quality == "ultra":
            beam_size = 10  # Ultra high beam size for maximum accuracy
        else:
            beam_size = 1
    
    # VAD ayarları
    if vad_threshold is None:
        vad_threshold = 0.5  # Varsayılan
        if long_form:
            vad_threshold = 0.3  # Uzun kayıtlarda daha hassas
    
    enable_vad = quality != "fastest"
    
    # Adaptif parametreler
    adaptive_params = _adaptive_parameters(audio_info)
    
    # Engine seçimi
    if engine == "auto":
        # Ses özelliklerine göre otomatik seçim
        if audio_info.snr_estimate and audio_info.snr_estimate < 15:
            engine = "whisper"  # Düşük kalitede Whisper daha iyi
        elif audio_info.duration > 3600:  # 1 saat+
            engine = "azure"  # Uzun kayıtlarda cloud servisler daha stabil
        else:
            engine = "whisper"
    
    # Ultra mode optimizasyonları
    if quality == "ultra":
        # Maximum accuracy için özel parametreler
        transcribe_kwargs = {
            'temperature': [0.0, 0.2, 0.4, 0.6, 0.8],  # Multiple temperature sampling
            'best_of': 5,  # En iyi 5 denemeden seç
            'beam_size': beam_size,
            'patience': 2.0,  # Daha sabırlı decode
            'suppress_tokens': [-1],  # Özel token bastırma
            'condition_on_previous_text': True,
            'compression_ratio_threshold': 2.4,
            'logprob_threshold': -1.0,
            'no_speech_threshold': 0.6,
            'word_timestamps': True,  # Ultra modda kelime zamanları
            'prepend_punctuations': "\"'([{-",
            'append_punctuations': "\"'.,:!?)]}"
        }
    else:
        # Uzun kayıt optimizasyonları
        transcribe_kwargs = kwargs.copy()
        if long_form:
            # Uzun kayıtlar için özel parametreler
            transcribe_kwargs.update({
                'chunk_length_s': 30,  # 30 saniyelik parçalar
                'batch_size': 4,  # Daha büyük batch
                'progress': True,  # İlerleme göster
            })
            if enable_vad:
                transcribe_kwargs.update({
                    'vad_min_silence_ms': 500,  # Daha uzun sessizlik gerekli
                    'vad_window_ms': 30  # Daha büyük pencere
                })

    # Transkripsiyon yap
    if engine == "whisper":
        raw_result = transcribe(
            processed_path,
            language=language,
            model_size=model_size,
            device=device,
            beam_size=adaptive_params.get('beam_size', beam_size),
            **transcribe_kwargs
        )
        model_used = f"whisper-{model_size}"
    elif engine == "azure":
        raw_result = _transcribe_azure(processed_path, language)
        model_used = "azure-speech"
    elif engine == "google":
        raw_result = _transcribe_google(processed_path, language)
        model_used = "google-speech"
    else:
        raise ValueError(f"Bilinmeyen engine: {engine}")
    
    # Güven skoru hesapla
    confidence = _calculate_confidence(raw_result.get("segments", []), model_used)
    
    # Kalite metrikleri
    quality_metrics = {
        "word_count": len(raw_result.get("text", "").split()),
        "segment_count": len(raw_result.get("segments", [])),
        "avg_segment_length": np.mean([
            len(s.get("text", "").split()) 
            for s in raw_result.get("segments", [])
        ]) if raw_result.get("segments") else 0,
        "processing_speed_rtf": (time.time() - start_time) / audio_info.duration if audio_info.duration > 0 else 0
    }
    
    # İçerik tipine göre ek işlemler
    segments = raw_result.get("segments", [])
    text = raw_result.get("text", "")
    
    if content_type == "meeting":
        # Toplantı için speaker labels ekle
        segments = _enhance_meeting_segments(segments)
    elif content_type == "lecture":
        # Ders için konu başlıkları belirle
        segments = _enhance_lecture_segments(segments)
    
    # Sonucu yapılandır - dict döndür (TranscriptionResult yerine uyumluluk için)
    result_dict = {
        'text': text,
        'segments': segments,
        'duration': raw_result.get("duration", audio_info.duration),
        'confidence': confidence,
        'audio_quality': _assess_audio_quality(audio_info),
        'model_used': model_used,
        'processing_time': time.time() - start_time,
        'quality_metrics': quality_metrics
    }
    
    # Geçici dosyayı temizle
    if processed_path != path and os.path.exists(processed_path):
        try:
            os.unlink(processed_path)
        except:
            pass
    
    return result_dict

def _transcribe_hybrid_advanced(
    path: str,
    language: str,
    audio_info: AudioInfo,
    **kwargs
) -> TranscriptionResult:
    """Hibrit model ile gelişmiş transkripsiyon"""
    start_time = time.time()
    
    # Farklı modelleri paralel çalıştır
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        
        # Whisper
        futures.append(
            executor.submit(transcribe, path, language=language, model_size="medium")
        )
        
        # Azure (varsa)
        if _HAS_AZURE and os.getenv("AZURE_SPEECH_KEY"):
            futures.append(
                executor.submit(_transcribe_azure, path, language)
            )
        
        # Google (varsa)  
        if _HAS_GOOGLE and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            futures.append(
                executor.submit(_transcribe_google, path, language)
            )
        
        # Sonuçları topla
        results = []
        for future in as_completed(futures, timeout=600):  # 10 dakika timeout
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[STT][Hibrit] Model hatası: {e}")
    
    if not results:
        raise RuntimeError("Hibrit modelde hiçbir model başarılı olamadı")
    
    # Sonuçları karşılaştır ve en iyisini seç
    best_result = max(results, key=lambda r: len(r.get("text", "")))
    
    confidence = _calculate_confidence(best_result.get("segments", []), "hybrid")
    
    return TranscriptionResult(
        text=best_result.get("text", ""),
        segments=best_result.get("segments", []),
        duration=best_result.get("duration", audio_info.duration),
        confidence=confidence,
        audio_info=audio_info,
        model_used="hybrid-multi",
        processing_time=time.time() - start_time,
        quality_metrics={
            "models_tried": len(results),
            "best_model_score": len(best_result.get("text", ""))
        }
    )

# =============================================================================
#                      İLERİ DÜZEY POST-PROCESSING
# =============================================================================
def _advanced_text_correction(text: str, language: str = "tr") -> str:
    """Gelişmiş metin düzeltme ve normalizasyon"""
    if not text:
        return ""
    
    # 1. NLP modülü ile temel düzeltme
    corrected = text
    if nlp and hasattr(nlp, "normalize_transcript"):
        try:
            corrected = nlp.normalize_transcript(text)
        except Exception:
            pass
    
    # 2. Dil özel düzeltmeler
    if language.startswith("tr"):
        # Türkçe özel düzeltmeler
        corrections = {
            # Yaygın STT hataları
            "bi": "bir", "bişey": "bir şey", "hiçbi": "hiçbir",
            "nası": "nasıl", "böle": "böyle", "şöle": "şöyle",
            "deil": "değil", "oluo": "oluyor", "diyo": "diyor",
            "gelio": "geliyor", "gidio": "gidiyor",
            
            # Teknik terimler
            "veri tabanı": "veritabanı", "veri base": "veritabanı",
            "data base": "veritabanı", "deyıtı": "data",
            "paiton": "Python", "payton": "Python",
            "cubernets": "Kubernetes", "kubernetis": "Kubernetes",
            
            # Yaygın kelimeler
            "toplantımız": "toplantımız", "görüşmemiz": "görüşmemiz",
        }
        
        for wrong, right in corrections.items():
            corrected = re.sub(rf"\b{re.escape(wrong)}\b", right, corrected, flags=re.IGNORECASE)
    
    # 3. Tekrarları temizle
    corrected = _remove_advanced_duplicates(corrected)
    
    # 4. Cümle yapısını düzelt
    corrected = _fix_sentence_structure(corrected)
    
    return corrected.strip()

def _remove_advanced_duplicates(text: str) -> str:
    """Gelişmiş tekrar temizleme"""
    sentences = re.split(r'[.!?]+', text)
    cleaned = []
    
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue
            
        # Önceki cümle ile benzerlik kontrolü
        is_duplicate = False
        if i > 0 and cleaned:
            prev_sent = cleaned[-1]
            similarity = _text_similarity(sent, prev_sent)
            if similarity > 0.8:  # %80+ benzer
                is_duplicate = True
        
        if not is_duplicate:
            cleaned.append(sent)
    
    return '. '.join(cleaned) + ('.' if cleaned else '')

def _text_similarity(text1: str, text2: str) -> float:
    """İki metin arasındaki benzerlik oranı"""
    if not text1 or not text2:
        return 0.0
    
    if _HAS_FUZZ:
        return fuzz.ratio(text1.lower(), text2.lower()) / 100.0
    
    # Basit jaccard benzerliği
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union if union > 0 else 0.0

def _fix_sentence_structure(text: str) -> str:
    """Cümle yapısını düzelt"""
    # Noktalama düzeltmeleri
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Noktalama öncesi boşluk
    text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)  # Noktalama sonrası boşluk
    text = re.sub(r'\s+', ' ', text)  # Çoklu boşlukları tek yap
    
    # Cümle başlarını büyük yap
    sentences = re.split(r'([.!?]+)', text)
    fixed = []
    
    for i, part in enumerate(sentences):
        if i % 2 == 0 and part.strip():  # Cümle metni
            part = part.strip()
            if part:
                part = part[0].upper() + part[1:] if len(part) > 1 else part.upper()
                fixed.append(part)
        elif part.strip():  # Noktalama
            fixed.append(part)
    
    return ''.join(fixed)

def _extract_speaker_segments(segments: List[Dict]) -> List[Dict]:
    """Konuşmacı segmentasyonu (basit versiyon)"""
    # Bu gerçek bir speaker diarization sistemi değil
    # Sadece sessizlik bazlı segment ayırma
    
    enhanced_segments = []
    current_speaker = "Konuşmacı_1"
    speaker_counter = 1
    
    for i, seg in enumerate(segments):
        if i > 0:
            prev_end = segments[i-1].get("end", 0)
            current_start = seg.get("start", 0)
            pause_duration = current_start - prev_end
            
            # 3 saniyeden uzun ara varsa konuşmacı değişti kabul et
            if pause_duration > 3.0:
                speaker_counter += 1
                current_speaker = f"Konuşmacı_{speaker_counter}"
        
        enhanced_seg = dict(seg)
        enhanced_seg["speaker"] = current_speaker
        enhanced_segments.append(enhanced_seg)
    
    return enhanced_segments

def _quality_assessment(result: TranscriptionResult) -> Dict[str, Any]:
    """Transkripsiyon kalitesini değerlendir"""
    assessment = {
        "overall_score": 0.0,
        "text_quality": 0.0,
        "timing_accuracy": 0.0,
        "completeness": 0.0,
        "issues": []
    }
    
    text = result.text
    segments = result.segments
    
    # 1. Metin kalitesi
    if text:
        word_count = len(text.split())
        # Çok kısa metinler şüpheli
        if word_count < 10:
            assessment["issues"].append("Çok kısa transkripsiyon")
            assessment["text_quality"] = 0.3
        else:
            # Özel terim varlığı
            terms_found = sum(1 for term in _load_custom_terms() 
                            if term.lower() in text.lower())
            term_ratio = terms_found / max(1, word_count / 50)  # 50 kelimede 1 terim bekleniyor
            
            # Tekrar oranı
            unique_words = len(set(text.lower().split()))
            repeat_ratio = 1 - (unique_words / word_count) if word_count > 0 else 0
            
            assessment["text_quality"] = min(1.0, 0.7 + 0.2 * min(1, term_ratio) - 0.3 * repeat_ratio)
    
    # 2. Zamanlama doğruluğu
    if segments:
        timing_issues = 0
        for i, seg in enumerate(segments):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            
            if end <= start:
                timing_issues += 1
                
            if i > 0:
                prev_end = segments[i-1].get("end", 0)
                if start < prev_end:  # Overlap
                    timing_issues += 1
        
        assessment["timing_accuracy"] = max(0, 1 - (timing_issues / len(segments)))
        
        if timing_issues > len(segments) * 0.1:  # %10'dan fazla problem
            assessment["issues"].append("Zamanlama sorunları")
    
    # 3. Tamlık
    expected_duration = result.duration
    if expected_duration > 0 and segments:
        covered_time = segments[-1].get("end", 0) - segments[0].get("start", 0)
        coverage = covered_time / expected_duration
        assessment["completeness"] = min(1.0, coverage)
        
        if coverage < 0.8:
            assessment["issues"].append("Eksik ses analizi")
    
    # Genel skor
    assessment["overall_score"] = (
        0.5 * assessment["text_quality"] +
        0.3 * assessment["timing_accuracy"] + 
        0.2 * assessment["completeness"]
    )
    
    return assessment

def post_process_result(result: TranscriptionResult, language: str = "tr") -> TranscriptionResult:
    """Sonuçları gelişmiş post-processing ile iyileştir"""
    
    # 1. Metin düzeltme
    corrected_text = _advanced_text_correction(result.text, language)
    
    # 2. Segment metinlerini düzelt
    corrected_segments = []
    for seg in result.segments:
        corrected_seg = dict(seg)
        corrected_seg["text"] = _advanced_text_correction(seg.get("text", ""), language)
        corrected_segments.append(corrected_seg)
    
    # 3. Konuşmacı bilgilerini ekle
    speaker_segments = _extract_speaker_segments(corrected_segments)
    
    # 4. Kalite değerlendirmesi
    enhanced_result = TranscriptionResult(
        text=corrected_text,
        segments=speaker_segments,
        duration=result.duration,
        confidence=result.confidence,
        audio_info=result.audio_info,
        model_used=result.model_used,
        processing_time=result.processing_time,
        word_level=result.word_level,
        speaker_info=[{"speaker": f"Konuşmacı_{i}", "segments": len([s for s in speaker_segments if s.get('speaker') == f"Konuşmacı_{i}"])} 
                     for i in range(1, max([int(s.get('speaker', 'Konuşmacı_1').split('_')[1]) for s in speaker_segments], default=[1]) + 1)],
        quality_metrics=result.quality_metrics
    )
    
    # Kalite değerlendirmesi ekle
    quality_assessment = _quality_assessment(enhanced_result)
    if enhanced_result.quality_metrics:
        enhanced_result.quality_metrics.update(quality_assessment)
    else:
        enhanced_result.quality_metrics = quality_assessment
    
    return enhanced_result

# =============================================================================
#                          KOLAY KULLANIM API'LERİ
# =============================================================================
def transcribe_simple(audio_path: str, language: str = "tr") -> str:
    """En basit kullanım: sadece metin döndür"""
    result = transcribe_advanced(audio_path, language=language, quality="balanced")
    return result.get('text', '') if isinstance(result, dict) else str(result)

def transcribe_with_speakers(audio_path: str, language: str = "tr") -> Dict[str, Any]:
    """Konuşmacı bilgileri ile transkripsiyon"""
    result = transcribe_advanced(audio_path, language=language, quality="highest")
    result = post_process_result(result, language)
    
    return {
        "text": result.text,
        "speakers": result.speaker_info,
        "segments": result.segments,
        "confidence": result.confidence
    }

def transcribe_for_meeting(audio_path: str, language: str = "tr") -> Dict[str, Any]:
    """Toplantı kayıtları için optimize edilmiş transkripsiyon"""
    result = transcribe_advanced(
        audio_path, 
        language=language, 
        quality="highest",
        preprocess=True,
        engine="auto"
    )
    result = post_process_result(result, language)
    
    # NLP modülü ile görev ve kararları çıkar
    tasks, decisions = [], []
    if nlp:
        try:
            if hasattr(nlp, "extract_tasks"):
                tasks = nlp.extract_tasks(result.text)
            if hasattr(nlp, "extract_decisions"):
                decisions = nlp.extract_decisions(result.text)
        except Exception:
            pass
    
    return {
        "transcript": result.text,
        "segments": result.segments,
        "speakers": result.speaker_info,
        "tasks": tasks,
        "decisions": decisions,
        "confidence": result.confidence,
        "quality_score": result.quality_metrics.get("overall_score", 0.0) if result.quality_metrics else 0.0,
        "processing_time": result.processing_time
    }

# Geriye uyumluluk için eski API
def transcribe_file(path: str, **kwargs) -> Dict:
    """Eski API ile uyumluluk"""
    result = transcribe_advanced(path, **kwargs)
    return asdict(result)
