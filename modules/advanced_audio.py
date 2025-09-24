# modules/advanced_audio.py - Ultra Advanced Audio Processing
"""
Ultra Gelişmiş Ses İşleme Modülü
===============================

Bu modül %99.9 doğruluk hedefine ulaşmak için:
- Spektral gürültü azaltma
- Adaptive filtering 
- Echo cancellation
- Bandwidth extension
- Psychoacoustic enhancement
- Multi-channel processing
- Real-time quality assessment

Kullanım:
    processor = UltraAudioProcessor()
    enhanced_audio = processor.enhance_for_maximum_accuracy(audio_path)
"""

import os
import numpy as np
import scipy.signal
import scipy.fftpack
from typing import Optional, Tuple, Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

try:
    import librosa
    import noisereduce as nr
    import webrtcvad
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    from pydub.silence import split_on_silence, detect_nonsilent
    import soundfile as sf
    _HAS_ADVANCED_AUDIO = True
except ImportError:
    _HAS_ADVANCED_AUDIO = False

try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    _HAS_TORCH_AUDIO = True
except ImportError:
    _HAS_TORCH_AUDIO = False


class SpectralNoiseReducer:
    """Gelişmiş spektral gürültü azaltma sistemi"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.noise_profile = None
        
    def estimate_noise_profile(self, audio: np.ndarray, silence_duration: float = 1.0) -> np.ndarray:
        """Sessizlik bölgelerinden gürültü profili çıkar"""
        # İlk ve son saniyelerden gürültü örneği al
        silence_samples = int(silence_duration * self.sr)
        
        if len(audio) > silence_samples * 2:
            noise_start = audio[:silence_samples]
            noise_end = audio[-silence_samples:]
            noise_sample = np.concatenate([noise_start, noise_end])
        else:
            noise_sample = audio[:len(audio)//4]  # İlk %25
            
        # Spektral analiz
        _, _, spectrogram = scipy.signal.spectrogram(
            noise_sample, self.sr, nperseg=1024, noverlap=512
        )
        
        # Gürültü profilini ortalama olarak hesapla
        self.noise_profile = np.mean(np.abs(spectrogram), axis=1)
        return self.noise_profile
    
    def adaptive_spectral_subtraction(self, audio: np.ndarray, alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
        """Adaptif spektral çıkarma algoritması"""
        if self.noise_profile is None:
            self.estimate_noise_profile(audio)
            
        # STFT dönüşümü
        f, t, stft_matrix = scipy.signal.stft(
            audio, self.sr, nperseg=1024, noverlap=512
        )
        
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        
        # Spektral çıkarma
        noise_mag = self.noise_profile[:, np.newaxis]
        
        # Adaptif çıkarma faktörü
        snr_estimate = magnitude / (noise_mag + 1e-10)
        adaptive_alpha = alpha * (1 - np.exp(-snr_estimate / 3))
        
        # Çıkarma işlemi
        cleaned_magnitude = magnitude - adaptive_alpha * noise_mag
        
        # Minimum eşik uygula
        cleaned_magnitude = np.maximum(
            cleaned_magnitude, 
            beta * magnitude
        )
        
        # Geri dönüşüm
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        _, cleaned_audio = scipy.signal.istft(
            cleaned_stft, self.sr, nperseg=1024, noverlap=512
        )
        
        return cleaned_audio.astype(np.float32)


class EchoCanceller:
    """Gelişmiş echo cancellation sistemi"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        
    def detect_echo_parameters(self, audio: np.ndarray) -> Dict[str, float]:
        """Echo parametrelerini otomatik tespit et"""
        # Autocorrelation analizi
        correlation = np.correlate(audio, audio, mode='full')
        correlation = correlation[correlation.size // 2:]
        
        # Echo delay tespiti
        peaks, _ = scipy.signal.find_peaks(
            correlation[int(0.05 * self.sr):],  # İlk 50ms'yi atla
            height=0.1 * np.max(correlation),
            distance=int(0.01 * self.sr)  # Min 10ms aralık
        )
        
        echo_delay = 0
        echo_strength = 0
        
        if len(peaks) > 0:
            echo_delay = (peaks[0] + int(0.05 * self.sr)) / self.sr
            echo_strength = correlation[peaks[0]] / correlation[0]
            
        return {
            'delay': echo_delay,
            'strength': echo_strength,
            'room_size': echo_delay * 343 / 2  # Yaklaşık oda boyutu (metre)
        }
    
    def adaptive_echo_cancellation(self, audio: np.ndarray) -> np.ndarray:
        """Adaptif echo cancellation"""
        echo_params = self.detect_echo_parameters(audio)
        
        if echo_params['strength'] < 0.1:
            return audio  # Zayıf echo, işlem gerekli değil
            
        delay_samples = int(echo_params['delay'] * self.sr)
        
        if delay_samples == 0 or delay_samples >= len(audio):
            return audio
            
        # Wiener filter yaklaşımı
        # Echo olan kısmı tespit et ve azalt
        enhanced_audio = audio.copy()
        
        for i in range(delay_samples, len(audio)):
            # Echo tahminini çıkar
            echo_estimate = echo_params['strength'] * audio[i - delay_samples]
            enhanced_audio[i] = audio[i] - 0.7 * echo_estimate
            
        return enhanced_audio


class BandwidthExtender:
    """Ses bant genişliği genişletici"""
    
    def __init__(self, sample_rate: int = 16000, target_rate: int = 48000):
        self.sr = sample_rate
        self.target_sr = target_rate
        
    def harmonic_extension(self, audio: np.ndarray) -> np.ndarray:
        """Harmonik analiz ile yüksek frekans ekleme"""
        if not _HAS_TORCH_AUDIO:
            return audio
            
        # PyTorch tensor'a çevir
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        
        # Resampling ile upsampling
        resampler = T.Resample(self.sr, self.target_sr)
        upsampled = resampler(audio_tensor)
        
        # Harmonik zenginleştirme
        # Temel frekansları tespit et ve harmoniklerini güçlendir
        stft = torch.stft(
            upsampled.squeeze(0), 
            n_fft=2048, 
            hop_length=512, 
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Yüksek frekanslarda harmonik ekleme
        enhanced_magnitude = magnitude.clone()
        n_freqs = magnitude.shape[0]
        
        # Alt yarıdaki güçlü frekansları üst yarıya kopyala (zayıflatılmış)
        for i in range(n_freqs // 4, n_freqs // 2):
            if i * 2 < n_freqs:
                enhanced_magnitude[i * 2] += 0.3 * magnitude[i]
                
        # Geri dönüşüm
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_audio = torch.istft(
            enhanced_stft, 
            n_fft=2048, 
            hop_length=512
        )
        
        return enhanced_audio.numpy()


class PsychoacousticEnhancer:
    """Psikosesitik tabanlı ses geliştirici"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        
    def critical_band_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Kritik bantlara göre ses geliştirme"""
        # Bark ölçeğine göre kritik bantlar
        bark_frequencies = [
            20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
            1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
            4400, 5300, 6400, 7700, 9500, 12000, 15500
        ]
        
        # Her bant için ayrı işleme
        enhanced_audio = np.zeros_like(audio)
        
        for i in range(len(bark_frequencies) - 1):
            low_freq = bark_frequencies[i]
            high_freq = bark_frequencies[i + 1]
            
            # Bant geçiren filtre
            sos = scipy.signal.butter(
                4, [low_freq, high_freq], 
                btype='band', fs=self.sr, output='sos'
            )
            
            band_audio = scipy.signal.sosfilt(sos, audio)
            
            # İnsan kulağının hassasiyet eğrisine göre güçlendirme
            # 1-4 kHz arasında daha fazla güçlendirme (konuşma için kritik)
            if 1000 <= low_freq <= 4000:
                enhancement_factor = 1.2
            elif 300 <= low_freq <= 1000 or 4000 <= low_freq <= 8000:
                enhancement_factor = 1.1
            else:
                enhancement_factor = 1.05
                
            enhanced_audio += enhancement_factor * band_audio
            
        return enhanced_audio


class AdvancedVAD:
    """İleri düzey Voice Activity Detection"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.energy_threshold = None
        self.spectral_threshold = None
        
    def multi_feature_vad(self, audio: np.ndarray) -> np.ndarray:
        """Çoklu özellik tabanlı VAD"""
        # 1. Enerji tabanlı
        energy_vad = self._energy_based_vad(audio)
        
        # 2. Spektral tabanlı  
        spectral_vad = self._spectral_based_vad(audio)
        
        # 3. Zero crossing rate tabanlı
        zcr_vad = self._zcr_based_vad(audio)
        
        # 4. Spektral centroid tabanlı
        centroid_vad = self._spectral_centroid_vad(audio)
        
        # Voting mechanism
        combined_vad = (
            0.3 * energy_vad + 
            0.3 * spectral_vad + 
            0.2 * zcr_vad + 
            0.2 * centroid_vad
        )
        
        # Threshold ve smoothing
        voice_mask = combined_vad > 0.5
        
        # Morphological operations
        from scipy import ndimage
        voice_mask = ndimage.binary_opening(voice_mask, iterations=2)
        voice_mask = ndimage.binary_closing(voice_mask, iterations=3)
        
        return voice_mask
        
    def _energy_based_vad(self, audio: np.ndarray) -> np.ndarray:
        """Enerji tabanlı VAD"""
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.01 * self.sr)     # 10ms hop
        
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
            
        energy = np.array(energy)
        
        if self.energy_threshold is None:
            # Adaptif eşik belirleme
            self.energy_threshold = np.percentile(energy, 30)
            
        # Normalize ve threshold
        energy_norm = energy / (np.max(energy) + 1e-10)
        return energy_norm > (self.energy_threshold / np.max(energy))
        
    def _spectral_based_vad(self, audio: np.ndarray) -> np.ndarray:
        """Spektral özellik tabanlı VAD"""
        if not _HAS_ADVANCED_AUDIO:
            return self._energy_based_vad(audio)
            
        # MFCC özellik çıkarma
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        
        # Spektral entropi
        spectral_entropy = []
        for frame in mfcc.T:
            # Her frame için entropi hesapla
            prob = np.abs(frame) / (np.sum(np.abs(frame)) + 1e-10)
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            spectral_entropy.append(entropy)
            
        spectral_entropy = np.array(spectral_entropy)
        
        if self.spectral_threshold is None:
            self.spectral_threshold = np.percentile(spectral_entropy, 40)
            
        return spectral_entropy > self.spectral_threshold
        
    def _zcr_based_vad(self, audio: np.ndarray) -> np.ndarray:
        """Zero crossing rate tabanlı VAD"""
        frame_length = int(0.025 * self.sr)
        hop_length = int(0.01 * self.sr)
        
        zcr = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            # Zero crossing sayısını hesapla
            zero_crossings = np.sum(np.diff(np.sign(frame)) != 0)
            zcr.append(zero_crossings / len(frame))
            
        zcr = np.array(zcr)
        
        # Konuşma genellikle 0.02-0.5 arasında ZCR'ye sahip
        return (zcr > 0.02) & (zcr < 0.5)
        
    def _spectral_centroid_vad(self, audio: np.ndarray) -> np.ndarray:
        """Spektral centroid tabanlı VAD"""
        if not _HAS_ADVANCED_AUDIO:
            return self._energy_based_vad(audio)
            
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        
        # Konuşma için tipik centroid değerleri: 1000-4000 Hz
        voice_range = (centroid > 1000) & (centroid < 4000)
        
        return voice_range


class UltraAudioProcessor:
    """Ultra gelişmiş ses işleme sistemi"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.noise_reducer = SpectralNoiseReducer(sample_rate)
        self.echo_canceller = EchoCanceller(sample_rate)
        self.bandwidth_extender = BandwidthExtender(sample_rate)
        self.psychoacoustic_enhancer = PsychoacousticEnhancer(sample_rate)
        self.advanced_vad = AdvancedVAD(sample_rate)
        
    def analyze_audio_quality(self, audio: np.ndarray) -> Dict[str, float]:
        """Detaylı ses kalitesi analizi"""
        analysis = {}
        
        # 1. SNR (Signal-to-Noise Ratio) tahmini
        analysis['snr_estimate'] = self._estimate_snr(audio)
        
        # 2. THD (Total Harmonic Distortion)
        analysis['thd'] = self._calculate_thd(audio)
        
        # 3. Spektral bant genişliği
        analysis['bandwidth'] = self._estimate_bandwidth(audio)
        
        # 4. Dinamik aralık
        analysis['dynamic_range'] = self._calculate_dynamic_range(audio)
        
        # 5. Konuşma oranı
        analysis['speech_ratio'] = self._estimate_speech_ratio(audio)
        
        # 6. Echo seviyesi
        echo_params = self.echo_canceller.detect_echo_parameters(audio)
        analysis['echo_strength'] = echo_params['strength']
        
        # 7. Genel kalite skoru (0-100)
        analysis['quality_score'] = self._calculate_quality_score(analysis)
        
        return analysis
        
    def enhance_for_maximum_accuracy(self, audio_path: str, target_quality: float = 95.0) -> Tuple[np.ndarray, Dict]:
        """Maksimum doğruluk için ses geliştirme"""
        # Ses dosyasını yükle
        if isinstance(audio_path, str):
            if _HAS_ADVANCED_AUDIO:
                audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            else:
                # Fallback yükleme
                from scipy.io import wavfile
                sr, audio = wavfile.read(audio_path)
                if sr != self.sr:
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * self.sr / sr))
                audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio_path
            
        original_quality = self.analyze_audio_quality(audio)
        processing_log = {'original_quality': original_quality, 'steps': []}
        
        # Aşama 1: Normalize et
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        processing_log['steps'].append('normalization')
        
        # Aşama 2: DC offset kaldır
        audio = audio - np.mean(audio)
        
        # Aşama 3: İleri düzey gürültü azaltma
        if original_quality['snr_estimate'] < 20:  # Düşük SNR
            audio = self.noise_reducer.adaptive_spectral_subtraction(audio, alpha=2.5, beta=0.05)
            processing_log['steps'].append('spectral_noise_reduction')
            
        # Aşama 4: Echo cancellation
        if original_quality['echo_strength'] > 0.15:
            audio = self.echo_canceller.adaptive_echo_cancellation(audio)
            processing_log['steps'].append('echo_cancellation')
            
        # Aşama 5: Bandwidth extension
        if original_quality['bandwidth'] < 7000:  # Dar bant
            try:
                audio = self.bandwidth_extender.harmonic_extension(audio)
                processing_log['steps'].append('bandwidth_extension')
            except Exception:
                pass  # Torch yoksa atla
                
        # Aşama 6: Psychoacoustic enhancement
        audio = self.psychoacoustic_enhancer.critical_band_enhancement(audio)
        processing_log['steps'].append('psychoacoustic_enhancement')
        
        # Aşama 7: Advanced VAD ile sessizlikleri temizle
        voice_mask = self.advanced_vad.multi_feature_vad(audio)
        
        # Sessizlik bölgelerinde gürültü azalt
        frame_length = len(voice_mask)
        audio_frames = np.array_split(audio, frame_length)
        
        enhanced_frames = []
        for i, frame in enumerate(audio_frames):
            if i < len(voice_mask) and not voice_mask[i]:
                # Sessizlik bölgesi - gürültü azalt
                frame = frame * 0.1
            enhanced_frames.append(frame)
            
        audio = np.concatenate(enhanced_frames)
        processing_log['steps'].append('advanced_vad_cleaning')
        
        # Aşama 8: Final normalization ve clipping önleme
        audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.95
        
        # Final kalite analizi
        final_quality = self.analyze_audio_quality(audio)
        processing_log['final_quality'] = final_quality
        processing_log['improvement'] = final_quality['quality_score'] - original_quality['quality_score']
        
        return audio, processing_log
        
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """SNR tahmini"""
        # Ses aktivitesi tespiti
        voice_mask = self.advanced_vad.multi_feature_vad(audio)
        
        if np.sum(voice_mask) == 0:
            return 0.0
            
        frame_length = len(audio) // len(voice_mask)
        
        signal_power = 0
        noise_power = 0
        signal_frames = 0
        noise_frames = 0
        
        for i, is_voice in enumerate(voice_mask):
            start_idx = i * frame_length
            end_idx = min((i + 1) * frame_length, len(audio))
            frame = audio[start_idx:end_idx]
            frame_power = np.mean(frame ** 2)
            
            if is_voice:
                signal_power += frame_power
                signal_frames += 1
            else:
                noise_power += frame_power
                noise_frames += 1
                
        if signal_frames == 0 or noise_frames == 0:
            return 20.0  # Varsayılan
            
        avg_signal_power = signal_power / signal_frames
        avg_noise_power = noise_power / noise_frames
        
        if avg_noise_power == 0:
            return 50.0  # Çok temiz sinyal
            
        snr_linear = avg_signal_power / avg_noise_power
        snr_db = 10 * np.log10(snr_linear + 1e-10)
        
        return max(0, snr_db)
        
    def _calculate_thd(self, audio: np.ndarray) -> float:
        """Total Harmonic Distortion hesaplama"""
        # FFT ile fundamental ve harmonikleri bul
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sr)
        magnitude = np.abs(fft)
        
        # Temel frekansı bul (en güçlü komponente yakın)
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # 100Hz - 4000Hz arası (konuşma aralığı)
        speech_range = (positive_freqs > 100) & (positive_freqs < 4000)
        if not np.any(speech_range):
            return 0.0
            
        fundamental_idx = np.argmax(positive_magnitude[speech_range])
        fundamental_freq = positive_freqs[speech_range][fundamental_idx]
        fundamental_power = positive_magnitude[speech_range][fundamental_idx] ** 2
        
        # Harmonikleri bul
        harmonic_power = 0
        for h in range(2, 6):  # 2., 3., 4., 5. harmonikler
            harmonic_freq = h * fundamental_freq
            if harmonic_freq < self.sr / 2:
                # En yakın frekans bileşenini bul
                harmonic_idx = np.argmin(np.abs(positive_freqs - harmonic_freq))
                harmonic_power += positive_magnitude[harmonic_idx] ** 2
                
        if fundamental_power == 0:
            return 0.0
            
        thd = np.sqrt(harmonic_power / fundamental_power) * 100
        return min(thd, 100.0)  # %100 ile sınırla
        
    def _estimate_bandwidth(self, audio: np.ndarray) -> float:
        """Spektral bant genişliği tahmini"""
        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/self.sr)
        magnitude = np.abs(fft)
        
        # Pozitif frekanslar
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        
        # Enerji eşiği (%5 seviyesinde)
        threshold = 0.05 * np.max(positive_magnitude)
        
        # Eşik üzerindeki en yüksek frekans
        above_threshold = positive_magnitude > threshold
        if not np.any(above_threshold):
            return 0.0
            
        max_freq_idx = np.max(np.where(above_threshold)[0])
        bandwidth = positive_freqs[max_freq_idx]
        
        return bandwidth
        
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Dinamik aralık hesaplama"""
        # RMS değerleri hesapla (pencereli)
        window_size = int(0.1 * self.sr)  # 100ms pencere
        rms_values = []
        
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            if rms > 1e-10:  # Sessizlik değil
                rms_values.append(rms)
                
        if len(rms_values) == 0:
            return 0.0
            
        rms_values = np.array(rms_values)
        max_rms = np.max(rms_values)
        min_rms = np.min(rms_values)
        
        if min_rms == 0:
            return 60.0  # Maksimum dinamik aralık
            
        dynamic_range_db = 20 * np.log10(max_rms / min_rms)
        return min(dynamic_range_db, 120.0)  # 120dB ile sınırla
        
    def _estimate_speech_ratio(self, audio: np.ndarray) -> float:
        """Konuşma oranı tahmini"""
        voice_mask = self.advanced_vad.multi_feature_vad(audio)
        return np.mean(voice_mask)
        
    def _calculate_quality_score(self, analysis: Dict[str, float]) -> float:
        """Genel kalite skoru hesaplama (0-100)"""
        score = 0.0
        
        # SNR skoru (40% ağırlık)
        snr_score = min(analysis['snr_estimate'] / 30.0 * 100, 100)
        score += 0.4 * snr_score
        
        # THD skoru (20% ağırlık) - düşük THD iyi
        thd_score = max(0, 100 - analysis['thd'] * 2)
        score += 0.2 * thd_score
        
        # Bandwidth skoru (20% ağırlık)
        bandwidth_score = min(analysis['bandwidth'] / 8000.0 * 100, 100)
        score += 0.2 * bandwidth_score
        
        # Speech ratio skoru (10% ağırlık)
        speech_score = analysis['speech_ratio'] * 100
        score += 0.1 * speech_score
        
        # Echo skoru (10% ağırlık) - düşük echo iyi
        echo_score = max(0, 100 - analysis['echo_strength'] * 200)
        score += 0.1 * echo_score
        
        return min(score, 100.0)


# Kolay kullanım fonksiyonları
def enhance_audio_for_ultra_accuracy(audio_path: str, target_quality: float = 95.0) -> Tuple[np.ndarray, Dict]:
    """Tek fonksiyon ile ultra doğruluk optimizasyonu"""
    processor = UltraAudioProcessor()
    return processor.enhance_for_maximum_accuracy(audio_path, target_quality)


def analyze_audio_quality_detailed(audio_path: str) -> Dict[str, Any]:
    """Detaylı ses kalitesi analizi"""
    processor = UltraAudioProcessor()
    
    # Ses yükle
    if _HAS_ADVANCED_AUDIO:
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    else:
        from scipy.io import wavfile
        sr, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32) / 32768.0
        
    return processor.analyze_audio_quality(audio)


if __name__ == "__main__":
    # Test kodu
    print("🎵 Ultra Advanced Audio Processing Test")
    print("=" * 50)
    
    if not _HAS_ADVANCED_AUDIO:
        print("⚠️ Gelişmiş ses kütüphaneleri eksik")
        print("pip install librosa noisereduce soundfile")
    else:
        print("✅ Tüm ses işleme kütüphaneleri hazır")
        print("🚀 Ultra accuracy audio processing ready!")