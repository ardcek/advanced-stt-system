# modules/long_form_processor.py
"""
Long-Form Audio Processor
========================

2-3 saatlik ses kayıtlarını işlemek için optimized sistem:
- Memory-efficient chunk processing
- Progressive STT + summarization
- Real-time progress tracking
- Hierarchical output generation
"""

import os
import math
import time
from typing import List, Dict, Tuple, Optional, Generator
from datetime import datetime, timedelta

class LongFormProcessor:
    """
    Uzun ses kayıtları için endüstriyel işlemci
    """
    
    def __init__(self, chunk_duration: int = 60, overlap_duration: int = 5):
        """
        Args:
            chunk_duration: Her chunk'ın süresi (saniye)
            overlap_duration: Chunk'lar arası overlap (saniye)
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.processing_stats = {}
        
    def estimate_audio_duration(self, audio_file: str) -> float:
        """Ses dosyasının süresini tahmin et (soundfile kullanarak)"""
        try:
            import soundfile as sf
            with sf.SoundFile(audio_file) as f:
                duration = len(f) / f.samplerate
            return duration
        except Exception as e:
            print(f"Duration estimation error: {e}")
            return 0.0
    
    def calculate_chunks(self, total_duration: float) -> List[Tuple[float, float]]:
        """Chunk başlangıç ve bitiş zamanlarını hesapla"""
        chunks = []
        current_start = 0.0
        
        while current_start < total_duration:
            current_end = min(current_start + self.chunk_duration, total_duration)
            chunks.append((current_start, current_end))
            
            # Overlap ile bir sonraki chunk'ın başlangıcını belirle
            current_start += self.chunk_duration - self.overlap_duration
            
        return chunks
    
    def extract_audio_chunk(self, audio_file: str, start_time: float, end_time: float, 
                          output_file: str) -> bool:
        """Belirli zaman aralığında audio chunk çıkar"""
        try:
            import soundfile as sf
            
            # Ana dosyayı oku
            data, samplerate = sf.read(audio_file)
            
            # Zaman indekslerini hesapla
            start_idx = int(start_time * samplerate)
            end_idx = int(end_time * samplerate)
            
            # Chunk'ı çıkar
            chunk_data = data[start_idx:end_idx]
            
            # Chunk'ı kaydet
            sf.write(output_file, chunk_data, samplerate)
            return True
            
        except Exception as e:
            print(f"Audio chunk extraction error: {e}")
            return False
    
    def process_single_chunk(self, chunk_file: str, chunk_index: int, 
                           start_time: float, end_time: float) -> Dict:
        """Tek bir chunk'ı STT + özetleme ile işle"""
        try:
            from .stt import transcribe_audio
            from .nlp import normalize_transcript
            from .industrial_summarizer import quick_summarize
            
            print(f"  🎬 Chunk {chunk_index}: {start_time:.1f}s - {end_time:.1f}s işleniyor...")
            
            # STT transkripsiyon
            transcript_start = time.time()
            transcript = transcribe_audio(chunk_file)
            transcript_time = time.time() - transcript_start
            
            # Normalizasyon
            normalized = normalize_transcript(transcript)
            
            # Özetleme (eğer yeterince uzunsa)
            summary = ""
            if len(normalized.split()) > 10:
                summary = quick_summarize(normalized, language="tr", max_sentences=2)
            else:
                summary = normalized
            
            # İstatistikler
            chunk_info = {
                'chunk_index': chunk_index,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'transcript': transcript,
                'normalized': normalized,
                'summary': summary,
                'word_count': len(normalized.split()),
                'processing_time': transcript_time,
                'rtf': transcript_time / (end_time - start_time) if (end_time - start_time) > 0 else 0
            }
            
            print(f"    ✅ {len(normalized.split())} kelime, RTF: {chunk_info['rtf']:.2f}")
            return chunk_info
            
        except Exception as e:
            print(f"    ❌ Chunk işleme hatası: {e}")
            return {
                'chunk_index': chunk_index,
                'start_time': start_time,
                'end_time': end_time,
                'error': str(e)
            }
    
    def process_long_audio(self, audio_file: str, output_dir: str = None) -> Dict:
        """
        Ana işleme fonksiyonu - uzun ses dosyasını tamamen işle
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(audio_file), "chunks")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎯 Uzun form audio işleme başlatılıyor: {audio_file}")
        
        # Audio süresini hesapla
        total_duration = self.estimate_audio_duration(audio_file)
        if total_duration == 0:
            return {'error': 'Audio duration estimation failed'}
        
        print(f"📊 Toplam süre: {total_duration:.1f} saniye ({total_duration/60:.1f} dakika)")
        
        # Chunk'ları hesapla
        chunks = self.calculate_chunks(total_duration)
        print(f"🔨 {len(chunks)} chunk oluşturulacak")
        
        # İşleme başlangıcı
        processing_start = time.time()
        processed_chunks = []
        
        try:
            for i, (start_time, end_time) in enumerate(chunks):
                chunk_filename = f"chunk_{i:03d}_{start_time:.1f}s-{end_time:.1f}s.wav"
                chunk_path = os.path.join(output_dir, chunk_filename)
                
                # Audio chunk'ını çıkar
                if self.extract_audio_chunk(audio_file, start_time, end_time, chunk_path):
                    # Chunk'ı işle
                    chunk_result = self.process_single_chunk(chunk_path, i, start_time, end_time)
                    processed_chunks.append(chunk_result)
                    
                    # Temporary chunk dosyasını sil (disk alanı için)
                    try:
                        os.remove(chunk_path)
                    except:
                        pass
                        
                else:
                    print(f"    ❌ Chunk {i} extraction failed")
                    
                # Progress update
                progress = (i + 1) / len(chunks) * 100
                print(f"📈 İlerleme: {progress:.1f}% ({i+1}/{len(chunks)})")
        
        except KeyboardInterrupt:
            print("🛑 İşleme kullanıcı tarafından durduruldu")
        
        processing_time = time.time() - processing_start
        
        # Sonuçları birleştir ve özet çıkar
        result = self.generate_final_report(processed_chunks, total_duration, processing_time)
        
        return result
    
    def generate_final_report(self, chunks: List[Dict], total_duration: float, 
                            processing_time: float) -> Dict:
        """Final rapor ve hierarchical özetleme"""
        print("\n🔄 Final rapor oluşturuluyor...")
        
        # Başarılı chunk'ları filtrele
        successful_chunks = [chunk for chunk in chunks if 'error' not in chunk]
        failed_chunks = [chunk for chunk in chunks if 'error' in chunk]
        
        if not successful_chunks:
            return {'error': 'No successful chunks processed'}
        
        # Tam transkripsiyon birleştir
        full_transcript = " ".join(chunk['normalized'] for chunk in successful_chunks)
        
        # Chunk özetlerini birleştir
        chunk_summaries = " ".join(chunk['summary'] for chunk in successful_chunks if chunk['summary'])
        
        # Hierarchical final summary
        try:
            from .industrial_summarizer import industrial_summarize
            final_summary = industrial_summarize(chunk_summaries, language="tr")
        except Exception as e:
            print(f"Final summarization error: {e}")
            final_summary = chunk_summaries[:500] + "..."
        
        # İstatistikler
        total_words = sum(chunk.get('word_count', 0) for chunk in successful_chunks)
        avg_rtf = sum(chunk.get('rtf', 0) for chunk in successful_chunks) / len(successful_chunks)
        
        result = {
            'audio_duration': total_duration,
            'processing_time': processing_time,
            'total_chunks': len(chunks),
            'successful_chunks': len(successful_chunks),
            'failed_chunks': len(failed_chunks),
            'total_words': total_words,
            'average_rtf': avg_rtf,
            'full_transcript': full_transcript,
            'chunk_summaries': chunk_summaries,
            'final_summary': final_summary,
            'chunk_details': successful_chunks,
            'processing_speed': total_duration / processing_time if processing_time > 0 else 0
        }
        
        # Raporu yazdır
        print(f"\n📋 İŞLEME RAPORU")
        print(f"⏱️  Audio süresi: {total_duration:.1f}s ({total_duration/60:.1f} dk)")
        print(f"🔄 İşleme süresi: {processing_time:.1f}s ({processing_time/60:.1f} dk)")
        print(f"⚡ İşleme hızı: {result['processing_speed']:.1f}x")
        print(f"📊 Chunk başarısı: {len(successful_chunks)}/{len(chunks)}")
        print(f"📝 Toplam kelime: {total_words}")
        print(f"📈 Ortalama RTF: {avg_rtf:.2f}")
        
        print(f"\n📋 FINAL ÖZET:")
        print(f"{final_summary}")
        
        return result

def process_long_form_audio(audio_file: str, chunk_duration: int = 60, 
                          overlap_duration: int = 5) -> Dict:
    """Uzun form audio işleme - dış interface"""
    processor = LongFormProcessor(chunk_duration=chunk_duration, 
                                overlap_duration=overlap_duration)
    return processor.process_long_audio(audio_file)

# Test fonksiyonu
if __name__ == "__main__":
    # Test için kısa bir audio dosyası kullan
    test_audio = "demo_medium.wav"  # Varsa
    if os.path.exists(test_audio):
        result = process_long_form_audio(test_audio, chunk_duration=30)
        print("Test tamamlandı!")
    else:
        print("Test audio dosyası bulunamadı")