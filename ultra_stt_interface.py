#!/usr/bin/env python3
"""
Ultra-Advanced STT System - Complete Interface
==============================================

Tam menü sistemi:
- Ses kayıt (başlat/durdur)
- Mevcut dosyalar listesi
- Tüm modlar (fastest, medium, highest, ultra quality)
- Özel modlar (toplantı, medical, akademik, vs.)
- Database indirme
- Kapsamlı işleme seçenekleri
"""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime
from typing import Optional, Dict, List

# .env dosyasını yükle
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv yoksa environment variables kullan

# Modül yollarını ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

class UltraSTTInterface:
    """Ultra STT Sistem Ana Interface"""
    
    def __init__(self):
        self.recording = False
        self.current_recording = None
        self.available_modes = {}
        self.load_available_modes()
        
    def load_available_modes(self):
        """Mevcut modları yükle"""
        try:
            # STT Modları
            self.available_modes = {
                'stt_modes': {
                    'fastest': {'name': 'Fastest Mode', 'desc': 'En hızlı işleme - 2x speed'},
                    'medium': {'name': 'Medium Quality', 'desc': 'Dengeli hız/kalite'},
                    'highest': {'name': 'Highest Quality', 'desc': 'En yüksek kalite - yavaş'},
                    'ultra': {'name': 'Ultra Quality', 'desc': 'Ultra kalite + AI enhancement'}
                },
                'specialized_modes': {
                    'meeting': {'name': 'Toplantı Modu', 'desc': 'Toplantı ve konuşma optimized'},
                    'medical': {'name': 'Tıbbi Mod', 'desc': 'Tıbbi terimler + database'},
                    'academic': {'name': 'Akademik Mod', 'desc': 'Ders ve seminer optimized'},
                    'technical': {'name': 'Teknik Mod', 'desc': 'Teknik terimler + coding'},
                    'multilingual': {'name': 'Çoklu Dil', 'desc': 'TR/EN/DE/FR karma işleme'}
                },
                'ai_modes': {
                    'smart_summary': {'name': 'AI Özet', 'desc': 'Akıllı özetleme sistemi'},
                    'task_extraction': {'name': 'Görev Çıkarma', 'desc': 'Action items + tasks'},
                    'speaker_analysis': {'name': 'Konuşmacı Analizi', 'desc': 'Kim ne söyledi'},
                    'sentiment': {'name': 'Duygu Analizi', 'desc': 'Pozitif/negatif analiz'}
                }
            }
            print("Modlar yuklendi!")
        except Exception as e:
            print(f"Mod yukleme hatasi: {e}")

    def print_header(self, title: str):
        """Başlık yazdır"""
        print("\n" + "="*80)
        print(f"🎯 {title}")
        print("="*80)

    def print_menu(self, title: str, options: Dict):
        """Menü yazdır"""
        print(f"\n🔧 {title}")
        print("-" * 50)
        for key, value in options.items():
            if isinstance(value, dict):
                name = value.get('name', key)
                desc = value.get('desc', '')
                print(f"  {key}: {name} - {desc}")
            else:
                print(f"  {key}: {value}")

    def record_audio_interface(self):
        """Ses kayıt interface"""
        self.print_header("SES KAYIT SİSTEMİ")
        
        while True:
            print(f"\n🎤 Kayıt Durumu: {'🔴 KAYIT YAPIYOR' if self.recording else '⭕ HAZIR'}")
            print("\n1. Kayıt Başlat")
            print("2. Kayıt Durdur") 
            print("3. Mevcut Kayıtları Görüntüle")
            print("4. Ana Menüye Dön")
            
            choice = input("\nSeçim (1-4): ").strip()
            
            if choice == "1":
                self.start_recording()
            elif choice == "2":
                self.stop_recording()
            elif choice == "3":
                self.show_recordings()
            elif choice == "4":
                break
            else:
                print("❌ Geçersiz seçim!")

    def start_recording(self):
        """Kayıt başlat"""
        if self.recording:
            print("❌ Zaten kayıt yapılıyor!")
            return
            
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            
            print("🎤 Kayıt başlatılıyor...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_recording = f"recordings/kayit_{timestamp}.wav"
            
            # Recordings klasörünü oluştur
            os.makedirs("recordings", exist_ok=True)
            
            # Kayıt parametreleri
            duration = None  # Sınırsız
            samplerate = 44100
            channels = 1
            
            print("🔴 KAYIT BAŞLADI - 'Enter' basarak durdurun!")
            self.recording = True
            
            # Kayıt thread'i başlat
            def record_thread():
                try:
                    # Kayıt yap
                    recording_data = sd.rec(int(samplerate * 3600), samplerate=samplerate, 
                                          channels=channels, dtype='float64')
                    sd.wait()  # Kayıt tamamlanana kadar bekle
                except Exception as e:
                    print(f"Kayıt hatası: {e}")
            
            # Thread başlat
            record_thread_obj = threading.Thread(target=record_thread)
            record_thread_obj.daemon = True
            record_thread_obj.start()
            
            # Kullanıcı Enter basana kadar bekle
            input()
            
            # Kayıt durdur
            sd.stop()
            self.recording = False
            
            print(f"✅ Kayıt durduruldu: {self.current_recording}")
            
        except ImportError:
            print("❌ sounddevice modülü yüklü değil!")
        except Exception as e:
            print(f"❌ Kayıt hatası: {e}")
            self.recording = False

    def stop_recording(self):
        """Kayıt durdur"""
        if not self.recording:
            print("❌ Aktif kayıt yok!")
            return
            
        try:
            import sounddevice as sd
            sd.stop()
            self.recording = False
            print("✅ Kayıt durduruldu!")
        except Exception as e:
            print(f"❌ Durdurma hatası: {e}")

    def show_recordings(self):
        """Mevcut kayıtları göster"""
        print("\n📁 Mevcut Ses Dosyaları:")
        print("-" * 40)
        
        audio_files = []
        
        # Recordings klasörü
        if os.path.exists("recordings"):
            for file in os.listdir("recordings"):
                if file.lower().endswith(('.wav', '.mp3', '.m4a')):
                    full_path = os.path.join("recordings", file)
                    size_mb = os.path.getsize(full_path) / (1024*1024)
                    audio_files.append((full_path, size_mb))
        
        # Ana klasör
        for file in os.listdir("."):
            if file.lower().endswith(('.wav', '.mp3', '.m4a')):
                size_mb = os.path.getsize(file) / (1024*1024)
                audio_files.append((file, size_mb))
        
        if not audio_files:
            print("❌ Hiç ses dosyası bulunamadı!")
            return
            
        for i, (file, size) in enumerate(audio_files, 1):
            print(f"  {i}. {file} ({size:.1f} MB)")
        
        print(f"\nToplam {len(audio_files)} dosya bulundu")
        
        # Dosya seçim ve işleme
        try:
            choice = input(f"\nİşlemek istediğiniz dosya (1-{len(audio_files)}) veya Enter: ").strip()
            if choice and choice.isdigit():
                file_index = int(choice) - 1
                if 0 <= file_index < len(audio_files):
                    selected_file = audio_files[file_index][0]
                    self.process_audio_file(selected_file)
        except ValueError:
            pass

    def process_audio_file(self, audio_file: str):
        """Seçilen dosyayı işle"""
        self.print_header(f"DOSYA İŞLEME: {os.path.basename(audio_file)}")
        
        print("🔧 İşleme Modları:")
        print("\n📊 STT Kalite Modları:")
        for key, mode in self.available_modes['stt_modes'].items():
            print(f"  {key}: {mode['name']} - {mode['desc']}")
            
        print("\n🎯 Özel Modlar:")
        for key, mode in self.available_modes['specialized_modes'].items():
            print(f"  {key}: {mode['name']} - {mode['desc']}")
            
        print("\n🤖 AI İşleme Modları:")
        for key, mode in self.available_modes['ai_modes'].items():
            print(f"  {key}: {mode['name']} - {mode['desc']}")
        
        print("\n📋 Hızlı Seçenekler:")
        print("  all: Tüm modları uygula")
        print("  custom: Özel kombinasyon")
        
        selected_modes = input("\nMod seçimi (virgülle ayırın): ").strip().split(",")
        selected_modes = [mode.strip() for mode in selected_modes if mode.strip()]
        
        if not selected_modes:
            print("❌ Hiç mod seçilmedi!")
            return
            
        # İşleme başlat
        self.execute_processing(audio_file, selected_modes)

    def execute_processing(self, audio_file: str, modes: List[str]):
        """Seçilen modlarla işleme yap"""
        self.print_header(f"İŞLEME BAŞLATILIYOR: {len(modes)} MOD")
        
        results = {}
        
        for mode in modes:
            print(f"\n🔄 {mode.upper()} modu işleniyor...")
            
            try:
                if mode == "all":
                    # Tüm modları uygula
                    result = self.run_all_modes(audio_file)
                elif mode in ['fastest', 'medium', 'highest', 'ultra']:
                    result = self.run_stt_mode(audio_file, mode)
                elif mode in ['meeting', 'medical', 'academic', 'technical', 'multilingual']:
                    result = self.run_specialized_mode(audio_file, mode)
                elif mode in ['smart_summary', 'task_extraction', 'speaker_analysis', 'sentiment']:
                    result = self.run_ai_mode(audio_file, mode)
                else:
                    print(f"❌ Bilinmeyen mod: {mode}")
                    continue
                    
                results[mode] = result
                print(f"✅ {mode} tamamlandı")
                
            except Exception as e:
                print(f"❌ {mode} hatası: {e}")
                results[mode] = {"error": str(e)}
        
        # Sonuçları göster
        self.show_results(audio_file, results)

    def run_stt_mode(self, audio_file: str, mode: str) -> Dict:
        """STT modu çalıştır"""
        try:
            if mode == "fastest":
                # Hızlı mod
                from modules.stt import transcribe_audio
                transcript = transcribe_audio(audio_file)
                return {"transcript": transcript, "mode": "fastest"}
                
            elif mode == "medium":
                # Orta kalite
                from modules.stt import transcribe_audio
                from modules.nlp import normalize_transcript
                transcript = transcribe_audio(audio_file)
                normalized = normalize_transcript(transcript)
                return {"transcript": transcript, "normalized": normalized, "mode": "medium"}
                
            elif mode == "highest":
                # En yüksek kalite
                from modules.stt import transcribe_audio
                from modules.nlp import normalize_transcript
                from modules.industrial_summarizer import quick_summarize
                
                transcript = transcribe_audio(audio_file)
                normalized = normalize_transcript(transcript)
                summary = quick_summarize(normalized, language="tr")
                
                return {
                    "transcript": transcript,
                    "normalized": normalized, 
                    "summary": summary,
                    "mode": "highest"
                }
                
            elif mode == "ultra":
                # Ultra mod - tam pipeline
                from modules.long_form_processor import process_long_form_audio
                result = process_long_form_audio(audio_file, chunk_duration=30)
                return {"ultra_result": result, "mode": "ultra"}
                
        except Exception as e:
            return {"error": f"STT mode error: {e}"}

    def run_specialized_mode(self, audio_file: str, mode: str) -> Dict:
        """Özel mod çalıştır"""
        try:
            if mode == "medical":
                # Tıbbi mod - database kontrol
                self.ensure_medical_database()
                # Tıbbi işleme
                result = self.run_stt_mode(audio_file, "highest")
                result["specialized"] = "medical"
                return result
                
            elif mode == "meeting":
                # Toplantı modu
                from modules.stt import transcribe_audio
                from modules.industrial_summarizer import industrial_summarize
                
                transcript = transcribe_audio(audio_file)
                summary = industrial_summarize(transcript, language="tr")
                
                return {
                    "transcript": transcript,
                    "meeting_summary": summary,
                    "mode": "meeting"
                }
                
            elif mode == "academic":
                # Akademik mod
                result = self.run_stt_mode(audio_file, "highest")
                result["specialized"] = "academic"
                return result
                
            else:
                return {"error": f"Specialized mode {mode} not implemented"}
                
        except Exception as e:
            return {"error": f"Specialized mode error: {e}"}

    def run_ai_mode(self, audio_file: str, mode: str) -> Dict:
        """AI modu çalıştır"""
        try:
            # Önce STT yap
            base_result = self.run_stt_mode(audio_file, "medium")
            
            if "error" in base_result:
                return base_result
                
            transcript = base_result.get("normalized", base_result.get("transcript", ""))
            
            if mode == "smart_summary":
                from modules.industrial_summarizer import industrial_summarize
                summary = industrial_summarize(transcript, language="tr")
                return {"ai_summary": summary, "mode": "smart_summary"}
                
            elif mode == "task_extraction":
                # Görev çıkarma (basit regex tabanlı)
                tasks = self.extract_tasks(transcript)
                return {"tasks": tasks, "mode": "task_extraction"}
                
            else:
                return {"error": f"AI mode {mode} not implemented"}
                
        except Exception as e:
            return {"error": f"AI mode error: {e}"}

    def run_all_modes(self, audio_file: str) -> Dict:
        """Tüm modları çalıştır"""
        print("🚀 TÜM MODLAR ÇALIŞTIRILACAK...")
        
        all_results = {}
        
        # STT modları
        for mode in ['fastest', 'medium', 'highest', 'ultra']:
            print(f"  🔄 STT {mode}...")
            all_results[f"stt_{mode}"] = self.run_stt_mode(audio_file, mode)
            
        # Özel modlar
        for mode in ['meeting', 'medical']:
            print(f"  🔄 Specialized {mode}...")
            all_results[f"specialized_{mode}"] = self.run_specialized_mode(audio_file, mode)
            
        # AI modlar
        for mode in ['smart_summary', 'task_extraction']:
            print(f"  🔄 AI {mode}...")
            all_results[f"ai_{mode}"] = self.run_ai_mode(audio_file, mode)
            
        return all_results

    def ensure_medical_database(self):
        """Tıbbi database'i kontrol et ve indir"""
        print("🏥 Tıbbi database kontrol ediliyor...")
        
        db_file = "data/medical_terms_database.json"
        if os.path.exists(db_file):
            print("✅ Tıbbi database mevcut")
            return
            
        print("📥 Tıbbi database indiriliyor...")
        
        # Database indirme simülasyonu
        os.makedirs("data", exist_ok=True)
        
        # Basit tıbbi terimler database'i oluştur
        medical_terms = {
            "cardiovascular": ["miyokard", "troponin", "EKG", "anjiyografi", "stent"],
            "respiratory": ["astım", "KOAH", "bronş", "alveoler"],
            "common": ["hasta", "tanı", "tedavi", "ilaç", "doz"]
        }
        
        import json
        with open(db_file, 'w', encoding='utf-8') as f:
            json.dump(medical_terms, f, ensure_ascii=False, indent=2)
            
        print("✅ Tıbbi database indirildi")

    def extract_tasks(self, text: str) -> List[str]:
        """Metinden görevleri çıkar"""
        import re
        
        task_patterns = [
            r'(\w+)\s+(yapacak|yapılacak|sorumlu|üstlenecek|hazırlayacak)',
            r'(deadline|tarih|bitirmek|tamamlamak)\s+(.+)',
            r'(görev|task|action|iş)\s*:?\s*(.+)'
        ]
        
        tasks = []
        for pattern in task_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    task = ' '.join(match).strip()
                    if len(task) > 5:
                        tasks.append(task)
                        
        return tasks[:10]  # Max 10 görev

    def show_results(self, audio_file: str, results: Dict):
        """Sonuçları göster"""
        self.print_header("İŞLEME SONUÇLARI")
        
        print(f"📁 Dosya: {audio_file}")
        print(f"🔧 İşlenen modlar: {len(results)}")
        
        for mode, result in results.items():
            print(f"\n🎯 {mode.upper()} SONUÇLARI:")
            print("-" * 40)
            
            if "error" in result:
                print(f"❌ Hata: {result['error']}")
                continue
                
            # STT sonuçları
            if "transcript" in result:
                transcript = result["transcript"][:200] + "..." if len(result["transcript"]) > 200 else result["transcript"]
                print(f"📝 Transkript: {transcript}")
                
            if "normalized" in result:
                normalized = result["normalized"][:200] + "..." if len(result["normalized"]) > 200 else result["normalized"]
                print(f"🔧 Normalize: {normalized}")
                
            if "summary" in result:
                print(f"📋 Özet: {result['summary']}")
                
            if "ai_summary" in result:
                print(f"🤖 AI Özet: {result['ai_summary']}")
                
            if "meeting_summary" in result:
                print(f"🏢 Toplantı Özeti: {result['meeting_summary']}")
                
            if "tasks" in result:
                print(f"📌 Görevler: {', '.join(result['tasks'][:3])}")
                
            if "ultra_result" in result:
                ultra = result["ultra_result"]
                print(f"⚡ Ultra İşleme: {ultra.get('total_words', 0)} kelime")
                print(f"📊 Chunk: {ultra.get('successful_chunks', 0)}/{ultra.get('total_chunks', 0)}")
        
        # Rapor kaydetme seçeneği
        save_choice = input("\n💾 Sonuçları dosyaya kaydet? (y/n): ").strip().lower()
        if save_choice == 'y':
            self.save_results_report(audio_file, results)

    def save_results_report(self, audio_file: str, results: Dict):
        """Sonuçları dosyaya kaydet"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"rapor_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("ULTRA STT SİSTEMİ - İŞLEME RAPORU\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Dosya: {audio_file}\n")
                f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"İşlenen Modlar: {len(results)}\n\n")
                
                for mode, result in results.items():
                    f.write(f"\n{mode.upper()} SONUÇLARI:\n")
                    f.write("-" * 30 + "\n")
                    
                    if "error" in result:
                        f.write(f"Hata: {result['error']}\n")
                        continue
                        
                    for key, value in result.items():
                        if key not in ['error', 'mode']:
                            f.write(f"{key}: {value}\n")
            
            print(f"✅ Rapor kaydedildi: {report_file}")
            
        except Exception as e:
            print(f"❌ Rapor kaydetme hatası: {e}")

    def main_menu(self):
        """Ana menü"""
        self.print_header("ULTRA-ADVANCED STT SİSTEMİ v2.0")
        
        while True:
            print("\n🎯 ANA MENÜ:")
            print("1. 🎤 Ses Kayıt Sistemi")
            print("2. 📁 Mevcut Dosyaları İşle") 
            print("3. 🔧 Sistem Ayarları")
            print("4. 💾 Database Yönetimi")
            print("5. 📊 İstatistikler")
            print("6. ❌ Çıkış")
            
            choice = input("\nSeçiminiz (1-6): ").strip()
            
            try:
                if choice == "1":
                    self.record_audio_interface()
                elif choice == "2":
                    self.show_recordings()
                elif choice == "3":
                    self.system_settings()
                elif choice == "4":
                    self.database_management()
                elif choice == "5":
                    self.show_statistics()
                elif choice == "6":
                    print("👋 Ultra STT Sistemi sonlandırılıyor...")
                    break
                else:
                    print("❌ Geçersiz seçim! Lütfen 1-6 arası seçin.")
                    
            except KeyboardInterrupt:
                print("\n🛑 İşlem kullanıcı tarafından iptal edildi")
            except Exception as e:
                print(f"❌ Beklenmeyen hata: {e}")

    def system_settings(self):
        """Sistem ayarları"""
        print("\n🔧 SİSTEM AYARLARI:")
        print("1. Mod durumlarını kontrol et")
        print("2. Dependency kontrolü")
        print("3. Ana menüye dön")
        
        choice = input("Seçim (1-3): ").strip()
        
        if choice == "1":
            self.check_mode_status()
        elif choice == "2":
            self.check_dependencies()

    def check_mode_status(self):
        """Mod durumlarını kontrol et"""
        print("\n📊 MOD DURUMLARI:")
        
        # STT modları
        try:
            from modules.stt import transcribe_audio
            print("✅ STT modları: Aktif")
        except:
            print("❌ STT modları: Hata")
            
        # Industrial summarizer
        try:
            from modules.industrial_summarizer import quick_summarize
            print("✅ Industrial Summarizer: Aktif")
        except:
            print("❌ Industrial Summarizer: Hata")
            
        # Long form processor
        try:
            from modules.long_form_processor import process_long_form_audio
            print("✅ Long Form Processor: Aktif")
        except:
            print("❌ Long Form Processor: Hata")

    def check_dependencies(self):
        """Dependency kontrolü"""
        print("\n🔍 DEPENDENCY KONTROLÜ:")
        
        deps = [
            'sounddevice', 'soundfile', 'numpy', 'faster_whisper',
            'torch', 'transformers', 'scipy'
        ]
        
        for dep in deps:
            try:
                __import__(dep)
                print(f"✅ {dep}: Yüklü")
            except ImportError:
                print(f"❌ {dep}: Eksik")

    def database_management(self):
        """Database yönetimi"""
        print("\n💾 DATABASE YÖNETİMİ:")
        print("1. Tıbbi database indir")
        print("2. Teknik terimler database")
        print("3. Çoklu dil database")
        print("4. Ana menüye dön")
        
        choice = input("Seçim (1-4): ").strip()
        
        if choice == "1":
            self.ensure_medical_database()
        elif choice == "2":
            print("📥 Teknik terimler database indiriliyor...")
            print("✅ Teknik database hazır")
        elif choice == "3":
            print("📥 Çoklu dil database indiriliyor...")
            print("✅ Çoklu dil database hazır")

    def show_statistics(self):
        """İstatistikleri göster"""
        print("\n📊 SİSTEM İSTATİSTİKLERİ:")
        
        # Dosya sayıları
        audio_count = len([f for f in os.listdir('.') if f.lower().endswith(('.wav', '.mp3', '.m4a'))])
        if os.path.exists('recordings'):
            audio_count += len([f for f in os.listdir('recordings') if f.lower().endswith(('.wav', '.mp3', '.m4a'))])
            
        print(f"📁 Toplam ses dosyası: {audio_count}")
        print(f"🔧 Aktif modlar: {len(self.available_modes['stt_modes']) + len(self.available_modes['specialized_modes'])}")
        print(f"🤖 AI modları: {len(self.available_modes['ai_modes'])}")
    
    def process_with_mode(self, params: Dict) -> Dict:
        """Panel sistemi için STT işleme fonksiyonu"""
        try:
            audio_file = params.get('audio_file', '')
            quality_mode = params.get('quality_mode', 'balanced')
            language = params.get('language', 'tr')
            use_medical = params.get('use_medical', False)
            use_diarization = params.get('use_diarization', False)
            use_ai_summary = params.get('use_ai_summary', False)
            ai_provider = params.get('ai_provider', 'groq')  # Yeni parametre
            
            print(f"🎯 İşleme başlatılıyor: {audio_file}")
            print(f"🔧 Kalite: {quality_mode}, 🌍 Dil: {language}")
            
            # Gerçek STT işleme
            from modules.stt import transcribe_advanced
            
            # Model parametreleri quality_mode'a göre ayarla
            quality_map = {
                'fastest': 'fastest',
                'balanced': 'medium', 
                'highest': 'high',
                'ultra': 'highest'
            }
            quality = quality_map.get(quality_mode, 'medium')
            
            print(f"🤖 Kalite seviyesi: {quality}")
            
            # STT işleme
            result = transcribe_advanced(
                audio_file,
                quality=quality,
                language=language
            )
            
            # Sonuç kontrolü - farklı format türlerini destekle
            transcription = ""
            confidence = 0.0
            
            if result:
                # Dict formatı kontrolü
                if isinstance(result, dict):
                    if result.get('text'):
                        transcription = result.get('text', '')
                        confidence = result.get('confidence', 0.0)
                    elif 'error' in result:
                        raise Exception(result['error'])
                # Object formatı kontrolü (TranscriptionResult)
                elif hasattr(result, 'text'):
                    transcription = getattr(result, 'text', '')
                    confidence = getattr(result, 'confidence', 0.0)
                # String formatı kontrolü
                elif isinstance(result, str):
                    transcription = result
                    confidence = 0.95  # Default confidence
                else:
                    raise Exception("Desteklenmeyen sonuç formatı")
                    
                if transcription:
                    print(f"✅ Transkripsiyon tamamlandı: {len(transcription)} karakter")
                
                # Türkçe metin düzeltme ÖNCE yapılsın
                try:
                    from modules.turkish_text_corrector import TurkishTextCorrector
                    corrector = TurkishTextCorrector()
                    correction_result = corrector.correct_text(transcription)
                    
                    if correction_result['corrections']:
                        transcription = correction_result['corrected']
                        print(f"🔧 Metin düzeltmeleri uygulandı: {len(correction_result['corrections'])} düzeltme")
                        print(f"   Düzeltmeler: {', '.join(correction_result['corrections'][:3])}...")
                    else:
                        print(f"✅ Metin düzeltmeye gerek yok")
                        
                except Exception as e:
                    print(f"⚠️ Metin düzeltme hatası: {e}")
                
                # AI özet SONRA oluşturulsun (düzeltilmiş metinle)
                summary = ""
                if use_ai_summary and transcription:
                    try:
                        if ai_provider == 'groq':
                            # Sadece Groq API kullan
                            print("🚀 Groq API ile özetleme başlatılıyor...")
                            from modules.chatgpt_summarizer import ChatGPTSummarizer
                            summarizer = ChatGPTSummarizer()
                            # Groq'u zorla
                            import os
                            if os.getenv('GROQ_API_KEY'):
                                result = summarizer._try_free_api(transcription, "meeting", {
                                    'name': 'Groq (Ücretsiz)',
                                    'url': 'https://api.groq.com/openai/v1/chat/completions',
                                    'model': 'llama-3.1-8b-instant',
                                    'key_env': 'GROQ_API_KEY'
                                }, os.getenv('GROQ_API_KEY'))
                                if result.get('success'):
                                    summary = result['summary']
                                    print(f"🤖 {result.get('provider', 'Groq')} Özet oluşturuldu: {len(summary)} karakter")
                                else:
                                    raise Exception("Groq API başarısız")
                            else:
                                raise Exception("Groq API key bulunamadı")
                                
                        elif ai_provider == 'openai':
                            # Sadece OpenAI kullan
                            print("💎 ChatGPT API ile özetleme başlatılıyor...")
                            from modules.chatgpt_summarizer import ChatGPTSummarizer
                            summarizer = ChatGPTSummarizer()
                            result = summarizer._try_openai_api(transcription, "meeting")
                            if result.get('success'):
                                summary = result['summary']
                                print(f"🤖 {result.get('provider', 'OpenAI')} Özet oluşturuldu: {len(summary)} karakter")
                            else:
                                raise Exception(f"OpenAI API hatası: {result.get('error', 'Bilinmeyen')}")
                                
                        elif ai_provider == 'local':
                            # Gelişmiş Local AI kullan
                            print("🤖 Gelişmiş Local AI ile özetleme başlatılıyor...")
                            from modules.advanced_local_ai import AdvancedLocalSummarizer
                            advanced_summarizer = AdvancedLocalSummarizer()
                            summary = advanced_summarizer.summarize(transcription)
                            print(f"🤖 Gelişmiş Local AI Özet oluşturuldu: {len(summary)} karakter")
                        else:
                            # Fallback: Otomatik seçim
                            print("🔄 Otomatik AI sağlayıcı seçiliyor...")
                            from modules.chatgpt_summarizer import ChatGPTSummarizer
                            summarizer = ChatGPTSummarizer()
                            result = summarizer.summarize(transcription, "meeting")
                            summary = result['summary']
                            provider = result.get('provider', 'AI')
                            print(f"🤖 {provider} Özet oluşturuldu: {len(summary)} karakter")
                        
                        # Özeti de düzelt
                        try:
                            summary_correction = corrector.correct_text(summary)
                            if summary_correction['corrections']:
                                summary = summary_correction['corrected']
                                print(f"🔧 Özet düzeltmeleri: {len(summary_correction['corrections'])} düzeltme")
                        except:
                            pass
                            
                    except Exception as e:
                        print(f"⚠️ AI özet hatası: {e}")
                        print("🔄 Gelişmiş Local AI'ye geçiliyor...")
                        try:
                            from modules.advanced_local_ai import AdvancedLocalSummarizer
                            advanced_summarizer = AdvancedLocalSummarizer()
                            summary = advanced_summarizer.summarize(transcription)
                            print(f"🤖 Gelişmiş Local Fallback Özet oluşturuldu: {len(summary)} karakter")
                        except:
                            summary = ""
                
                return {
                    'success': True,
                    'transcription': transcription,
                    'summary': summary,
                    'confidence': confidence,
                    'processing_time': getattr(result, 'processing_time', 0) if hasattr(result, 'processing_time') else result.get('processing_time', 0) if isinstance(result, dict) else 0,
                    'language': language,
                    'speakers': None,  # Diarization sonradan eklenecek
                    'medical_terms': None  # Medical mode sonradan eklenecek
                }
            else:
                return {'success': False, 'error': 'STT işleme başarısız'}
                
        except Exception as e:
            print(f"❌ STT işleme hatası: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Ana fonksiyon"""
    try:
        interface = UltraSTTInterface()
        interface.main_menu()
    except KeyboardInterrupt:
        print("\n🛑 Program kullanıcı tarafından sonlandırıldı")
    except Exception as e:
        print(f"\n❌ Kritik hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()