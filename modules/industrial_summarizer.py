# modules/industrial_summarizer.py
"""
Industrial-Grade Summarization System
====================================

Dependency-free, robust summarization system for 2-3 hour audio transcripts.
- Rule-based extractive summarization
- Hierarchical processing for long content  
- Keyword extraction and sentence scoring
- No external dependencies (no transformers/torch conflicts)
- Optimized for Turkish and English content
"""

import re
import math
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict

class IndustrialSummarizer:
    """
    Endüstriyel özet sistemi - dependency-free, hızlı ve güvenilir
    """
    
    def __init__(self, language: str = "tr"):
        self.language = language
        self.sentence_min_length = 10
        self.sentence_max_length = 200
        
        # Türkçe stop words
        self.tr_stopwords = {
            'bir', 'bu', 'da', 'de', 'den', 'dır', 'dir', 'du', 'dü', 'için', 
            'ile', 'ise', 'ki', 'mi', 'mu', 'mü', 've', 'veya', 'ya', 'ancak',
            'ama', 'fakat', 'lakin', 'şu', 'o', 'ben', 'sen', 'biz', 'siz',
            'onlar', 'şey', 'gibi', 'kadar', 'daha', 'en', 'çok', 'az', 'hiç',
            'var', 'yok', 'olan', 'oldu', 'olacak', 'etti', 'etmek', 'yapmak',
            'yapılan', 'yapılacak', 'olan', 'olmuş', 'olması', 'edildi', 'edilen'
        }
        
        # English stop words
        self.en_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if'
        }
        
        # Importance indicators (Turkish)
        self.importance_keywords = {
            'önemli', 'kritik', 'acil', 'gerekli', 'zorunlu', 'mutlaka', 'kesinlikle',
            'sonuç', 'karar', 'kararlaştırdık', 'belirlendi', 'planlandı', 'hedef',
            'amaç', 'görev', 'sorumluluk', 'deadline', 'tarih', 'bütçe', 'maliyet',
            'problem', 'sorun', 'çözüm', 'öneri', 'görüş', 'fikir', 'toplantı',
            'meeting', 'action', 'task', 'responsible', 'deadline', 'budget', 'cost',
            'issue', 'solution', 'proposal', 'decision', 'important', 'critical'
        }
        
        # Action words
        self.action_words = {
            'yapacak', 'yapılacak', 'gerçekleştirilecek', 'uygulanacak', 'başlanacak',
            'bitirilecek', 'tamamlanacak', 'devam edilecek', 'kontrol edilecek',
            'sorumlu', 'üstleniyor', 'yönetecek', 'koordine', 'takip', 'hazırlayacak',
            'will do', 'responsible', 'manage', 'coordinate', 'follow', 'prepare',
            'implement', 'execute', 'complete', 'finish', 'start', 'continue'
        }

    def clean_text(self, text: str) -> str:
        """Metni temizle ve normalize et"""
        if not text:
            return ""
            
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Tekrar eden noktalama işaretlerini düzelt
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)
        
        return text

    def split_sentences(self, text: str) -> List[str]:
        """Metni cümlelere böl"""
        text = self.clean_text(text)
        
        # Cümle ayırıcıları
        sentence_endings = r'[.!?]+'
        sentences = re.split(sentence_endings, text)
        
        # Temizle ve filtrele
        valid_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= self.sentence_min_length and len(sent) <= self.sentence_max_length:
                # Çok kısa veya çok uzun cümleleri atla
                word_count = len(sent.split())
                if 3 <= word_count <= 50:
                    valid_sentences.append(sent)
        
        return valid_sentences

    def extract_keywords(self, text: str, top_k: int = 20) -> List[Tuple[str, int]]:
        """Anahtar kelimeleri çıkar"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Stop words'leri filtrele
        stopwords = self.tr_stopwords if self.language == 'tr' else self.en_stopwords
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Frekans hesapla
        word_freq = Counter(filtered_words)
        
        return word_freq.most_common(top_k)

    def score_sentence(self, sentence: str, keywords: List[str], 
                      importance_boost: float = 2.0) -> float:
        """Cümleyi puanla"""
        if not sentence:
            return 0.0
            
        words = set(re.findall(r'\b\w+\b', sentence.lower()))
        score = 0.0
        
        # Anahtar kelime puanı
        keyword_score = sum(1 for word in words if word in keywords)
        score += keyword_score
        
        # Önem göstergesi puanı
        importance_score = sum(importance_boost for word in words 
                             if word in self.importance_keywords)
        score += importance_score
        
        # Aksiyon puanı
        action_score = sum(1.5 for word in words if word in self.action_words)
        score += action_score
        
        # Sayı puanı (tarih, rakam, yüzde vb.)
        number_score = len(re.findall(r'\b\d+\b', sentence)) * 0.5
        score += number_score
        
        # Cümle uzunluğu normalizasyonu
        word_count = len(sentence.split())
        if word_count > 0:
            score = score / math.sqrt(word_count)  # Normalize by length
            
        return score

    def extract_top_sentences(self, text: str, target_ratio: float = 0.3) -> List[str]:
        """En önemli cümleleri çıkar"""
        sentences = self.split_sentences(text)
        if not sentences:
            return []
            
        # Anahtar kelimeleri bul
        keywords = [word for word, freq in self.extract_keywords(text, top_k=15)]
        
        # Cümleleri puanla
        sentence_scores = []
        for sent in sentences:
            score = self.score_sentence(sent, keywords)
            sentence_scores.append((sent, score))
        
        # En yüksek puanlı cümleleri seç
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        target_count = max(1, int(len(sentences) * target_ratio))
        target_count = min(target_count, 10)  # Max 10 cümle
        
        top_sentences = [sent for sent, score in sentence_scores[:target_count]]
        
        return top_sentences

    def summarize(self, text: str, max_sentences: int = 5) -> str:
        """Ana özetleme fonksiyonu - geliştirilmiş versiyon"""
        if not text or not text.strip():
            return ""
            
        text = self.clean_text(text)
        word_count = len(text.split())
        
        # Çok kısa metinler için özet gerekmiyor
        if word_count < 30:
            return text
            
        # Özel toplantı/konuşma formatı tespiti
        if self._is_meeting_speech(text):
            return self._create_structured_summary(text)
            
        # Normal özetleme
        ratio = 0.6 if word_count < 100 else 0.4 if word_count < 500 else 0.25
        top_sentences = self.extract_top_sentences(text, target_ratio=ratio)
        
        # Max sentence limit uygula
        if len(top_sentences) > max_sentences:
            top_sentences = top_sentences[:max_sentences]
            
        # Cümleleri birleştir ve düzelt
        if top_sentences:
            clean_sentences = []
            for sentence in top_sentences:
                clean_sentence = sentence.strip()
                if clean_sentence and not clean_sentence.endswith(('.', '!', '?')):
                    clean_sentence += '.'
                clean_sentences.append(clean_sentence)
            
            summary = ' '.join(clean_sentences)
        else:
            summary = ""
        
        return summary.strip()
    
    def _is_meeting_speech(self, text: str) -> bool:
        """Toplantı/konuşma metni olup olmadığını tespit et"""
        meeting_indicators = [
            "hoş geldiniz", "toplantı", "gündem", "başlık", "hedef", 
            "ekip", "çalışıyoruz", "teşekkür", "öncelik", "görüş", "paylaş"
        ]
        text_lower = text.lower()
        found_indicators = sum(1 for indicator in meeting_indicators if indicator in text_lower)
        return found_indicators >= 4
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Metni cümlelere ayır"""
        import re
        # Basit cümle ayırma (nokta, ünlem, soru işareti)
        sentences = re.split(r'[.!?]+', text)
        # Boş cümleleri temizle ve whitespace kaldır
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_structured_summary(self, text: str) -> str:
        """Toplantı metinleri için detaylı ve profesyonel özet"""
        # Çok daha detaylı tema analizi
        themes = {
            'toplantı_konusu': ['toplantı', 'toplanma', 'burada', 'nedeni', 'amaç', 'hedef'],
            'teşekkür': ['teşekkür', 'özveri', 'katkı', 'göstermiş', 'birrer'],
            'başarı_tanımı': ['başarı', 'yalnızca', 'rakam', 'çalış', 'değer', 'ölçül'],
            'gündem': ['gündem', 'başlık', 'birinci', 'ikinci', 'üç', 'mevcut', 'proje', 'öncelik'],
            'katılım': ['görüş', 'paylaş', 'çekin', 'fikir', 'alışveriş', 'farklı', 'bakış'],
            'ortak_hedef': ['ortak', 'hedef', 'bireysel', 'katkı', 'toplam', 'hazır']
        }
        
        sentences = self._split_into_sentences(text)
        categorized = {theme: [] for theme in themes}
        all_sentences = []  # Tüm cümleleri saklayalım
        
        # Cümleleri kategorilere ayır ve skorla
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_clean = sentence.strip()
            if len(sentence_clean) < 15:  # Çok kısa cümleleri atla
                continue
                
            all_sentences.append(sentence_clean)
            best_theme = None
            max_matches = 0
            
            for theme, keywords in themes.items():
                matches = sum(1 for keyword in keywords if keyword in sentence_lower)
                if matches > max_matches:
                    max_matches = matches
                    best_theme = theme
            
            if best_theme and max_matches > 0:
                categorized[best_theme].append(sentence_clean)
        
        # ChatGPT benzeri profesyonel özet oluştur
        summary_parts = []
        
        # 1. Toplantı Konusu
        if categorized['toplantı_konusu']:
            best_sentence = self._get_best_sentence(categorized['toplantı_konusu'], ['hedef', 'netleştirmek'])
            summary_parts.append(f"**Toplantı Konusu:**\n{best_sentence}")
        
        # 2. Teşekkür ve Takdir
        if categorized['teşekkür']:
            best_sentence = self._get_best_sentence(categorized['teşekkür'], ['teşekkür', 'özveri'])
            summary_parts.append(f"**Teşekkür ve Takdir:**\n{best_sentence}")
        
        # 3. Başarı Tanımı
        if categorized['başarı_tanımı']:
            best_sentence = self._get_best_sentence(categorized['başarı_tanımı'], ['başarı', 'rakam'])
            summary_parts.append(f"**Başarı Yaklaşımı:**\n{best_sentence}")
        
        # 4. Gündem Maddeleri
        if categorized['gündem']:
            gündem_sentences = categorized['gündem'][:2]  # En fazla 2 cümle
            gündem_text = ' '.join(gündem_sentences)
            if 'üç' in gündem_text.lower() or '3' in gündem_text:
                summary_parts.append(f"**Toplantı Gündemi:**\n{gündem_sentences[0]}")
        
        # 5. Katılım ve Açıklık
        if categorized['katılım']:
            best_sentence = self._get_best_sentence(categorized['katılım'], ['görüş', 'paylaş'])
            summary_parts.append(f"**Katılım ve Açıklık:**\n{best_sentence}")
        
        # 6. Ortak Amaç
        if categorized['ortak_hedef']:
            best_sentence = self._get_best_sentence(categorized['ortak_hedef'], ['ortak', 'hedef'])
            summary_parts.append(f"**Ortak Amaç:**\n{best_sentence}")
        
        # Eğer yeterince kategori bulunamadıysa, önemli cümleleri ekle
        if len(summary_parts) < 3:
            # En önemli 4-5 cümleyi bul
            important_sentences = self._extract_most_important_sentences(all_sentences, 5)
            for i, sentence in enumerate(important_sentences, 1):
                if len(summary_parts) < 6:  # Max 6 kategori
                    summary_parts.append(f"**Nokta {i}:** {sentence}")
        
        if summary_parts:
            # Düzgün formatlanmış özet döndür
            formatted_summary = '\n\n'.join(summary_parts)
            
            # Türkçe düzeltmeler uygula
            formatted_summary = self._apply_turkish_corrections(formatted_summary)
            
            return formatted_summary
        else:
            # Fallback: basit özet
            simple_summary = self._create_simple_summary(all_sentences)
            return self._apply_turkish_corrections(simple_summary)
    
    def _apply_turkish_corrections(self, text: str) -> str:
        """Özete Türkçe düzeltmeler uygula"""
        # Temel Türkçe düzeltmeler
        corrections = {
            'birrer birrer': 'birer birer',
            'birrlikte': 'birlikte', 
            'birrbirrimize': 'birbirimize',
            'birreysel': 'bireysel',
            'birlgi': 'bilgi',
            'getirebirlmek': 'getirebilmek',
            'Bugün ki': 'Bugünkü',
            'bugün ki': 'bugünkü',
            'özgüleri': 'özveri',
            'durumlu': 'durum'
        }
        
        corrected_text = text
        for wrong, correct in corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)
            
        return corrected_text
    
    def _get_best_sentence(self, sentences: List[str], priority_keywords: List[str]) -> str:
        """Kategori içinden en iyi cümleyi seç"""
        if not sentences:
            return ""
        
        # Öncelikli kelimeleri içeren cümleyi bul
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in priority_keywords):
                return sentence
        
        # Bulamazsa en uzun cümleyi döndür (genellikle daha bilgilendirici)
        return max(sentences, key=len)
    
    def _extract_most_important_sentences(self, sentences: List[str], count: int) -> List[str]:
        """En önemli cümleleri skorlayarak çıkar"""
        if not sentences:
            return []
        
        scored_sentences = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Önem kelimeleri skorlaması
            for keyword in self.importance_keywords:
                if keyword in sentence_lower:
                    score += 2
            
            # Aksiyon kelimeleri skorlaması
            for keyword in self.action_words:
                if keyword in sentence_lower:
                    score += 1.5
                    
            # Uzunluk skorlaması (çok kısa veya çok uzun cümleler düşük skor)
            word_count = len(sentence.split())
            if 10 <= word_count <= 30:
                score += 1
            elif word_count > 30:
                score -= 0.5
                
            scored_sentences.append((sentence, score))
        
        # En yüksek skorlu cümleleri seç
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in scored_sentences[:count]]
    
    def _create_simple_summary(self, sentences: List[str]) -> str:
        """Basit özet oluştur (fallback)"""
        if not sentences:
            return ""
        
        important = self._extract_most_important_sentences(sentences, 4)
        if important:
            return '. '.join(important) + '.'
        else:
            # Son çare: ilk birkaç cümle
            return '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else '. '.join(sentences) + '.'

    def hierarchical_summarize(self, text: str, chunk_size: int = 1000) -> str:
        """
        Çok uzun metinler için hiyerarşik özetleme
        1. Metni parçalara böl
        2. Her parçayı özetle
        3. Parça özetlerini birleştir ve tekrar özetle
        """
        if not text or not text.strip():
            return ""
            
        word_count = len(text.split())
        
        # Kısa metinler için normal özetleme
        if word_count <= chunk_size:
            return self.summarize(text)
            
        # Metni parçalara böl
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        print(f"Hiyerarşik özetleme: {len(chunks)} parça işleniyor...")
        
        # Her parçayı özetle
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  Parça {i+1}/{len(chunks)} özetleniyor...")
            chunk_summary = self.summarize(chunk, max_sentences=3)
            if chunk_summary:
                chunk_summaries.append(chunk_summary)
                
        # Parça özetlerini birleştir
        combined_summary = ' '.join(chunk_summaries)
        
        # Eğer birleştirilmiş özet hala çok uzunsa, tekrar özetle
        if len(combined_summary.split()) > chunk_size:
            print("Final özetleme yapılıyor...")
            final_summary = self.summarize(combined_summary, max_sentences=8)
        else:
            final_summary = combined_summary
            
        return final_summary

def quick_summarize(text: str, language: str = "tr", max_sentences: int = 5) -> str:
    """Hızlı özet fonksiyonu - dış dünyaya export edilen interface"""
    summarizer = IndustrialSummarizer(language=language)
    return summarizer.summarize(text, max_sentences=max_sentences)

def industrial_summarize(text: str, language: str = "tr") -> str:
    """Endüstriyel özet fonksiyonu - uzun metinler için"""
    summarizer = IndustrialSummarizer(language=language)
    return summarizer.hierarchical_summarize(text, chunk_size=800)

# Test fonksiyonu
if __name__ == "__main__":
    test_text = """
    Bugünkü toplantımızda çok önemli kararlar aldık. Backend geliştirme işini Ali üstlendi, 
    frontend kısmını da Ayşe yapacak. Toplam bütçemiz 50 bin lira olarak belirlendi. 
    Projeyi 3 hafta içinde bitirmek zorundayız. Veritabanı tasarımını pazartesi günü 
    tamamlayacağız. Test işlemlerini Mehmet koordine edecek. Her salı saat 14:00'te 
    haftalık rapor toplantısı yapacağız. Müşteri sunumumuz gelecek cuma günü olacak. 
    Herkes görevlerini tam zamanında yerine getirmeli. Kalite kontrol süreçleri de 
    devreye alınacak. Proje yöneticisi haftalık raporlar hazırlayacak.
    """
    
    summarizer = IndustrialSummarizer()
    summary = summarizer.summarize(test_text)
    print("Özet:", summary)