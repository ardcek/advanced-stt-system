#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Türkçe Metin Düzeltici - Turkish Text Corrector
===============================================

STT çıktılarındaki yaygın Türkçe hatalarını düzeltir.
"""

import re
from typing import Dict, List, Tuple

class TurkishTextCorrector:
    """Türkçe STT çıktıları için metin düzeltici"""
    
    def __init__(self):
        self.common_corrections = {
            # Yaygın STT hataları
            "özgüleri": "özveri",
            "durumlu": "durum", 
            "yakışı": "akışı",
            "ilerdiğimizden": "ilerlediğimizden",
            "uygulamak isterim": "vurgulamak isterim",
            "değer verdiğimiz": "değer kattığımızla",
            "günlerimizde": "gündemimizde",
            "çeyreğe": "çeyreğe",
            "arttırıcı": "artırıcı",
            "zemini": "zemin",
            "yapmaya hazırım": "yapmaya hazırım",
            
            # Yeni tespit edilen hatalar
            "birrer birrer": "birer birer",
            "birrlikte": "birlikte", 
            "birrbirrimize": "birbirimize",
            "birlgi": "bilgi",
            "işbirrliği": "işbirliği",
            "getirebirlmek": "getirebilmek",
            "birreysel": "bireysel",
            "Bugün ki": "Bugünkü",
            
            # Noktalama düzeltmeleri
            " ,": ",",
            " .": ".",
            " ;": ";",
            " :": ":",
            " !": "!",
            " ?": "?",
            
            # Kelime bütünlüğü
            "iş birliği": "işbirliği",
            "fikir alışverişi": "fikir alışverişi",
            "bakış açıları": "bakış açıları",
            
            # Yaygın konuşma dili hataları
            "bi": "bir",
            "şey": "şey",
            "yani": "yani",
            "işte": "işte"
        }
        
        # Kelime bazlı düzeltmeler
        self.word_corrections = {
            "özgüleri": "özveri",
            "durumlu": "durum",
            "yakışı": "akışı", 
            "ilerdiğimizden": "ilerlediğimizden",
            "günlerimizde": "gündemimizde",
            "arttırıcı": "artırıcı",
            "zemini": "zemin",
            "birrer": "birer",
            "birrlikte": "birlikte",
            "birrbirrimize": "birbirimize", 
            "birlgi": "bilgi",
            "işbirrliği": "işbirliği",
            "getirebirlmek": "getirebilmek",
            "birreysel": "bireysel"
        }
        
        # Cümle yapısı düzeltmeleri
        self.sentence_patterns = [
            (r"Bugün ki günlerimizde", "Bugünkü gündemimizde"),
            (r"bilgi yakışı değil", "bilgi akışı değil"),
            (r"şunu özellikle uygulamak isterim", "şunu özellikle vurgulamak isterim"),
            (r"değer verdiğimiz katkılarla", "değer kattığımızla"),
            (r"üzerime düşene", "üzerime düşeni")
        ]
    
    def correct_text(self, text: str) -> Dict[str, str]:
        """
        Metni düzelt ve düzeltme detaylarını döndür
        
        Args:
            text: Düzeltilecek metin
            
        Returns:
            Dict: Orijinal, düzeltilmiş metin ve değişiklikler
        """
        original_text = text
        corrected_text = text
        corrections_made = []
        
        # 1. Regex tabanlı düzeltmeler (daha etkili)
        regex_corrections = [
            (r'\bbirrer\b', 'birer'),
            (r'\bbirrlikte\b', 'birlikte'),
            (r'\bbirrbirrimize\b', 'birbirimize'),
            (r'\bbirlgi\b', 'bilgi'),
            (r'\bişbirrliği\b', 'işbirliği'),
            (r'\bgetirebirlmek\b', 'getirebilmek'),
            (r'\bbirreysel\b', 'bireysel'),
            (r'\bBugün ki\b', 'Bugünkü'),
            (r'\bözgüleri\b', 'özveri'),
            (r'\bdurumlu\b', 'durum'),
            (r'\byakışı\b', 'akışı'),
            (r'\bilerdiğimizden\b', 'ilerlediğimizden'),
            (r'\barttırıcı\b', 'artırıcı'),
            (r'\bzemini\b', 'zemin')
        ]
        
        for pattern, replacement in regex_corrections:
            if re.search(pattern, corrected_text, re.IGNORECASE):
                corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
                corrections_made.append(f"{pattern.replace('\\b', '')} → {replacement}")
        
        # 2. Kelime bazlı düzeltmeler
        for wrong, correct in self.word_corrections.items():
            if wrong in corrected_text:
                corrected_text = corrected_text.replace(wrong, correct)
                if f"{wrong} → {correct}" not in corrections_made:
                    corrections_made.append(f"{wrong} → {correct}")
        
        # 3. Cümle yapısı düzeltmeleri
        for pattern, replacement in self.sentence_patterns:
            if re.search(pattern, corrected_text, re.IGNORECASE):
                corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
                corrections_made.append(f"Cümle yapısı düzeltildi")
        
        # 4. Genel düzeltmeler
        for wrong, correct in self.common_corrections.items():
            if wrong in corrected_text:
                corrected_text = corrected_text.replace(wrong, correct)
                if f"{wrong} → {correct}" not in corrections_made:
                    corrections_made.append(f"{wrong} → {correct}")
        
        # 5. Noktalama düzenleme
        corrected_text = self._fix_punctuation(corrected_text)
        
        # 6. Boşluk düzenleme
        corrected_text = self._fix_spacing(corrected_text)
        
        return {
            'original': original_text,
            'corrected': corrected_text,
            'corrections': corrections_made,
            'improvement_score': self._calculate_improvement_score(original_text, corrected_text)
        }
    
    def _fix_punctuation(self, text: str) -> str:
        """Noktalama işaretlerini düzelt"""
        # Boşluk + noktalama → noktalama
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        
        # Noktalama + cümle başı → büyük harf
        text = re.sub(r'([.!?])\s+([a-züğışöç])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
        
        # Paragraf başı büyük harf
        text = re.sub(r'^([a-züğışöç])', lambda m: m.group(1).upper(), text, flags=re.MULTILINE)
        
        return text
    
    def _fix_spacing(self, text: str) -> str:
        """Boşluk düzenlemesi"""
        # Çoklu boşlukları tekil yap
        text = re.sub(r'\s+', ' ', text)
        
        # Satır başı/sonu boşlukları temizle
        text = text.strip()
        
        return text
    
    def _calculate_improvement_score(self, original: str, corrected: str) -> float:
        """Düzeltme skorunu hesapla"""
        if original == corrected:
            return 1.0
        
        # Basit benzerlik skoru
        original_words = set(original.lower().split())
        corrected_words = set(corrected.lower().split())
        
        common_words = original_words.intersection(corrected_words)
        total_words = original_words.union(corrected_words)
        
        if len(total_words) == 0:
            return 1.0
            
        return len(common_words) / len(total_words)
    
    def suggest_corrections(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Düzeltme önerileri listesi
        
        Returns:
            List[Tuple]: (hatalı_kelime, önerilen_düzeltme, açıklama)
        """
        suggestions = []
        
        for wrong, correct in self.word_corrections.items():
            if wrong in text:
                suggestions.append((wrong, correct, "Yaygın STT hatası"))
        
        return suggestions

# Test fonksiyonu
def test_corrector():
    """Test fonksiyonu"""
    corrector = TurkishTextCorrector()
    
    test_text = """Evet, arkadaşlar, hepinize hoş geldiniz. Bugün burada toplanmamızın nedeni, 
    önümüzdeki döneme dair hedeflerimizi netleştirmek ve ekip olarak aynı çizgide ilerdiğimizden emin olmaktır. 
    Arkadaşlar, öncelikle son dönemde göstermiş olduğunuz özgüleri ve katkılar için hepinize birer birer teşekkür ediyorum."""
    
    result = corrector.correct_text(test_text)
    
    print("🔍 Türkçe Metin Düzeltici Test")
    print("=" * 50)
    print(f"📝 Orijinal: {result['original'][:100]}...")
    print(f"✅ Düzeltilmiş: {result['corrected'][:100]}...")
    print(f"🔧 Düzeltmeler: {result['corrections']}")
    print(f"📊 İyileştirme Skoru: {result['improvement_score']:.2f}")

if __name__ == "__main__":
    test_corrector()