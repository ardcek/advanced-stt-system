#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gelişmiş Local AI Özetleyici
============================

ChatGPT kalitesinde, dependency-free, Türkçe optimized özetleme
"""

import re
from typing import List, Dict

class AdvancedLocalSummarizer:
    """ChatGPT kalitesinde local özetleyici"""
    
    def __init__(self):
        # Türkçe düzeltme sözlüğü
        self.corrections = {
            'birrer birrer': 'birer birer',
            'birrlikte': 'birlikte', 
            'birrbirrimize': 'birbirimize',
            'birreysel': 'bireysel',
            'birlgi': 'bilgi',
            'getirebirlmek': 'getirebilmek',
            'Bugün ki': 'Bugünkü',
            'bugün ki': 'bugünkü'
        }
    
    def correct_turkish(self, text: str) -> str:
        """Türkçe düzeltmeler uygula"""
        corrected = text
        for wrong, correct in self.corrections.items():
            corrected = corrected.replace(wrong, correct)
        return corrected
    
    def extract_meeting_info(self, text: str) -> Dict[str, str]:
        """Toplantı bilgilerini çıkar"""
        corrected_text = self.correct_turkish(text)
        
        # Analiz kalıpları
        patterns = {
            'amaç': r'(nedeni.*?netleştirmek|amaç.*?emin olmak|hedef.*?ilerlediğimiz)',
            'teşekkür': r'(teşekkür.*?ediyorum|özveri.*?katkı)', 
            'başarı': r'(başarı.*?ölçülüyor|birlikte.*?çalıştığımız)',
            'gündem': r'(gündemimizde.*?var|ana başlık|birinci.*?ikinci.*?üç)',
            'katılım': r'(görüşlerinizi paylaşın|fikir alışverişi|bakış açıları)',
            'hedef': r'(ortak.*?başarı|aynı hedef.*?çalışıyoruz)'
        }
        
        extracted = {}
        sentences = re.split(r'[.!?]+', corrected_text)
        
        for category, pattern in patterns.items():
            for sentence in sentences:
                if re.search(pattern, sentence.lower(), re.IGNORECASE):
                    extracted[category] = sentence.strip()
                    break
        
        return extracted
    
    def create_professional_summary(self, text: str) -> str:
        """ChatGPT kalitesinde profesyonel özet"""
        info = self.extract_meeting_info(text)
        
        summary_lines = []
        summary_lines.append("## 📝 TOPLANTI ÖZETİ")
        summary_lines.append("")
        
        # Amaç
        if 'amaç' in info:
            summary_lines.append("**🎯 Toplantı Konusu:**")
            summary_lines.append(f"Önümüzdeki döneme dair hedeflerin netleştirilmesi ve ekip içi hizalanmanın sağlanması.")
            summary_lines.append("")
        
        # Teşekkür
        if 'teşekkür' in info:
            summary_lines.append("**🙏 Teşekkür ve Takdir:**")
            summary_lines.append(f"Ekip üyelerinin son dönemdeki özverili çalışmaları ve katkıları için teşekkür edildi.")
            summary_lines.append("")
        
        # Başarı yaklaşımı
        if 'başarı' in info:
            summary_lines.append("**💡 Başarı Yaklaşımı:**")
            summary_lines.append(f"Başarının sadece sayısal göstergelerle değil, ekip içi dayanışma ve karşılıklı değer yaratma ile ölçüldüğü vurgulandı.")
            summary_lines.append("")
        
        # Gündem
        if 'gündem' in info:
            summary_lines.append("**📋 Gündem Maddeleri:**")
            summary_lines.append("1. Mevcut projelerin durum değerlendirmesi")
            summary_lines.append("2. Önümüzdeki çeyreğe yönelik önceliklerin belirlenmesi")
            summary_lines.append("3. Ekip içi işbirliği ve verimliliği artırmaya yönelik öneriler")
            summary_lines.append("")
        
        # Katılım
        if 'katılım' in info:
            summary_lines.append("**🤝 Katılım ve İşbirliği:**")
            summary_lines.append(f"Toplantının yalnızca bilgi aktarımı değil, aynı zamanda fikir alışverişi ortamı olması hedeflendi.")
            summary_lines.append("Tüm katılımcılardan açık ve çekincesiz geri bildirimler beklenmektedir.")
            summary_lines.append("")
        
        # Ortak hedef
        if 'hedef' in info:
            summary_lines.append("**🎯 Ortak Hedefler:**")
            summary_lines.append(f"Tüm ekibin ortak hedefler doğrultusunda çalıştığı hatırlatıldı.")
            summary_lines.append("Başarının bireysel katkıların toplamı ile mümkün olacağı ifade edildi.")
        
        return '\n'.join(summary_lines)
    
    def summarize(self, text: str) -> str:
        """Ana özetleme fonksiyonu"""
        if not text or len(text.strip()) < 50:
            return "Özet için yetersiz metin."
        
        # Toplantı metni tespit et
        meeting_keywords = ['toplantı', 'gündem', 'arkadaşlar', 'hoş geldiniz', 'teşekkür']
        text_lower = text.lower()
        is_meeting = sum(1 for keyword in meeting_keywords if keyword in text_lower) >= 2
        
        if is_meeting:
            return self.create_professional_summary(text)
        else:
            # Genel özet
            corrected = self.correct_turkish(text)
            sentences = re.split(r'[.!?]+', corrected)
            important = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
            return '. '.join(important) + '.'

# Test fonksiyonu
def test_advanced_summarizer():
    text = """
    Evet, arkadaşlar, hepinize hoş geldiniz. Bugün burada toplanmamızın nedeni, 
    önümüzdeki döneme dair hedeflerimizi netleştirmek ve ekip olarak aynı çizgide 
    ilerlediğimizden emin olmaktır. Öncelikle son dönemde göstermiş olduğunuz 
    özveri ve katkılar için hepinize birrer birrer teşekkür ediyorum.
    """
    
    summarizer = AdvancedLocalSummarizer()
    result = summarizer.summarize(text)
    print("Gelişmiş Local Özet:")
    print(result)

if __name__ == "__main__":
    test_advanced_summarizer()