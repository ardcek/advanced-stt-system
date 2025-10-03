#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGPT Özetleyici - ChatGPT Summarizer
=======================================

OpenAI ChatGPT API + Ücretsiz Alternatifler
"""

import os
import json
import requests
from typing import Dict, Optional

# .env dosyasını yükle
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv yoksa environment variables kullan

class ChatGPTSummarizer:
    """ChatGPT API ile gelişmiş özetleme + Ücretsiz alternatifler"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.openai_url = "https://api.openai.com/v1/chat/completions"
        
        # Ücretsiz API alternatifleri
        self.free_apis = [
            {
                'name': 'Groq (Ücretsiz)',
                'url': 'https://api.groq.com/openai/v1/chat/completions',
                'model': 'llama-3.1-8b-instant',  # Groq'un desteklediği model
                'key_env': 'GROQ_API_KEY'
            },
            {
                'name': 'Together AI (Ücretsiz)',
                'url': 'https://api.together.xyz/v1/chat/completions', 
                'model': 'meta-llama/Llama-2-7b-chat-hf',
                'key_env': 'TOGETHER_API_KEY'
            },
            {
                'name': 'Hugging Face (Ücretsiz)',
                'url': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-large',
                'model': 'microsoft/DialoGPT-large',
                'key_env': 'HF_API_KEY'
            }
        ]
        
    def summarize(self, text: str, summary_type: str = "meeting") -> Dict[str, str]:
        """
        ChatGPT ile gelişmiş özetleme - ücretsiz alternatiflerle
        
        Args:
            text: Özetlenecek metin
            summary_type: Özet türü (meeting, academic, medical)
            
        Returns:
            Dict: Özet ve metadata
        """
        
        # 1. OpenAI API dene (eğer key varsa)
        if self.api_key:
            result = self._try_openai_api(text, summary_type)
            if result['success']:
                return result
                
        # 2. Ücretsiz API'ları dene
        for api in self.free_apis:
            api_key = os.getenv(api['key_env'])
            if api_key:
                print(f"🔄 {api['name']} deneniyor...")
                result = self._try_free_api(text, summary_type, api, api_key)
                if result['success']:
                    return result
        
        # 3. Son çare: gelişmiş local özetleme
        print("🔄 Gelişmiş local özetleyiciye geçiliyor...")
        return self._advanced_local_summary(text, summary_type)
    
    def _try_openai_api(self, text: str, summary_type: str) -> Dict[str, str]:
        """OpenAI API dene"""
        try:
            prompt = self._create_prompt(text, summary_type)
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 400,  # Daha az token
                'temperature': 0.3
            }
            
            response = requests.post(self.openai_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                summary = result['choices'][0]['message']['content'].strip()
                return {
                    'summary': summary,
                    'type': 'openai_gpt',
                    'success': True,
                    'provider': 'OpenAI GPT-3.5-turbo'
                }
            elif response.status_code == 429:
                print(f"⚠️ OpenAI Rate Limit - Ücretsiz limiti aştınız")
                return {'success': False, 'error': 'Rate limit exceeded'}
            else:
                print(f"⚠️ OpenAI API hatası: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Detay: {error_detail}")
                except:
                    pass
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            print(f"⚠️ OpenAI API çağrı hatası: {e}")
            return {'success': False}
    
    def _try_free_api(self, text: str, summary_type: str, api: dict, api_key: str) -> Dict[str, str]:
        """Ücretsiz API'ları dene"""
        try:
            prompt = self._create_prompt(text, summary_type)
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            if 'huggingface' in api['url']:
                # Hugging Face farklı format
                data = {"inputs": prompt}
            else:
                # OpenAI uyumlu format
                data = {
                    'model': api['model'],
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 600,
                    'temperature': 0.3
                }
            
            response = requests.post(api['url'], headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'huggingface' in api['url']:
                    summary = result[0]['generated_text'] if result else ""
                else:
                    summary = result['choices'][0]['message']['content'].strip()
                    
                return {
                    'summary': summary,
                    'type': 'free_api',
                    'success': True,
                    'provider': api['name']
                }
            else:
                print(f"⚠️ {api['name']} hatası: {response.status_code}")
                return {'success': False}
                
        except Exception as e:
            print(f"⚠️ {api['name']} çağrı hatası: {e}")
            return {'success': False}
    
    def _advanced_local_summary(self, text: str, summary_type: str) -> Dict[str, str]:
        """Gelişmiş local özetleme - ChatGPT benzeri kalite"""
        try:
            # Industrial summarizer import et
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from industrial_summarizer import IndustrialSummarizer
            
            summarizer = IndustrialSummarizer(language="tr")
            summary = summarizer.summarize(text, max_sentences=8)
            
            # ChatGPT benzeri formatlama
            formatted_summary = self._format_like_chatgpt(summary, summary_type)
            
            return {
                'summary': formatted_summary,
                'type': 'advanced_local',
                'success': True,
                'provider': 'Gelişmiş Local AI'
            }
            
        except Exception as e:
            print(f"⚠️ Local özetleme hatası: {e}")
            return {
                'summary': text[:300] + "...",
                'type': 'fallback',
                'success': True,
                'provider': 'Basit Fallback'
            }
    
    def _format_like_chatgpt(self, summary: str, summary_type: str) -> str:
        """Local özeti ChatGPT benzeri formata çevir"""
        if summary_type == "meeting":
            return f"""📝 **TOPLANTI ÖZETİ**

**🎯 Ana Konular:**
{summary}

**✅ Sonuç:**
Toplantı başarıyla tamamlanmış, ana hedefler ve görevler netleştirilmiştir."""
        
        elif summary_type == "academic":
            return f"""📚 **AKADEMİK ÖZET**

**Konu:** {summary[:100]}...
**Ana Noktalar:** {summary}
**Sonuç:** Önemli akademik bulgular ve çıkarımlar elde edilmiştir."""
        
        else:
            return f"""📋 **ÖZET**

{summary}

**Not:** Bu özet gelişmiş local AI tarafından oluşturulmuştur."""
    
    def _create_prompt(self, text: str, summary_type: str) -> str:
        """Özetleme promptu oluştur"""
        prompts = {
            "meeting": f"""Sen profesyonel bir toplantı analisti ve özetleme uzmanısın. Aşağıdaki Türkçe toplantı transkripsiyonunu analiz et ve profesyonel, detaylı bir özet oluştur.

TOPLANTI TRANSKRİPSİYONU:
{text}

Lütfen şu formatta özetle:

📝 **TOPLANTI ÖZETİ**

**🎯 Toplantı Konusu:**
[Toplantının temel amacı ve konusu]

**� Teşekkür ve Takdir:**
[Ekip ve katkılara dair teşekkürler]

**💡 Başarı Yaklaşımı:**
[Başarının nasıl tanımlandığı ve ölçüleceği]

**�📋 Gündem Maddeleri:**
1. [Birinci gündem maddesi]
2. [İkinci gündem maddesi]  
3. [Üçüncü gündem maddesi]

**🤝 Katılım ve İşbirliği:**
[Katılım beklentileri ve iletişim vurguları]

**🎯 Ortak Hedefler:**
[Ortak amaçlar ve sonraki adımlar]

ÖNEMLİ: Sadece transkripsiyonda geçen bilgileri kullan. Türkçe yazım hatalarını düzelt (örn: "birrer" → "birer", "birrlikte" → "birlikte"). Özeti Türkçe yaz.""",

            "academic": f"""Sen akademik metin analisti uzmanısın. Bu Türkçe akademik içeriği bilimsel sunum formatında özetle:

İÇERİK:
{text}

📚 **AKADEMİK ÖZET**

**Konu:** [Ana akademik konu]
**Amaç:** [Hedef ve amaç]
**Ana Bulgular:** [Önemli akademik bulgular]
**Sonuç:** [Çıkarım ve öneriler]

Türkçe yazım hatalarını düzelt.""",

            "medical": f"""Sen tıbbi metin analisti uzmanısın. Bu Türkçe tıbbi içeriği professional tıbbi formatta özetle:

TIBBİ İÇERİK:
{text}

🏥 **TIBBİ ÖZET**

**Konu:** [Tıbbi konu]
**Ana Bulgular:** [Önemli tıbbi bilgiler]
**Öneriler:** [Tıbbi öneriler]
**Sonuç:** [Özet çıkarım]

Türkçe yazım hatalarını düzelt."""
        }
        
        return prompts.get(summary_type, prompts["meeting"])
    
    def _call_openai_api(self, prompt: str) -> Optional[str]:
        """OpenAI API çağrısı"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 500,
            'temperature': 0.3
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"⚠️ OpenAI API hatası: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"⚠️ API çağrı hatası: {e}")
            return None
    
    def _fallback_summary(self, text: str) -> str:
        """API başarısız olursa basit özetleme"""
        # Basit kural tabanlı özetleme
        sentences = text.split('.')
        important_sentences = []
        
        keywords = [
            'hedef', 'amaç', 'toplantı', 'gündem', 'başarı', 
            'proje', 'öncelik', 'teşekkür', 'çalış', 'ekip'
        ]
        
        for sentence in sentences[:10]:  # İlk 10 cümleye bak
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in keywords):
                important_sentences.append(sentence)
                
        if important_sentences:
            return '. '.join(important_sentences[:3]) + '.'
        else:
            return sentences[0] + '.' if sentences else text[:200] + '...'

# Test fonksiyonu
def test_chatgpt_summarizer():
    """Test fonksiyonu"""
    summarizer = ChatGPTSummarizer()
    
    test_text = """
    Evet, arkadaşlar, hepinize hoş geldiniz. Bugün burada toplanmamızın nedeni, 
    önümüzdeki döneme dair hedeflerimizi netleştirmek ve ekip olarak aynı çizgide 
    ilerlediğimizden emin olmaktır. Öncelikle son dönemde göstermiş olduğunuz 
    özveri ve katkılar için hepinize teşekkür ediyorum.
    """
    
    result = summarizer.summarize(test_text, "meeting")
    
    print("🤖 ChatGPT Özetleyici Test")
    print("=" * 50)
    print(f"📝 Özet: {result['summary']}")
    print(f"🔧 Tip: {result['type']}")
    if 'error' in result:
        print(f"⚠️ Hata: {result['error']}")

if __name__ == "__main__":
    test_chatgpt_summarizer()