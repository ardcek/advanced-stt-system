# modules/nlp.py
from __future__ import annotations

"""
Metin normalizasyonu, özetleme ve bilgi çıkarımı (görev / karar) yardımcıları.

Öne çıkanlar
------------
- normalize_transcript: Kural tabanlı + (opsiyonel) fuzzy düzeltme, tekrar kırpma,
  noktalama/boşluk toparlama, cümle bazlı tekilleştirme.
- summarize_text: Token-bilinçli parçalara bölüp (overlap ile) ara özetler üretir,
  sonra ikinci geçişte nihai özeti çıkarır. Uzun metinlerde "token indices" uyarısını engeller.
- extract_tasks / extract_decisions: Emir kipleri, aksiyon ve karar kelimeleri, isimle çağrı
  (Ali:, Ayşe,) gibi sinyallere bakarak cümle cümle görev/karar listesi çıkarır.
- summarize_by_windows: Çok uzun kayıtlar için (örn. 10 dk) zaman penceresi bazlı bölüm özetleri.

Bağımlılıklar
-------------
- transformers  (HF pipeline için)
- rapidfuzz     (opsiyonel, fuzzy düzeltme – yoksa devre dışı)
"""

import os
import re
from typing import Dict, List, Tuple, Iterable, Optional

# ==============================
#  HF Summarizer (lazy load)
# ==============================

_SUMM_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"   # hızlı ve yeterince iyi; dilersen "facebook/bart-large-cnn"
_SUMMARIZER = None          # transformers.pipeline instance
_TOKENIZER = None           # AutoTokenizer (çoğu pipeline.tokenizer olarak tutar)

def _get_summarizer():
    """
    HF summarizer'ı ve tokenizer'ı lazy yükler.
    Dönen: (pipeline, tokenizer|None)
    """
    global _SUMMARIZER, _TOKENIZER
    if _SUMMARIZER is None:
        from transformers import pipeline
        _SUMMARIZER = pipeline("summarization", model=_SUMM_MODEL_NAME, device=-1)  # CPU kullan
        _TOKENIZER = getattr(_SUMMARIZER, "tokenizer", None)
    return _SUMMARIZER, _TOKENIZER


# ==============================
#  Opsiyonel fuzzy (rapidfuzz)
# ==============================

try:
    from rapidfuzz import fuzz  # type: ignore
    _HAS_FUZZ = True
except Exception:
    _HAS_FUZZ = False

# Fuzzy yakınsaması için kanonik terimler (özel adlar + teknik terimler)
_CANON_TERMS = [
    "Python", "NumPy", "Pandas", "PostgreSQL", "Docker", "Kubernetes",
    "Arda", "Ayşe", "Mehmet", "Zeynep",
    "özet", "toplantı", "önbellek", "güncelleme", "doküman", "metrik",
    "senaryo", "görseller", "kullanım", "oturum", "öneri", "teslim", "ekran",
]

# ======================================
#  Kural tabanlı düzeltmeler (base fix)
#  (kök -> doğru); ek/apos korunur
# ======================================

_BASE_FIX: Dict[str, str] = {
    # Özel isim / teknoloji yanlış okunuşları
    "payton": "Python", "paiton": "Python", "paton": "Python", "pyton": "Python", "piton": "Python",
    "numpy": "NumPy", "nampay": "NumPy",
    "pandas": "Pandas",
    "postgre": "PostgreSQL", "postgress": "PostgreSQL",
    "docker": "Docker", "kubernetis": "Kubernetes", "kubernettes": "Kubernetes",

    "marda": "Arda",
    "ayse": "Ayşe", "yaşe": "Ayşe", "ya işe": "Ayşe",
    "znp": "Zeynep", "znhp": "Zeynep", "zenef": "Zeynep", "znep": "Zeynep", "zenep": "Zeynep",

    # Toplantı/ders bağlamı
    "özür": "özet", "özetle": "özetle",  # "özetle" doğru; normalize ederken korunur
    "toplandığı": "toplantı", "toplandigi": "toplantı",
    "uygulamasındık": "uygulamasındaki", "uygulamasindık": "uygulamasındaki",

    # Performans/teknik
    "önbellik": "önbellek", "onbellik": "önbellek", "ön bellek": "önbellek", "on bellek": "önbellek",
    "görsenleri": "görselleri", "görsenler": "görseller",
    "duvan": "duman", "yünceleme": "güncelleme", "günceleme": "güncelleme",
    "metrix": "metrik", "önleri": "öneri", "kullanılım": "kullanım", "otorum": "oturum",
    "çık deme": "yükleme", "geç çık": "geç yükleme", "çağrışımlar": "çağrılar",

    # Yaygın konuşma/yazım hataları
    "gönderilicek": "gönderilecek", "harve": "hale", "topluğa": "topluca", "topluğa at": "topluca at",
    "dükümana": "dokümana", "dökümana": "dokümana",
    "görüntülünü": "görüntülerini",
    "ayakkadaş": "arkadaş", "dinli": "dilini", "haftalıkla": "haftalık",
}

# Kullanıcı ek sözlüğü (corrections.txt içeriği: "yanlış => doğru")
def _load_user_pairs(path: str = "corrections.txt") -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=>" not in line:
                    continue
                wrong, right = [x.strip() for x in line.split("=>", 1)]
                if wrong and right:
                    pairs[wrong] = right
    return pairs

# Ekleri/apostrofu koru: Python’u, önbellekten, Ayşe’ye...
_SUFFIX_RE = (
    r"(?P<apos>['’]?)"
    r"(?P<suffix>(?:lar|ler|da|de|ta|te|dan|den|tan|ten|a|e|ı|i|u|ü|ya|ye|yı|yi|yu|yü|"
    r"ın|in|un|ün|sı|si|su|sü)?)"
)

def _compile_rules(base: Dict[str, str]) -> List[Tuple[re.Pattern, callable]]:
    rules: List[Tuple[re.Pattern, callable]] = []
    for wrong, right in base.items():
        pat = re.compile(rf"\b{re.escape(wrong)}{_SUFFIX_RE}\b", re.IGNORECASE)
        def _mk(right=right):
            return lambda m: f"{right}{m.group('apos')}{m.group('suffix') or ''}"
        rules.append((pat, _mk()))
    return rules

# ==============================
#  Normalize yardımcıları
# ==============================

def _normalize_space_punct(t: str) -> str:
    # Boşlukları sadeleştir, noktalama etrafını toparla
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,;:\.\!\?])", r"\1", t)
    t = re.sub(r"([\(])\s+", r"\1", t)
    t = re.sub(r"\s+([\)])", r"\1", t)
    return t

def _collapse_letter_repeats(t: str) -> str:
    # Harf uzatmalarını kırp: "çoook" -> "çok"
    return re.sub(r"([A-Za-zÇĞİÖŞÜçğıöşü])\1{2,}", r"\1", t)

def _collapse_word_repeats(t: str) -> str:
    # Aynı kelimenin ardışık tekrarlarını 2 ile sınırla
    words = t.split()
    out, prev, cnt = [], None, 0
    for w in words:
        if w.lower() == (prev or "").lower():
            cnt += 1
            if cnt <= 2:
                out.append(w)
        else:
            prev, cnt = w, 1
            out.append(w)
    return " ".join(out)

def _sentence_split(t: str) -> List[str]:
    # Basit ve dayanıklı cümle bölücü
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", (t or "").strip())
    return [p.strip() for p in parts if p.strip()]

def _canon(s: str) -> str:
    # Tekilleştirme anahtarı (noktalama ve boşluklardan arındırılmış, küçük harf)
    return re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]+", "", s.lower())

def _fuzzy_fix_token(tok: str) -> str:
    if not _HAS_FUZZ or len(tok) < 3:
        return tok
    m = re.match(rf"^(?P<root>[A-Za-zÇĞİÖŞÜçğıöşü]+){_SUFFIX_RE}$", tok)
    if not m:
        root, apos, suf = tok, "", ""
    else:
        root = m.group("root")
        apos = m.group("apos") or ""
        suf = m.group("suffix") or ""
    best, score = root, 0
    for cand in _CANON_TERMS:
        sc = fuzz.ratio(root.lower(), cand.lower())  # type: ignore
        if sc > score:
            best, score = cand, sc
    return f"{best}{apos}{suf}" if score >= 88 else tok

# ==============================
#  ANA NORMALİZE
# ==============================

def normalize_transcript(text: str) -> str:
    """
    STT çıktısını kurumsal metne yaklaştırır:
    - kullanıcı sözlüğü + kural tabanlı düzeltme (ek/apos korunur)
    - harf/kelime tekrar kırpma
    - opsiyonel fuzzy düzeltme (rapidfuzz varsa)
    - cümle bazlı ardışık tekrar tekilleştirme
    """
    t = _normalize_space_punct(text or "")
    if not t:
        return ""

    # 1) kullanıcı sözlüğü + baz düzeltmeler
    rules = _compile_rules({**_BASE_FIX, **_load_user_pairs()})
    for pat, repl in rules:
        t = pat.sub(repl, t)

    # 2) harf/kelime tekrarları
    t = _collapse_letter_repeats(t)
    t = _collapse_word_repeats(t)

    # 3) opsiyonel fuzzy: tek kelimelik tuhaflıklar
    if _HAS_FUZZ:
        t = " ".join(_fuzzy_fix_token(tok) for tok in t.split())

    # 4) cümle bazında ardışık tekrar kırpma
    sents = _sentence_split(t)
    out_s, last_key = [], ""
    for s in sents:
        key = _canon(s)
        if not key:
            continue
        if key == last_key:
            continue
        out_s.append(s)
        last_key = key

    return " ".join(out_s)


def normalize_transcript_advanced(text: str, language: str = "tr", fix_spelling: bool = True, fix_foreign_terms: bool = True) -> str:
    """
    Gelişmiş normalizasyon - çoklu dil desteği ve yazım düzeltme ile
    
    Args:
        text: Ham transkripsiyon metni
        language: Kaynak dili (tr, en, de, fr, es, it, la)
        fix_spelling: Yazım hatalarını düzelt
        fix_foreign_terms: Yabancı terim düzeltme
    """
    if not text or not text.strip():
        return ""
        
    # Temel normalizasyon
    normalized = normalize_transcript(text)
    
    if not fix_spelling and not fix_foreign_terms:
        return normalized
        
    # Dile özel düzeltmeler
    if language == "tr":
        normalized = _fix_turkish_spelling(normalized, fix_foreign_terms)
    elif language == "en":
        normalized = _fix_english_spelling(normalized)
    elif language in ["de", "fr", "es", "it", "la"]:
        normalized = _fix_foreign_language_spelling(normalized, language)
        
    return normalized


def _load_custom_terms() -> Dict[str, str]:
    """Custom terms dosyasından düzeltme çiftleri yükle"""
    corrections = {}
    
    # corrections.txt dosyasından
    corrections.update(_load_user_pairs("corrections.txt"))
    
    # custom_terms.txt dosyasından da kelime çiftleri çıkar
    terms_path = "custom_terms.txt"
    if os.path.exists(terms_path):
        with open(terms_path, "r", encoding="utf-8") as f:
            current_section = ""
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    if "# " in line:
                        current_section = line.replace("#", "").strip().lower()
                    continue
                
                # Terim bazlı otomatik düzeltme kuralları oluştur
                term = line.strip()
                if term and len(term) > 2:
                    # Yaygın yanlış telaffuz/yazım varyantları oluştur
                    if "İngilizce" in current_section or "English" in current_section:
                        corrections.update(_generate_english_variants(term))
                    elif "Latince" in current_section or "Latin" in current_section:
                        corrections.update(_generate_latin_variants(term))
                    elif "Almanca" in current_section or "German" in current_section:
                        corrections.update(_generate_german_variants(term))
                    elif "Fransızca" in current_section or "French" in current_section:
                        corrections.update(_generate_french_variants(term))
    
    return corrections


def _generate_english_variants(term: str) -> Dict[str, str]:
    """İngilizce terim için yaygın yanlış varyantlar oluştur"""
    variants = {}
    term_lower = term.lower()
    
    # Genel İngilizce ses değişimleri
    common_mistakes = [
        # th -> t/d değişimi
        ("th", "t"), ("th", "d"),
        # v -> w değişimi  
        ("v", "w"), ("w", "v"),
        # -tion -> -şın değişimi
        ("tion", "şın"), ("tion", "şan"),
        # double letters
        ("ll", "l"), ("ss", "s"), ("tt", "t"),
        # i/e karışımı
        ("i", "e"), ("e", "i")
    ]
    
    for old, new in common_mistakes:
        if old in term_lower:
            variant = term_lower.replace(old, new)
            if variant != term_lower and len(variant) > 2:
                variants[variant] = term
    
    # Specific corrections
    specific_corrections = {
        "artifishal": "artificial",
        "intelijent": "intelligent", 
        "algoritm": "algorithm",
        "daytabase": "database",
        "servır": "server",
        "klient": "client",
        "interfeys": "interface",
        "javascirpt": "JavaScript",
        "reakt": "React"
    }
    
    # Terimi specific corrections'ta ara
    for wrong, right in specific_corrections.items():
        if term_lower == right.lower():
            variants[wrong] = term
    
    return variants


def _generate_latin_variants(term: str) -> Dict[str, str]:
    """Latince terim için Türkçe telaffuz varyantları"""
    variants = {}
    term_lower = term.lower()
    
    latin_adaptations = [
        # c -> k/s değişimi
        ("c", "k"), ("c", "s"),
        # ae -> e değişimi
        ("ae", "e"), ("ae", "a"),
        # ph -> f değişimi  
        ("ph", "f"),
        # x -> ks değişimi
        ("x", "ks"),
        # qu -> kw değişimi
        ("qu", "kw"), ("qu", "ku")
    ]
    
    for old, new in latin_adaptations:
        if old in term_lower:
            variant = term_lower.replace(old, new)
            if variant != term_lower:
                variants[variant] = term
    
    return variants


def _generate_german_variants(term: str) -> Dict[str, str]:
    """Almanca terim için Türkçe telaffuz varyantları"""
    variants = {}
    term_lower = term.lower()
    
    german_adaptations = [
        # sch -> ş değişimi
        ("sch", "ş"), ("sch", "sh"),
        # ß -> ss değişimi
        ("ß", "ss"), ("ß", "s"),
        # ü -> u değişimi
        ("ü", "u"), ("ä", "a"), ("ö", "o"),
        # w -> v değişimi
        ("w", "v"), ("v", "w"),
        # z -> ts değişimi
        ("z", "ts"), ("z", "s")
    ]
    
    for old, new in german_adaptations:
        if old in term_lower:
            variant = term_lower.replace(old, new)
            if variant != term_lower:
                variants[variant] = term
    
    return variants


def _generate_french_variants(term: str) -> Dict[str, str]:
    """Fransızca terim için Türkçe telaffuz varyantları"""
    variants = {}
    term_lower = term.lower()
    
    french_adaptations = [
        # Silent letters
        ("ent", "an"), ("ent", "ant"),
        # Nasal sounds  
        ("on", "ın"), ("an", "an"),
        ("in", "in"), ("un", "un"),
        # j -> ş/zh değişimi
        ("j", "ş"), ("j", "zh"), ("j", "c"),
        # ou -> u değişimi
        ("ou", "u"), ("au", "o"),
        # ç -> s değişimi
        ("ç", "s")
    ]
    
    for old, new in french_adaptations:
        if old in term_lower:
            variant = term_lower.replace(old, new)
            if variant != term_lower:
                variants[variant] = term
    
    return variants


def _fix_turkish_spelling(text: str, fix_foreign: bool = True) -> str:
    """Türkçe yazım düzeltmeleri - gelişmiş versiyon"""
    
    # Türkçe özel düzeltmeler
    tr_fixes = {
        # Yaygın hatalı yazımlar
        "birşey": "bir şey", "birşeyi": "bir şeyi", 
        "herşey": "her şey", "herşeyi": "her şeyi",
        "birsürü": "bir sürü", "hiçbirşey": "hiç bir şey",
        
        # Teknik terimler
        "yazilim": "yazılım", "gelişitrme": "geliştirme",
        "uygulamş": "uygulama", "programş": "program",
        
        # Yaygın kelime hataları
        "hemde": "hem de", "yinede": "yine de", "kesinlikle": "kesinlikle",
        "teşekkürler": "teşekkürler", "merhebe": "merhaba",
    }
    
    # Yabancı terim düzeltmeleri
    if fix_foreign:
        # Custom terms dosyasından yükle
        custom_corrections = _load_custom_terms()
        tr_fixes.update(custom_corrections)
    
    # Düzeltmeleri uygula (büyük/küçük harf duyarsız)
    for wrong, right in tr_fixes.items():
        # Kelime sınırlarına dikkat et
        pattern = r'\b' + re.escape(wrong) + r'\b'
        text = re.sub(pattern, right, text, flags=re.IGNORECASE)
    
    return text


def _fix_english_spelling(text: str) -> str:
    """İngilizce yazım düzeltmeleri - kapsamlı versiyon"""
    
    # Temel yazım hataları
    en_fixes = {
        # ie/ei karışımları
        "recieve": "receive", "recieved": "received", "reciever": "receiver",
        "beleive": "believe", "beleived": "believed", "beleiver": "believer",
        "acheive": "achieve", "acheived": "achieved", "acheiver": "achiever",
        
        # Separate kelimesi
        "seperate": "separate", "seperately": "separately", "seperation": "separation",
        
        # -ly/-ally sonekleri
        "definately": "definitely", "defiantly": "definitely",
        "probaly": "probably", "realy": "really", "usualy": "usually",
        
        # Double letters
        "occured": "occurred", "occurence": "occurrence", "occuring": "occurring",
        "begining": "beginning", "stoped": "stopped", "planed": "planned",
        
        # -ment/-ement karışımı
        "managment": "management", "developement": "development", 
        "arrangment": "arrangement", "judgement": "judgment",
        
        # Environment
        "enviroment": "environment", "enviornment": "environment",
        "goverment": "government", "govenment": "government",
        
        # Teknik terimler
        "algoritm": "algorithm", "algorythm": "algorithm",
        "artifical": "artificial", "artifishal": "artificial",
        "inteligent": "intelligent", "intelijent": "intelligent",
        
        # Akademik terimler
        "knowlege": "knowledge", "knowladge": "knowledge",
        "reserch": "research", "reasearch": "research",
        "analize": "analyze", "analise": "analyze",
        "critisism": "criticism", "critisise": "criticise",
        
        # Diğer yaygın hatalar
        "wierd": "weird", "freind": "friend", "thier": "their",
        "whitch": "which", "witch": "which", "becuase": "because",
        "necesary": "necessary", "neccessary": "necessary",
        "buisness": "business", "bussiness": "business",
        "intresting": "interesting", "interresting": "interesting"
    }
    
    # Custom terms dosyasından İngilizce düzeltmeler
    custom_terms = _load_custom_terms()
    en_fixes.update(custom_terms)
    
    # Düzeltmeleri uygula
    for wrong, right in en_fixes.items():
        pattern = r'\b' + re.escape(wrong) + r'\b'
        text = re.sub(pattern, right, text, flags=re.IGNORECASE)
        
    return text


def _fix_foreign_language_spelling(text: str, language: str) -> str:
    """Diğer diller için kapsamlı yazım düzeltmeleri"""
    
    # Almanca
    if language == "de":
        de_fixes = {
            # das/dass karışıklığı
            "dass": "dass", "das": "das",
            # seit/seid karışıklığı  
            "seit": "seit", "seid": "seid",
            # wird/wirt karışıklığı
            "wirt": "wird", "wird": "wird",
            # Yaygın yazım hataları
            "gesundheit": "Gesundheit", "gesundhait": "Gesundheit",
            "volkswagen": "Volkswagen", "folkvagen": "Volkswagen", 
            "deutschland": "Deutschland", "doyçland": "Deutschland",
            "kindergarten": "Kindergarten", "kindegarten": "Kindergarten",
            # Umlaut düzeltmeleri
            "muench": "München", "munchen": "München",
            "koeln": "Köln", "koln": "Köln"
        }
        for wrong, right in de_fixes.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            text = re.sub(pattern, right, text, flags=re.IGNORECASE)
    
    # Fransızca
    elif language == "fr":
        fr_fixes = {
            "bonjur": "bonjour", "bonzhur": "bonjour",
            "mersi": "merci", "mersy": "merci", 
            "silvuple": "s'il vous plaît", "silvu ple": "s'il vous plaît",
            "reson": "raison", "detr": "d'être",
            "se la vi": "c'est la vie", "sela vi": "c'est la vie"
        }
        for wrong, right in fr_fixes.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            text = re.sub(pattern, right, text, flags=re.IGNORECASE)
    
    # İspanyolca
    elif language == "es":
        es_fixes = {
            "ola": "hola", "grasiyas": "gracias", "grasyas": "gracias",
            "por favur": "por favor", "porfavor": "por favor",
            "de nade": "de nada", "denade": "de nada",
            "buenas nohes": "buenas noches", "nohes": "noches",
            "asta la vista": "hasta la vista", "astala vista": "hasta la vista"
        }
        for wrong, right in es_fixes.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            text = re.sub(pattern, right, text, flags=re.IGNORECASE)
    
    # İtalyanca
    elif language == "it":
        it_fixes = {
            "chao": "ciao", "çao": "ciao",
            "gratsie": "grazie", "gratsye": "grazie",
            "per favore": "per favore", "perfavore": "per favore",
            "molto bene": "molto bene", "moltobene": "molto bene",
            "arrivederci": "arrivederci", "arivederci": "arrivederci"
        }
        for wrong, right in it_fixes.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            text = re.sub(pattern, right, text, flags=re.IGNORECASE)
    
    # Latince
    elif language == "la":
        la_fixes = {
            "vise versa": "vice versa", "vise": "vice", "versa": "versa",
            "etselera": "et cetera", "etsetera": "et cetera", "etc": "et cetera",
            "eksampıl gratiya": "exempli gratia", "eksampıl": "exempli", "gratiya": "gratia",
            "perse": "per se", "persente": "per se",
            "adsurdum": "ad absurdum", "adbsurdum": "ad absurdum",
            "sinkua non": "sine qua non", "sinekua non": "sine qua non",
            "defakto": "de facto", "de fakto": "de facto",
            "dejure": "de jure", "de cure": "de jure"
        }
        for wrong, right in la_fixes.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            text = re.sub(pattern, right, text, flags=re.IGNORECASE)
    
    return text


# ==============================
#  ÖZETLEME (token-bilinçli)
# ==============================

def _token_len(text: str) -> int:
    """
    Tokenizer yoksa kaba tahmin: ~1.3 kelime ≈ 1 token.
    """
    _, tok = _get_summarizer()
    if tok is None:
        return int(max(1, len(text.split()) / 1.3))
    return len(tok.encode(text, add_special_tokens=False))

def _chunks_by_tokens(text: str, max_tokens: int = 500, overlap: int = 50) -> Iterable[str]:
    """
    Metni modelin bağlam sınırını aşmayacak şekilde token bazlı parçalara böler.
    Overlap, bağlamın sürekliliğini sağlar.
    """
    summarizer, tok = _get_summarizer()
    if tok is None:
        # Tokenizer yoksa karakter bazlı güvenli parçalama (daha küçük)
        chunk_size = 1200
        step = chunk_size - 150
        t = re.sub(r"\s+", " ", (text or "")).strip()
        for i in range(0, len(t), max(1, step)):
            yield t[i:i + chunk_size]
        return

    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return
    step = max_tokens - max(0, overlap)
    for start in range(0, len(ids), max(1, step)):
        sub_ids = ids[start:start + max_tokens]
        yield tok.decode(sub_ids, skip_special_tokens=True)

def _safe_summarize_chunk(s: str, max_length: int, min_length: int) -> str:
    """
    Tek bir parçayı özetler; hata/taşma durumunda karakter-bazlı küçüğe düşer.
    """
    summarizer, _ = _get_summarizer()
    s = s.strip()
    if not s:
        return ""
    try:
        return summarizer(s, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
    except Exception:
        # Aşırı uzun/parazitli parça; biraz kesip tekrar dene
        short = s[:1600]
        return summarizer(short, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]

def summarize_text(text: str, max_length: Optional[int] = None, min_length: Optional[int] = None, language: str = "tr") -> str:
    """
    Uzun metinleri token-bilinçli parçalara böler, parça özetlerini birleştirir ve
    ikinci geçişte nihai özeti üretir. Kısa metinde tek atış yapar.

    Not: Çok kısa metinde özet anlamsız olacağından metnin kendisini döndürür.
    """
    t = (text or "").strip()
    if not t:
        return ""
    if len(t.split()) < 20:
        return t

    in_tokens = _token_len(t)

    # Hedef uzunluğu girdiye göre dinamik belirle
    if max_length is None or min_length is None:
        # Çıkış uzunluğu: giriş tokenlarının ~%55–60'ı; sınırlar 40–190
        max_length = max(40, min(190, int(in_tokens * 0.58)))
        min_length = max(20, int(max_length * 0.45))

    # Tokenizer bağlamına göre güvenli sınır belirle (daha konservatif)
    _, tok = _get_summarizer()
    model_ctx = getattr(tok, "model_max_length", 1024) if tok is not None else 1024
    safe_tokens = min(500, max(200, model_ctx - 200))  # daha büyük emniyet payı

    if in_tokens <= safe_tokens:
        return _safe_summarize_chunk(t, max_length, min_length)

    # 1) Parça parça özet
    partials: List[str] = []
    for piece in _chunks_by_tokens(t, max_tokens=safe_tokens, overlap=100):
        ps = _safe_summarize_chunk(piece, max_length, min_length)
        if ps:
            partials.append(ps)

    if not partials:
        return ""

    merged = " ".join(partials)

    # 2) Birleştirilmiş ara özetleri tekrar kısalt (nihai özet)
    final_max = max(80, min(220, int(max_length * 0.9)))
    final_min = max(40, int(final_max * 0.5))
    return _safe_summarize_chunk(merged, final_max, final_min)


def summarize_long_content(text: str, max_length: int = 2000, language: str = "tr", content_mode: str = "auto") -> str:
    """
    Uzun kayıtlar (2-3 saat) için gelişmiş özetleme
    
    Args:
        text: Tam transkripsiyon metni
        max_length: Maksimum özet uzunluğu (kelime)
        language: İçerik dili
        content_mode: meeting, lecture, interview, auto
    """
    if not text or not text.strip():
        return ""
        
    words = text.split()
    word_count = len(words)
    
    # Çok uzun içerik için hierarchical summarization
    if word_count > 5000:  # 5000+ kelime için chunk-based yaklaşım
        
        # İçerik tipine göre chunk boyutunu ayarla
        if content_mode == "lecture":
            chunk_size = 2000  # Dersler için büyük bölümler
        elif content_mode == "meeting":
            chunk_size = 1500  # Toplantılar için orta bölümler  
        else:
            chunk_size = 1800  # Varsayılan
            
        # Metni anlamlı parçalara böl
        chunks = _smart_chunk_text(text, chunk_size)
        
        # Her chunk için özet
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"   📋 Bölüm {i+1}/{len(chunks)} özetleniyor...")
            
            chunk_summary = summarize_text(
                chunk, 
                max_length=int(max_length / len(chunks) * 1.2),  # Her chunk için pay
                language=language
            )
            if chunk_summary:
                chunk_summaries.append(chunk_summary)
        
        # Chunk özetlerini birleştir ve final özet
        if chunk_summaries:
            combined = " ".join(chunk_summaries)
            final_summary = summarize_text(
                combined, 
                max_length=max_length,
                language=language
            )
            
            # İçerik tipine göre özet formatı
            if content_mode == "lecture":
                return _format_lecture_summary(final_summary)
            elif content_mode == "meeting":
                return _format_meeting_summary(final_summary)
            else:
                return final_summary
                
        else:
            return "Özet oluşturulamadı."
    
    else:
        # Daha kısa içerik için normal özetleme
        return summarize_text(text, max_length=max_length, language=language)


def _smart_chunk_text(text: str, target_size: int) -> List[str]:
    """Metni anlamlı noktalarda böl (paragraf, cümle sınırları)"""
    
    # Önce paragraflara böl
    paragraphs = text.split('\n\n')
    if len(paragraphs) == 1:
        # Paragraf yoksa cümlelere böl
        sentences = _sentence_split(text)
        return _group_sentences(sentences, target_size)
    
    # Paragrafları grupla
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para.split())
        
        if current_size + para_size > target_size and current_chunk:
            # Chunk'ı tamamla
            chunks.append(' '.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    # Son chunk'ı ekle
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def _group_sentences(sentences: List[str], target_size: int) -> List[str]:
    """Cümleleri hedef boyutta gruplara topla"""
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence.split())
        
        if current_size + sentence_size > target_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks


def _format_lecture_summary(summary: str) -> str:
    """Ders özetlerini akademik formatta düzenle"""
    return f"📚 DERS ÖZETİ\n\n{summary}\n\n🎯 Ana konular ve öğrenme hedefleri yukarıda özetlenmiştir."


def _format_meeting_summary(summary: str) -> str:
    """Toplantı özetlerini iş formatında düzenle"""  
    return f"🏢 TOPLANTI ÖZETİ\n\n{summary}\n\n📋 Kararlar ve aksiyon maddeleri belirlenen konular dahilindedir."


# ==============================
#  EĞİTİM İÇERİĞİ MODELLERİ
# ==============================

def extract_educational_content(text: str, language: str = "tr") -> Dict:
    """
    Eğitim içeriği çıkarma - dersler için özelleştirilmiş
    
    Returns:
        {
            'topics': List[str],          # Ana konular
            'definitions': List[Dict],    # Tanımlar
            'examples': List[Dict],       # Örnekler
            'questions': List[str],       # Sorular
            'key_points': List[str],      # Önemli noktalar
            'formulas': List[str],        # Formüller/denklemler
            'references': List[str]       # Referanslar
        }
    """
    
    result = {
        'topics': [],
        'definitions': [],
        'examples': [],
        'questions': [],
        'key_points': [],
        'formulas': [],
        'references': []
    }
    
    sentences = _sentence_split(text)
    
    for sentence in sentences:
        sent_lower = sentence.lower()
        
        # Konu başlıklarını tespit et
        if _is_topic_header(sentence, language):
            result['topics'].append(sentence.strip())
        
        # Tanımları tespit et
        definition = _extract_definition(sentence, language)
        if definition:
            result['definitions'].append(definition)
        
        # Örnekleri tespit et
        example = _extract_example(sentence, language)
        if example:
            result['examples'].append(example)
        
        # Soruları tespit et
        if _is_question(sentence, language):
            result['questions'].append(sentence.strip())
        
        # Önemli noktaları tespit et
        if _is_key_point(sentence, language):
            result['key_points'].append(sentence.strip())
        
        # Formül/denklem tespit et
        formula = _extract_formula(sentence)
        if formula:
            result['formulas'].append(formula)
        
        # Referans tespit et
        reference = _extract_reference(sentence, language)
        if reference:
            result['references'].append(reference)
    
    return result


def _is_topic_header(sentence: str, language: str) -> bool:
    """Konu başlığı olup olmadığını kontrol et"""
    
    sent_lower = sentence.lower()
    
    if language == "tr":
        topic_indicators = [
            "konu", "başlık", "bölüm", "ünite", "ders", "konumuz",
            "bugünkü konu", "şimdi", "geçiyoruz", "anlatacağımız"
        ]
    else:
        topic_indicators = [
            "topic", "chapter", "section", "subject", "lesson",
            "today's topic", "now we", "moving to", "next topic"
        ]
    
    return any(indicator in sent_lower for indicator in topic_indicators)


def _extract_definition(sentence: str, language: str) -> Optional[Dict]:
    """Tanım çıkarma"""
    
    sent_lower = sentence.lower()
    
    if language == "tr":
        definition_patterns = [
            r"(.+?)\s+(dir|dır|dur|dür|demektir|anlamına gelir|olarak tanımlanır)",
            r"(.+?)\s+(nedir|ne demek|ne anlama gelir)",
            r"tanım\s*:\s*(.+)",
            r"(.+?)\s+dediğimiz"
        ]
    else:
        definition_patterns = [
            r"(.+?)\s+is\s+(defined as|an?)\s+(.+)",
            r"definition\s*:\s*(.+)",
            r"(.+?)\s+means\s+(.+)",
            r"we\s+call\s+(.+?)\s+(.+)"
        ]
    
    for pattern in definition_patterns:
        match = re.search(pattern, sent_lower, re.IGNORECASE)
        if match:
            if len(match.groups()) >= 2:
                term = match.group(1).strip()
                definition = match.group(2).strip() if len(match.groups()) == 2 else match.group(3).strip()
                return {'term': term, 'definition': definition, 'sentence': sentence}
    
    return None


def _extract_example(sentence: str, language: str) -> Optional[Dict]:
    """Örnek çıkarma"""
    
    sent_lower = sentence.lower()
    
    if language == "tr":
        example_indicators = [
            "örnek", "mesela", "diyelim", "varsayalım", "farz edelim",
            "örneğin", "bir örnek", "örnekler", "misal"
        ]
    else:
        example_indicators = [
            "example", "for instance", "for example", "such as",
            "let's say", "suppose", "consider", "imagine"
        ]
    
    if any(indicator in sent_lower for indicator in example_indicators):
        return {'text': sentence.strip(), 'type': 'example'}
    
    return None


def _is_question(sentence: str, language: str) -> bool:
    """Soru olup olmadığını kontrol et"""
    
    # Soru işareti kontrolü
    if '?' in sentence:
        return True
    
    sent_lower = sentence.lower()
    
    if language == "tr":
        question_starters = [
            "ne", "neden", "nasıl", "nerede", "ne zaman", "kim", "hangi",
            "kaç", "ne kadar", "niye", "niçin"
        ]
    else:
        question_starters = [
            "what", "why", "how", "where", "when", "who", "which",
            "how many", "how much", "can", "is", "are", "do", "does"
        ]
    
    words = sent_lower.split()
    if words and words[0] in question_starters:
        return True
    
    return False


def _is_key_point(sentence: str, language: str) -> bool:
    """Önemli nokta olup olmadığını kontrol et"""
    
    sent_lower = sentence.lower()
    
    if language == "tr":
        key_indicators = [
            "önemli", "dikkat", "not", "unutmayın", "hatırlayın",
            "vurgulamak istiyorum", "özellikle", "kesinlikle", "mutlaka"
        ]
    else:
        key_indicators = [
            "important", "note", "remember", "keep in mind", "crucial",
            "essential", "key point", "significant", "emphasize"
        ]
    
    return any(indicator in sent_lower for indicator in key_indicators)


def _extract_formula(sentence: str) -> Optional[str]:
    """Formül/denklem çıkarma"""
    
    # Matematiksel sembol kontrolü
    math_symbols = ['=', '+', '-', '*', '/', '^', '√', '∑', '∫', 'π', '∞', '∆', '∇']
    
    if any(symbol in sentence for symbol in math_symbols):
        # Basit formül tespiti - daha karmaşık regex'ler eklenebilir
        formula_pattern = r'[A-Za-z0-9\s]*[=+\-*/^√∑∫πα∞∆∇][A-Za-z0-9\s\(\)]*'
        match = re.search(formula_pattern, sentence)
        if match:
            return match.group(0).strip()
    
    return None


def _extract_reference(sentence: str, language: str) -> Optional[str]:
    """Referans çıkarma"""
    
    sent_lower = sentence.lower()
    
    if language == "tr":
        ref_indicators = [
            "kaynak", "referans", "literatür", "kitap", "makale",
            "yazar", "araştırma", "çalışma", "sayfa"
        ]
    else:
        ref_indicators = [
            "reference", "source", "literature", "book", "article",
            "author", "research", "study", "page", "according to"
        ]
    
    if any(indicator in sent_lower for indicator in ref_indicators):
        return sentence.strip()
    
    # ISBN, DOI pattern kontrolü
    isbn_pattern = r'ISBN[:\s]*[\d-]{10,17}'
    doi_pattern = r'DOI[:\s]*[\d\.\/a-zA-Z]+'
    
    if re.search(isbn_pattern, sentence, re.IGNORECASE) or re.search(doi_pattern, sentence, re.IGNORECASE):
        return sentence.strip()
    
    return None


def create_student_summary(text: str, educational_content: Dict, language: str = "tr") -> str:
    """Öğrenciler için özelleştirilmiş özet oluştur"""
    
    if language == "tr":
        summary_parts = ["# 📚 DERS NOTLARI\n"]
        
        # Ana konular
        if educational_content['topics']:
            summary_parts.append("## 🎯 İşlenen Konular")
            for i, topic in enumerate(educational_content['topics'][:5], 1):
                summary_parts.append(f"{i}. {topic}")
            summary_parts.append("")
        
        # Önemli tanımlar
        if educational_content['definitions']:
            summary_parts.append("## 📖 Önemli Tanımlar")
            for definition in educational_content['definitions'][:3]:
                summary_parts.append(f"**{definition['term']}:** {definition['definition']}")
            summary_parts.append("")
        
        # Örnekler
        if educational_content['examples']:
            summary_parts.append("## 💡 Örnekler")
            for i, example in enumerate(educational_content['examples'][:3], 1):
                summary_parts.append(f"{i}. {example['text']}")
            summary_parts.append("")
        
        # Önemli noktalar
        if educational_content['key_points']:
            summary_parts.append("## ⚠️ Önemli Noktalar")
            for point in educational_content['key_points'][:5]:
                summary_parts.append(f"• {point}")
            summary_parts.append("")
        
        # Sorular
        if educational_content['questions']:
            summary_parts.append("## ❓ Derste Sorulan Sorular")
            for question in educational_content['questions'][:3]:
                summary_parts.append(f"• {question}")
            summary_parts.append("")
        
        # Formüller
        if educational_content['formulas']:
            summary_parts.append("## 🧮 Formüller")
            for formula in educational_content['formulas']:
                summary_parts.append(f"• `{formula}`")
            summary_parts.append("")
        
        # Genel özet
        general_summary = summarize_text(text, language=language)
        summary_parts.append("## 📋 Genel Özet")
        summary_parts.append(general_summary)
        
    else:  # İngilizce
        summary_parts = ["# 📚 LECTURE NOTES\n"]
        
        if educational_content['topics']:
            summary_parts.append("## 🎯 Covered Topics")
            for i, topic in enumerate(educational_content['topics'][:5], 1):
                summary_parts.append(f"{i}. {topic}")
            summary_parts.append("")
        
        if educational_content['definitions']:
            summary_parts.append("## 📖 Key Definitions")
            for definition in educational_content['definitions'][:3]:
                summary_parts.append(f"**{definition['term']}:** {definition['definition']}")
            summary_parts.append("")
        
        # Diğer bölümler benzer şekilde İngilizce
        general_summary = summarize_text(text, language=language)
        summary_parts.append("## 📋 General Summary")
        summary_parts.append(general_summary)
    
    return "\n".join(summary_parts)


# ==============================
#  GÖREV / KARAR ÇIKARIMI
# ==============================

_ACTION_WORDS = [
    "yap", "et", "hazırla", "gönder", "kontrol et", "oluştur", "paylaş", "ara", "planla",
    "takip et", "tamamla", "güncelle", "raporla", "topla", "değerlendir", "belirle",
    "kur", "çöz", "ilet", "onayla", "çıkar", "düzenle", "ölç", "sıkıştır", "yaz", "eşleştir"
]
_DEADLINE_WORDS = [
    "bugün", "yarın", "haftaya", "son tarih", "deadline", "gün sonuna",
    "hafta sonuna", "pazartesi", "salı", "çarşamba", "perşembe", "cuma"
]
_EMIR_KIPI = re.compile(r"(alım|elim|ınız|iniz)$", re.IGNORECASE)
_NAME_CALL = re.compile(r"^(ali|ayşe|mehmet|zeynep)\b[,:]?", re.IGNORECASE)

def extract_tasks(text: str, language: str = "tr") -> List[str]:
    """
    Emir kipleri, aksiyon kelimeleri, zaman/dedline sinyalleri ve isimle çağrı paternine
    göre cümle bazlı görev listesi çıkarır. Tekilleştirir.
    """
    t = normalize_transcript(text)
    tasks: List[str] = []
    seen: set[str] = set()

    # Dile özel aksiyon kelimeleri
    action_words = _ACTION_WORDS
    if language == "en":
        action_words.extend([
            "do", "make", "create", "send", "check", "update", "prepare", "call", 
            "plan", "follow up", "complete", "report", "collect", "evaluate"
        ])
    elif language in ["de", "fr", "es", "it"]:
        # Diğer Avrupa dilleri için temel kelimeler
        if language == "de":
            action_words.extend(["machen", "erstellen", "senden", "prüfen"])
        elif language == "fr":
            action_words.extend(["faire", "créer", "envoyer", "vérifier"])

    for sent in _sentence_split(t):
        s = sent.strip()
        if not s:
            continue
        ls = s.lower()

        rule1 = any(w in ls for w in action_words)
        rule2 = "gerekiyor" in ls or "lazım" in ls or bool(_EMIR_KIPI.search(ls))
        rule3 = any(w in ls for w in _DEADLINE_WORDS)
        rule4 = bool(_NAME_CALL.match(ls))  # "Ali," "Ayşe:" vb.

        # İngilizce için ek kurallar
        if language == "en":
            rule2 = rule2 or "need to" in ls or "should" in ls or "must" in ls
            rule3 = rule3 or any(w in ls for w in ["today", "tomorrow", "deadline", "by"])

        if rule1 or rule2 or rule3 or rule4:
            key = _canon(ls)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(s)

    return tasks

_DECISION_WORDS = [
    "karar", "mutabık", "onaylandı", "kabul edildi", "netleştirildi", "belirlendi",
    "seçildi", "uygulanacak", "devreye alınacak", "sürüm tarihi", "planlandı"
]

def extract_decisions(text: str, language: str = "tr") -> List[str]:
    """
    Karar kelimelerinin geçtiği cümleleri yakalayıp tekilleştirir.
    """
    t = normalize_transcript(text)
    decisions: List[str] = []
    seen: set[str] = set()

    # Dile özel karar kelimeleri
    decision_words = _DECISION_WORDS
    if language == "en":
        decision_words.extend([
            "decided", "agreed", "approved", "determined", "resolved", "concluded",
            "settled", "finalized", "confirmed", "established"
        ])
    elif language == "de":
        decision_words.extend([
            "entschieden", "beschlossen", "vereinbart", "bestimmt"
        ])
    elif language == "fr":
        decision_words.extend([
            "décidé", "convenu", "approuvé", "déterminé"
        ])

    for sent in _sentence_split(t):
        s = sent.strip()
        if not s:
            continue
        ls = s.lower()
        if any(w in ls for w in decision_words):
            key = _canon(ls)
            if key in seen:
                continue
            seen.add(key)
            decisions.append(s)

    return decisions


# ==============================
#  PENCERE ÖZETLERİ (uzun kayıtlar)
# ==============================

def summarize_by_windows(segments: List[Dict], window_sec: int = 600, language: str = "tr") -> List[Dict]:
    """
    Zaman pencerelerine göre (örn. 10 dk) özet üretir.
    segments: [{"start": float, "end": float, "text": str}, ...]
    return:  [{"start": sec, "end": sec, "summary": str}]
    """
    if not segments:
        return []

    windows: List[Dict] = []
    bucket: List[Dict] = []
    start, end = 0.0, float(window_sec)

    for s in segments:
        # segment pencereyi aşıyorsa flush et
        while s["end"] > end:
            text = " ".join(x["text"] for x in bucket).strip()
            windows.append({
                "start": start,
                "end": end,
                "summary": summarize_text(text, language=language) if text else ""
            })
            start, end, bucket = end, end + window_sec, []
        bucket.append(s)

    # son kuyruk
    if bucket:
        last_end = segments[-1]["end"]
        text = " ".join(x["text"] for x in bucket).strip()
        windows.append({
            "start": start,
            "end": last_end,
            "summary": summarize_text(text, language=language) if text else ""
        })

    return windows
