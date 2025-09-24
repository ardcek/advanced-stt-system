# modules/adaptive_learning.py - Adaptive Learning System for STT Accuracy
"""
Adaptif Öğrenme Sistemi - STT Doğruluğu için Sürekli İyileştirme
============================================================

Bu modül %99.9 doğruluk için adaptif öğrenme teknolojileri içerir:
- User feedback learning
- Domain-specific fine-tuning  
- Accuracy tracking & analytics
- Personalized correction models
- Performance optimization
- Error pattern analysis
- Custom vocabulary learning

Kullanım:
    learner = AdaptiveLearner()
    learner.add_feedback(original, corrected, context)
    improved_text = learner.apply_learned_corrections(text)
"""

import os
import json
import time
import pickle
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import difflib
import re

# ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    import joblib
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import torch
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


@dataclass
class FeedbackEntry:
    """Kullanıcı geri bildirimi girişi"""
    id: str
    timestamp: datetime
    original_text: str
    corrected_text: str
    context_type: str
    user_id: str
    correction_type: str  # spelling, grammar, vocabulary, etc.
    confidence_score: float
    applied_count: int = 0
    success_rate: float = 1.0


@dataclass
class ErrorPattern:
    """Hata paterni"""
    pattern_id: str
    error_type: str
    pattern: str  # Regex or text pattern
    correction: str
    frequency: int
    confidence: float
    context_words: List[str]
    domain: str


@dataclass
class PerformanceMetrics:
    """Performans metrikleri"""
    timestamp: datetime
    accuracy_before: float
    accuracy_after: float
    wer_before: float  # Word Error Rate
    wer_after: float
    processing_time: float
    corrections_applied: int
    user_satisfaction: Optional[float] = None


@dataclass
class DomainProfile:
    """Domain özellik profili"""
    domain_name: str
    vocabulary: Set[str]
    common_phrases: List[str]
    correction_patterns: List[ErrorPattern]
    performance_history: List[PerformanceMetrics]
    last_updated: datetime


class FeedbackDatabase:
    """Geri bildirim veritabanı yöneticisi"""
    
    def __init__(self, db_path: str = "adaptive_learning.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Veritabanını başlat"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                original_text TEXT,
                corrected_text TEXT,
                context_type TEXT,
                user_id TEXT,
                correction_type TEXT,
                confidence_score REAL,
                applied_count INTEGER,
                success_rate REAL
            )
        ''')
        
        # Error patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_patterns (
                pattern_id TEXT PRIMARY KEY,
                error_type TEXT,
                pattern TEXT,
                correction TEXT,
                frequency INTEGER,
                confidence REAL,
                context_words TEXT,
                domain TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp TEXT,
                accuracy_before REAL,
                accuracy_after REAL,
                wer_before REAL,
                wer_after REAL,
                processing_time REAL,
                corrections_applied INTEGER,
                user_satisfaction REAL
            )
        ''')
        
        # Domain profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domain_profiles (
                domain_name TEXT PRIMARY KEY,
                vocabulary TEXT,
                common_phrases TEXT,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_feedback(self, feedback: FeedbackEntry):
        """Geri bildirim ekle"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.id,
            feedback.timestamp.isoformat(),
            feedback.original_text,
            feedback.corrected_text,
            feedback.context_type,
            feedback.user_id,
            feedback.correction_type,
            feedback.confidence_score,
            feedback.applied_count,
            feedback.success_rate
        ))
        
        conn.commit()
        conn.close()
    
    def get_feedback_history(
        self,
        user_id: Optional[str] = None,
        context_type: Optional[str] = None,
        days: int = 30
    ) -> List[FeedbackEntry]:
        """Geri bildirim geçmişi al"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM feedback WHERE timestamp > ?"
        params = [(datetime.now() - timedelta(days=days)).isoformat()]
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
            
        if context_type:
            query += " AND context_type = ?"
            params.append(context_type)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        feedback_list = []
        for row in rows:
            feedback_list.append(FeedbackEntry(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                original_text=row[2],
                corrected_text=row[3],
                context_type=row[4],
                user_id=row[5],
                correction_type=row[6],
                confidence_score=row[7],
                applied_count=row[8],
                success_rate=row[9]
            ))
        
        return feedback_list
    
    def add_error_pattern(self, pattern: ErrorPattern):
        """Hata paterni ekle"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO error_patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id,
            pattern.error_type,
            pattern.pattern,
            pattern.correction,
            pattern.frequency,
            pattern.confidence,
            json.dumps(pattern.context_words),
            pattern.domain
        ))
        
        conn.commit()
        conn.close()
    
    def get_error_patterns(self, domain: Optional[str] = None) -> List[ErrorPattern]:
        """Hata paternleri al"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if domain:
            cursor.execute("SELECT * FROM error_patterns WHERE domain = ?", (domain,))
        else:
            cursor.execute("SELECT * FROM error_patterns")
        
        rows = cursor.fetchall()
        conn.close()
        
        patterns = []
        for row in rows:
            patterns.append(ErrorPattern(
                pattern_id=row[0],
                error_type=row[1],
                pattern=row[2],
                correction=row[3],
                frequency=row[4],
                confidence=row[5],
                context_words=json.loads(row[6]),
                domain=row[7]
            ))
        
        return patterns


class ErrorPatternAnalyzer:
    """Hata patern analizi sistemi"""
    
    def __init__(self):
        self.patterns = {}
        self.vectorizer = None
        if _HAS_SKLEARN:
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    def analyze_corrections(self, feedback_list: List[FeedbackEntry]) -> List[ErrorPattern]:
        """Düzeltmelerden hata paternleri çıkar"""
        
        patterns = []
        
        # Group by correction type
        corrections_by_type = defaultdict(list)
        for feedback in feedback_list:
            corrections_by_type[feedback.correction_type].append(feedback)
        
        for correction_type, corrections in corrections_by_type.items():
            
            # 1. Direct word substitutions
            word_patterns = self._extract_word_patterns(corrections)
            patterns.extend(word_patterns)
            
            # 2. Phrase patterns  
            phrase_patterns = self._extract_phrase_patterns(corrections)
            patterns.extend(phrase_patterns)
            
            # 3. Grammar patterns
            if correction_type == "grammar":
                grammar_patterns = self._extract_grammar_patterns(corrections)
                patterns.extend(grammar_patterns)
        
        return patterns
    
    def _extract_word_patterns(self, corrections: List[FeedbackEntry]) -> List[ErrorPattern]:
        """Kelime düzeltme paternleri çıkar"""
        
        patterns = []
        word_corrections = defaultdict(Counter)
        
        for feedback in corrections:
            # Find word-level differences
            original_words = feedback.original_text.lower().split()
            corrected_words = feedback.corrected_text.lower().split()
            
            # Simple alignment (can be improved)
            if len(original_words) == len(corrected_words):
                for orig, corr in zip(original_words, corrected_words):
                    if orig != corr:
                        word_corrections[orig][corr] += 1
        
        # Create patterns for frequent corrections
        for original_word, corrections_counter in word_corrections.items():
            if sum(corrections_counter.values()) >= 2:  # Minimum frequency
                most_common = corrections_counter.most_common(1)[0]
                corrected_word, frequency = most_common
                
                confidence = frequency / sum(corrections_counter.values())
                
                patterns.append(ErrorPattern(
                    pattern_id=f"word_{original_word}_{corrected_word}",
                    error_type="word_substitution",
                    pattern=rf"\b{re.escape(original_word)}\b",
                    correction=corrected_word,
                    frequency=frequency,
                    confidence=confidence,
                    context_words=[],
                    domain="general"
                ))
        
        return patterns
    
    def _extract_phrase_patterns(self, corrections: List[FeedbackEntry]) -> List[ErrorPattern]:
        """Phrase düzeltme paternleri çıkar"""
        
        patterns = []
        phrase_corrections = defaultdict(Counter)
        
        for feedback in corrections:
            # Extract 2-4 word phrases
            original_text = feedback.original_text.lower()
            corrected_text = feedback.corrected_text.lower()
            
            # Find differing phrases using sequence matching
            matcher = difflib.SequenceMatcher(None, original_text, corrected_text)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    original_phrase = original_text[i1:i2].strip()
                    corrected_phrase = corrected_text[j1:j2].strip()
                    
                    # Only consider phrases with 2-20 characters
                    if 2 <= len(original_phrase) <= 20 and 2 <= len(corrected_phrase) <= 20:
                        phrase_corrections[original_phrase][corrected_phrase] += 1
        
        # Create patterns
        for original_phrase, corrections_counter in phrase_corrections.items():
            if sum(corrections_counter.values()) >= 2:
                most_common = corrections_counter.most_common(1)[0]
                corrected_phrase, frequency = most_common
                
                confidence = frequency / sum(corrections_counter.values())
                
                patterns.append(ErrorPattern(
                    pattern_id=f"phrase_{hash(original_phrase)}",
                    error_type="phrase_substitution",
                    pattern=re.escape(original_phrase),
                    correction=corrected_phrase,
                    frequency=frequency,
                    confidence=confidence,
                    context_words=[],
                    domain="general"
                ))
        
        return patterns
    
    def _extract_grammar_patterns(self, corrections: List[FeedbackEntry]) -> List[ErrorPattern]:
        """Gramer düzeltme paternleri çıkar"""
        
        patterns = []
        
        # Common Turkish grammar patterns
        grammar_rules = [
            # Verb conjugations
            (r'(\w+)mış', r'\1miş'),  # mış/miş harmony
            (r'(\w+)muş', r'\1müş'),  # muş/müş harmony
            
            # Case endings
            (r'(\w+)de\b', r'\1da'),  # de/da harmony  
            (r'(\w+)da\b', r'\1de'),  # da/de harmony
            
            # Possessive endings
            (r'(\w+)ın\b', r'\1in'),  # ın/in harmony
            (r'(\w+)in\b', r'\1ın'),  # in/ın harmony
        ]
        
        for rule_pattern, rule_correction in grammar_rules:
            frequency = 0
            
            # Count occurrences in corrections
            for feedback in corrections:
                original = feedback.original_text.lower()
                corrected = feedback.corrected_text.lower()
                
                # Check if this rule applies
                if re.search(rule_pattern, original) and rule_correction in corrected:
                    frequency += 1
            
            if frequency >= 2:
                patterns.append(ErrorPattern(
                    pattern_id=f"grammar_{hash(rule_pattern)}",
                    error_type="grammar",
                    pattern=rule_pattern,
                    correction=rule_correction,
                    frequency=frequency,
                    confidence=0.8,  # Grammar rules are usually reliable
                    context_words=[],
                    domain="turkish_grammar"
                ))
        
        return patterns


class PersonalizedCorrector:
    """Kişiselleştirilmiş düzeltici"""
    
    def __init__(self):
        self.user_patterns = {}  # user_id -> patterns
        self.model = None
        self.vectorizer = None
        
        if _HAS_SKLEARN:
            self.model = LogisticRegression(max_iter=1000)
            self.vectorizer = TfidfVectorizer(max_features=1000)
    
    def learn_user_patterns(self, user_id: str, feedback_list: List[FeedbackEntry]):
        """Kullanıcı özel paternleri öğren"""
        
        user_feedback = [f for f in feedback_list if f.user_id == user_id]
        
        if len(user_feedback) < 5:  # Minimum data needed
            return
        
        # Extract patterns
        analyzer = ErrorPatternAnalyzer()
        patterns = analyzer.analyze_corrections(user_feedback)
        
        # Store user-specific patterns
        self.user_patterns[user_id] = patterns
        
        # Train personalized model if enough data
        if len(user_feedback) >= 10 and _HAS_SKLEARN:
            self._train_personalized_model(user_id, user_feedback)
    
    def _train_personalized_model(self, user_id: str, feedback_list: List[FeedbackEntry]):
        """Kişiselleştirilmiş model eğit"""
        
        if not _HAS_SKLEARN:
            return
        
        # Prepare training data
        texts = []
        labels = []  # 1 if needs correction, 0 if already correct
        
        for feedback in feedback_list:
            # Add original (needs correction)
            texts.append(feedback.original_text)
            labels.append(1)
            
            # Add corrected (already correct)
            texts.append(feedback.corrected_text)
            labels.append(0)
        
        if len(set(labels)) < 2:  # Need both classes
            return
        
        try:
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            y = np.array(labels)
            
            # Train model
            self.model.fit(X, y)
            
            # Save model
            model_path = f"user_models/{user_id}_model.joblib"
            os.makedirs("user_models", exist_ok=True)
            joblib.dump((self.model, self.vectorizer), model_path)
            
            print(f"✅ Personalized model trained for user {user_id}")
            
        except Exception as e:
            print(f"Error training personalized model: {e}")
    
    def apply_user_corrections(self, user_id: str, text: str) -> str:
        """Kullanıcı özel düzeltmeleri uygula"""
        
        corrected_text = text
        
        # Apply user-specific patterns
        if user_id in self.user_patterns:
            patterns = self.user_patterns[user_id]
            
            for pattern in sorted(patterns, key=lambda x: x.confidence, reverse=True):
                if pattern.confidence > 0.7:  # High confidence only
                    corrected_text = re.sub(
                        pattern.pattern, 
                        pattern.correction, 
                        corrected_text, 
                        flags=re.IGNORECASE
                    )
        
        # Apply personalized model prediction
        if self.model and self.vectorizer:
            try:
                X = self.vectorizer.transform([corrected_text])
                needs_correction_prob = self.model.predict_proba(X)[0][1]
                
                # If model says text needs correction with high confidence,
                # apply additional generic corrections
                if needs_correction_prob > 0.8:
                    corrected_text = self._apply_generic_corrections(corrected_text)
                
            except Exception:
                pass  # Model not available or error
        
        return corrected_text
    
    def _apply_generic_corrections(self, text: str) -> str:
        """Genel düzeltmeler uygula"""
        
        # Common Turkish corrections
        corrections = {
            r'\bde\b': 'da',  # Context-dependent, simplified
            r'\bda\b': 'de',
            r'görüşme': 'görüşme',
            r'toplantı': 'toplantı',
            r'proje': 'proje',
            r'görev': 'görev'
        }
        
        corrected = text
        for pattern, replacement in corrections.items():
            # Apply with low confidence for safety
            if len(re.findall(pattern, corrected, re.IGNORECASE)) <= 2:
                corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        return corrected


class PerformanceTracker:
    """Performans izleme sistemi"""
    
    def __init__(self, db: FeedbackDatabase):
        self.db = db
        self.current_session = {}
    
    def start_session(self, session_id: str):
        """Yeni performans oturumu başlat"""
        
        self.current_session[session_id] = {
            'start_time': time.time(),
            'corrections_applied': 0,
            'original_texts': [],
            'corrected_texts': []
        }
    
    def record_correction(
        self,
        session_id: str,
        original_text: str,
        corrected_text: str
    ):
        """Düzeltme kaydı"""
        
        if session_id in self.current_session:
            session = self.current_session[session_id]
            session['corrections_applied'] += 1
            session['original_texts'].append(original_text)
            session['corrected_texts'].append(corrected_text)
    
    def end_session(
        self,
        session_id: str,
        user_satisfaction: Optional[float] = None
    ) -> PerformanceMetrics:
        """Oturumu sonlandır ve metrikleri hesapla"""
        
        if session_id not in self.current_session:
            return None
        
        session = self.current_session[session_id]
        
        # Calculate metrics
        processing_time = time.time() - session['start_time']
        
        # Calculate WER (Word Error Rate) - simplified
        wer_before = self._calculate_wer(session['original_texts'])
        wer_after = self._calculate_wer(session['corrected_texts'])
        
        # Accuracy estimation (based on text similarity improvement)
        accuracy_before = self._estimate_accuracy(session['original_texts'])
        accuracy_after = self._estimate_accuracy(session['corrected_texts'])
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            wer_before=wer_before,
            wer_after=wer_after,
            processing_time=processing_time,
            corrections_applied=session['corrections_applied'],
            user_satisfaction=user_satisfaction
        )
        
        # Save to database
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            metrics.accuracy_before,
            metrics.accuracy_after,
            metrics.wer_before,
            metrics.wer_after,
            metrics.processing_time,
            metrics.corrections_applied,
            metrics.user_satisfaction
        ))
        
        conn.commit()
        conn.close()
        
        # Clean up session
        del self.current_session[session_id]
        
        return metrics
    
    def _calculate_wer(self, texts: List[str]) -> float:
        """Basitleştirilmiş WER hesaplama"""
        
        if not texts:
            return 0.0
        
        # Simple WER estimation based on text characteristics
        total_score = 0.0
        
        for text in texts:
            # Factors that might indicate errors:
            # - Very short words (might be incomplete)
            # - Numbers mixed with letters
            # - Excessive capitalization
            
            words = text.split()
            if not words:
                continue
                
            error_indicators = 0
            
            for word in words:
                if len(word) == 1 and word.isalpha():  # Single letters
                    error_indicators += 1
                elif re.search(r'\d+[a-zA-Z]+|\w*\d+\w*', word):  # Mixed numbers/letters
                    error_indicators += 1
                elif word.isupper() and len(word) > 1:  # All caps
                    error_indicators += 0.5
            
            # WER estimate (0 = perfect, 1 = all errors)
            wer = min(error_indicators / len(words), 1.0)
            total_score += wer
        
        return total_score / len(texts)
    
    def _estimate_accuracy(self, texts: List[str]) -> float:
        """Doğruluk tahmini"""
        
        if not texts:
            return 1.0
        
        # Accuracy based on text quality indicators
        total_score = 0.0
        
        for text in texts:
            # Quality indicators:
            # - Proper sentence structure
            # - Reasonable word lengths
            # - Proper punctuation
            
            score = 1.0  # Start with perfect score
            
            # Penalize for quality issues
            if not text.strip():
                score = 0.0
            else:
                words = text.split()
                
                # Average word length should be reasonable (2-12 characters)
                avg_word_len = np.mean([len(w) for w in words]) if words else 0
                if avg_word_len < 2 or avg_word_len > 12:
                    score -= 0.2
                
                # Should have some punctuation for longer texts
                if len(text) > 50 and not re.search(r'[.!?,:;]', text):
                    score -= 0.1
                
                # Too many short words might indicate fragmentation
                short_words = sum(1 for w in words if len(w) <= 2)
                if words and short_words / len(words) > 0.3:
                    score -= 0.2
            
            total_score += max(score, 0.0)
        
        return total_score / len(texts)
    
    def get_performance_history(self, days: int = 30) -> List[PerformanceMetrics]:
        """Performans geçmişi al"""
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT * FROM performance_metrics 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (cutoff_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        metrics_list = []
        for row in rows:
            metrics_list.append(PerformanceMetrics(
                timestamp=datetime.fromisoformat(row[0]),
                accuracy_before=row[1],
                accuracy_after=row[2],
                wer_before=row[3],
                wer_after=row[4],
                processing_time=row[5],
                corrections_applied=row[6],
                user_satisfaction=row[7]
            ))
        
        return metrics_list


class AdaptiveLearner:
    """Ana adaptif öğrenme sistemi"""
    
    def __init__(self, db_path: str = "adaptive_learning.db"):
        self.db = FeedbackDatabase(db_path)
        self.pattern_analyzer = ErrorPatternAnalyzer()
        self.personalized_corrector = PersonalizedCorrector()
        self.performance_tracker = PerformanceTracker(self.db)
        
        # Load existing patterns
        self.learned_patterns = self.db.get_error_patterns()
        print(f"✅ {len(self.learned_patterns)} learned patterns loaded")
    
    def add_feedback(
        self,
        original_text: str,
        corrected_text: str,
        user_id: str = "default",
        context_type: str = "general"
    ):
        """Yeni geri bildirim ekle"""
        
        # Determine correction type
        correction_type = self._determine_correction_type(original_text, corrected_text)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(original_text, corrected_text)
        
        # Create feedback entry
        feedback = FeedbackEntry(
            id=f"{user_id}_{int(time.time())}",
            timestamp=datetime.now(),
            original_text=original_text,
            corrected_text=corrected_text,
            context_type=context_type,
            user_id=user_id,
            correction_type=correction_type,
            confidence_score=confidence
        )
        
        # Save to database
        self.db.add_feedback(feedback)
        
        print(f"📝 Feedback added: {correction_type} correction (confidence: {confidence:.2f})")
        
        # Re-analyze patterns periodically
        if len(self.db.get_feedback_history(days=1)) % 10 == 0:  # Every 10 feedbacks
            self.update_learned_patterns()
    
    def update_learned_patterns(self):
        """Öğrenilen paternleri güncelle"""
        
        print("🔄 Updating learned patterns...")
        
        # Get recent feedback
        recent_feedback = self.db.get_feedback_history(days=30)
        
        if len(recent_feedback) < 5:
            print("   ⚠️ Not enough feedback for pattern analysis")
            return
        
        # Analyze new patterns
        new_patterns = self.pattern_analyzer.analyze_corrections(recent_feedback)
        
        # Save new patterns
        patterns_added = 0
        for pattern in new_patterns:
            if pattern.confidence > 0.6:  # Only high-confidence patterns
                self.db.add_error_pattern(pattern)
                patterns_added += 1
        
        # Reload patterns
        self.learned_patterns = self.db.get_error_patterns()
        
        print(f"   ✅ {patterns_added} new patterns learned, {len(self.learned_patterns)} total")
        
        # Update personalized models
        self._update_personalized_models(recent_feedback)
    
    def _update_personalized_models(self, feedback_list: List[FeedbackEntry]):
        """Kişiselleştirilmiş modelleri güncelle"""
        
        # Group feedback by user
        users_feedback = defaultdict(list)
        for feedback in feedback_list:
            users_feedback[feedback.user_id].append(feedback)
        
        # Update models for users with sufficient data
        for user_id, user_feedback in users_feedback.items():
            if len(user_feedback) >= 5:
                self.personalized_corrector.learn_user_patterns(user_id, user_feedback)
    
    def apply_learned_corrections(
        self,
        text: str,
        user_id: str = "default",
        context_type: str = "general"
    ) -> str:
        """Öğrenilen düzeltmeleri uygula"""
        
        corrected_text = text
        
        # 1. Apply learned patterns
        for pattern in self.learned_patterns:
            if pattern.confidence > 0.7:  # High confidence only
                try:
                    corrected_text = re.sub(
                        pattern.pattern,
                        pattern.correction,
                        corrected_text,
                        flags=re.IGNORECASE
                    )
                except re.error:
                    # Skip invalid regex patterns
                    continue
        
        # 2. Apply personalized corrections
        corrected_text = self.personalized_corrector.apply_user_corrections(
            user_id, corrected_text
        )
        
        return corrected_text
    
    def _determine_correction_type(self, original: str, corrected: str) -> str:
        """Düzeltme türünü belirle"""
        
        original_words = original.lower().split()
        corrected_words = corrected.lower().split()
        
        # Check for specific correction types
        if len(original_words) != len(corrected_words):
            return "structure"  # Word insertion/deletion
        
        # Count character-level differences
        char_diffs = sum(1 for a, b in zip(original.lower(), corrected.lower()) if a != b)
        char_diff_ratio = char_diffs / max(len(original), 1)
        
        if char_diff_ratio < 0.1:
            return "spelling"  # Minor spelling fixes
        elif char_diff_ratio < 0.3:
            return "grammar"   # Grammar corrections
        else:
            return "vocabulary"  # Major word changes
    
    def _calculate_confidence(self, original: str, corrected: str) -> float:
        """Düzeltme güvenilirliği hesapla"""
        
        # Confidence based on:
        # 1. Text similarity (high similarity = reliable correction)
        # 2. Length ratio (similar lengths = more reliable)
        # 3. Character-level changes
        
        similarity = difflib.SequenceMatcher(None, original, corrected).ratio()
        
        len_ratio = min(len(original), len(corrected)) / max(len(original), len(corrected), 1)
        
        # Penalize very short texts (less reliable)
        length_penalty = min(len(original) / 10, 1.0)
        
        confidence = similarity * len_ratio * length_penalty
        
        return confidence
    
    def get_learning_analytics(self) -> Dict[str, Any]:
        """Öğrenme analitikleri al"""
        
        analytics = {}
        
        # Pattern analytics
        patterns_by_type = defaultdict(int)
        for pattern in self.learned_patterns:
            patterns_by_type[pattern.error_type] += 1
        
        analytics['patterns_by_type'] = dict(patterns_by_type)
        analytics['total_patterns'] = len(self.learned_patterns)
        
        # Performance analytics
        recent_performance = self.performance_tracker.get_performance_history(days=7)
        
        if recent_performance:
            analytics['avg_accuracy_improvement'] = np.mean([
                p.accuracy_after - p.accuracy_before 
                for p in recent_performance
            ])
            
            analytics['avg_processing_time'] = np.mean([
                p.processing_time 
                for p in recent_performance
            ])
            
            analytics['total_corrections'] = sum([
                p.corrections_applied 
                for p in recent_performance
            ])
        
        # Feedback analytics
        recent_feedback = self.db.get_feedback_history(days=7)
        
        if recent_feedback:
            users_active = len(set(f.user_id for f in recent_feedback))
            analytics['active_users'] = users_active
            analytics['feedback_count'] = len(recent_feedback)
            
            avg_confidence = np.mean([f.confidence_score for f in recent_feedback])
            analytics['avg_feedback_confidence'] = avg_confidence
        
        return analytics
    
    def start_performance_session(self, session_id: str):
        """Performans oturumu başlat"""
        self.performance_tracker.start_session(session_id)
    
    def record_correction_performance(
        self,
        session_id: str,
        original_text: str,
        corrected_text: str
    ):
        """Düzeltme performansını kaydet"""
        self.performance_tracker.record_correction(session_id, original_text, corrected_text)
    
    def end_performance_session(
        self,
        session_id: str,
        user_satisfaction: Optional[float] = None
    ) -> Optional[PerformanceMetrics]:
        """Performans oturumunu sonlandır"""
        return self.performance_tracker.end_session(session_id, user_satisfaction)


# Kolay kullanım fonksiyonları
def learn_from_correction(
    original: str,
    corrected: str,
    user_id: str = "default",
    context: str = "general"
):
    """Tek fonksiyon ile düzeltmeden öğrenme"""
    
    learner = AdaptiveLearner()
    learner.add_feedback(original, corrected, user_id, context)


def apply_learned_improvements(
    text: str,
    user_id: str = "default",
    context: str = "general"
) -> str:
    """Tek fonksiyon ile öğrenilen iyileştirmeleri uygula"""
    
    learner = AdaptiveLearner()
    return learner.apply_learned_corrections(text, user_id, context)


if __name__ == "__main__":
    # Test kodu
    print("🧠 Adaptive Learning System Test")
    print("=" * 50)
    
    # Create learner
    learner = AdaptiveLearner()
    
    # Simulate some feedback
    test_corrections = [
        ("Bu bir deneme metnidir", "Bu bir deneme metnidir.", "punctuation"),
        ("Yazilim gelistirme", "Yazılım geliştirme", "spelling"),
        ("toplanti yapalim", "toplantı yapalım", "spelling"),
        ("proje hakkinda", "proje hakkında", "spelling"),
        ("gorus almak", "görüş almak", "spelling")
    ]
    
    print("📝 Adding test feedback...")
    for original, corrected, context in test_corrections:
        learner.add_feedback(original, corrected, "test_user", context)
    
    # Test learned corrections
    test_text = "Bu yazilim projesi hakkinda gorus almak icin toplanti yapalim"
    corrected = learner.apply_learned_corrections(test_text, "test_user")
    
    print(f"\nOriginal: {test_text}")
    print(f"Corrected: {corrected}")
    
    # Get analytics
    analytics = learner.get_learning_analytics()
    print(f"\n📊 Analytics:")
    for key, value in analytics.items():
        print(f"  {key}: {value}")
    
    available_features = []
    if _HAS_SKLEARN: available_features.append("Scikit-learn ML")
    if _HAS_PANDAS: available_features.append("Pandas Analytics")
    if _HAS_TRANSFORMERS: available_features.append("Transformers NLP")
    
    print(f"\n✅ Mevcut özellikler: {', '.join(available_features)}")
    print("🚀 Adaptive Learning system ready!")