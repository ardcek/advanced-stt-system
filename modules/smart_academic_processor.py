"""
Smart Academic Processing Module
Öğrenci derslerine özel ultra-gelişmiş transkripsiyon sistemi

Bu modül, öğrencilerin 2-3 saatlik ders kayıtlarını mükemmel doğrulukla
yazıya dökebilmesi için özel olarak tasarlanmıştır.

Features:
- Hoca konuşma paternlerini analiz eder
- Akademik terminoloji database'i
- Ders yapısını anlar (giriş, ana konular, sonuç)
- Teknik terimler için özel işleme
- Öğrenci notları formatında çıktı

Made by Mehmet Arda Çekiç © 2025
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import sqlite3
import re
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter
import openai
from transformers import pipeline, AutoTokenizer, AutoModel
import spacy
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AcademicTerm:
    """Academic terminology data structure"""
    term: str
    definition: str
    subject: str
    difficulty_level: int  # 1-5
    pronunciation: str
    related_terms: List[str]
    frequency_score: float

@dataclass
class LectureSegment:
    """Lecture segment with academic structure"""
    start_time: float
    end_time: float
    content: str
    segment_type: str  # introduction, main_topic, example, conclusion, q_and_a
    confidence: float
    key_terms: List[str]
    summary: str
    importance_score: float

@dataclass
class ProfessorSpeechPattern:
    """Professor speech pattern analysis"""
    speech_rate: float
    pause_patterns: List[float]
    emphasis_words: List[str]
    filler_words: List[str]
    speaking_style: str  # lecture, discussion, presentation
    clarity_score: float

class SmartAcademicProcessor:
    """
    Ultra-advanced academic transcription processor
    Specifically designed for student lecture recordings
    """
    
    def __init__(self):
        self.academic_terms_db = self._initialize_academic_database()
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # Academic-specific models
        self.academic_classifier = None
        self.technical_term_extractor = None
        
        # Subject-specific terminology databases
        self.subject_databases = {
            "mathematics": self._load_math_terms(),
            "physics": self._load_physics_terms(),
            "chemistry": self._load_chemistry_terms(),
            "biology": self._load_biology_terms(),
            "computer_science": self._load_cs_terms(),
            "engineering": self._load_engineering_terms(),
            "medicine": self._load_medical_terms(),
            "economics": self._load_economics_terms(),
            "history": self._load_history_terms(),
            "literature": self._load_literature_terms()
        }
        
        # Lecture structure patterns
        self.lecture_patterns = {
            "introduction": [
                "today we will cover", "in this lecture", "let's begin with",
                "first, let's discuss", "overview of", "introduction to"
            ],
            "main_content": [
                "moving on to", "next topic", "another important point",
                "let's examine", "consider the following", "for example"
            ],
            "conclusion": [
                "in summary", "to conclude", "in conclusion", "to wrap up",
                "key takeaways", "important points to remember"
            ],
            "questions": [
                "any questions?", "do you understand?", "is this clear?",
                "questions about", "let me clarify", "good question"
            ]
        }
        
        logger.info("Smart Academic Processor initialized successfully")
    
    def _initialize_academic_database(self) -> sqlite3.Connection:
        """Initialize academic terminology database"""
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE academic_terms (
                    id INTEGER PRIMARY KEY,
                    term TEXT UNIQUE NOT NULL,
                    definition TEXT,
                    subject TEXT,
                    difficulty_level INTEGER,
                    pronunciation TEXT,
                    related_terms TEXT,
                    frequency_score REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE lecture_patterns (
                    id INTEGER PRIMARY KEY,
                    pattern TEXT,
                    category TEXT,
                    confidence REAL
                )
            """)
            
            # Load pre-built academic terms
            self._populate_academic_database(conn)
            
            conn.commit()
            return conn
            
        except Exception as e:
            logger.error(f"Failed to initialize academic database: {e}")
            raise
    
    def _populate_academic_database(self, conn: sqlite3.Connection):
        """Populate database with academic terms"""
        academic_terms = [
            # Mathematics
            ("derivative", "Rate of change of a function", "mathematics", 3, "dɪˈrɪvətɪv", "integral,calculus,function", 0.95),
            ("integral", "Antiderivative of a function", "mathematics", 3, "ˈɪntɪɡrəl", "derivative,calculus,area", 0.92),
            ("logarithm", "Power to which base must be raised", "mathematics", 2, "ˈlɒɡərɪðəm", "exponential,power,base", 0.88),
            ("polynomial", "Expression with variables and coefficients", "mathematics", 2, "pɒlɪˈnəʊmɪəl", "function,variable,coefficient", 0.85),
            
            # Physics
            ("quantum mechanics", "Physics of atomic and subatomic particles", "physics", 5, "ˈkwɒntəm mɪˈkænɪks", "wave function,uncertainty principle", 0.98),
            ("thermodynamics", "Study of heat and temperature", "physics", 4, "θɜːməʊdaɪˈnæmɪks", "entropy,heat,temperature", 0.94),
            ("electromagnetic", "Related to electricity and magnetism", "physics", 3, "ɪlektrəʊmæɡˈnetɪk", "electric field,magnetic field", 0.91),
            ("kinetic energy", "Energy of motion", "physics", 2, "kɪˈnetɪk ˈenədʒi", "potential energy,motion,momentum", 0.87),
            
            # Computer Science
            ("algorithm", "Step-by-step procedure for calculations", "computer_science", 2, "ˈælɡərɪðəm", "data structure,complexity,efficiency", 0.96),
            ("data structure", "Way of organizing and storing data", "computer_science", 3, "ˈdeɪtə ˈstrʌktʃə", "algorithm,array,tree,graph", 0.93),
            ("recursion", "Function calling itself", "computer_science", 4, "rɪˈkɜːʃən", "base case,recursive case,stack", 0.89),
            ("polymorphism", "Ability to take multiple forms", "computer_science", 4, "pɒlɪˈmɔːfɪzəm", "inheritance,encapsulation,abstraction", 0.86),
            
            # Chemistry
            ("stoichiometry", "Calculation of reactants and products", "chemistry", 4, "stɔɪkɪˈɒmɪtri", "mole,reaction,balance", 0.92),
            ("electronegativity", "Tendency to attract electrons", "chemistry", 3, "ɪlektrəʊneɡəˈtɪvɪti", "periodic table,bond,electron", 0.88),
            ("catalyst", "Substance that speeds up reaction", "chemistry", 2, "ˈkætəlɪst", "reaction rate,activation energy", 0.85),
            
            # Biology
            ("photosynthesis", "Process of converting light to chemical energy", "biology", 3, "fəʊtəʊˈsɪnθəsɪs", "chloroplast,glucose,carbon dioxide", 0.94),
            ("mitochondria", "Powerhouse of the cell", "biology", 3, "maɪtəˈkɒndrɪə", "cellular respiration,ATP,organelle", 0.91),
            ("DNA replication", "Process of copying DNA", "biology", 4, "ˌdiːenˈeɪ replɪˈkeɪʃən", "helicase,polymerase,primer", 0.95)
        ]
        
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO academic_terms 
            (term, definition, subject, difficulty_level, pronunciation, related_terms, frequency_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, academic_terms)
    
    def _load_math_terms(self) -> Dict[str, AcademicTerm]:
        """Load mathematics-specific terminology"""
        terms = {
            "calculus": AcademicTerm("calculus", "Mathematical study of continuous change", "mathematics", 4, "ˈkælkjʊləs", 
                                   ["derivative", "integral", "limit"], 0.96),
            "matrix": AcademicTerm("matrix", "Rectangular array of numbers", "mathematics", 3, "ˈmeɪtrɪks",
                                 ["determinant", "eigenvalue", "linear algebra"], 0.89),
            "topology": AcademicTerm("topology", "Study of geometric properties", "mathematics", 5, "təˈpɒlədʒi",
                                   ["continuity", "homeomorphism", "space"], 0.82)
        }
        return terms
    
    def _load_physics_terms(self) -> Dict[str, AcademicTerm]:
        """Load physics-specific terminology"""
        terms = {
            "relativity": AcademicTerm("relativity", "Einstein's theory of space-time", "physics", 5, "relə'tɪvəti",
                                     ["space-time", "mass-energy", "gravity"], 0.97),
            "wave function": AcademicTerm("wave function", "Quantum state description", "physics", 5, "weɪv ˈfʌŋkʃən",
                                        ["quantum mechanics", "probability", "measurement"], 0.94),
            "entropy": AcademicTerm("entropy", "Measure of disorder", "physics", 4, "ˈentrəpi",
                                  ["thermodynamics", "statistical mechanics", "information"], 0.91)
        }
        return terms
    
    def _load_chemistry_terms(self) -> Dict[str, AcademicTerm]:
        """Load chemistry-specific terminology"""
        terms = {
            "orbital": AcademicTerm("orbital", "Region where electrons are likely found", "chemistry", 4, "ˈɔːbɪtəl",
                                  ["electron", "quantum number", "atomic structure"], 0.93),
            "equilibrium": AcademicTerm("equilibrium", "State of balanced reactions", "chemistry", 3, "iːkwɪˈlɪbrɪəm",
                                      ["reaction rate", "Le Chatelier", "concentration"], 0.89),
            "molarity": AcademicTerm("molarity", "Moles of solute per liter", "chemistry", 2, "məʊˈlærəti",
                                   ["concentration", "solution", "mole"], 0.86)
        }
        return terms
    
    def _load_biology_terms(self) -> Dict[str, AcademicTerm]:
        """Load biology-specific terminology"""
        terms = {
            "evolution": AcademicTerm("evolution", "Change in heritable traits", "biology", 3, "iːvəˈluːʃən",
                                    ["natural selection", "adaptation", "species"], 0.95),
            "ecosystem": AcademicTerm("ecosystem", "Community of living organisms", "biology", 2, "ˈiːkəʊsɪstəm",
                                    ["biodiversity", "food chain", "environment"], 0.88),
            "homeostasis": AcademicTerm("homeostasis", "Maintenance of internal balance", "biology", 4, "həʊmɪə'steɪsɪs",
                                      ["regulation", "feedback", "physiological"], 0.91)
        }
        return terms
    
    def _load_cs_terms(self) -> Dict[str, AcademicTerm]:
        """Load computer science terminology"""
        terms = {
            "machine learning": AcademicTerm("machine learning", "AI that learns from data", "computer_science", 4, 
                                           "məˈʃiːn ˈlɜːnɪŋ", ["neural network", "training", "algorithm"], 0.98),
            "database": AcademicTerm("database", "Organized collection of data", "computer_science", 2, "ˈdeɪtəbeɪs",
                                   ["SQL", "query", "relational"], 0.92),
            "encryption": AcademicTerm("encryption", "Process of encoding information", "computer_science", 3, "ɪnˈkrɪpʃən",
                                     ["cryptography", "security", "key"], 0.87)
        }
        return terms
    
    def _load_engineering_terms(self) -> Dict[str, AcademicTerm]:
        """Load engineering terminology"""
        terms = {
            "stress analysis": AcademicTerm("stress analysis", "Study of internal forces", "engineering", 4, "stres əˈnæləsɪs",
                                          ["strain", "material", "load"], 0.89),
            "circuit": AcademicTerm("circuit", "Path for electrical current", "engineering", 2, "ˈsɜːkɪt",
                                  ["voltage", "current", "resistance"], 0.94),
            "optimization": AcademicTerm("optimization", "Process of making something optimal", "engineering", 3, "ɒptəmaɪˈzeɪʃən",
                                       ["constraint", "objective function", "efficiency"], 0.86)
        }
        return terms
    
    def _load_medical_terms(self) -> Dict[str, AcademicTerm]:
        """Load medical terminology"""
        terms = {
            "pathophysiology": AcademicTerm("pathophysiology", "Study of disease processes", "medicine", 5, "pæθəʊfɪziˈɒlədʒi",
                                          ["disease", "mechanism", "dysfunction"], 0.93),
            "pharmacokinetics": AcademicTerm("pharmacokinetics", "Drug movement in body", "medicine", 4, "fɑːməkəʊkɪˈnetɪks",
                                           ["absorption", "metabolism", "excretion"], 0.88),
            "diagnosis": AcademicTerm("diagnosis", "Identification of disease", "medicine", 3, "daɪəɡˈnəʊsɪs",
                                    ["symptom", "examination", "test"], 0.95)
        }
        return terms
    
    def _load_economics_terms(self) -> Dict[str, AcademicTerm]:
        """Load economics terminology"""
        terms = {
            "macroeconomics": AcademicTerm("macroeconomics", "Study of economy as whole", "economics", 3, "mækrəʊɪkəˈnɒmɪks",
                                         ["GDP", "inflation", "unemployment"], 0.91),
            "supply and demand": AcademicTerm("supply and demand", "Economic model of price determination", "economics", 2, 
                                            "səˈplaɪ ænd dɪˈmɑːnd", ["market", "equilibrium", "price"], 0.94),
            "elasticity": AcademicTerm("elasticity", "Responsiveness of demand to price", "economics", 3, "ɪlæˈstɪsəti",
                                     ["price elasticity", "demand", "responsive"], 0.87)
        }
        return terms
    
    def _load_history_terms(self) -> Dict[str, AcademicTerm]:
        """Load history terminology"""
        terms = {
            "historiography": AcademicTerm("historiography", "Study of historical writing", "history", 4, "hɪstɔːrɪˈɒɡrəfi",
                                         ["methodology", "interpretation", "sources"], 0.82),
            "primary source": AcademicTerm("primary source", "Direct evidence from time period", "history", 2, 
                                         "ˈpraɪməri sɔːs", ["evidence", "document", "artifact"], 0.89),
            "chronology": AcademicTerm("chronology", "Arrangement in time sequence", "history", 2, "krəˈnɒlədʒi",
                                     ["timeline", "sequence", "dating"], 0.85)
        }
        return terms
    
    def _load_literature_terms(self) -> Dict[str, AcademicTerm]:
        """Load literature terminology"""
        terms = {
            "metaphor": AcademicTerm("metaphor", "Figure of speech comparing two things", "literature", 2, "ˈmetəfə",
                                   ["simile", "imagery", "figurative"], 0.88),
            "narrative": AcademicTerm("narrative", "Account of connected events", "literature", 2, "ˈnærətɪv",
                                    ["story", "plot", "structure"], 0.91),
            "allegory": AcademicTerm("allegory", "Story with hidden meaning", "literature", 3, "ˈæləɡəri",
                                   ["symbolism", "metaphor", "interpretation"], 0.84)
        }
        return terms
    
    async def process_academic_lecture(self, transcript: str, audio_metadata: Dict) -> Dict:
        """
        Process academic lecture transcript with intelligent analysis
        
        Args:
            transcript: Raw transcript text
            audio_metadata: Metadata about audio (duration, quality, etc.)
            
        Returns:
            Processed academic transcript with structure and analysis
        """
        try:
            logger.info("Starting academic lecture processing...")
            
            # Detect subject area
            subject = await self._detect_subject_area(transcript)
            logger.info(f"Detected subject: {subject}")
            
            # Analyze professor speech patterns
            speech_patterns = await self._analyze_speech_patterns(transcript, audio_metadata)
            
            # Segment the lecture
            lecture_segments = await self._segment_lecture(transcript, subject)
            
            # Extract and enhance technical terms
            enhanced_terms = await self._enhance_technical_terms(transcript, subject)
            
            # Generate academic summary
            academic_summary = await self._generate_academic_summary(transcript, subject, lecture_segments)
            
            # Create student-friendly notes
            student_notes = await self._create_student_notes(transcript, lecture_segments, enhanced_terms)
            
            # Generate study materials
            study_materials = await self._generate_study_materials(transcript, enhanced_terms, subject)
            
            result = {
                "original_transcript": transcript,
                "enhanced_transcript": enhanced_terms["enhanced_text"],
                "subject_area": subject,
                "professor_speech_patterns": speech_patterns.__dict__,
                "lecture_segments": [seg.__dict__ for seg in lecture_segments],
                "academic_summary": academic_summary,
                "student_notes": student_notes,
                "study_materials": study_materials,
                "key_terms_explained": enhanced_terms["explained_terms"],
                "lecture_structure": self._analyze_lecture_structure(lecture_segments),
                "processing_metadata": {
                    "processing_time": datetime.now().isoformat(),
                    "audio_duration": audio_metadata.get("duration", 0),
                    "confidence_score": self._calculate_overall_confidence(lecture_segments),
                    "terminology_complexity": enhanced_terms["complexity_score"]
                }
            }
            
            logger.info("Academic lecture processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in academic lecture processing: {e}")
            raise
    
    async def _detect_subject_area(self, transcript: str) -> str:
        """Detect the subject area of the lecture"""
        try:
            # Count subject-specific terms
            subject_scores = defaultdict(int)
            
            for subject, terms_dict in self.subject_databases.items():
                for term_name, term_obj in terms_dict.items():
                    # Check for term and related terms in transcript
                    if term_name.lower() in transcript.lower():
                        subject_scores[subject] += term_obj.frequency_score * 2
                    
                    for related in term_obj.related_terms:
                        if related.lower() in transcript.lower():
                            subject_scores[subject] += term_obj.frequency_score * 0.5
            
            # Also check academic terms database
            cursor = self.academic_terms_db.cursor()
            cursor.execute("SELECT subject, term, frequency_score FROM academic_terms")
            for subject, term, freq_score in cursor.fetchall():
                if term.lower() in transcript.lower():
                    subject_scores[subject] += freq_score
            
            # Return subject with highest score
            if subject_scores:
                return max(subject_scores, key=subject_scores.get)
            else:
                return "general"
                
        except Exception as e:
            logger.error(f"Error detecting subject area: {e}")
            return "general"
    
    async def _analyze_speech_patterns(self, transcript: str, metadata: Dict) -> ProfessorSpeechPattern:
        """Analyze professor's speech patterns"""
        try:
            # Analyze speech rate (words per minute)
            word_count = len(transcript.split())
            duration_minutes = metadata.get("duration", 3600) / 60  # Convert to minutes
            speech_rate = word_count / duration_minutes if duration_minutes > 0 else 100
            
            # Detect filler words
            filler_words = ["um", "uh", "er", "ah", "you know", "like", "so", "well", "actually", "basically"]
            detected_fillers = [word for word in filler_words if word in transcript.lower()]
            
            # Detect emphasis patterns (words in caps, repeated words)
            emphasis_words = re.findall(r'\b[A-Z]{2,}\b', transcript)
            
            # Analyze pause patterns (based on punctuation)
            sentences = re.split(r'[.!?]+', transcript)
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 20
            
            # Determine speaking style based on patterns
            question_marks = transcript.count('?')
            exclamations = transcript.count('!')
            
            if question_marks > word_count * 0.02:  # More than 2% questions
                speaking_style = "interactive"
            elif exclamations > word_count * 0.01:  # More than 1% exclamations
                speaking_style = "enthusiastic"
            elif avg_sentence_length > 25:
                speaking_style = "formal_lecture"
            else:
                speaking_style = "conversational"
            
            # Calculate clarity score based on filler word density
            filler_density = sum(transcript.lower().count(f) for f in detected_fillers) / word_count
            clarity_score = max(0.0, 1.0 - (filler_density * 10))  # Scale to 0-1
            
            return ProfessorSpeechPattern(
                speech_rate=speech_rate,
                pause_patterns=[avg_sentence_length],
                emphasis_words=emphasis_words[:10],  # Top 10 emphasis words
                filler_words=detected_fillers,
                speaking_style=speaking_style,
                clarity_score=clarity_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing speech patterns: {e}")
            return ProfessorSpeechPattern(100.0, [20.0], [], [], "conversational", 0.8)
    
    async def _segment_lecture(self, transcript: str, subject: str) -> List[LectureSegment]:
        """Segment lecture into structured parts"""
        try:
            segments = []
            sentences = re.split(r'[.!?]+', transcript)
            current_time = 0.0
            time_per_sentence = 5.0  # Estimate 5 seconds per sentence
            
            current_segment = {"content": "", "type": "introduction", "start": 0.0, "sentences": []}
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                sentence = sentence.strip()
                current_segment["content"] += sentence + ". "
                current_segment["sentences"].append(sentence)
                
                # Detect segment transitions
                segment_type = self._classify_sentence_type(sentence)
                
                # If type changes or segment gets too long, create new segment
                if (segment_type != current_segment["type"] or 
                    len(current_segment["sentences"]) > 10):
                    
                    # Finish current segment
                    if current_segment["content"].strip():
                        key_terms = self._extract_key_terms_from_text(current_segment["content"], subject)
                        summary = await self._summarize_segment(current_segment["content"])
                        importance = self._calculate_importance_score(current_segment["content"], key_terms)
                        
                        segment = LectureSegment(
                            start_time=current_segment["start"],
                            end_time=current_time,
                            content=current_segment["content"].strip(),
                            segment_type=current_segment["type"],
                            confidence=0.85,  # Default confidence
                            key_terms=key_terms,
                            summary=summary,
                            importance_score=importance
                        )
                        segments.append(segment)
                    
                    # Start new segment
                    current_segment = {
                        "content": sentence + ". ",
                        "type": segment_type,
                        "start": current_time,
                        "sentences": [sentence]
                    }
                
                current_time += time_per_sentence
            
            # Don't forget the last segment
            if current_segment["content"].strip():
                key_terms = self._extract_key_terms_from_text(current_segment["content"], subject)
                summary = await self._summarize_segment(current_segment["content"])
                importance = self._calculate_importance_score(current_segment["content"], key_terms)
                
                segment = LectureSegment(
                    start_time=current_segment["start"],
                    end_time=current_time,
                    content=current_segment["content"].strip(),
                    segment_type=current_segment["type"],
                    confidence=0.85,
                    key_terms=key_terms,
                    summary=summary,
                    importance_score=importance
                )
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error segmenting lecture: {e}")
            return []
    
    def _classify_sentence_type(self, sentence: str) -> str:
        """Classify sentence into lecture segment type"""
        sentence_lower = sentence.lower()
        
        # Check introduction patterns
        for pattern in self.lecture_patterns["introduction"]:
            if pattern in sentence_lower:
                return "introduction"
        
        # Check conclusion patterns
        for pattern in self.lecture_patterns["conclusion"]:
            if pattern in sentence_lower:
                return "conclusion"
        
        # Check question patterns
        for pattern in self.lecture_patterns["questions"]:
            if pattern in sentence_lower:
                return "q_and_a"
        
        # Check for examples
        if any(word in sentence_lower for word in ["example", "for instance", "such as", "like"]):
            return "example"
        
        # Default to main content
        return "main_topic"
    
    def _extract_key_terms_from_text(self, text: str, subject: str) -> List[str]:
        """Extract key terms from text segment"""
        key_terms = []
        text_lower = text.lower()
        
        # Check subject-specific terms
        if subject in self.subject_databases:
            for term_name, term_obj in self.subject_databases[subject].items():
                if term_name.lower() in text_lower:
                    key_terms.append(term_name)
        
        # Check general academic terms
        cursor = self.academic_terms_db.cursor()
        cursor.execute("SELECT term FROM academic_terms WHERE subject = ? OR subject = 'general'", (subject,))
        for (term,) in cursor.fetchall():
            if term.lower() in text_lower:
                key_terms.append(term)
        
        return list(set(key_terms))  # Remove duplicates
    
    async def _summarize_segment(self, content: str) -> str:
        """Generate summary for lecture segment"""
        try:
            # Use OpenAI to generate summary
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "You are an expert at creating concise academic summaries. Summarize the following lecture segment in 1-2 sentences, focusing on key concepts."
                }, {
                    "role": "user", 
                    "content": content
                }],
                max_tokens=100,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating segment summary: {e}")
            # Fallback: simple extractive summary
            sentences = content.split('.')
            return sentences[0] + '.' if sentences else "Summary not available."
    
    def _calculate_importance_score(self, content: str, key_terms: List[str]) -> float:
        """Calculate importance score for segment"""
        try:
            score = 0.5  # Base score
            
            # Increase score for key terms
            score += len(key_terms) * 0.1
            
            # Increase score for length (longer segments often more important)
            word_count = len(content.split())
            if word_count > 50:
                score += 0.2
            elif word_count > 100:
                score += 0.3
            
            # Increase score for certain phrases
            important_phrases = ["important", "key point", "remember", "crucial", "essential", "main idea"]
            for phrase in important_phrases:
                if phrase in content.lower():
                    score += 0.1
            
            return min(1.0, score)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating importance score: {e}")
            return 0.5
    
    def _analyze_lecture_structure(self, segments: List[LectureSegment]) -> Dict:
        """Analyze overall lecture structure"""
        try:
            structure = {
                "total_segments": len(segments),
                "segment_types": {},
                "duration_distribution": {},
                "key_topics": [],
                "engagement_patterns": []
            }
            
            # Count segment types
            for segment in segments:
                seg_type = segment.segment_type
                structure["segment_types"][seg_type] = structure["segment_types"].get(seg_type, 0) + 1
            
            # Calculate duration distribution
            total_duration = sum(seg.end_time - seg.start_time for seg in segments)
            for segment in segments:
                duration = segment.end_time - segment.start_time
                percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
                seg_type = segment.segment_type
                
                if seg_type not in structure["duration_distribution"]:
                    structure["duration_distribution"][seg_type] = 0
                structure["duration_distribution"][seg_type] += percentage
            
            # Extract key topics (most frequent key terms)
            all_key_terms = []
            for segment in segments:
                all_key_terms.extend(segment.key_terms)
            
            term_counts = Counter(all_key_terms)
            structure["key_topics"] = [term for term, count in term_counts.most_common(10)]
            
            # Analyze engagement patterns based on Q&A segments
            qa_segments = [s for s in segments if s.segment_type == "q_and_a"]
            structure["engagement_patterns"] = {
                "qa_frequency": len(qa_segments),
                "avg_qa_importance": np.mean([s.importance_score for s in qa_segments]) if qa_segments else 0,
                "interactive_ratio": len(qa_segments) / len(segments) if segments else 0
            }
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing lecture structure: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_confidence(self, segments: List[LectureSegment]) -> float:
        """Calculate overall confidence score for the lecture processing"""
        try:
            if not segments:
                return 0.0
            
            # Weight by importance and duration
            weighted_scores = []
            total_weight = 0
            
            for segment in segments:
                duration = segment.end_time - segment.start_time
                weight = duration * segment.importance_score
                weighted_scores.append(segment.confidence * weight)
                total_weight += weight
            
            return sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.75
    
    async def _enhance_technical_terms(self, transcript: str, subject: str) -> Dict:
        """Enhance transcript with technical term explanations"""
        try:
            enhanced_text = transcript
            explained_terms = {}
            complexity_scores = []
            
            # Process subject-specific terms
            if subject in self.subject_databases:
                for term_name, term_obj in self.subject_databases[subject].items():
                    if term_name.lower() in transcript.lower():
                        # Add explanation in parentheses after first occurrence
                        pattern = re.compile(re.escape(term_name), re.IGNORECASE)
                        if pattern.search(enhanced_text):
                            replacement = f"{term_name} ({term_obj.definition})"
                            enhanced_text = pattern.sub(replacement, enhanced_text, count=1)
                            explained_terms[term_name] = {
                                "definition": term_obj.definition,
                                "pronunciation": term_obj.pronunciation,
                                "difficulty": term_obj.difficulty_level,
                                "related_terms": term_obj.related_terms
                            }
                            complexity_scores.append(term_obj.difficulty_level)
            
            # Process general academic terms from database
            cursor = self.academic_terms_db.cursor()
            cursor.execute("""
                SELECT term, definition, pronunciation, difficulty_level, related_terms 
                FROM academic_terms 
                WHERE subject = ? OR subject = 'general'
            """, (subject,))
            
            for term, definition, pronunciation, difficulty, related_terms in cursor.fetchall():
                if term.lower() in transcript.lower() and term not in explained_terms:
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    if pattern.search(enhanced_text):
                        replacement = f"{term} ({definition})"
                        enhanced_text = pattern.sub(replacement, enhanced_text, count=1)
                        explained_terms[term] = {
                            "definition": definition,
                            "pronunciation": pronunciation,
                            "difficulty": difficulty,
                            "related_terms": related_terms.split(',') if related_terms else []
                        }
                        complexity_scores.append(difficulty)
            
            avg_complexity = np.mean(complexity_scores) if complexity_scores else 2.5
            
            return {
                "enhanced_text": enhanced_text,
                "explained_terms": explained_terms,
                "complexity_score": avg_complexity,
                "total_terms_explained": len(explained_terms)
            }
            
        except Exception as e:
            logger.error(f"Error enhancing technical terms: {e}")
            return {
                "enhanced_text": transcript,
                "explained_terms": {},
                "complexity_score": 2.5,
                "total_terms_explained": 0
            }
    
    async def _generate_academic_summary(self, transcript: str, subject: str, 
                                       segments: List[LectureSegment]) -> Dict:
        """Generate comprehensive academic summary"""
        try:
            # Extract key points from high-importance segments
            key_segments = [s for s in segments if s.importance_score > 0.7]
            
            # Generate overall summary using AI
            summary_prompt = f"""
            Summarize this {subject} lecture transcript focusing on:
            1. Main concepts and theories
            2. Key definitions and terminology
            3. Important examples or applications
            4. Conclusions and takeaways
            
            Transcript: {transcript[:2000]}...
            """
            
            try:
                response = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "system",
                        "content": "You are an expert academic summarizer. Create clear, structured summaries for students."
                    }, {
                        "role": "user",
                        "content": summary_prompt
                    }],
                    max_tokens=500,
                    temperature=0.3
                )
                
                ai_summary = response.choices[0].message.content.strip()
            except:
                ai_summary = "AI summary not available."
            
            # Create structured summary
            summary = {
                "overview": ai_summary,
                "key_concepts": [],
                "main_topics": [],
                "important_segments": [],
                "difficulty_level": "intermediate"
            }
            
            # Extract key concepts from segments
            all_terms = []
            for segment in key_segments:
                all_terms.extend(segment.key_terms)
                summary["important_segments"].append({
                    "time": f"{segment.start_time//60:.0f}:{segment.start_time%60:02.0f}",
                    "topic": segment.segment_type.replace('_', ' ').title(),
                    "summary": segment.summary,
                    "importance": segment.importance_score
                })
            
            # Most frequent terms become key concepts
            term_counts = Counter(all_terms)
            summary["key_concepts"] = [term for term, count in term_counts.most_common(5)]
            
            # Topic extraction from segment types
            topic_types = [s.segment_type for s in segments if s.segment_type != "q_and_a"]
            topic_counts = Counter(topic_types)
            summary["main_topics"] = [topic.replace('_', ' ').title() for topic, _ in topic_counts.most_common(3)]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating academic summary: {e}")
            return {"overview": "Summary generation failed", "key_concepts": [], "main_topics": [], "important_segments": []}
    
    async def _create_student_notes(self, transcript: str, segments: List[LectureSegment], 
                                  enhanced_terms: Dict) -> Dict:
        """Create student-friendly lecture notes"""
        try:
            notes = {
                "structured_outline": [],
                "bullet_points": [],
                "definitions": enhanced_terms["explained_terms"],
                "timestamps": [],
                "study_questions": [],
                "key_takeaways": []
            }
            
            # Create structured outline from segments
            current_section = {"title": "", "content": [], "subsections": []}
            
            for segment in segments:
                if segment.segment_type == "introduction":
                    if current_section["title"]:
                        notes["structured_outline"].append(current_section)
                    current_section = {
                        "title": "Introduction",
                        "content": [segment.summary],
                        "subsections": [],
                        "timestamp": f"{segment.start_time//60:.0f}:{segment.start_time%60:02.0f}"
                    }
                elif segment.segment_type == "main_topic":
                    # Start new section or add to current
                    if len(current_section["content"]) > 3:  # Too much content, start new section
                        if current_section["title"]:
                            notes["structured_outline"].append(current_section)
                        current_section = {
                            "title": f"Topic {len(notes['structured_outline']) + 1}",
                            "content": [segment.summary],
                            "subsections": [],
                            "timestamp": f"{segment.start_time//60:.0f}:{segment.start_time%60:02.0f}"
                        }
                    else:
                        current_section["content"].append(segment.summary)
                elif segment.segment_type == "example":
                    current_section["subsections"].append({
                        "type": "example",
                        "content": segment.summary,
                        "timestamp": f"{segment.start_time//60:.0f}:{segment.start_time%60:02.0f}"
                    })
                elif segment.segment_type == "conclusion":
                    current_section["subsections"].append({
                        "type": "conclusion", 
                        "content": segment.summary,
                        "timestamp": f"{segment.start_time//60:.0f}:{segment.start_time%60:02.0f}"
                    })
            
            # Don't forget the last section
            if current_section["title"]:
                notes["structured_outline"].append(current_section)
            
            # Create bullet points from all segments
            for segment in segments:
                if segment.importance_score > 0.6:  # Only important points
                    notes["bullet_points"].append(f"• {segment.summary}")
                    notes["timestamps"].append({
                        "time": f"{segment.start_time//60:.0f}:{segment.start_time%60:02.0f}",
                        "content": segment.summary[:100] + "..." if len(segment.summary) > 100 else segment.summary
                    })
            
            # Generate study questions
            key_terms = list(enhanced_terms["explained_terms"].keys())
            if key_terms:
                notes["study_questions"] = [
                    f"What is {term}?" for term in key_terms[:5]
                ] + [
                    f"Explain the relationship between {key_terms[i]} and {key_terms[i+1]}" 
                    for i in range(min(3, len(key_terms)-1))
                ]
            
            # Extract key takeaways from conclusion segments
            conclusion_segments = [s for s in segments if s.segment_type == "conclusion"]
            for segment in conclusion_segments:
                notes["key_takeaways"].append(segment.summary)
            
            # If no conclusions, use highest importance segments
            if not notes["key_takeaways"]:
                high_importance = sorted(segments, key=lambda x: x.importance_score, reverse=True)[:3]
                notes["key_takeaways"] = [s.summary for s in high_importance]
            
            return notes
            
        except Exception as e:
            logger.error(f"Error creating student notes: {e}")
            return {"structured_outline": [], "bullet_points": [], "definitions": {}, "timestamps": [], "study_questions": [], "key_takeaways": []}
    
    async def _generate_study_materials(self, transcript: str, enhanced_terms: Dict, subject: str) -> Dict:
        """Generate additional study materials for students"""
        try:
            study_materials = {
                "flashcards": [],
                "practice_questions": [],
                "additional_reading": [],
                "concept_map": {},
                "difficulty_assessment": "intermediate"
            }
            
            # Create flashcards from explained terms
            for term, details in enhanced_terms["explained_terms"].items():
                flashcard = {
                    "front": term,
                    "back": details["definition"],
                    "pronunciation": details.get("pronunciation", ""),
                    "difficulty": details.get("difficulty", 3),
                    "subject": subject
                }
                study_materials["flashcards"].append(flashcard)
            
            # Generate practice questions using AI
            if enhanced_terms["explained_terms"]:
                try:
                    question_prompt = f"""
                    Create 5 practice questions for a {subject} lecture covering these key terms:
                    {', '.join(list(enhanced_terms["explained_terms"].keys())[:5])}
                    
                    Make questions at different difficulty levels (easy, medium, hard).
                    Format: Q: [question] | A: [answer] | Level: [easy/medium/hard]
                    """
                    
                    response = await openai.ChatCompletion.acreate(
                        model="gpt-3.5-turbo",
                        messages=[{
                            "role": "system",
                            "content": "You are an educational content creator. Create clear, helpful practice questions."
                        }, {
                            "role": "user",
                            "content": question_prompt
                        }],
                        max_tokens=400,
                        temperature=0.7
                    )
                    
                    # Parse the response into structured questions
                    questions_text = response.choices[0].message.content.strip()
                    for line in questions_text.split('\n'):
                        if '|' in line and line.startswith('Q:'):
                            parts = line.split('|')
                            if len(parts) >= 3:
                                question = parts[0].replace('Q:', '').strip()
                                answer = parts[1].replace('A:', '').strip()
                                level = parts[2].replace('Level:', '').strip()
                                
                                study_materials["practice_questions"].append({
                                    "question": question,
                                    "answer": answer,
                                    "difficulty": level,
                                    "subject": subject
                                })
                
                except Exception as e:
                    logger.error(f"Error generating practice questions: {e}")
            
            # Create concept map from related terms
            concept_map = {}
            for term, details in enhanced_terms["explained_terms"].items():
                related = details.get("related_terms", [])
                concept_map[term] = [r for r in related if r in enhanced_terms["explained_terms"]]
            
            study_materials["concept_map"] = concept_map
            
            # Assess overall difficulty
            difficulties = [details.get("difficulty", 3) for details in enhanced_terms["explained_terms"].values()]
            if difficulties:
                avg_difficulty = np.mean(difficulties)
                if avg_difficulty <= 2:
                    study_materials["difficulty_assessment"] = "beginner"
                elif avg_difficulty <= 3.5:
                    study_materials["difficulty_assessment"] = "intermediate" 
                else:
                    study_materials["difficulty_assessment"] = "advanced"
            
            # Subject-specific additional reading suggestions
            reading_suggestions = {
                "mathematics": ["Khan Academy Calculus", "MIT OpenCourseWare Mathematics", "Wolfram MathWorld"],
                "physics": ["Feynman Lectures on Physics", "MIT Physics Course Materials", "PhET Interactive Simulations"],
                "chemistry": ["Khan Academy Chemistry", "ChemSpider Database", "NIST Chemistry WebBook"],
                "biology": ["Khan Academy Biology", "NCBI Bookshelf", "Biology Online Dictionary"],
                "computer_science": ["MIT OpenCourseWare CS", "Stanford CS Course Materials", "Algorithm Visualizer"],
                "engineering": ["MIT OpenCourseWare Engineering", "Engineering ToolBox", "ASME Resources"],
                "medicine": ["PubMed Central", "Medical Subject Headings (MeSH)", "Medscape Reference"],
                "economics": ["Khan Academy Economics", "Federal Reserve Economic Data", "MIT Economics Courses"],
                "history": ["Stanford History Education Group", "Library of Congress", "National Archives"],
                "literature": ["Poetry Foundation", "Project Gutenberg", "Literary Theory and Criticism"]
            }
            
            study_materials["additional_reading"] = reading_suggestions.get(subject, ["Wikipedia", "Google Scholar", "Course Textbook"])
            
            return study_materials
            
        except Exception as e:
            logger.error(f"Error generating study materials: {e}")
            return {"flashcards": [], "practice_questions": [], "additional_reading": [], "concept_map": {}, "difficulty_assessment": "intermediate"}

    def close(self):
        """Close database connections"""
        try:
            if self.academic_terms_db:
                self.academic_terms_db.close()
        except Exception as e:
            logger.error(f"Error closing academic processor: {e}")

# Export the main class
__all__ = ['SmartAcademicProcessor', 'LectureSegment', 'AcademicTerm', 'ProfessorSpeechPattern']

if __name__ == "__main__":
    # Test the academic processor
    async def test_processor():
        processor = SmartAcademicProcessor()
        
        # Sample academic transcript
        sample_transcript = """
        Today we will cover derivatives in calculus. The derivative of a function represents the rate of change. 
        For example, if we have a function f(x) = x squared, the derivative would be 2x. 
        This is a fundamental concept in mathematics. Are there any questions about this topic?
        Let's move on to integrals, which are the opposite of derivatives.
        """
        
        sample_metadata = {"duration": 300, "quality": "high"}
        
        result = await processor.process_academic_lecture(sample_transcript, sample_metadata)
        
        print("=== ACADEMIC PROCESSING RESULTS ===")
        print(f"Subject: {result['subject_area']}")
        print(f"Segments: {len(result['lecture_segments'])}")
        print(f"Key Terms: {len(result['key_terms_explained'])}")
        print(f"Study Materials: {len(result['study_materials']['flashcards'])} flashcards")
        
        processor.close()
    
    # Run test
    asyncio.run(test_processor())