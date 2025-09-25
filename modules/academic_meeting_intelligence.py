"""
Academic & Meeting Intelligence AI
Akademik ve toplantı içeriği için özel yapay zeka sistemi

Bu modül hem öğrenci derslerini hem de toplantıları akıllı şekilde analiz eder:
- Lecture structuring ve academic content analysis
- Meeting action items ve decision extraction
- Key points extraction ve intelligent summarization
- Speaker insights ve participation analysis
- Academic terminology explanation ve context

Made by Mehmet Arda Çekiç © 2025
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import json
import re
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter
import openai
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AcademicInsight:
    """Academic content insight"""
    insight_type: str  # concept, definition, example, conclusion
    content: str
    confidence: float
    timestamp: Optional[float] = None
    speaker: Optional[str] = None
    importance_score: float = 0.0
    related_concepts: List[str] = field(default_factory=list)

@dataclass
class MeetingActionItem:
    """Meeting action item with details"""
    description: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    priority: str = "medium"  # low, medium, high, urgent
    status: str = "identified"  # identified, assigned, in_progress, completed
    timestamp: Optional[float] = None
    context: str = ""
    confidence: float = 0.0

@dataclass
class KeyConcept:
    """Key concept with relationships"""
    concept: str
    definition: str
    importance_score: float
    mentions: int
    related_concepts: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)

@dataclass
class IntelligentSummary:
    """Intelligent summary with structured insights"""
    content_type: str  # academic, meeting, mixed
    executive_summary: str
    key_insights: List[AcademicInsight]
    main_concepts: List[KeyConcept]
    structure_analysis: Dict
    speaker_contributions: Dict
    actionable_items: List[MeetingActionItem]
    learning_objectives: List[str]
    follow_up_suggestions: List[str]

class AcademicMeetingAI:
    """
    Advanced AI system for academic and meeting content analysis
    Provides intelligent insights, structuring, and actionable outputs
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize AI system with required models"""
        
        # Set OpenAI API key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Initialize NLP models
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            logger.warning("NLTK data not available, using fallback")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Initialize transformers models
        self.sentiment_analyzer = None
        self.summarizer = None
        self.question_answerer = None
        
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            logger.info("Sentiment analyzer loaded")
        except:
            logger.warning("Could not load sentiment analyzer")
        
        try:
            self.summarizer = pipeline("summarization", 
                                     model="facebook/bart-large-cnn")
            logger.info("Summarizer loaded")
        except:
            logger.warning("Could not load summarizer")
        
        # Academic patterns
        self.academic_patterns = {
            'definitions': [
                r'(.+?) is defined as (.+)',
                r'(.+?) means (.+)',
                r'(.+?) refers to (.+)',
                r'we define (.+?) as (.+)',
                r'(.+?) can be understood as (.+)'
            ],
            'concepts': [
                r'the concept of (.+)',
                r'the principle of (.+)',
                r'the theory of (.+)',
                r'the law of (.+)',
                r'the phenomenon of (.+)'
            ],
            'examples': [
                r'for example, (.+)',
                r'for instance, (.+)',
                r'such as (.+)',
                r'consider (.+)',
                r'let\'s look at (.+)'
            ],
            'conclusions': [
                r'in conclusion, (.+)',
                r'therefore, (.+)',
                r'thus, (.+)',
                r'as a result, (.+)',
                r'consequently, (.+)'
            ]
        }
        
        # Meeting patterns
        self.meeting_patterns = {
            'action_items': [
                r'we need to (.+)',
                r'action item[:\s]+(.+)',
                r'(?:someone|[A-Za-z]+) will (.+)',
                r'(?:someone|[A-Za-z]+) should (.+)',
                r'let\'s (.+)',
                r'(?:someone|[A-Za-z]+) can (.+)',
                r'task[:\s]+(.+)',
                r'todo[:\s]+(.+)'
            ],
            'decisions': [
                r'we (?:decide|decided) (?:to )?(.+)',
                r'the decision is (.+)',
                r'we\'ll go with (.+)',
                r'final decision[:\s]+(.+)',
                r'agreed[:\s]+(.+)',
                r'resolution[:\s]+(.+)'
            ],
            'deadlines': [
                r'by (.+?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|next week|next month))',
                r'deadline[:\s]+(.+)',
                r'due (?:by|on) (.+)',
                r'before (.+)',
                r'until (.+)'
            ],
            'assignments': [
                r'([A-Za-z]+) will (.+)',
                r'([A-Za-z]+) is responsible for (.+)',
                r'assign (?:this )?to ([A-Za-z]+)',
                r'([A-Za-z]+) can (?:handle|take care of) (.+)',
                r'([A-Za-z]+) should (.+)'
            ]
        }
        
        logger.info("Academic & Meeting AI initialized successfully")
    
    async def analyze_content(self, transcript: str, content_type: str = "auto",
                            speaker_segments: Optional[List] = None,
                            metadata: Optional[Dict] = None) -> IntelligentSummary:
        """
        Analyze content with advanced AI to extract insights and structure
        
        Args:
            transcript: Full transcript text
            content_type: 'academic', 'meeting', or 'auto' for auto-detection
            speaker_segments: Optional speaker-separated segments
            metadata: Additional metadata about the content
            
        Returns:
            Comprehensive intelligent summary with insights
        """
        try:
            logger.info("Starting intelligent content analysis...")
            
            # Auto-detect content type if needed
            if content_type == "auto":
                content_type = await self._detect_content_type(transcript)
            
            logger.info(f"Analyzing as {content_type} content")
            
            # Extract key concepts and insights
            key_concepts = await self._extract_key_concepts(transcript, content_type)
            academic_insights = await self._extract_academic_insights(transcript, content_type)
            
            # Analyze structure
            structure_analysis = await self._analyze_content_structure(
                transcript, content_type, speaker_segments)
            
            # Extract actionable items
            actionable_items = await self._extract_actionable_items(
                transcript, content_type, speaker_segments)
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                transcript, content_type, key_concepts, academic_insights)
            
            # Analyze speaker contributions if available
            speaker_contributions = {}
            if speaker_segments:
                speaker_contributions = await self._analyze_speaker_contributions(
                    speaker_segments, content_type)
            
            # Generate learning objectives and follow-up suggestions
            learning_objectives = await self._generate_learning_objectives(
                transcript, content_type, key_concepts)
            
            follow_up_suggestions = await self._generate_follow_up_suggestions(
                transcript, content_type, actionable_items, key_concepts)
            
            # Create intelligent summary
            summary = IntelligentSummary(
                content_type=content_type,
                executive_summary=executive_summary,
                key_insights=academic_insights,
                main_concepts=key_concepts,
                structure_analysis=structure_analysis,
                speaker_contributions=speaker_contributions,
                actionable_items=actionable_items,
                learning_objectives=learning_objectives,
                follow_up_suggestions=follow_up_suggestions
            )
            
            logger.info("Intelligent content analysis completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            raise
    
    async def _detect_content_type(self, transcript: str) -> str:
        """Detect whether content is academic or meeting-based"""
        try:
            academic_indicators = [
                'lecture', 'lesson', 'professor', 'student', 'theory', 'concept',
                'definition', 'formula', 'equation', 'principle', 'law',
                'chapter', 'textbook', 'homework', 'assignment', 'exam'
            ]
            
            meeting_indicators = [
                'meeting', 'agenda', 'action item', 'deadline', 'task',
                'responsibility', 'project', 'budget', 'timeline', 'milestone',
                'decision', 'vote', 'agreement', 'proposal', 'presentation'
            ]
            
            text_lower = transcript.lower()
            
            academic_score = sum(1 for indicator in academic_indicators if indicator in text_lower)
            meeting_score = sum(1 for indicator in meeting_indicators if indicator in text_lower)
            
            # Also check for typical academic structures
            if any(pattern in text_lower for pattern in ['today we will cover', 'in this lecture', 'let\'s begin with']):
                academic_score += 2
            
            # Check for meeting structures
            if any(pattern in text_lower for pattern in ['let\'s start the meeting', 'agenda item', 'next steps']):
                meeting_score += 2
            
            if academic_score > meeting_score:
                return "academic"
            elif meeting_score > academic_score:
                return "meeting"
            else:
                return "mixed"
                
        except Exception as e:
            logger.error(f"Error detecting content type: {e}")
            return "mixed"
    
    async def _extract_key_concepts(self, transcript: str, content_type: str) -> List[KeyConcept]:
        """Extract and analyze key concepts from content"""
        try:
            concepts = []
            
            # Use TF-IDF to find important terms
            sentences = sent_tokenize(transcript)
            if len(sentences) < 2:
                sentences = [transcript]
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2 if len(sentences) > 5 else 1
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get average TF-IDF scores for each term
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                
                # Get top terms
                top_indices = np.argsort(mean_scores)[-20:][::-1]  # Top 20 terms
                
                for idx in top_indices:
                    term = feature_names[idx]
                    score = mean_scores[idx]
                    
                    if score > 0.1 and len(term) > 2:  # Minimum threshold
                        # Find mentions and context
                        mentions = transcript.lower().count(term.lower())
                        
                        # Try to find definition
                        definition = await self._find_term_definition(term, transcript)
                        
                        # Find examples
                        examples = await self._find_term_examples(term, transcript)
                        
                        # Find speakers who mentioned this term
                        speakers = []  # Would be populated with speaker data if available
                        
                        concept = KeyConcept(
                            concept=term,
                            definition=definition,
                            importance_score=float(score),
                            mentions=mentions,
                            examples=examples[:3],  # Top 3 examples
                            speakers=speakers
                        )
                        
                        concepts.append(concept)
                        
            except Exception as tfidf_error:
                logger.warning(f"TF-IDF analysis failed: {tfidf_error}")
                # Fallback: use pattern matching
                concepts = await self._extract_concepts_by_patterns(transcript, content_type)
            
            # Sort by importance and return top concepts
            concepts.sort(key=lambda x: x.importance_score, reverse=True)
            return concepts[:15]  # Top 15 concepts
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {e}")
            return []
    
    async def _find_term_definition(self, term: str, transcript: str) -> str:
        """Find definition for a term in the transcript"""
        try:
            for pattern in self.academic_patterns['definitions']:
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    if term.lower() in match.group().lower():
                        # Extract the definition part
                        groups = match.groups()
                        if len(groups) >= 2:
                            # Check which group contains the term
                            if term.lower() in groups[0].lower():
                                return groups[1].strip()
                            elif term.lower() in groups[1].lower():
                                return groups[0].strip()
            
            # Fallback: look for sentences containing the term
            sentences = sent_tokenize(transcript)
            for sentence in sentences:
                if (term.lower() in sentence.lower() and 
                    any(word in sentence.lower() for word in ['is', 'means', 'refers', 'defined'])):
                    return sentence.strip()
            
            return f"Key term mentioned {transcript.lower().count(term.lower())} times in content"
            
        except Exception as e:
            logger.error(f"Error finding definition for {term}: {e}")
            return "Definition not found"
    
    async def _find_term_examples(self, term: str, transcript: str) -> List[str]:
        """Find examples related to a term"""
        try:
            examples = []
            
            for pattern in self.academic_patterns['examples']:
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(transcript), match.end() + 100)
                    context = transcript[context_start:context_end]
                    
                    if term.lower() in context.lower():
                        example = match.group(1).strip()
                        if len(example) > 10 and example not in examples:
                            examples.append(example)
            
            return examples[:5]  # Return up to 5 examples
            
        except Exception as e:
            logger.error(f"Error finding examples for {term}: {e}")
            return []
    
    async def _extract_concepts_by_patterns(self, transcript: str, content_type: str) -> List[KeyConcept]:
        """Fallback method to extract concepts using patterns"""
        try:
            concepts = []
            
            # Academic concept patterns
            if content_type in ["academic", "mixed"]:
                for pattern in self.academic_patterns['concepts']:
                    matches = re.finditer(pattern, transcript, re.IGNORECASE)
                    for match in matches:
                        concept_name = match.group(1).strip()
                        if len(concept_name) > 3:
                            concept = KeyConcept(
                                concept=concept_name,
                                definition=f"Academic concept from {content_type} content",
                                importance_score=0.5,
                                mentions=transcript.lower().count(concept_name.lower())
                            )
                            concepts.append(concept)
            
            # Meeting concepts (projects, tasks, etc.)
            if content_type in ["meeting", "mixed"]:
                meeting_terms = ['project', 'task', 'deadline', 'budget', 'timeline', 'milestone']
                for term in meeting_terms:
                    if term in transcript.lower():
                        concept = KeyConcept(
                            concept=term,
                            definition=f"Meeting-related concept",
                            importance_score=0.4,
                            mentions=transcript.lower().count(term)
                        )
                        concepts.append(concept)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error in pattern-based concept extraction: {e}")
            return []
    
    async def _extract_academic_insights(self, transcript: str, content_type: str) -> List[AcademicInsight]:
        """Extract academic insights and important points"""
        try:
            insights = []
            
            # Extract definitions
            for pattern in self.academic_patterns['definitions']:
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    insight = AcademicInsight(
                        insight_type="definition",
                        content=match.group().strip(),
                        confidence=0.8,
                        importance_score=0.9
                    )
                    insights.append(insight)
            
            # Extract examples
            for pattern in self.academic_patterns['examples']:
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    # Get context around the example
                    start = max(0, match.start() - 50)
                    end = min(len(transcript), match.end() + 100)
                    context = transcript[start:end].strip()
                    
                    insight = AcademicInsight(
                        insight_type="example",
                        content=context,
                        confidence=0.7,
                        importance_score=0.6
                    )
                    insights.append(insight)
            
            # Extract conclusions
            for pattern in self.academic_patterns['conclusions']:
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    # Get the full sentence
                    sentences = sent_tokenize(transcript)
                    for sentence in sentences:
                        if match.group() in sentence:
                            insight = AcademicInsight(
                                insight_type="conclusion",
                                content=sentence.strip(),
                                confidence=0.85,
                                importance_score=0.95
                            )
                            insights.append(insight)
                            break
            
            # Use AI to extract additional insights if OpenAI is available
            if hasattr(openai, 'api_key') and openai.api_key:
                try:
                    ai_insights = await self._extract_ai_insights(transcript, content_type)
                    insights.extend(ai_insights)
                except Exception as ai_error:
                    logger.warning(f"AI insight extraction failed: {ai_error}")
            
            # Sort by importance and confidence
            insights.sort(key=lambda x: x.importance_score * x.confidence, reverse=True)
            return insights[:20]  # Top 20 insights
            
        except Exception as e:
            logger.error(f"Error extracting academic insights: {e}")
            return []
    
    async def _extract_ai_insights(self, transcript: str, content_type: str) -> List[AcademicInsight]:
        """Use AI to extract additional insights"""
        try:
            insights = []
            
            prompt = f"""
            Analyze this {content_type} content and extract key insights:
            1. Important concepts and definitions
            2. Key examples and applications
            3. Main conclusions or outcomes
            4. Critical points for understanding
            
            Content: {transcript[:2000]}...
            
            Format each insight as: TYPE|CONTENT|IMPORTANCE(0-1)
            """
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "You are an expert at analyzing academic and meeting content to extract key insights."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse AI response
            ai_text = response.choices[0].message.content.strip()
            for line in ai_text.split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        try:
                            insight_type = parts[0].strip().lower()
                            content = parts[1].strip()
                            importance = float(parts[2].strip())
                            
                            insight = AcademicInsight(
                                insight_type=insight_type,
                                content=content,
                                confidence=0.8,
                                importance_score=importance
                            )
                            insights.append(insight)
                        except ValueError:
                            continue
            
            return insights
            
        except Exception as e:
            logger.error(f"Error in AI insight extraction: {e}")
            return []
    
    async def _analyze_content_structure(self, transcript: str, content_type: str,
                                       speaker_segments: Optional[List] = None) -> Dict:
        """Analyze the structure and organization of content"""
        try:
            structure = {
                'total_segments': 0,
                'content_flow': [],
                'topic_transitions': [],
                'engagement_patterns': {},
                'information_density': 0.0
            }
            
            sentences = sent_tokenize(transcript)
            structure['total_segments'] = len(sentences)
            
            # Analyze content flow
            if content_type == "academic":
                # Look for academic structure patterns
                intro_patterns = ['today we will', 'in this lecture', 'let\'s begin', 'first, we\'ll discuss']
                main_patterns = ['next topic', 'moving on', 'another important', 'let\'s examine']
                conclusion_patterns = ['in conclusion', 'to summarize', 'finally', 'to wrap up']
                
                current_section = "introduction"
                for i, sentence in enumerate(sentences):
                    sentence_lower = sentence.lower()
                    
                    if any(pattern in sentence_lower for pattern in intro_patterns):
                        current_section = "introduction"
                    elif any(pattern in sentence_lower for pattern in main_patterns):
                        current_section = "main_content"
                    elif any(pattern in sentence_lower for pattern in conclusion_patterns):
                        current_section = "conclusion"
                    
                    structure['content_flow'].append({
                        'sentence_index': i,
                        'section': current_section,
                        'content': sentence[:100] + "..." if len(sentence) > 100 else sentence
                    })
            
            elif content_type == "meeting":
                # Analyze meeting flow
                agenda_patterns = ['agenda', 'first item', 'next item', 'moving to']
                discussion_patterns = ['let\'s discuss', 'what do you think', 'opinions on']
                decision_patterns = ['we decide', 'agreed', 'final decision']
                
                current_phase = "opening"
                for i, sentence in enumerate(sentences):
                    sentence_lower = sentence.lower()
                    
                    if any(pattern in sentence_lower for pattern in agenda_patterns):
                        current_phase = "agenda_review"
                    elif any(pattern in sentence_lower for pattern in discussion_patterns):
                        current_phase = "discussion"
                    elif any(pattern in sentence_lower for pattern in decision_patterns):
                        current_phase = "decision_making"
                    
                    structure['content_flow'].append({
                        'sentence_index': i,
                        'phase': current_phase,
                        'content': sentence[:100] + "..." if len(sentence) > 100 else sentence
                    })
            
            # Calculate information density (complex sentences per total sentences)
            complex_sentence_count = 0
            for sentence in sentences:
                # Count as complex if it has multiple clauses, technical terms, or is long
                if (len(sentence.split()) > 20 or 
                    sentence.count(',') > 2 or
                    any(word in sentence.lower() for word in ['however', 'therefore', 'consequently', 'furthermore'])):
                    complex_sentence_count += 1
            
            structure['information_density'] = complex_sentence_count / len(sentences) if sentences else 0
            
            # Analyze topic transitions using sentence similarity
            if len(sentences) > 1:
                try:
                    # Simple similarity based on word overlap
                    transitions = []
                    for i in range(len(sentences) - 1):
                        current_words = set(word_tokenize(sentences[i].lower()))
                        next_words = set(word_tokenize(sentences[i + 1].lower()))
                        
                        # Remove stop words
                        current_words -= self.stop_words
                        next_words -= self.stop_words
                        
                        # Calculate Jaccard similarity
                        if current_words and next_words:
                            similarity = len(current_words & next_words) / len(current_words | next_words)
                            
                            # Low similarity indicates topic transition
                            if similarity < 0.2:
                                transitions.append({
                                    'position': i + 1,
                                    'similarity': similarity,
                                    'transition_type': 'major' if similarity < 0.1 else 'minor'
                                })
                    
                    structure['topic_transitions'] = transitions
                    
                except Exception as similarity_error:
                    logger.warning(f"Topic transition analysis failed: {similarity_error}")
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing content structure: {e}")
            return {'total_segments': 0, 'content_flow': [], 'topic_transitions': [], 'information_density': 0.0}
    
    async def _extract_actionable_items(self, transcript: str, content_type: str,
                                      speaker_segments: Optional[List] = None) -> List[MeetingActionItem]:
        """Extract actionable items and tasks from content"""
        try:
            action_items = []
            
            # Extract action items using patterns
            for pattern in self.meeting_patterns['action_items']:
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    description = match.group(1).strip()
                    
                    # Try to extract assignee
                    assignee = None
                    for assign_pattern in self.meeting_patterns['assignments']:
                        assign_match = re.search(assign_pattern, match.group(), re.IGNORECASE)
                        if assign_match:
                            assignee = assign_match.group(1)
                            break
                    
                    # Try to extract deadline
                    deadline = None
                    for deadline_pattern in self.meeting_patterns['deadlines']:
                        deadline_match = re.search(deadline_pattern, match.group(), re.IGNORECASE)
                        if deadline_match:
                            deadline = deadline_match.group(1)
                            break
                    
                    # Determine priority based on keywords
                    priority = "medium"
                    description_lower = description.lower()
                    if any(word in description_lower for word in ['urgent', 'asap', 'immediately', 'critical']):
                        priority = "urgent"
                    elif any(word in description_lower for word in ['high priority', 'important', 'crucial']):
                        priority = "high"
                    elif any(word in description_lower for word in ['low priority', 'when possible', 'eventually']):
                        priority = "low"
                    
                    # Get context (surrounding text)
                    start = max(0, match.start() - 100)
                    end = min(len(transcript), match.end() + 100)
                    context = transcript[start:end].strip()
                    
                    action_item = MeetingActionItem(
                        description=description,
                        assignee=assignee,
                        deadline=deadline,
                        priority=priority,
                        context=context,
                        confidence=0.8
                    )
                    
                    action_items.append(action_item)
            
            # For academic content, extract learning tasks
            if content_type in ["academic", "mixed"]:
                academic_tasks = await self._extract_academic_tasks(transcript)
                action_items.extend(academic_tasks)
            
            # Remove duplicates and sort by priority
            unique_items = []
            seen_descriptions = set()
            
            for item in action_items:
                if item.description not in seen_descriptions:
                    unique_items.append(item)
                    seen_descriptions.add(item.description)
            
            # Sort by priority
            priority_order = {"urgent": 4, "high": 3, "medium": 2, "low": 1}
            unique_items.sort(key=lambda x: priority_order.get(x.priority, 2), reverse=True)
            
            return unique_items[:15]  # Top 15 action items
            
        except Exception as e:
            logger.error(f"Error extracting actionable items: {e}")
            return []
    
    async def _extract_academic_tasks(self, transcript: str) -> List[MeetingActionItem]:
        """Extract academic tasks and assignments from content"""
        try:
            tasks = []
            
            academic_task_patterns = [
                r'homework[:\s]+(.+)',
                r'assignment[:\s]+(.+)',
                r'read (?:chapter|pages?) (.+)',
                r'study (.+)',
                r'review (.+)',
                r'practice (.+)',
                r'complete (.+)',
                r'prepare (?:for )?(.+)'
            ]
            
            for pattern in academic_task_patterns:
                matches = re.finditer(pattern, transcript, re.IGNORECASE)
                for match in matches:
                    description = match.group(1).strip()
                    
                    # Academic tasks are typically medium priority
                    task = MeetingActionItem(
                        description=f"Academic task: {description}",
                        assignee="Students",
                        priority="medium",
                        context=match.group(),
                        confidence=0.7
                    )
                    
                    tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error extracting academic tasks: {e}")
            return []
    
    async def _analyze_speaker_contributions(self, speaker_segments: List, 
                                          content_type: str) -> Dict:
        """Analyze how each speaker contributed to the content"""
        try:
            contributions = {}
            
            for segment in speaker_segments:
                speaker_id = segment.get('speaker_id', 'Unknown')
                content = segment.get('content', '')
                
                if speaker_id not in contributions:
                    contributions[speaker_id] = {
                        'total_words': 0,
                        'key_concepts_introduced': [],
                        'questions_asked': 0,
                        'decisions_made': 0,
                        'action_items_created': 0,
                        'speaking_time': 0.0,
                        'contribution_type': 'participant'
                    }
                
                # Count words
                word_count = len(content.split())
                contributions[speaker_id]['total_words'] += word_count
                
                # Count questions
                question_count = content.count('?')
                contributions[speaker_id]['questions_asked'] += question_count
                
                # Check for decision-making language
                decision_indicators = ['we decide', 'final decision', 'agreed', 'resolved']
                if any(indicator in content.lower() for indicator in decision_indicators):
                    contributions[speaker_id]['decisions_made'] += 1
                
                # Check for action item creation
                action_indicators = ['action item', 'we need to', 'task', 'assignment']
                if any(indicator in content.lower() for indicator in action_indicators):
                    contributions[speaker_id]['action_items_created'] += 1
                
                # Add speaking time if available
                if hasattr(segment, 'duration'):
                    contributions[speaker_id]['speaking_time'] += segment.duration
                
                # Determine contribution type
                if word_count > 100:  # Substantial contribution
                    if any(word in content.lower() for word in ['explain', 'definition', 'concept', 'theory']):
                        contributions[speaker_id]['contribution_type'] = 'educator'
                    elif question_count > 2:
                        contributions[speaker_id]['contribution_type'] = 'questioner'
                    elif contributions[speaker_id]['decisions_made'] > 0:
                        contributions[speaker_id]['contribution_type'] = 'decision_maker'
            
            return contributions
            
        except Exception as e:
            logger.error(f"Error analyzing speaker contributions: {e}")
            return {}
    
    async def _generate_executive_summary(self, transcript: str, content_type: str,
                                        key_concepts: List[KeyConcept],
                                        insights: List[AcademicInsight]) -> str:
        """Generate executive summary using AI and extracted insights"""
        try:
            # Try AI-powered summary first
            if hasattr(openai, 'api_key') and openai.api_key:
                try:
                    concepts_text = ", ".join([c.concept for c in key_concepts[:5]])
                    
                    prompt = f"""
                    Create a comprehensive executive summary for this {content_type} content.
                    Key concepts covered: {concepts_text}
                    
                    Focus on:
                    1. Main objectives and outcomes
                    2. Key decisions or learning points
                    3. Important concepts and their significance
                    4. Next steps or conclusions
                    
                    Content: {transcript[:1500]}...
                    
                    Keep the summary concise but comprehensive (200-300 words).
                    """
                    
                    response = await openai.ChatCompletion.acreate(
                        model="gpt-3.5-turbo",
                        messages=[{
                            "role": "system",
                            "content": f"You are an expert at creating {content_type} summaries."
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        max_tokens=400,
                        temperature=0.3
                    )
                    
                    return response.choices[0].message.content.strip()
                    
                except Exception as ai_error:
                    logger.warning(f"AI summary generation failed: {ai_error}")
            
            # Fallback: rule-based summary
            summary_parts = []
            
            # Add content type and overview
            if content_type == "academic":
                summary_parts.append(f"This academic content covers {len(key_concepts)} key concepts.")
            elif content_type == "meeting":
                summary_parts.append(f"This meeting discussion addressed {len(key_concepts)} main topics.")
            else:
                summary_parts.append(f"This content combines academic and meeting elements.")
            
            # Add key concepts
            if key_concepts:
                top_concepts = [c.concept for c in key_concepts[:3]]
                summary_parts.append(f"Primary concepts include: {', '.join(top_concepts)}.")
            
            # Add insights
            high_importance_insights = [i for i in insights if i.importance_score > 0.8]
            if high_importance_insights:
                summary_parts.append(f"Key insights: {len(high_importance_insights)} critical points were identified.")
            
            # Add conclusion
            if content_type == "academic":
                summary_parts.append("The content provides educational value through structured concept presentation.")
            else:
                summary_parts.append("The discussion concluded with actionable outcomes and clear next steps.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"Summary of {content_type} content with {len(key_concepts)} key concepts discussed."
    
    async def _generate_learning_objectives(self, transcript: str, content_type: str,
                                          key_concepts: List[KeyConcept]) -> List[str]:
        """Generate learning objectives based on content analysis"""
        try:
            objectives = []
            
            if content_type in ["academic", "mixed"]:
                # Academic learning objectives
                for concept in key_concepts[:5]:
                    if concept.definition and concept.definition != "Definition not found":
                        objectives.append(f"Understand the concept of {concept.concept}")
                    
                    if concept.examples:
                        objectives.append(f"Apply {concept.concept} through practical examples")
                
                # Add general academic objectives
                if any("formula" in transcript.lower() or "equation" in transcript.lower()):
                    objectives.append("Master relevant formulas and equations")
                
                if any("problem" in transcript.lower() or "solve" in transcript.lower()):
                    objectives.append("Develop problem-solving skills in the subject area")
            
            if content_type in ["meeting", "mixed"]:
                # Meeting learning objectives
                objectives.append("Understand key decisions made and their rationale")
                objectives.append("Identify action items and responsibilities")
                
                if any("strategy" in transcript.lower() or "plan" in transcript.lower()):
                    objectives.append("Comprehend strategic planning discussions")
            
            # Limit to top 8 objectives
            return objectives[:8]
            
        except Exception as e:
            logger.error(f"Error generating learning objectives: {e}")
            return [f"Understand the main concepts from this {content_type} content"]
    
    async def _generate_follow_up_suggestions(self, transcript: str, content_type: str,
                                            action_items: List[MeetingActionItem],
                                            key_concepts: List[KeyConcept]) -> List[str]:
        """Generate follow-up suggestions and next steps"""
        try:
            suggestions = []
            
            # Based on action items
            if action_items:
                urgent_items = [item for item in action_items if item.priority == "urgent"]
                if urgent_items:
                    suggestions.append(f"Address {len(urgent_items)} urgent action items immediately")
                
                if any(item.deadline for item in action_items):
                    suggestions.append("Review deadlines for assigned tasks and create calendar reminders")
            
            # Based on content type
            if content_type in ["academic", "mixed"]:
                if key_concepts:
                    suggestions.append("Review and study the key concepts identified")
                    suggestions.append("Create flashcards or study materials for important terms")
                
                if any("homework" in transcript.lower() or "assignment" in transcript.lower()):
                    suggestions.append("Complete assigned homework and practice exercises")
                
                suggestions.append("Seek clarification on any unclear concepts from instructor")
            
            if content_type in ["meeting", "mixed"]:
                suggestions.append("Distribute meeting summary to all participants")
                suggestions.append("Schedule follow-up meetings if necessary")
                
                if action_items:
                    suggestions.append("Track progress on action items and report back")
            
            # General suggestions
            suggestions.append("Document key insights for future reference")
            suggestions.append("Share relevant information with team members or classmates")
            
            return suggestions[:10]  # Limit to 10 suggestions
            
        except Exception as e:
            logger.error(f"Error generating follow-up suggestions: {e}")
            return ["Review the content and identify next steps"]

# Export main class
__all__ = ['AcademicMeetingAI', 'AcademicInsight', 'MeetingActionItem', 'KeyConcept', 'IntelligentSummary']

if __name__ == "__main__":
    # Test the Academic & Meeting AI
    async def test_ai_system():
        ai_system = AcademicMeetingAI()
        
        # Sample academic content
        academic_text = """
        Today we will cover the concept of derivatives in calculus. A derivative is defined as 
        the rate of change of a function with respect to its variable. For example, if we have 
        f(x) = x^2, then the derivative f'(x) = 2x. This is a fundamental principle in mathematics.
        In conclusion, understanding derivatives is essential for advanced calculus.
        """
        
        # Sample meeting content
        meeting_text = """
        Let's start today's project meeting. First agenda item is the budget review. 
        We need to reduce costs by 10% this quarter. John will analyze the expenses and 
        report back by Friday. The decision is to postpone the marketing campaign. 
        Action item: Sarah should contact the vendors about pricing.
        """
        
        print("=== ACADEMIC & MEETING AI TEST ===")
        
        try:
            # Test academic analysis
            academic_result = await ai_system.analyze_content(academic_text, "academic")
            print(f"Academic Analysis:")
            print(f"- Key concepts: {len(academic_result.main_concepts)}")
            print(f"- Insights: {len(academic_result.key_insights)}")
            print(f"- Learning objectives: {len(academic_result.learning_objectives)}")
            
            # Test meeting analysis
            meeting_result = await ai_system.analyze_content(meeting_text, "meeting")
            print(f"\nMeeting Analysis:")
            print(f"- Key concepts: {len(meeting_result.main_concepts)}")
            print(f"- Action items: {len(meeting_result.actionable_items)}")
            print(f"- Follow-up suggestions: {len(meeting_result.follow_up_suggestions)}")
            
            print("\nTest completed successfully!")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run test
    asyncio.run(test_ai_system())