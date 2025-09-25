"""
Student-Friendly Output Formats
Öğrenciler için optimize edilmiş çıktı formatları ve görselleştirmeler

Bu modül öğrencilerin en çok ihtiyaç duyduğu formatları sağlar:
- Structured notes with bullet points and chapters
- Searchable transcripts with timestamp navigation
- Visual mind maps and concept connections
- Study guides and flashcards
- Interactive HTML outputs with navigation
- PDF exports with professional formatting

Made by Mehmet Arda Çekiç © 2025
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import jinja2
from jinja2 import Environment, DictLoader
import markdown
from markdown.extensions import codehilite, toc
import pdfkit
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus.tableofcontents import TableOfContents
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
import numpy as np
from collections import defaultdict, Counter
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StudyNote:
    """Individual study note with metadata"""
    title: str
    content: str
    note_type: str  # concept, definition, example, important, question
    timestamp: Optional[float] = None
    chapter: Optional[str] = None
    importance_level: int = 1  # 1-5 scale
    tags: List[str] = field(default_factory=list)
    related_notes: List[str] = field(default_factory=list)

@dataclass
class ChapterSection:
    """Chapter or section organization"""
    chapter_id: str
    title: str
    start_time: float
    end_time: float
    duration: float
    notes: List[StudyNote]
    subsections: List['ChapterSection'] = field(default_factory=list)
    summary: str = ""
    key_concepts: List[str] = field(default_factory=list)

@dataclass
class StudyMaterial:
    """Complete study material package"""
    title: str
    subject: str
    date_created: datetime
    total_duration: float
    chapters: List[ChapterSection]
    all_notes: List[StudyNote]
    concept_map: Dict
    glossary: Dict[str, str]
    study_questions: List[str]
    flashcards: List[Dict]
    metadata: Dict

class StudentFriendlyFormatter:
    """
    Advanced formatter creating student-optimized outputs
    Multiple formats: HTML, PDF, Markdown, Interactive study guides
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize formatter with output directory"""
        
        self.output_dir = Path(output_dir) if output_dir else Path("./study_materials")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 templating
        self.setup_templates()
        
        # Color schemes for different note types
        self.color_schemes = {
            'concept': {'bg': '#E3F2FD', 'border': '#1976D2', 'text': '#0D47A1'},
            'definition': {'bg': '#F3E5F5', 'border': '#7B1FA2', 'text': '#4A148C'},
            'example': {'bg': '#E8F5E8', 'border': '#388E3C', 'text': '#1B5E20'},
            'important': {'bg': '#FFF3E0', 'border': '#F57C00', 'text': '#E65100'},
            'question': {'bg': '#FFEBEE', 'border': '#D32F2F', 'text': '#B71C1C'}
        }
        
        logger.info("Student-Friendly Formatter initialized")
    
    def setup_templates(self):
        """Setup HTML templates for different output formats"""
        
        # Main study guide template
        study_guide_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }} - Study Guide</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    line-height: 1.6; 
                    margin: 0; 
                    background: #f5f5f5;
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                .header { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 2rem; 
                    text-align: center;
                }
                .nav { 
                    background: #333; 
                    padding: 1rem; 
                    position: sticky; 
                    top: 0;
                    z-index: 100;
                }
                .nav ul { 
                    list-style: none; 
                    margin: 0; 
                    padding: 0; 
                    display: flex;
                    flex-wrap: wrap;
                }
                .nav li { 
                    margin-right: 2rem; 
                }
                .nav a { 
                    color: white; 
                    text-decoration: none; 
                    padding: 0.5rem 1rem;
                    border-radius: 4px;
                    transition: background 0.3s;
                }
                .nav a:hover { 
                    background: #555; 
                }
                .content { 
                    display: flex; 
                }
                .sidebar { 
                    width: 250px; 
                    background: #f8f9fa; 
                    padding: 1rem;
                    border-right: 1px solid #dee2e6;
                }
                .main { 
                    flex: 1; 
                    padding: 2rem; 
                }
                .chapter { 
                    margin-bottom: 3rem;
                    border-left: 4px solid #667eea;
                    padding-left: 1rem;
                }
                .chapter-title { 
                    color: #333; 
                    border-bottom: 2px solid #667eea; 
                    padding-bottom: 0.5rem;
                    margin-bottom: 1rem;
                }
                .note { 
                    margin: 1rem 0; 
                    padding: 1rem; 
                    border-radius: 8px; 
                    border-left: 4px solid;
                    transition: transform 0.2s;
                }
                .note:hover {
                    transform: translateX(5px);
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                .note-concept { 
                    background: {{ color_schemes.concept.bg }}; 
                    border-color: {{ color_schemes.concept.border }};
                    color: {{ color_schemes.concept.text }};
                }
                .note-definition { 
                    background: {{ color_schemes.definition.bg }}; 
                    border-color: {{ color_schemes.definition.border }};
                    color: {{ color_schemes.definition.text }};
                }
                .note-example { 
                    background: {{ color_schemes.example.bg }}; 
                    border-color: {{ color_schemes.example.border }};
                    color: {{ color_schemes.example.text }};
                }
                .note-important { 
                    background: {{ color_schemes.important.bg }}; 
                    border-color: {{ color_schemes.important.border }};
                    color: {{ color_schemes.important.text }};
                }
                .note-question { 
                    background: {{ color_schemes.question.bg }}; 
                    border-color: {{ color_schemes.question.border }};
                    color: {{ color_schemes.question.text }};
                }
                .timestamp { 
                    font-size: 0.8rem; 
                    color: #666; 
                    float: right;
                    background: rgba(0,0,0,0.1);
                    padding: 0.2rem 0.5rem;
                    border-radius: 4px;
                }
                .search-box { 
                    width: 100%; 
                    padding: 0.5rem; 
                    margin-bottom: 1rem;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .concept-map { 
                    text-align: center; 
                    margin: 2rem 0; 
                }
                .glossary { 
                    background: #f8f9fa; 
                    padding: 1rem; 
                    border-radius: 8px;
                    margin: 2rem 0;
                }
                .glossary-term { 
                    font-weight: bold; 
                    color: #495057; 
                }
                .flashcard { 
                    background: white; 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 1rem; 
                    margin: 1rem 0;
                    cursor: pointer;
                    transition: all 0.3s;
                }
                .flashcard:hover { 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15); 
                    transform: translateY(-2px);
                }
                .progress-bar { 
                    background: #e9ecef; 
                    border-radius: 4px; 
                    overflow: hidden;
                    margin: 1rem 0;
                }
                .progress-fill { 
                    background: linear-gradient(90deg, #28a745, #20c997); 
                    height: 20px; 
                    transition: width 0.3s;
                }
                @media (max-width: 768px) {
                    .content { flex-direction: column; }
                    .sidebar { width: 100%; }
                    .nav ul { flex-direction: column; }
                    .nav li { margin-right: 0; margin-bottom: 0.5rem; }
                }
            </style>
            <script>
                function searchNotes() {
                    const query = document.getElementById('searchBox').value.toLowerCase();
                    const notes = document.querySelectorAll('.note');
                    let visibleCount = 0;
                    
                    notes.forEach(note => {
                        const text = note.textContent.toLowerCase();
                        if (text.includes(query)) {
                            note.style.display = 'block';
                            visibleCount++;
                        } else {
                            note.style.display = query ? 'none' : 'block';
                        }
                    });
                    
                    document.getElementById('searchResults').textContent = 
                        query ? `Found ${visibleCount} matching notes` : '';
                }
                
                function toggleFlashcard(card) {
                    const answer = card.querySelector('.flashcard-answer');
                    const isVisible = answer.style.display !== 'none';
                    answer.style.display = isVisible ? 'none' : 'block';
                }
                
                function jumpToTime(seconds) {
                    // This would integrate with audio/video player if available
                    console.log(`Jump to ${seconds} seconds`);
                }
            </script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{{ title }}</h1>
                    <p>{{ subject }} • {{ date_created.strftime('%B %d, %Y') }}</p>
                    <p>Duration: {{ (total_duration/60)|round(1) }} minutes • {{ chapters|length }} chapters</p>
                </div>
                
                <nav class="nav">
                    <ul>
                        <li><a href="#overview">Overview</a></li>
                        <li><a href="#chapters">Chapters</a></li>
                        <li><a href="#concepts">Concept Map</a></li>
                        <li><a href="#glossary">Glossary</a></li>
                        <li><a href="#flashcards">Flashcards</a></li>
                        <li><a href="#questions">Study Questions</a></li>
                    </ul>
                </nav>
                
                <div class="content">
                    <div class="sidebar">
                        <input type="text" id="searchBox" class="search-box" 
                               placeholder="Search notes..." onkeyup="searchNotes()">
                        <div id="searchResults"></div>
                        
                        <h3>Quick Navigation</h3>
                        <ul>
                        {% for chapter in chapters %}
                            <li><a href="#chapter-{{ chapter.chapter_id }}">{{ chapter.title }}</a></li>
                        {% endfor %}
                        </ul>
                        
                        <h3>Study Progress</h3>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 0%" id="studyProgress"></div>
                        </div>
                    </div>
                    
                    <div class="main">
                        <section id="overview">
                            <h2>Study Guide Overview</h2>
                            <p>This study guide contains {{ all_notes|length }} notes organized into {{ chapters|length }} chapters.</p>
                            <p>Use the search function to quickly find specific topics, or navigate through chapters sequentially.</p>
                        </section>
                        
                        <section id="chapters">
                            <h2>Chapter Contents</h2>
                            {% for chapter in chapters %}
                            <div class="chapter" id="chapter-{{ chapter.chapter_id }}">
                                <h3 class="chapter-title">
                                    {{ chapter.title }}
                                    <span class="timestamp">{{ (chapter.start_time/60)|round(1) }}-{{ (chapter.end_time/60)|round(1) }} min</span>
                                </h3>
                                
                                {% if chapter.summary %}
                                <div class="note note-important">
                                    <strong>Chapter Summary:</strong> {{ chapter.summary }}
                                </div>
                                {% endif %}
                                
                                {% for note in chapter.notes %}
                                <div class="note note-{{ note.note_type }}">
                                    {% if note.timestamp %}
                                    <span class="timestamp" onclick="jumpToTime({{ note.timestamp }})">{{ (note.timestamp/60)|round(1) }}min</span>
                                    {% endif %}
                                    <h4>{{ note.title }}</h4>
                                    <div>{{ note.content|safe }}</div>
                                    {% if note.tags %}
                                    <div style="margin-top: 0.5rem;">
                                        {% for tag in note.tags %}
                                        <span style="background: rgba(0,0,0,0.1); padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem; margin-right: 0.5rem;">{{ tag }}</span>
                                        {% endfor %}
                                    </div>
                                    {% endif %}
                                </div>
                                {% endfor %}
                            </div>
                            {% endfor %}
                        </section>
                        
                        <section id="concepts" class="concept-map">
                            <h2>Concept Map</h2>
                            <div id="conceptMapContainer">
                                <!-- Concept map visualization would be inserted here -->
                                <p>Interactive concept map showing relationships between key topics.</p>
                            </div>
                        </section>
                        
                        <section id="glossary" class="glossary">
                            <h2>Glossary</h2>
                            {% for term, definition in glossary.items() %}
                            <div style="margin-bottom: 1rem;">
                                <span class="glossary-term">{{ term }}:</span> {{ definition }}
                            </div>
                            {% endfor %}
                        </section>
                        
                        <section id="flashcards">
                            <h2>Study Flashcards</h2>
                            {% for card in flashcards %}
                            <div class="flashcard" onclick="toggleFlashcard(this)">
                                <div class="flashcard-question">
                                    <strong>Q:</strong> {{ card.front }}
                                </div>
                                <div class="flashcard-answer" style="display: none; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #eee;">
                                    <strong>A:</strong> {{ card.back }}
                                </div>
                            </div>
                            {% endfor %}
                        </section>
                        
                        <section id="questions">
                            <h2>Study Questions</h2>
                            {% for question in study_questions %}
                            <div class="note note-question">
                                <strong>{{ loop.index }}.</strong> {{ question }}
                            </div>
                            {% endfor %}
                        </section>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Simple notes template
        notes_template = """
        # {{ title }} - Study Notes
        
        **Subject:** {{ subject }}  
        **Date:** {{ date_created.strftime('%B %d, %Y') }}  
        **Duration:** {{ (total_duration/60)|round(1) }} minutes
        
        ---
        
        ## Table of Contents
        {% for chapter in chapters %}
        - [{{ chapter.title }}](#{{ chapter.title|lower|replace(' ', '-') }})
        {% endfor %}
        
        ---
        
        {% for chapter in chapters %}
        ## {{ chapter.title }}
        
        **Time:** {{ (chapter.start_time/60)|round(1) }}-{{ (chapter.end_time/60)|round(1) }} minutes
        
        {% if chapter.summary %}
        **Summary:** {{ chapter.summary }}
        {% endif %}
        
        {% for note in chapter.notes %}
        ### {{ note.title }}
        
        {% if note.note_type == 'definition' %}🔍 **Definition**{% endif %}
        {% if note.note_type == 'concept' %}💡 **Concept**{% endif %}
        {% if note.note_type == 'example' %}📝 **Example**{% endif %}
        {% if note.note_type == 'important' %}⚠️ **Important**{% endif %}
        {% if note.note_type == 'question' %}❓ **Question**{% endif %}
        
        {{ note.content }}
        
        {% if note.timestamp %}*Timestamp: {{ (note.timestamp/60)|round(1) }} minutes*{% endif %}
        
        {% endfor %}
        
        {% endfor %}
        
        ---
        
        ## Glossary
        
        {% for term, definition in glossary.items() %}
        **{{ term }}:** {{ definition }}
        
        {% endfor %}
        
        ## Study Questions
        
        {% for question in study_questions %}
        {{ loop.index }}. {{ question }}
        {% endfor %}
        """
        
        self.templates = {
            'study_guide': study_guide_template,
            'notes': notes_template
        }
        
        # Create Jinja2 environment
        self.jinja_env = Environment(
            loader=DictLoader(self.templates),
            autoescape=True
        )
        
        # Add custom filters
        self.jinja_env.filters['round'] = round
    
    async def create_study_materials(self, transcript: str, processing_results: Dict,
                                   title: str = "Study Session",
                                   subject: str = "General") -> StudyMaterial:
        """Create comprehensive study materials from processing results"""
        try:
            logger.info("Creating comprehensive study materials...")
            
            # Extract data from processing results
            speaker_segments = processing_results.get('speaker_segments', [])
            academic_analysis = processing_results.get('academic_analysis', {})
            meeting_insights = processing_results.get('meeting_insights', {})
            
            # Create chapters from content structure
            chapters = await self._create_chapters(transcript, processing_results)
            
            # Extract all notes
            all_notes = []
            for chapter in chapters:
                all_notes.extend(chapter.notes)
            
            # Create concept map
            concept_map = await self._create_concept_map(processing_results)
            
            # Create glossary
            glossary = await self._create_glossary(processing_results)
            
            # Generate study questions
            study_questions = await self._generate_study_questions(processing_results)
            
            # Create flashcards
            flashcards = await self._create_flashcards(processing_results)
            
            # Create study material object
            study_material = StudyMaterial(
                title=title,
                subject=subject,
                date_created=datetime.now(),
                total_duration=processing_results.get('processing_metadata', {}).get('total_duration', 0),
                chapters=chapters,
                all_notes=all_notes,
                concept_map=concept_map,
                glossary=glossary,
                study_questions=study_questions,
                flashcards=flashcards,
                metadata=processing_results.get('processing_metadata', {})
            )
            
            logger.info(f"Created study materials with {len(chapters)} chapters and {len(all_notes)} notes")
            return study_material
            
        except Exception as e:
            logger.error(f"Error creating study materials: {e}")
            raise
    
    async def _create_chapters(self, transcript: str, processing_results: Dict) -> List[ChapterSection]:
        """Create organized chapters from content"""
        try:
            chapters = []
            
            # Try to use academic analysis if available
            academic_analysis = processing_results.get('academic_analysis', {})
            if academic_analysis and 'lecture_segments' in academic_analysis:
                segments = academic_analysis['lecture_segments']
                
                current_chapter = None
                chapter_counter = 1
                
                for segment in segments:
                    segment_type = segment.get('segment_type', 'main_topic')
                    start_time = segment.get('start_time', 0)
                    end_time = segment.get('end_time', 0)
                    content = segment.get('content', '')
                    
                    # Start new chapter for introduction or major topic changes
                    if (segment_type == 'introduction' or 
                        (current_chapter and len(current_chapter.notes) > 5)):
                        
                        if current_chapter:
                            chapters.append(current_chapter)
                        
                        # Create new chapter
                        chapter_title = f"Chapter {chapter_counter}"
                        if segment_type == 'introduction':
                            chapter_title = "Introduction"
                        elif segment_type == 'conclusion':
                            chapter_title = "Conclusion"
                        else:
                            chapter_title = f"Topic {chapter_counter}"
                        
                        current_chapter = ChapterSection(
                            chapter_id=f"ch_{chapter_counter}",
                            title=chapter_title,
                            start_time=start_time,
                            end_time=end_time,
                            duration=0,
                            notes=[],
                            summary=segment.get('summary', '')
                        )
                        chapter_counter += 1
                    
                    if current_chapter:
                        # Create note from segment
                        note_type = self._classify_note_type(content, segment_type)
                        note_title = self._extract_note_title(content)
                        
                        note = StudyNote(
                            title=note_title,
                            content=content,
                            note_type=note_type,
                            timestamp=start_time,
                            chapter=current_chapter.chapter_id,
                            importance_level=int(segment.get('importance_score', 0.5) * 5),
                            tags=segment.get('key_terms', [])
                        )
                        
                        current_chapter.notes.append(note)
                        current_chapter.end_time = end_time
                        current_chapter.duration = current_chapter.end_time - current_chapter.start_time
                
                # Don't forget the last chapter
                if current_chapter:
                    chapters.append(current_chapter)
            
            # Fallback: create chapters from transcript structure
            if not chapters:
                chapters = await self._create_chapters_from_transcript(transcript)
            
            return chapters
            
        except Exception as e:
            logger.error(f"Error creating chapters: {e}")
            return []
    
    def _classify_note_type(self, content: str, segment_type: str) -> str:
        """Classify note type based on content"""
        content_lower = content.lower()
        
        # Check for definitions
        if any(word in content_lower for word in ['is defined as', 'means', 'refers to', 'definition']):
            return 'definition'
        
        # Check for examples
        if any(word in content_lower for word in ['for example', 'for instance', 'such as']):
            return 'example'
        
        # Check for questions
        if '?' in content:
            return 'question'
        
        # Check for important information
        if any(word in content_lower for word in ['important', 'crucial', 'key', 'remember']):
            return 'important'
        
        # Default to concept
        return 'concept'
    
    def _extract_note_title(self, content: str) -> str:
        """Extract appropriate title from note content"""
        # Take first meaningful phrase (up to first sentence or 50 characters)
        sentences = content.split('.')
        first_sentence = sentences[0].strip()
        
        if len(first_sentence) > 50:
            # Take first 50 characters and find last complete word
            truncated = first_sentence[:50]
            last_space = truncated.rfind(' ')
            if last_space > 20:  # Make sure we don't truncate too much
                return truncated[:last_space] + "..."
        
        return first_sentence if first_sentence else "Study Note"
    
    async def _create_chapters_from_transcript(self, transcript: str) -> List[ChapterSection]:
        """Fallback method to create chapters from raw transcript"""
        try:
            # Split transcript into roughly equal chunks
            words = transcript.split()
            words_per_chapter = max(200, len(words) // 5)  # 5 chapters max, 200 words min
            
            chapters = []
            for i in range(0, len(words), words_per_chapter):
                chunk_words = words[i:i + words_per_chapter]
                chunk_text = ' '.join(chunk_words)
                
                # Estimate timing (assuming 150 words per minute)
                start_time = (i / 150) * 60
                end_time = ((i + len(chunk_words)) / 150) * 60
                
                chapter = ChapterSection(
                    chapter_id=f"ch_{len(chapters) + 1}",
                    title=f"Section {len(chapters) + 1}",
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    notes=[],
                    summary=chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                )
                
                # Create a single note for this chapter
                note = StudyNote(
                    title=f"Section {len(chapters) + 1} Content",
                    content=chunk_text,
                    note_type='concept',
                    timestamp=start_time,
                    chapter=chapter.chapter_id,
                    importance_level=3
                )
                
                chapter.notes.append(note)
                chapters.append(chapter)
            
            return chapters
            
        except Exception as e:
            logger.error(f"Error creating chapters from transcript: {e}")
            return []
    
    async def _create_concept_map(self, processing_results: Dict) -> Dict:
        """Create concept map data structure"""
        try:
            concept_map = {
                'nodes': [],
                'edges': [],
                'visualization_data': None
            }
            
            # Extract concepts from processing results
            concepts = []
            
            # From academic analysis
            academic_analysis = processing_results.get('academic_analysis', {})
            if 'key_terms_explained' in academic_analysis:
                for term, details in academic_analysis['key_terms_explained'].items():
                    concepts.append({
                        'id': term,
                        'label': term,
                        'type': 'concept',
                        'definition': details.get('definition', ''),
                        'related': details.get('related_terms', [])
                    })
            
            # From meeting insights
            meeting_insights = processing_results.get('meeting_insights', {})
            if 'key_topics' in meeting_insights:
                for topic in meeting_insights['key_topics']:
                    concepts.append({
                        'id': topic,
                        'label': topic,
                        'type': 'topic',
                        'definition': f'Key topic discussed: {topic}',
                        'related': []
                    })
            
            # Create nodes
            for concept in concepts:
                concept_map['nodes'].append({
                    'id': concept['id'],
                    'label': concept['label'],
                    'type': concept['type'],
                    'definition': concept['definition']
                })
            
            # Create edges based on relationships
            for concept in concepts:
                for related in concept.get('related', []):
                    if any(node['id'] == related for node in concept_map['nodes']):
                        concept_map['edges'].append({
                            'source': concept['id'],
                            'target': related,
                            'relationship': 'related'
                        })
            
            # Generate visualization if NetworkX is available
            try:
                concept_map['visualization_data'] = await self._generate_concept_visualization(concept_map)
            except Exception as viz_error:
                logger.warning(f"Could not generate concept visualization: {viz_error}")
            
            return concept_map
            
        except Exception as e:
            logger.error(f"Error creating concept map: {e}")
            return {'nodes': [], 'edges': [], 'visualization_data': None}
    
    async def _generate_concept_visualization(self, concept_map: Dict) -> str:
        """Generate concept map visualization"""
        try:
            if not concept_map['nodes']:
                return None
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for node in concept_map['nodes']:
                G.add_node(node['id'], label=node['label'], type=node['type'])
            
            # Add edges
            for edge in concept_map['edges']:
                if G.has_node(edge['source']) and G.has_node(edge['target']):
                    G.add_edge(edge['source'], edge['target'])
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes
            node_colors = []
            for node in G.nodes():
                node_data = next((n for n in concept_map['nodes'] if n['id'] == node), {})
                if node_data.get('type') == 'concept':
                    node_colors.append('#E3F2FD')
                else:
                    node_colors.append('#F3E5F5')
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9)
            nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6, width=2)
            
            # Draw labels
            labels = {}
            for node in G.nodes():
                node_data = next((n for n in concept_map['nodes'] if n['id'] == node), {})
                labels[node] = node_data.get('label', node)[:15]  # Truncate long labels
            
            nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
            
            plt.title("Concept Map", size=16, weight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error generating concept visualization: {e}")
            return None
    
    async def _create_glossary(self, processing_results: Dict) -> Dict[str, str]:
        """Create glossary from processing results"""
        try:
            glossary = {}
            
            # From academic analysis
            academic_analysis = processing_results.get('academic_analysis', {})
            if 'key_terms_explained' in academic_analysis:
                for term, details in academic_analysis['key_terms_explained'].items():
                    definition = details.get('definition', '')
                    if definition and definition != "Definition not found":
                        glossary[term] = definition
            
            # From medical AI if available
            if 'medical_analysis' in processing_results:
                medical_terms = processing_results['medical_analysis'].get('explained_terms', {})
                for term, details in medical_terms.items():
                    definition = details.get('definition', '')
                    if definition:
                        glossary[term] = definition
            
            # Sort alphabetically
            sorted_glossary = dict(sorted(glossary.items()))
            
            return sorted_glossary
            
        except Exception as e:
            logger.error(f"Error creating glossary: {e}")
            return {}
    
    async def _generate_study_questions(self, processing_results: Dict) -> List[str]:
        """Generate study questions from content"""
        try:
            questions = []
            
            # From academic analysis
            academic_analysis = processing_results.get('academic_analysis', {})
            if 'study_materials' in academic_analysis:
                existing_questions = academic_analysis['study_materials'].get('practice_questions', [])
                for q in existing_questions:
                    questions.append(q.get('question', ''))
            
            # Generate additional questions from key terms
            if 'key_terms_explained' in academic_analysis:
                for term, details in list(academic_analysis['key_terms_explained'].items())[:5]:
                    questions.append(f"What is {term} and why is it important?")
                    if details.get('related_terms'):
                        questions.append(f"How does {term} relate to {details['related_terms'][0]}?")
            
            # From meeting insights
            meeting_insights = processing_results.get('meeting_insights', {})
            if 'action_items' in meeting_insights:
                if meeting_insights['action_items']:
                    questions.append("What are the key action items from this session?")
                    questions.append("Who is responsible for implementing the discussed changes?")
            
            # Remove duplicates and limit
            unique_questions = list(set(questions))
            return unique_questions[:15]  # Limit to 15 questions
            
        except Exception as e:
            logger.error(f"Error generating study questions: {e}")
            return ["What were the main topics covered in this session?"]
    
    async def _create_flashcards(self, processing_results: Dict) -> List[Dict]:
        """Create flashcards from processing results"""
        try:
            flashcards = []
            
            # From academic analysis
            academic_analysis = processing_results.get('academic_analysis', {})
            if 'study_materials' in academic_analysis:
                existing_cards = academic_analysis['study_materials'].get('flashcards', [])
                flashcards.extend(existing_cards)
            
            # Create cards from glossary
            if 'key_terms_explained' in academic_analysis:
                for term, details in list(academic_analysis['key_terms_explained'].items())[:10]:
                    definition = details.get('definition', '')
                    if definition and definition != "Definition not found":
                        flashcards.append({
                            'front': f"What is {term}?",
                            'back': definition,
                            'category': 'definition'
                        })
            
            # Limit to prevent overwhelming
            return flashcards[:20]
            
        except Exception as e:
            logger.error(f"Error creating flashcards: {e}")
            return []
    
    async def export_html_study_guide(self, study_material: StudyMaterial, 
                                    filename: Optional[str] = None) -> str:
        """Export study material as interactive HTML study guide"""
        try:
            if not filename:
                filename = f"{study_material.title.lower().replace(' ', '_')}_study_guide.html"
            
            output_path = self.output_dir / filename
            
            # Render template
            template = self.jinja_env.get_template('study_guide')
            html_content = template.render(
                title=study_material.title,
                subject=study_material.subject,
                date_created=study_material.date_created,
                total_duration=study_material.total_duration,
                chapters=study_material.chapters,
                all_notes=study_material.all_notes,
                concept_map=study_material.concept_map,
                glossary=study_material.glossary,
                study_questions=study_material.study_questions,
                flashcards=study_material.flashcards,
                color_schemes=self.color_schemes
            )
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML study guide exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting HTML study guide: {e}")
            raise
    
    async def export_markdown_notes(self, study_material: StudyMaterial,
                                  filename: Optional[str] = None) -> str:
        """Export study material as Markdown notes"""
        try:
            if not filename:
                filename = f"{study_material.title.lower().replace(' ', '_')}_notes.md"
            
            output_path = self.output_dir / filename
            
            # Render template
            template = self.jinja_env.get_template('notes')
            markdown_content = template.render(
                title=study_material.title,
                subject=study_material.subject,
                date_created=study_material.date_created,
                total_duration=study_material.total_duration,
                chapters=study_material.chapters,
                glossary=study_material.glossary,
                study_questions=study_material.study_questions
            )
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Markdown notes exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting Markdown notes: {e}")
            raise
    
    async def export_pdf_study_guide(self, study_material: StudyMaterial,
                                   filename: Optional[str] = None) -> str:
        """Export study material as PDF"""
        try:
            if not filename:
                filename = f"{study_material.title.lower().replace(' ', '_')}_study_guide.pdf"
            
            output_path = self.output_dir / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(output_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph(study_material.title, title_style))
            story.append(Paragraph(f"{study_material.subject} • {study_material.date_created.strftime('%B %d, %Y')}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Table of Contents
            story.append(Paragraph("Table of Contents", styles['Heading2']))
            for i, chapter in enumerate(study_material.chapters):
                story.append(Paragraph(f"{i+1}. {chapter.title}", styles['Normal']))
            story.append(PageBreak())
            
            # Chapters
            for chapter in study_material.chapters:
                story.append(Paragraph(chapter.title, styles['Heading2']))
                
                if chapter.summary:
                    story.append(Paragraph(f"Summary: {chapter.summary}", styles['Normal']))
                    story.append(Spacer(1, 12))
                
                for note in chapter.notes:
                    # Note title
                    story.append(Paragraph(note.title, styles['Heading3']))
                    
                    # Note content
                    story.append(Paragraph(note.content, styles['Normal']))
                    
                    if note.timestamp:
                        story.append(Paragraph(f"Time: {note.timestamp/60:.1f} minutes", styles['Italic']))
                    
                    story.append(Spacer(1, 12))
                
                story.append(PageBreak())
            
            # Glossary
            if study_material.glossary:
                story.append(Paragraph("Glossary", styles['Heading2']))
                for term, definition in study_material.glossary.items():
                    story.append(Paragraph(f"<b>{term}:</b> {definition}", styles['Normal']))
                    story.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"PDF study guide exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting PDF study guide: {e}")
            raise
    
    async def export_json_data(self, study_material: StudyMaterial,
                             filename: Optional[str] = None) -> str:
        """Export study material as structured JSON data"""
        try:
            if not filename:
                filename = f"{study_material.title.lower().replace(' ', '_')}_data.json"
            
            output_path = self.output_dir / filename
            
            # Convert to serializable format
            data = {
                'title': study_material.title,
                'subject': study_material.subject,
                'date_created': study_material.date_created.isoformat(),
                'total_duration': study_material.total_duration,
                'chapters': [],
                'glossary': study_material.glossary,
                'study_questions': study_material.study_questions,
                'flashcards': study_material.flashcards,
                'concept_map': study_material.concept_map,
                'metadata': study_material.metadata
            }
            
            # Add chapters
            for chapter in study_material.chapters:
                chapter_data = {
                    'chapter_id': chapter.chapter_id,
                    'title': chapter.title,
                    'start_time': chapter.start_time,
                    'end_time': chapter.end_time,
                    'duration': chapter.duration,
                    'summary': chapter.summary,
                    'key_concepts': chapter.key_concepts,
                    'notes': []
                }
                
                # Add notes
                for note in chapter.notes:
                    note_data = {
                        'title': note.title,
                        'content': note.content,
                        'note_type': note.note_type,
                        'timestamp': note.timestamp,
                        'chapter': note.chapter,
                        'importance_level': note.importance_level,
                        'tags': note.tags,
                        'related_notes': note.related_notes
                    }
                    chapter_data['notes'].append(note_data)
                
                data['chapters'].append(chapter_data)
            
            # Write JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON data exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting JSON data: {e}")
            raise

# Export main class
__all__ = ['StudentFriendlyFormatter', 'StudyNote', 'ChapterSection', 'StudyMaterial']

if __name__ == "__main__":
    # Test the student-friendly formatter
    async def test_formatter():
        formatter = StudentFriendlyFormatter()
        
        # Mock processing results
        mock_results = {
            'academic_analysis': {
                'key_terms_explained': {
                    'calculus': {'definition': 'Mathematical study of continuous change'},
                    'derivative': {'definition': 'Rate of change of a function'}
                },
                'study_materials': {
                    'flashcards': [
                        {'front': 'What is calculus?', 'back': 'Mathematical study of continuous change'}
                    ]
                }
            },
            'processing_metadata': {
                'total_duration': 3600  # 1 hour
            }
        }
        
        # Mock transcript
        mock_transcript = """
        Today we will study calculus. Calculus is defined as the mathematical study of continuous change.
        The derivative is the rate of change of a function. For example, if f(x) = x^2, then f'(x) = 2x.
        This is a fundamental concept in mathematics that we'll use throughout the course.
        """
        
        print("=== STUDENT-FRIENDLY FORMATTER TEST ===")
        
        try:
            # Create study materials
            study_material = await formatter.create_study_materials(
                transcript=mock_transcript,
                processing_results=mock_results,
                title="Calculus Fundamentals",
                subject="Mathematics"
            )
            
            print(f"Study material created:")
            print(f"- Title: {study_material.title}")
            print(f"- Chapters: {len(study_material.chapters)}")
            print(f"- Notes: {len(study_material.all_notes)}")
            print(f"- Glossary terms: {len(study_material.glossary)}")
            print(f"- Flashcards: {len(study_material.flashcards)}")
            
            # Export formats
            html_path = await formatter.export_html_study_guide(study_material)
            print(f"- HTML exported: {html_path}")
            
            md_path = await formatter.export_markdown_notes(study_material)
            print(f"- Markdown exported: {md_path}")
            
            json_path = await formatter.export_json_data(study_material)
            print(f"- JSON exported: {json_path}")
            
            print("\nTest completed successfully!")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    # Run test
    asyncio.run(test_formatter())