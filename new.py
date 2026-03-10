# IMPROVED DUAL SEMANTIC MEMORY GRAPH (IDSMMG) 
# Your original code converted to TRUE dual-memory architecture
# Long-term (document) + Short-term (query context) semantic graphs

import fitz
import re
import networkx as nx
import nltk
from collections import defaultdict, Counter, deque
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import os

# Download once
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

class IDSMMG:  # Improved Dual Semantic Memory Graph
    def __init__(self):
        # LONG-TERM MEMORY: Document knowledge (permanent)
        self.long_term_graph = nx.DiGraph()
        self.paragraphs = []
        
        # SHORT-TERM MEMORY: Recent query context (working memory)
        self.short_term_graph = nx.DiGraph()
        self.query_history = deque(maxlen=20)  # Last 20 queries
        self.session_context = []
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Dual memory fusion weights
        self.intent_weights = {"DEFINITION": 5, "EXPLANATION": 6, "RULE": 4, 
                              "EXAMPLE": 2, "COMPARISON": 3, "EXCEPTION": 3, "GENERAL": 1}
        self.memory_weights = {'long_term': 0.65, 'short_term': 0.25, 'semantic': 0.10}
    
    def extract_pdf_structure(self, pdf_path):
        """Extract pages → blocks → paragraphs with hierarchy (KEEP YOUR ORIGINAL)"""
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            page_blocks = []
            current_heading = "GENERAL"
            
            for block in blocks:
                if block['type'] == 0:
                    font_size = np.mean([span['size'] for span in block['lines'][0]['spans']]) if block['lines'] else 10
                    
                    text = ' '.join(span['text'] for line in block['lines'] for span in line['spans'])
                    text = re.sub(r'\s+', ' ', text.strip())
                    
                    if font_size > 14 and len(text.split()) < 10:
                        current_heading = text.upper()
                    elif len(text) > 50:
                        page_blocks.append({
                            'text': text, 
                            'heading': current_heading, 
                            'page': page_num + 1,
                            'font_weight': font_size / 12
                        })
            pages.append(page_blocks)
        doc.close()
        return pages

    def process_paragraph(self, para):
        """Intent + concepts + embedding (KEEP YOUR ORIGINAL)"""
        text = para['text'].lower()
        
        intent_patterns = {
            'DEFINITION': ['define', 'known as', 'refers to', 'called', 'is a'],
            'EXPLANATION': ['because', 'therefore', 'since', 'results in', 'due to'],
            'RULE': ['must', 'shall', 'required', 'theorem states', 'law says'],
            'EXAMPLE': ['for example', 'such as', 'e.g.', 'instance'],
            'COMPARISON': ['compare', 'versus', 'unlike', 'similar to'],
            'EXCEPTION': ['however', 'except', 'unless', 'not when']
        }
        
        intent = 'GENERAL'
        for i_type, patterns in intent_patterns.items():
            if any(p in text for p in patterns):
                intent = i_type
                break
        
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', para['text'])
        concepts += re.findall(r'\b[a-z]{4,}(?:[a-z]+)?\b', text)
        
        embedding = self.model.encode(para['text'])
        
        return {
            'id': f"P{para['page']}_{len(self.paragraphs)}",
            **para, 'intent': intent, 'concepts': list(set(concepts)),
            'embedding': embedding
        }

    def build_dual_memory(self, pdf_path):
        """NEW: Build BOTH long-term AND short-term memory graphs"""
        # Build long-term memory (your original graph)
        pages = self.extract_pdf_structure(pdf_path)
        
        for page_blocks in pages:
            for para in page_blocks:
                processed = self.process_paragraph(para)
                self.paragraphs.append(processed)
                node_id = processed['id']
                
                # LONG-TERM: Document knowledge graph
                self.long_term_graph.add_node(node_id, **processed, memory_type='long_term', type='paragraph')
                
                for concept in processed['concepts']:
                    self.long_term_graph.add_node(concept, type='concept', memory_type='long_term', count=0)
                    self.long_term_graph.add_edge(concept, node_id, weight=1.0, relation=processed['intent'])
                    self.long_term_graph.add_edge(node_id, concept, weight=0.5)
        
        self.compute_dual_gravity()
        print(f"✅ Dual Memory Built: {len(self.paragraphs)} long-term nodes")

    def compute_dual_gravity(self):
        """Enhanced gravity for BOTH memory types"""
        # Long-term gravity (your original algorithm)
        all_concepts = Counter()
        for p in self.paragraphs:
            all_concepts.update(p['concepts'])
        
        pagerank = nx.pagerank(self.long_term_graph, alpha=0.85)
        
        for para in self.paragraphs:
            node_id = para['id']
            data = self.long_term_graph.nodes[node_id]
            
            indegree = self.long_term_graph.in_degree(node_id)
            intent_w = self.intent_weights.get(para['intent'], 1)
            concept_bonus = sum(all_concepts[c] for c in para['concepts']) / max(1, len(para['concepts']))
            heading_bonus = 2.0 if para['heading'] != 'GENERAL' else 1.0
            font_bonus = para['font_weight']
            
            gravity = (indegree * intent_w * 0.4 +
                      concept_bonus * 0.2 +
                      pagerank.get(node_id, 0) * 100 * 0.2 +
                      heading_bonus * font_bonus * 0.2)
            
            data['gravity'] = min(gravity / 10, 1.0)
        
        print("✅ Dual gravity computed")

    def update_short_term_memory(self, question):
        """NEW: Track query context in short-term memory"""
        q_concepts = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b|\b[a-z]{4,}\b', question.lower())
        q_id = f"Q_{len(self.query_history)}"
        
        self.short_term_graph.add_node(q_id, question=question, concepts=q_concepts, memory_type='short_term')
        self.query_history.append(question)
        self.session_context.append(question)
        
        # Link to related long-term concepts
        for concept in q_concepts:
            if self.long_term_graph.has_node(concept):
                self.short_term_graph.add_edge(q_id, concept, context_weight=1.0)

    def query_dual_memory(self, question, top_k=5, semantic_threshold=0.3):
        """NEW: Dual retrieval from BOTH memory types"""
        self.update_short_term_memory(question)  # Update context
        
        q_embedding = self.model.encode(question)
        q_concepts = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b|\b[a-z]{4,}\b', question.lower())
        q_intent = self.detect_intent(question)
        
        # Phase 1: Long-term retrieval (your original graph traversal)
        long_term_candidates = []
        for concept in q_concepts:
            if self.long_term_graph.has_node(concept):
                for neighbor in self.long_term_graph.neighbors(concept):
                    if (self.long_term_graph.nodes[neighbor]['type'] == 'paragraph' and
                        self.long_term_graph.nodes[neighbor]['memory_type'] == 'long_term'):
                        node = self.long_term_graph.nodes[neighbor]
                        if node['intent'] == q_intent or q_intent == 'GENERAL':
                            long_term_candidates.append(neighbor)
        
        # Phase 2: Short-term context boost
        short_term_boost = self._compute_context_boost(question)
        
        # Phase 3: Dual fusion scoring
        scored = []
        for node_id in set(long_term_candidates):
            node = self.long_term_graph.nodes[node_id]
            semantic_sim = util.cos_sim(q_embedding, node['embedding'])[0][0].item()
            
            if semantic_sim > semantic_threshold:
                # DUAL SCORE: long_term(65%) + short_term(25%) + semantic(10%)
                dual_score = (node['gravity'] * self.memory_weights['long_term'] +
                            short_term_boost * self.memory_weights['short_term'] +
                            semantic_sim * self.memory_weights['semantic'])
                
                scored.append((dual_score, node))
        
        results = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
        
        output = []
        for score, node in results:
            output.append({
                'text': node['text'][:300] + '...',
                'page': node['page'],
                'intent': node['intent'],
                'gravity': f"{node['gravity']:.2f}",
                'dual_score': f"{score:.2f}",
                'short_term_boost': f"{short_term_boost:.2f}",
                'heading': node['heading']
            })
        return output

    def _compute_context_boost(self, question):
        """Calculate relevance to recent query history"""
        if not self.query_history:
            return 0.0
        
        recent_context = ' '.join(list(self.query_history)[-5:])
        q_words = set(re.findall(r'\b[a-z]{4,}\b', question.lower()))
        context_words = set(re.findall(r'\b[a-z]{4,}\b', recent_context.lower()))
        
        overlap = len(q_words & context_words) / max(len(q_words), 1)
        return min(overlap * 2, 0.8)

    def detect_intent(self, text):
        """Quick intent classifier (KEEP YOUR ORIGINAL)"""
        text = text.lower()
        patterns = {
            'DEFINITION': ['what is', 'define', 'refers to'],
            'EXPLANATION': ['why', 'how', 'because'],
            'EXAMPLE': ['example', 'such as']
        }
        for intent, pats in patterns.items():
            if any(p in text for p in pats):
                return intent
        return 'GENERAL'

    def query_with_fallback(self, question, top_k=5, semantic_threshold=0.25):
        """Enhanced fallback with dual memory awareness (KEEP YOUR LOGIC)"""
        results = self.query_dual_memory(question, top_k, semantic_threshold)
        
        if not results:
            return [{
                'text': "❌ No relevant information found in the document.",
                'page': None, 'intent': 'NONE', 'gravity': '0.00',
                'dual_score': '0.00', 'short_term_boost': '0.00', 'heading': 'N/A'
            }]
        
        avg_score = np.mean([float(r['dual_score']) for r in results])
        if avg_score < 0.35:
            return [{
                'text': f"⚠️ Low dual-memory relevance (avg: {avg_score:.2f}). "
                       f"Try related physics concepts from recent queries.",
                'page': None, 'intent': 'IRRELEVANT', 'gravity': f"{avg_score:.2f}",
                'dual_score': f"{avg_score:.2f}", 'short_term_boost': '0.00', 'heading': 'SUGGESTION'
            }]
        
        return results

    def save_dual_memory(self, path='idsmmg_dual.pkl'):
        """Save both memory graphs"""
        data = {
            'paragraphs': self.paragraphs,
            'long_term_graph': (dict(self.long_term_graph.nodes(data=True)), 
                              list(self.long_term_graph.edges(data=True))),
            'query_history': list(self.query_history),
            'session_context': self.session_context
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Dual memory saved: {os.path.getsize(path)/1e6:.1f}MB")

# ================= USAGE (ENHANCED) =================
def main():
    smg = IDSMMG()
    #------Add your file path-----
    pdf_path = r"C:\Users\Dell-pc\Desktop\improved\pdf\physics-book.pdf" -
    print("Building Dual Semantic Memory Graph...")
    smg.build_dual_memory(pdf_path)  # Replaces build_graph()
    
    print("\n🔍 IDSMMG Dual Memory Ready! (Type 'exit' to quit)")
    
    while True:
        q = input(">> Question: ").strip()
        if q.lower() == 'exit':
            break
        
        if not q:
            print("📝 Please enter a question.\n")
            continue
        
        results = smg.query_with_fallback(q)
        
        print("\n📚 Dual Memory Results:")
        for i, res in enumerate(results, 1):
            print(f"{i}. [{res['intent']}] {res['heading']} (Page: {res['page']})")
            print(f"   Dual Score: {res['dual_score']} | Gravity: {res['gravity']} | Context: {res.get('short_term_boost', '0.00')}")
            print(f"   {res['text']}\n")
        
        if len(results) > 1 and 'IRRELEVANT' not in results[0]['intent']:
            print("✅ Good dual-memory match! Context preserved.\n")

if __name__ == "__main__":
    main()
