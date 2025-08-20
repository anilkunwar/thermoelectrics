import streamlit as st
import sqlite3
import pandas as pd
from io import BytesIO
import logging
import spacy
from spacy.matcher import Matcher
from transformers import AutoTokenizer, AutoModel
import torch
import re
from pymatgen.core.composition import Composition
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import defaultdict, Counter
from math import log2
from fuzzywuzzy import fuzz
from datetime import datetime
import os
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import networkx as nx
import io
from cleantext import clean
import time
import psutil

# Define database directory and files
DB_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_DB_FILE = os.path.join(DB_DIR, "thermoelectric_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "thermoelectric_universe.db")

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'thermoelectric_inspector.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page config
st.set_page_config(page_title="Thermoelectric Data Inspector and NER Analysis", layout="wide")

# Initialize Streamlit app
st.title("Thermoelectric Data Inspector and NER Analysis")
st.markdown("""
This tool inspects `thermoelectric_universe.db`, performs lightweight NER to extract chemical formulas and classify materials as p-type or n-type, and generates visualizations. Analysis starts only after pressing 'Inspect and Analyze Database'.

**Date and Time**: 09:05 PM CEST, Wednesday, August 20, 2025
""")

# Dependency check
st.sidebar.header("Setup")
st.sidebar.markdown("""
**Dependencies**:
- `pip install streamlit spacy transformers torch pymatgen pandas scikit-learn fuzzywuzzy python-Levenshtein wordcloud matplotlib networkx clean-text psutil`
- `python -m spacy download en_core_web_sm`
""")

# Initialize session state
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "relevance_threshold" not in st.session_state:
    st.session_state.relevance_threshold = 30.0
if "pmi_threshold" not in st.session_state:
    st.session_state.pmi_threshold = 1.0
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.7
if "strict_validation" not in st.session_state:
    st.session_state.strict_validation = False
if "material_types" not in st.session_state:
    st.session_state.material_types = ["p-type", "n-type", "neutral"]
if "max_terms" not in st.session_state:
    st.session_state.max_terms = 20
if "context_window" not in st.session_state:
    st.session_state.context_window = 50
if "min_edge_weight" not in st.session_state:
    st.session_state.min_edge_weight = 2

# Constants
P_TYPE_KEYWORDS = [
    "p-type", "p-doped", "hole-doped", "positive seebeck", "hole carrier", "hole conduction",
    "p-type semiconductor", "acceptor doping", "hole transport", "antimony", "boron", "gallium", "indium", "aluminum"
]
N_TYPE_KEYWORDS = [
    "n-type", "n-doped", "electron-doped", "negative seebeck", "electron carrier", "electron conduction",
    "n-type semiconductor", "donor doping", "electron transport", "selenium", "phosphorus", "arsenic"
]
THERMOELECTRIC_KEYWORDS = [
    "seebeck coefficient", "thermopower", "power factor", "zt", "figure of merit",
    "thermoelectric", "thermoelectric material", "band gap", "electrical conductivity",
    "thermal conductivity", "carrier concentration", "carrier mobility"
]
SYNONYM_MAPPING = {
    "p-doped": "p-type", "hole-doped": "p-type", "acceptor-doped": "p-type", "hole conduction": "p-type",
    "n-doped": "n-type", "electron-doped": "n-type", "donor-doped": "n-type", "electron transport": "n-type",
    "seebeck": "seebeck coefficient", "thermopower": "seebeck coefficient",
    "figure of merit": "zt"
}
TERM_WEIGHTS = {
    "p-type": 2.0, "n-type": 2.0, "seebeck coefficient": 2.5, "zt": 2.5, "thermoelectric": 1.5,
    "antimony": 1.5, "selenium": 1.5, "boron": 1.5, "phosphorus": 1.5
}
UNIT_VARIANTS = ["microvolt/K", "μV/K", "mV/K", "V/K", "microvolts per Kelvin", "microvolts/Kelvin"]
MATERIAL_FORMULA_PATTERN = r'\b(?:[A-Z][a-z]?(?:[0-9.]+)?(?:\((?:[A-Z][a-z]?[0-9.]+)+\)[0-9.]+)?(?:[-_/][A-Z][a-z]?)?){1,}\b'
SEEBECK_VALUE_PATTERN = r'(-?\d*\.?\d+)\s*(microvolt/K|μV/K|mV/K|V/K|microvolts per Kelvin|microvolts/Kelvin)'
P_TYPE_PATTERN = r'\b(?:' + '|'.join(P_TYPE_KEYWORDS + list(SYNONYM_MAPPING.keys())) + r')\b'
N_TYPE_PATTERN = r'\b(?:' + '|'.join(N_TYPE_KEYWORDS + list(SYNONYM_MAPPING.keys())) + r')\b'

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_usage = psutil.Process().memory_info().rss / 1024**2  # MB
    st.session_state.log_buffer.append(f"[{timestamp}] {message} (Memory: {memory_usage:.2f} MB)")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(f"{message} (Memory: {memory_usage:.2f} MB)")

# Text preprocessing
def preprocess_text(text):
    try:
        if not isinstance(text, str):
            return ""
        return clean(text, no_urls=True, no_emails=True, no_punct=False, replace_with_url="", replace_with_email="", lowercase=True, no_numbers=False)[:5000]
    except Exception as e:
        update_log(f"Text preprocessing error: {str(e)}")
        return ""

# Load spaCy model
@st.cache_resource
def load_spacy():
    try:
        update_log("Loading spaCy en_core_web_sm...")
        start_time = time.time()
        nlp = spacy.load("en_core_web_sm")
        matcher = Matcher(nlp.vocab)
        material_patterns = [
            [{"LOWER": {"IN": ["p-type", "p-doped", "hole-doped", "acceptor-doped", "hole conduction"]}, "OP": "?"}, {"LOWER": {"IN": ["material", "semiconductor", "thermoelectric"]}, "OP": "?"}],
            [{"LOWER": {"IN": ["n-type", "n-doped", "electron-doped", "donor-doped", "electron transport"]}, "OP": "?"}, {"LOWER": {"IN": ["material", "semiconductor", "thermoelectric"]}, "OP": "?"}],
            [{"LOWER": {"IN": ["material", "semiconductor"]}, "OP": "?"}, {"LOWER": {"IN": ["with", "having"]}}, {"LOWER": {"IN": ["p-type", "n-type", "hole-doped", "electron-doped"]}}]
        ]
        matcher.add("P_TYPE", [material_patterns[0]])
        matcher.add("N_TYPE", [material_patterns[1]])
        matcher.add("MATERIAL_VARIANTS", [material_patterns[2]])
        thermo_patterns = [[{"LOWER": kw} for kw in phrase.split()] for phrase in THERMOELECTRIC_KEYWORDS]
        matcher.add("THERMO_PHRASES", thermo_patterns)
        update_log(f"Loaded spaCy en_core_web_sm in {time.time() - start_time:.2f} seconds")
        return nlp, matcher
    except Exception as e:
        st.error(f"Failed to load spaCy: {e}. Install: `python -m spacy download en_core_web_sm`")
        logging.error(f"spaCy loading error: {e}")
        st.stop()

# Initialize database
def initialize_metadata_db(db_file=METADATA_DB_FILE):
    try:
        conn = sqlite3.connect(db_file, timeout=10)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                matched_terms TEXT,
                relevance_prob REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS materials (
                paper_id TEXT,
                formula TEXT,
                material_type TEXT,
                seebeck_value REAL,
                unit TEXT,
                context TEXT,
                score REAL,
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_id ON materials(paper_id)")
        conn.commit()
        conn.close()
        update_log(f"Initialized metadata database schema for {db_file}")
    except Exception as e:
        update_log(f"Failed to initialize {db_file}: {str(e)}")
        st.error(f"Failed to initialize {db_file}: {str(e)}")

initialize_metadata_db()

# Preserve chemical formulas
def preserve_formulas(text):
    formulas = re.findall(MATERIAL_FORMULA_PATTERN, text, re.IGNORECASE)
    formula_map = {}
    for formula in formulas:
        placeholder = f"FORMULA_{hash(formula)}"
        text = text.replace(formula, placeholder)
        formula_map[placeholder] = formula
    return text, formula_map

# Extract terms (simplified, no SciBERT attentions)
def extract_terms(texts, query, similarity_threshold=0.7, top_n=20):
    try:
        term_counts = Counter()
        key_terms = set(THERMOELECTRIC_KEYWORDS + P_TYPE_KEYWORDS + N_TYPE_KEYWORDS + list(SYNONYM_MAPPING.keys()))
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            text = preprocess_text(text)
            doc = nlp(text)
            
            for term in key_terms:
                term_counts[term] += len(re.findall(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
            
            formulas = set()
            for match in re.finditer(MATERIAL_FORMULA_PATTERN, text, re.IGNORECASE):
                formula = match.group(0).replace(" ", "").replace("-", "").replace("_", "")
                try:
                    if st.session_state.strict_validation:
                        Composition(formula)
                    formulas.add(formula)
                except:
                    if any(fuzz.ratio(formula.lower(), f.lower()) > 70 for f in formulas + [query]):
                        formulas.add(formula)
            key_terms.update(formulas)
            
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower()
                if phrase in key_terms or any(fuzz.ratio(phrase, k.lower()) > 70 for k in key_terms):
                    term_counts[phrase] += 1
        
        return {term: count for term, count in term_counts.most_common(top_n) if count > 1}
    except Exception as e:
        update_log(f"Term extraction error: {str(e)}")
        return {}

# Generate histogram
def generate_term_histogram(term_counts, max_terms=20):
    try:
        if not term_counts:
            update_log("No terms for histogram generation")
            return None
        terms = list(term_counts.keys())[:max_terms]
        counts = list(term_counts.values())[:max_terms]
        fig = plt.figure(figsize=(10, 6))
        plt.bar(terms, counts, color='skyblue')
        plt.xlabel("Terms")
        plt.ylabel("Frequency")
        plt.title("Top Terms and Phrases in Thermoelectric Data")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    except Exception as e:
        update_log(f"Histogram generation error: {str(e)}")
        return None
    finally:
        plt.close()

# Generate word cloud
def generate_word_cloud_from_terms(term_counts):
    try:
        if not term_counts:
            update_log("No terms for word cloud generation")
            return None
        custom_stopwords = set(STOPWORDS) | {'et', 'al', 'fig', 'figure', 'table', 'equation', 'http', 'https', 'arxiv', 'journal', 'volume', 'doi', 'published'}
        filtered_terms = {term: count for term, count in term_counts.items() if term not in custom_stopwords}
        wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10, max_font_size=150).generate_from_frequencies(filtered_terms)
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        return fig
    except Exception as e:
        update_log(f"Word cloud generation error: {str(e)}")
        return None
    finally:
        plt.close()

# Generate word co-occurrence network
def generate_word_network(texts, context_window=50, min_edge_weight=2):
    try:
        G = nx.Graph()
        key_terms = set(THERMOELECTRIC_KEYWORDS + P_TYPE_KEYWORDS + N_TYPE_KEYWORDS + list(SYNONYM_MAPPING.keys()))
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            text = preprocess_text(text)
            doc = nlp(text)
            formulas = set()
            for match in re.finditer(MATERIAL_FORMULA_PATTERN, text, re.IGNORECASE):
                formula = match.group(0).replace(" ", "").replace("-", "").replace("_", "")
                try:
                    if st.session_state.strict_validation:
                        Composition(formula)
                    formulas.add(formula)
                except:
                    if any(fuzz.ratio(formula.lower(), f.lower()) > 70 for f in formulas):
                        formulas.add(formula)
            key_terms.update(formulas)
            words = [w.lower() for w in text.split() if w.lower() in key_terms]
            for i, word1 in enumerate(words):
                for j, word2 in enumerate(words[i+1:], start=i+1):
                    if j - i <= context_window and word1 != word2:
                        if G.has_edge(word1, word2):
                            G[word1][word2]['weight'] += 1
                        else:
                            G.add_edge(word1, word2, weight=1)
        G_filtered = nx.Graph()
        for u, v, data in G.edges(data=True):
            if data['weight'] >= min_edge_weight:
                G_filtered.add_edge(u, v, weight=data['weight'])
        fig = plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G_filtered, k=0.5, iterations=50)
        nx.draw(G_filtered, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, edge_color='gray')
        nx.draw_network_edges(G_filtered, pos, edge_color='gray', width=[data['weight'] * 0.5 for u, v, data in G_filtered.edges(data=True)])
        plt.title("Word Co-occurrence Network")
        return fig
    except Exception as e:
        update_log(f"Word network generation error: {str(e)}")
        return None
    finally:
        plt.close()

# Compute PMI
def compute_pmi(text, formula, keyword, total_words, word_counts):
    try:
        formula_count = len(re.findall(r'\b' + re.escape(formula) + r'\b', text, re.IGNORECASE))
        keyword_count = word_counts.get(keyword.lower(), 0)
        co_occurrence = len(re.findall(rf'\b{re.escape(formula)}\b.*?\b{keyword}\b|\b{keyword}\b.*?\b{re.escape(formula)}\b', text, re.IGNORECASE))
        p_formula = formula_count / total_words
        p_keyword = keyword_count / total_words
        p_joint = co_occurrence / total_words
        if p_formula == 0 or p_keyword == 0 or p_joint == 0:
            return 0.0
        pmi = log2(p_joint / (p_formula * p_keyword))
        return pmi
    except Exception as e:
        update_log(f"PMI computation error for {formula}, {keyword}: {str(e)}")
        return 0.0

# Simplified SciBERT scoring
def score_text_with_scibert(texts, query, pmi_threshold, similarity_threshold, tokenizer, model, device, nlp, matcher):
    try:
        inputs = tokenizer(
            [preprocess_text(t) for t in texts],
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        results = []
        query_terms = [query] + THERMOELECTRIC_KEYWORDS
        
        for idx, text in enumerate(texts):
            start_time = time.time()
            if time.time() - start_time > 20:
                update_log(f"Timeout processing text {idx}")
                results.append((0.0, [], [], []))
                continue
                
            text = preprocess_text(text)
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents][:20]
            word_counts = defaultdict(int)
            for word in text.lower().split():
                word_counts[word] += 1
            total_words = sum(word_counts.values())
            
            relevance_scores = []
            matched_terms = []
            matched_formulas = []
            seebeck_values = []
            
            for sentence in sentences:
                sentence_score = 0.0
                sentence_terms = []
                sentence_formulas = []
                
                formula_matches = re.finditer(MATERIAL_FORMULA_PATTERN, sentence, re.IGNORECASE)
                for match in formula_matches:
                    matched_formula = match.group(0).replace(" ", "").replace("-", "").replace("_", "")
                    try:
                        if st.session_state.strict_validation:
                            Composition(matched_formula)
                        sentence_formulas.append(matched_formula)
                        sentence_terms.append(matched_formula)
                    except:
                        if any(fuzz.ratio(matched_formula.lower(), f.lower()) > 70 for f in matched_formulas + query_terms):
                            sentence_formulas.append(matched_formula)
                            sentence_terms.append(matched_formula)
                
                seebeck_matches = re.finditer(SEEBECK_VALUE_PATTERN, sentence, re.IGNORECASE)
                for match in seebeck_matches:
                    value = float(match.group(1))
                    unit = match.group(2)
                    if 'mV/K' in unit:
                        value *= 1000
                    elif 'V/K' in unit:
                        value *= 1e6
                    if -1000 <= value <= 1000:
                        seebeck_values.append(value)
                        sentence_terms.append(match.group(0))
                
                for keyword in THERMOELECTRIC_KEYWORDS + list(SYNONYM_MAPPING.keys()):
                    if re.search(r'\b' + re.escape(keyword) + r'\b', sentence, re.IGNORECASE):
                        mapped_keyword = SYNONYM_MAPPING.get(keyword.lower(), keyword.lower())
                        pmi = compute_pmi(sentence, query, mapped_keyword, total_words, word_counts)
                        if pmi > pmi_threshold:
                            weight = TERM_WEIGHTS.get(mapped_keyword, 1.0)
                            sentence_score += weight * pmi
                            sentence_terms.append(mapped_keyword)
                
                relevance_scores.append(sentence_score)
                matched_terms.extend(sentence_terms)
                matched_formulas.extend(sentence_formulas)
            
            relevance_score = min(sum(relevance_scores) / (len(sentences) + 1e-6), 1.0)
            matched_terms = list(set(matched_terms))
            matched_formulas = list(set(matched_formulas))
            results.append((relevance_score, matched_terms, matched_formulas, seebeck_values))
            update_log(f"Processed text {idx} in {time.time() - start_time:.2f} seconds")
        
        return results
    except Exception as e:
        update_log(f"SciBERT scoring error: {str(e)}")
        return [(0.0, [], [], []) for _ in texts]

# Inspect universe database
def inspect_universe_db(db_file=UNIVERSE_DB_FILE):
    try:
        conn = sqlite3.connect(db_file, timeout=10)
        query_sql = "SELECT id, title, authors, year, content FROM papers"
        df = pd.read_sql_query(query_sql, conn)
        conn.close()
        invalid_papers = []
        for idx, row in df.iterrows():
            if not isinstance(row["content"], str) or not row["content"].strip():
                invalid_papers.append(row["id"])
        if invalid_papers:
            update_log(f"Invalid content in papers: {', '.join(invalid_papers)}")
            st.warning(f"Found {len(invalid_papers)} papers with invalid or empty content. Check logs for details.")
        update_log(f"Loaded {len(df)} papers from {db_file}")
        return df[df["content"].apply(lambda x: isinstance(x, str) and x.strip())]
    except Exception as e:
        update_log(f"Error loading {db_file}: {str(e)}")
        st.error(f"Error loading {db_file}: {str(e)}")
        return pd.DataFrame()

# Perform NER and classification
def analyze_universe_db(db_file=UNIVERSE_DB_FILE, query="thermoelectric materials", material_types=["p-type", "n-type", "neutral"], pmi_threshold=1.0, similarity_threshold=0.7, relevance_threshold=30.0):
    try:
        start_time = time.time()
        update_log("Starting database analysis...")
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        update_log(f"SciBERT loaded on {device} in {time.time() - start_time:.2f} seconds")
        
        X_train = [
            "p-type semiconductor antimony", "n-type material selenium", "hole-doped", "electron-doped",
            "positive seebeck coefficient", "negative seebeck coefficient", "hole conduction material",
            "electron transport semiconductor", "thermoelectric Bi2Te3 n-type", "p-type PbTe doping",
            "thermoelectric performance", "charge carrier concentration"
        ]
        y_train = ["p-type", "n-type", "p-type", "n-type", "p-type", "n-type", "p-type", "n-type", "n-type", "p-type", "neutral", "neutral"]
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train_vec, y_train)
        update_log("Trained Random Forest classifier")
        
        nlp, matcher = load_spacy()
        
        df = inspect_universe_db(db_file)
        if df.empty:
            update_log(f"No valid data found in {db_file}")
            return [], [], {}
        
        metadata = []
        papers = []
        term_counts = Counter()
        batch_size = 3
        progress_bar = st.progress(0)
        
        for i in range(0, len(df), batch_size):
            batch_df = df[i:i+batch_size]
            batch_texts = batch_df["content"]
            update_log(f"Processing batch {i//batch_size + 1} of {len(df)//batch_size + 1}")
            
            batch_term_counts = extract_terms(batch_texts, query, similarity_threshold, st.session_state.max_terms)
            term_counts.update(batch_term_counts)
            
            batch_results = score_text_with_scibert(batch_texts, query, pmi_threshold, similarity_threshold, tokenizer, model, device, nlp, matcher)
            
            for idx, (row, (relevance_score, matched_terms, matched_formulas, seebeck_values)) in enumerate(zip(batch_df.itertuples(), batch_results)):
                paper_id = row.id
                update_log(f"Analyzing paper {paper_id}")
                try:
                    if not matched_formulas or relevance_score * 100 < relevance_threshold:
                        continue
                    text = preprocess_text(row.content)
                    doc = nlp(text)
                    matches = matcher(doc)
                    p_type_count = len(re.finditer(P_TYPE_PATTERN, text, re.IGNORECASE))
                    n_type_count = len(re.finditer(N_TYPE_PATTERN, text, re.IGNORECASE))
                    
                    for match_id, start, end in matches:
                        rule_id = nlp.vocab.strings[match_id]
                        if rule_id in ["P_TYPE", "MATERIAL_VARIANTS"]:
                            p_type_count += 1
                        elif rule_id == "N_TYPE":
                            n_type_count += 1
                    
                    material_type = "neutral"
                    p_type_score = p_type_count * 2.0
                    n_type_score = n_type_count * 2.0
                    X_vec = vectorizer.transform([text])
                    clf_pred = clf.predict(X_vec)[0]
                    clf_prob = clf.predict_proba(X_vec)[0]
                    p_type_prob = clf_prob[0] if clf_pred == "p-type" else clf_prob[1]
                    n_type_prob = clf_prob[1] if clf_pred == "n-type" else clf_prob[0]
                    p_type_score += p_type_prob * 3.0
                    n_type_score += n_type_prob * 3.0
                    
                    if seebeck_values:
                        positive_count = sum(1 for v in seebeck_values if v > 0)
                        negative_count = sum(1 for v in seebeck_values if v < 0)
                        if positive_count > negative_count:
                            p_type_score += 3.0
                            p_type_prob = max(p_type_prob, 0.7)
                        elif negative_count > positive_count:
                            n_type_score += 3.0
                            n_type_prob = max(n_type_prob, 0.7)
                    
                    word_counts = defaultdict(int)
                    for word in text.lower().split():
                        word_counts[word] += 1
                    total_words = sum(word_counts.values())
                    
                    for formula in matched_formulas:
                        formula_pos = text.lower().find(formula.lower())
                        if formula_pos != -1:
                            context_window = text[max(0, formula_pos - 50):formula_pos + len(formula) + 50]
                            pmi_scores = [compute_pmi(context_window, formula, kw, total_words, word_counts) for kw in P_TYPE_KEYWORDS + N_TYPE_KEYWORDS]
                            if any(pmi > pmi_threshold for pmi in pmi_scores):
                                if any(kw in context_window.lower() for kw in P_TYPE_KEYWORDS + list(SYNONYM_MAPPING.keys())):
                                    p_type_score += 2.0
                                if any(kw in context_window.lower() for kw in N_TYPE_KEYWORDS + list(SYNONYM_MAPPING.keys())):
                                    n_type_score += 2.0
                    
                    if p_type_score > n_type_score + 0.05:
                        material_type = "p-type"
                    elif n_type_score > p_type_score + 0.05:
                        material_type = "n-type"
                    
                    if material_type not in material_types:
                        continue
                    
                    papers.append({
                        "id": row.id,
                        "title": row.title,
                        "authors": row.authors,
                        "year": row.year,
                        "matched_terms": ", ".join(matched_terms),
                        "relevance_prob": relevance_score * 100
                    })
                    
                    for matched_formula in matched_formulas:
                        formula_match = re.search(r'\b' + re.escape(matched_formula) + r'\b', text, re.IGNORECASE)
                        if formula_match:
                            snippet_start = max(0, formula_match.start() - 100)
                            snippet_end = min(len(text), formula_match.end() + 100)
                            snippet = text[snippet_start:snippet_end]
                            metadata.append({
                                "paper_id": row.id,
                                "formula": matched_formula,
                                "material_type": material_type,
                                "p_type_prob": p_type_prob,
                                "n_type_prob": n_type_prob,
                                "seebeck_value": seebeck_values[0] if seebeck_values else None,
                                "unit": "μV/K" if seebeck_values else None,
                                "context": snippet,
                                "score": relevance_score,
                                "title": row.title,
                                "year": row.year
                            })
                except Exception as e:
                    update_log(f"Error processing paper {paper_id}: {str(e)}")
                    continue
                
                progress_bar.progress((i + idx + 1) / len(df))
        
        update_log(f"Extracted {len(metadata)} materials from {len(papers)} papers in {time.time() - start_time:.2f} seconds")
        return papers, metadata, term_counts
    except Exception as e:
        update_log(f"Error analyzing {db_file}: {str(e)}")
        st.error(f"Error analyzing {db_file}: {str(e)}")
        return [], [], {}

# Save to SQLite
def save_to_sqlite(papers_df, materials_list, db_file=METADATA_DB_FILE):
    try:
        initialize_metadata_db(db_file)
        conn = sqlite3.connect(db_file, timeout=10)
        papers_df.to_sql("papers", conn, if_exists="replace", index=False)
        materials_df = pd.DataFrame(materials_list)
        if not materials_df.empty:
            materials_df.to_sql("materials", conn, if_exists="append", index=False)
        conn.close()
        update_log(f"Saved {len(papers_df)} papers and {len(materials_list)} materials to {db_file}")
        return f"Saved to {db_file}"
    except Exception as e:
        update_log(f"SQLite save failed: {str(e)}")
        st.error(f"SQLite save failed: {str(e)}")
        return f"Failed to save to SQLite: {str(e)}"

# Streamlit UI
st.sidebar.header("Analysis Parameters")
query = st.sidebar.text_input("Query (e.g., thermoelectric materials)", "thermoelectric materials")
material_types = st.sidebar.multiselect("Material Types", ["p-type", "n-type", "neutral"], default=st.session_state.material_types)
relevance_threshold = st.sidebar.slider("Minimum Relevance Score (%)", 0.0, 100.0, st.session_state.relevance_threshold)
pmi_threshold = st.sidebar.slider("PMI Threshold", 0.0, 5.0, st.session_state.pmi_threshold)
similarity_threshold = st.sidebar.slider("Term Similarity Threshold", 0.5, 1.0, st.session_state.similarity_threshold)
strict_validation = st.sidebar.checkbox("Strict Formula Validation (pymatgen)", st.session_state.strict_validation)
max_terms = st.sidebar.slider("Max Terms in Histogram/Word Cloud", 10, 50, st.session_state.max_terms)
context_window = st.sidebar.slider("Context Window for Network (words)", 10, 100, st.session_state.context_window)
min_edge_weight = st.sidebar.slider("Minimum Edge Weight for Network", 1, 10, st.session_state.min_edge_weight)
analyze_button = st.sidebar.button("Inspect and Analyze Database")

st.session_state.material_types = material_types
st.session_state.relevance_threshold = relevance_threshold
st.session_state.pmi_threshold = pmi_threshold
st.session_state.similarity_threshold = similarity_threshold
st.session_state.strict_validation = strict_validation
st.session_state.max_terms = max_terms
st.session_state.context_window = context_window
st.session_state.min_edge_weight = min_edge_weight

# Display database contents
st.subheader("Database Contents (thermoelectric_universe.db)")
universe_df = inspect_universe_db()
if not universe_df.empty:
    st.dataframe(universe_df[["id", "title", "authors", "year"]], use_container_width=True)
    st.info(f"Loaded {len(universe_df)} valid papers")
else:
    st.warning("No valid data found in thermoelectric_universe.db. Please ensure the database is populated.")

# Debug mode
if st.sidebar.checkbox("Debug Mode: Test Term Extraction"):
    debug_text = "Bi2Te3 is an n-type thermoelectric material with a Seebeck coefficient of -200 μV/K."
    nlp, matcher = load_spacy()
    debug_terms = extract_terms([debug_text], query, similarity_threshold, st.session_state.max_terms)
    st.write("Debug Term Extraction:", debug_terms)

# Process analysis
if analyze_button:
    if not query.strip():
        st.error("Please provide a valid query.")
    else:
        with st.spinner("Analyzing thermoelectric_universe.db..."):
            papers, metadata, term_counts = analyze_universe_db(
                UNIVERSE_DB_FILE, query, material_types, pmi_threshold, similarity_threshold, relevance_threshold
            )
        
        if not universe_df.empty:
            st.subheader("Histogram of Common Terms and Phrases")
            if term_counts:
                histogram_fig = generate_term_histogram(term_counts, max_terms=st.session_state.max_terms)
                if histogram_fig:
                    st.pyplot(histogram_fig)
                    buf = io.BytesIO()
                    histogram_fig.savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button(
                        label="Download Histogram PNG",
                        data=buf,
                        file_name="thermoelectric_terms_histogram.png",
                        mime="image/png"
                    )
                    plt.close(histogram_fig)
                else:
                    st.warning("Failed to generate histogram. Check logs for details.")
            else:
                st.warning("No key terms detected. Try lowering similarity_threshold, disabling strict_validation, or checking content in thermoelectric_universe.db.")
            
            st.subheader("Word Cloud of Common Terms and Phrases")
            if term_counts:
                wordcloud_fig = generate_word_cloud_from_terms(term_counts)
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                    buf = io.BytesIO()
                    wordcloud_fig.savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button(
                        label="Download Word Cloud PNG",
                        data=buf,
                        file_name="thermoelectric_wordcloud.png",
                        mime="image/png"
                    )
                    plt.close(wordcloud_fig)
                else:
                    st.warning("Failed to generate word cloud. Check logs for details.")
            else:
                st.warning("No key terms detected. Try lowering similarity_threshold, disabling strict_validation, or checking content in thermoelectric_universe.db.")
            
            st.subheader("Word Co-occurrence Network")
            network_fig = generate_word_network(universe_df["content"], context_window=st.session_state.context_window, min_edge_weight=st.session_state.min_edge_weight)
            if network_fig:
                st.pyplot(network_fig)
                buf = io.BytesIO()
                network_fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button(
                    label="Download Network PNG",
                    data=buf,
                    file_name="thermoelectric_word_network.png",
                    mime="image/png"
                )
                plt.close(network_fig)
            else:
                st.warning("Failed to generate word network. Check logs for details.")
        
        if not metadata:
            st.warning("No materials extracted. Adjust query, thresholds, or ensure thermoelectric_universe.db contains valid data.")
        else:
            st.success(f"Extracted {len(metadata)} materials from {len(papers)} papers!")
            papers_df = pd.DataFrame(papers)
            materials_df = pd.DataFrame(metadata)
            
            st.subheader("Extracted Materials")
            st.dataframe(
                materials_df[["paper_id", "title", "year", "formula", "material_type", "p_type_prob", "n_type_prob", "seebeck_value", "unit", "context", "score"]],
                use_container_width=True
            )
            
            st.subheader("Papers")
            st.dataframe(
                papers_df[["id", "title", "authors", "year", "matched_terms", "relevance_prob"]],
                use_container_width=True
            )
            
            csv = materials_df.to_csv(index=False)
            st.download_button(
                label="Download Materials CSV",
                data=csv,
                file_name="thermoelectric_materials.csv",
                mime="text/csv"
            )
            
            sqlite_status = save_to_sqlite(papers_df, metadata)
            st.info(sqlite_status)
            if sqlite_status.startswith("Saved"):
                try:
                    conn_source = sqlite3.connect(METADATA_DB_FILE, timeout=10)
                    conn_target = sqlite3.connect(":memory:")
                    conn_source.backup(conn_target)
                    sqlite_data = BytesIO()
                    for line in conn_target.iterdump():
                        sqlite_data.write(line.encode('utf-8'))
                    conn_source.close()
                    conn_target.close()
                    sqlite_data.seek(0)
                    st.download_button(
                        label="Download thermoelectric_metadata.db",
                        data=sqlite_data.getvalue(),
                        file_name="thermoelectric_metadata.db",
                        mime="application/x-sqlite3"
                    )
                except Exception as e:
                    st.error(f"Failed to prepare thermoelectric_metadata.db for download: {str(e)}")
                    update_log(f"Failed to prepare thermoelectric_metadata.db for download: {str(e)}")

        log_container = st.empty()
        log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)
