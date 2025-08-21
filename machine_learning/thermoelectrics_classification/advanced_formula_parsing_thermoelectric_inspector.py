import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Span, Doc
from spacy.util import filter_spans
from spacy.matcher import PhraseMatcher
from collections import Counter
import re
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px
import logging
import networkx as nx
from wordcloud import WordCloud
from nltk import ngrams
from itertools import chain, combinations
import math
import glob
import uuid
import seaborn as sns
import psutil

# Try to import pymatgen for material formula parsing
try:
    from pymatgen.core.composition import Composition
    from pymatgen.core.periodic_table import Element
    PYMAGEN_AVAILABLE = True
except ImportError:
    PYMAGEN_AVAILABLE = False
    st.warning("pymatgen is not installed. Material formula standardization will be limited. Install with: pip install pymatgen")

# Matplotlib configuration
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 1.5,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 200,
    'savefig.transparent': True
})

# Directory and logging setup
# Set up logging
DB_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=os.path.join(DB_DIR, 'thermoelectric_ner_analysis.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit configuration
st.set_page_config(page_title="Thermoelectric Material Classification Tool", layout="wide")
st.title("Thermoelectric Material Classification and Analysis Tool")
st.markdown("""
This tool inspects SQLite databases, extracts common terms and phrases related to thermoelectric materials, 
performs rule-based NER analysis using SciBERT, and specifically classifies materials as p-type or n-type.
Select or upload a database, then use the tabs to inspect the database, analyze terms, extract entities, 
or classify materials.

**Date and Time**: 03:50 AM CEST, Thursday, August 21, 2025

**Dependencies**:
- `pip install streamlit pandas sqlite3 spacy transformers torch nltk networkx wordcloud seaborn matplotlib psutil plotly`
- `python -m spacy download en_core_web_lg` (or `en_core_web_sm` as fallback)
- `pip install pymatgen` (for enhanced material formula parsing)
""")

# -----------------------------
# Enhanced Formula NER and Material Classification
# -----------------------------

# Default synonyms for material classification
DEFAULT_SYNONYMS = {
    "p-type": ["p-type", "positive type", "positive thermoelectric", "hole conducting", 
               "hole transport", "p doped", "p-type semiconductor", "p-type material",
               "hole carrier", "acceptor doped"],
    "n-type": ["n-type", "negative type", "negative thermoelectric", "electron conducting",
               "electron transport", "n doped", "n-type semiconductor", "n-type material",
               "electron carrier", "donor doped"]
}

@Language.component("formula_ner")
def formula_ner(doc):
    """
    Enhanced formula NER component using regex patterns
    """
    # Improved formula pattern to capture complex material formulas
    formula_pattern = r'\b(?:[A-Z][a-z]?(?:\d*\.?\d*)?)+(?:[-_][A-Z][a-z]?(?:\d*\.?\d*)?)*\b'
    spans = []
    
    for match in re.finditer(formula_pattern, doc.text):
        # Additional validation to exclude single elements unless they're common in context
        text = match.group()
        if len(text) > 2 or text in ['Bi', 'Te', 'Sb', 'Se', 'Pb', 'Sn', 'Co', 'Si', 'Ge']:
            span = doc.char_span(match.start(), match.end(), label="FORMULA")
            if span:
                spans.append(span)
    
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

def build_material_matcher(nlp, synonyms):
    """
    Build phrase matcher for material types
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for canonical, variants in synonyms.items():
        patterns = [nlp.make_doc(v) for v in variants]
        matcher.add(canonical, patterns)
    return matcher

@Language.component("material_matcher")
def material_matcher(doc):
    """
    Material matcher component using phrase matching
    """
    matcher = doc._.material_matcher
    matches = matcher(doc)
    spans = []
    
    for match_id, start, end in matches:
        canonical = doc.vocab.strings[match_id]
        span = Span(doc, start, end, label="MATERIAL_TYPE")
        span._.norm = canonical
        spans.append(span)
    
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

def link_formula_to_material(doc, max_distance=200):
    """
    Link formulas to their nearest material type with distance threshold
    """
    formulas = [ent for ent in doc.ents if ent.label_ == "FORMULA"]
    materials = [ent for ent in doc.ents if ent.label_ == "MATERIAL_TYPE"]
    
    pairs = []
    for f in formulas:
        nearest_material = None
        min_distance = float("inf")
        
        for m in materials:
            distance = abs(f.start_char - m.start_char)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_material = m
        
        if nearest_material:
            pairs.append({
                "formula": f.text,
                "material_type": nearest_material._.norm,
                "distance": min_distance,
                "confidence": "high" if min_distance < 50 else "medium"
            })
        else:
            pairs.append({
                "formula": f.text,
                "material_type": "unknown",
                "distance": float("inf"),
                "confidence": "low"
            })
    
    return pairs

def standardize_material_formula(formula):
    """
    Standardize material formula using pymatgen if available
    """
    if not formula or not isinstance(formula, str):
        return formula
    
    # Basic cleaning
    formula = re.sub(r'\s+', '', formula)
    formula = re.sub(r'[\(\)\[\]\{\}]', '', formula)
    
    if PYMAGEN_AVAILABLE:
        try:
            comp = Composition(formula)
            return comp.reduced_formula
        except Exception:
            pass
    
    # Basic normalization
    formula = re.sub(r'([A-Z][a-z]?)(\d*\.?\d*)', lambda m: m.group(1) + (m.group(2) if m.group(2) else ""), formula)
    
    # Handle common substitutions
    substitutions = {
        'Bi2Te3': 'Bi₂Te₃', 'PbTe': 'PbTe', 'SnSe': 'SnSe', 
        'CoSb3': 'CoSb₃', 'SiGe': 'SiGe', 'Zn4Sb3': 'Zn₄Sb₃',
        'Mg2Si': 'Mg₂Si', 'Cu2Se': 'Cu₂Se'
    }
    
    for orig, sub in substitutions.items():
        formula = re.sub(orig, sub, formula, flags=re.IGNORECASE)
    
    # Convert numbers to subscripts
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    formula = formula.translate(subscript_map)
    
    return formula

def load_spacy_model(synonyms=None):
    """
    Load and configure spaCy model with custom components
    """
    if synonyms is None:
        synonyms = DEFAULT_SYNONYMS
    
    try:
        nlp = spacy.load("en_core_web_lg")
    except Exception:
        nlp = spacy.load("en_core_web_sm")
    
    # Add custom components
    if not nlp.has_pipe("formula_ner"):
        nlp.add_pipe("formula_ner", last=True)
    
    # Build and add material matcher
    matcher = build_material_matcher(nlp, synonyms)
    if not nlp.has_pipe("material_matcher"):
        nlp.add_pipe("material_matcher", last=True)
    
    # Set extensions
    if not Doc.has_extension("material_matcher"):
        Doc.set_extension("material_matcher", default=None)
    Doc.set_extension("material_matcher", default=matcher, force=True)
    
    if not Span.has_extension("norm"):
        Span.set_extension("norm", default=None)
    
    nlp.max_length = 500_000
    return nlp

# Load spaCy model with enhanced capabilities
nlp = load_spacy_model(DEFAULT_SYNONYMS)

# Load SciBERT
try:
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model.eval()
except Exception as e:
    st.error(f"Failed to load SciBERT: {e}. Install: `pip install transformers torch`")
    st.stop()

# Initialize session state
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "ner_results" not in st.session_state:
    st.session_state.ner_results = None
if "raw_common_terms" not in st.session_state:
    st.session_state.raw_common_terms = None
if "common_terms" not in st.session_state:
    st.session_state.common_terms = None
if "db_file" not in st.session_state:
    st.session_state.db_file = None
if "term_counts" not in st.session_state:
    st.session_state.term_counts = None
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "csv_filename" not in st.session_state:
    st.session_state.csv_filename = None
if "material_classifications" not in st.session_state:
    st.session_state.material_classifications = None
if "formula_mappings" not in st.session_state:
    st.session_state.formula_mappings = None

def update_log(message):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_usage = psutil.Process().memory_info().rss / 1024**2  # MB
    log_message = f"[{timestamp}] {message} (Memory: {memory_usage:.2f} MB)"
    st.session_state.log_buffer.append(log_message)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(log_message)

# [Keep all your existing functions like inspect_database, extract_common_terms, 
# perform_ner_on_terms, plot_word_cloud, etc. but add the enhanced material classification]

def extract_material_classifications_enhanced(db_file):
    """
    Enhanced material classification using spaCy NER and formula matching
    """
    try:
        update_log("Starting enhanced p-type/n-type material classification")
        conn = sqlite3.connect(db_file)
        query = "SELECT id, title, year, content FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            update_log("No valid papers found for material classification")
            return pd.DataFrame(), pd.DataFrame()
        
        material_classifications = []
        formula_mappings = []
        
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            content = row["content"]
            
            if not content or len(content) < 100:
                continue
            
            # Process with enhanced spaCy model
            try:
                if len(content) > nlp.max_length:
                    content = content[:nlp.max_length]
                
                doc = nlp(content)
                
                # Extract formula-material pairs
                pairs = link_formula_to_material(doc)
                
                for pair in pairs:
                    if pair["material_type"] != "unknown":
                        standardized_formula = standardize_material_formula(pair["formula"])
                        
                        formula_mappings.append({
                            "paper_id": row["id"],
                            "title": row["title"],
                            "year": row["year"],
                            "original_formula": pair["formula"],
                            "standardized_formula": standardized_formula,
                            "material_type": pair["material_type"],
                            "confidence": pair["confidence"],
                            "distance": pair["distance"]
                        })
                        
                        material_classifications.append({
                            "paper_id": row["id"],
                            "title": row["title"],
                            "year": row["year"],
                            "material": standardized_formula,
                            "classification": pair["material_type"],
                            "confidence": pair["confidence"],
                            "context": f"Formula: {pair['formula']}, Type: {pair['material_type']}"
                        })
            
            except Exception as e:
                update_log(f"Error processing paper {row['id']}: {str(e)}")
                continue
            
            progress_value = min((i + 1) / len(df), 1.0)
            progress_bar.progress(progress_value)
        
        update_log(f"Extracted {len(material_classifications)} material classifications")
        update_log(f"Extracted {len(formula_mappings)} formula mappings")
        
        return pd.DataFrame(material_classifications), pd.DataFrame(formula_mappings)
    
    except Exception as e:
        update_log(f"Error in enhanced material classification: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def plot_enhanced_material_analysis(material_df, formula_df):
    """
    Create enhanced visualizations for material analysis
    """
    if material_df.empty:
        return None, None, None, None
    
    # Material classification distribution
    fig_class_dist = px.pie(
        material_df, 
        names='classification', 
        title='Distribution of Material Types',
        hole=0.3
    )
    
    # Confidence level distribution
    fig_confidence = px.pie(
        material_df, 
        names='confidence', 
        title='Confidence Levels of Classifications',
        hole=0.3
    )
    
    # Top materials by type
    top_materials = material_df.groupby(['material', 'classification']).size().reset_index(name='count')
    top_materials = top_materials.nlargest(15, 'count')
    
    fig_top_materials = px.bar(
        top_materials,
        x='material',
        y='count',
        color='classification',
        title='Top 15 Materials by Type',
        labels={'material': 'Material', 'count': 'Frequency'}
    )
    fig_top_materials.update_layout(xaxis_tickangle=-45)
    
    # Formula standardization analysis
    if not formula_df.empty:
        standardization_effect = formula_df.groupby('original_formula')['standardized_formula'].nunique().reset_index()
        standardization_effect.columns = ['original', 'unique_standardized']
        standardization_effect = standardization_effect[standardization_effect['unique_standardized'] > 1]
        
        if not standardization_effect.empty:
            fig_standardization = px.bar(
                standardization_effect,
                x='original',
                y='unique_standardized',
                title='Formula Standardization Effect',
                labels={'original': 'Original Formula', 'unique_standardized': 'Unique Standardized Forms'}
            )
            fig_standardization.update_layout(xaxis_tickangle=-45)
        else:
            fig_standardization = None
    else:
        fig_standardization = None
    
    return fig_class_dist, fig_confidence, fig_top_materials, fig_standardization

# Main app
st.header("Select or Upload Database")
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
db_options = [os.path.basename(f) for f in db_files] + ["Upload a new .db file"]
db_selection = st.selectbox("Select Database", db_options, key="db_select")
uploaded_file = None
if db_selection == "Upload a new .db file":
    uploaded_file = st.file_uploader("Upload SQLite Database (.db)", type=["db"], key="db_upload")
    if uploaded_file:
        temp_db_path = os.path.join(DB_DIR, f"uploaded_{uuid.uuid4().hex}.db")
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.db_file = temp_db_path
        update_log(f"Uploaded database saved as {temp_db_path}")
else:
    if db_selection:
        st.session_state.db_file = os.path.join(DB_DIR, db_selection)
        update_log(f"Selected database: {db_selection}")

if st.session_state.db_file:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Database Inspection", "Common Terms Analysis", "NER Analysis", "Material Classification", "Enhanced Formula Analysis"])
    
    # [Your existing tabs 1-3 code remains unchanged]
    
    with tab4:
        st.header("Material Classification Analysis (p-type vs n-type)")
        # [Your existing tab4 code remains unchanged]
    
    with tab5:
        st.header("🧪 Enhanced Formula-Material Analysis")
        
        st.info("""
        This enhanced analysis uses spaCy NER with custom components to:
        - Detect material formulas using advanced regex patterns
        - Identify material types using phrase matching with synonyms
        - Link formulas to their nearest material type mentions
        - Standardize formulas using pymatgen
        """)
        
        with st.sidebar:
            st.subheader("Advanced Parameters")
            max_link_distance = st.slider("Maximum Link Distance (characters)", 50, 500, 200, 
                                         help="Maximum distance between formula and material type for linking")
            confidence_threshold = st.selectbox("Confidence Threshold", ["high", "medium", "low"], 
                                              index=0, help="Minimum confidence level to include")
        
        if st.button("Run Enhanced Analysis", key="enhanced_analysis"):
            with st.spinner("Running enhanced formula-material analysis..."):
                material_df, formula_df = extract_material_classifications_enhanced(st.session_state.db_file)
                st.session_state.material_classifications = material_df
                st.session_state.formula_mappings = formula_df
            
            if material_df.empty:
                st.warning("No material classifications found.")
            else:
                st.success(f"Extracted {len(material_df)} material classifications and {len(formula_df)} formula mappings!")
                
                # Filter by confidence
                filtered_material = material_df[material_df['confidence'] == confidence_threshold]
                filtered_formula = formula_df[formula_df['confidence'] == confidence_threshold]
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Classifications", len(material_df))
                with col2:
                    st.metric("High Confidence", len(material_df[material_df['confidence'] == 'high']))
                with col3:
                    st.metric("p-type Materials", len(material_df[material_df['classification'] == 'p-type']))
                with col4:
                    st.metric("n-type Materials", len(material_df[material_df['classification'] == 'n-type']))
                
                # Visualizations
                st.subheader("📈 Visualizations")
                fig1, fig2, fig3, fig4 = plot_enhanced_material_analysis(filtered_material, filtered_formula)
                
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
                if fig4:
                    st.plotly_chart(fig4, use_container_width=True)
                
                # Data tables
                st.subheader("🔍 Formula-Material Mappings")
                st.dataframe(filtered_formula[['paper_id', 'original_formula', 'standardized_formula', 
                                             'material_type', 'confidence', 'distance']].head(50))
                
                st.subheader("📋 Material Classifications")
                st.dataframe(filtered_material[['paper_id', 'material', 'classification', 'confidence']].head(50))
                
                # Download buttons
                material_csv = filtered_material.to_csv(index=False)
                formula_csv = filtered_formula.to_csv(index=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Material Classifications", 
                        material_csv, 
                        "enhanced_material_classifications.csv", 
                        "text/csv"
                    )
                with col2:
                    st.download_button(
                        "📥 Download Formula Mappings", 
                        formula_csv, 
                        "formula_material_mappings.csv", 
                        "text/csv"
                    )
        
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="enhanced_logs")

else:
    st.warning("Select or upload a database file.")
