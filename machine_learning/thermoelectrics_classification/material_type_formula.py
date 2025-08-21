import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Span
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
    st.warning("pymatgen is not installed. Material formula standardization will be limited. Install with: `pip install pymatgen`")

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
DB_DIR = "/home/kindness/workstation/declarmima_members/anil_kunwar/projects/word_graph1/conferenceFME/debuggin1/material_type"
logging.basicConfig(
    filename=os.path.join(DB_DIR, 'thermoelectric_ner.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Streamlit configuration
st.set_page_config(page_title="Thermoelectric Material Classification Tool", layout="wide")
st.title("Thermoelectric Material Classification and Analysis Tool")
st.markdown("""
This tool inspects SQLite databases, extracts common terms and phrases related to thermoelectric materials, 
performs rule-based NER analysis using SciBERT, classifies materials as p-type or n-type, and allows users to input a chemical formula to check its classification.

**Date and Time**: 08:22 PM CEST, Thursday, August 21, 2025

**Dependencies**:
- `pip install streamlit pandas sqlite3 spacy transformers torch nltk networkx wordcloud seaborn matplotlib psutil plotly pymatgen`
- `python -m spacy download en_core_web_lg` (or `en_core_web_sm` as fallback)
""")

# Load spaCy
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    st.warning(f"Failed to load 'en_core_web_lg': {e}. Using 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e2:
        st.error(f"Failed to load spaCy: {e2}. Install: `python -m spacy download en_core_web_sm`")
        st.stop()

# Custom tokenizer for hyphenated phrases
@Language.component("custom_tokenizer")
def custom_tokenizer(doc):
    hyphenated_phrases = ["p-type", "n-type", "thermoelectric-material", "Seebeck-coefficient", "figure-of-merit"]
    for phrase in hyphenated_phrases:
        if phrase.lower() in doc.text.lower():
            with doc.retokenize() as retokenizer:
                for match in re.finditer(rf'\b{re.escape(phrase)}\b', doc.text, re.IGNORECASE):
                    start_char, end_char = match.span()
                    start_token = None
                    for token in doc:
                        if token.idx >= start_char:
                            start_token = token.i
                            break
                    if start_token is not None:
                        retokenizer.merge(doc[start_token:start_token+len(phrase.split('-'))])
    return doc

# Add rule-based NER for chemical formulas
@Language.component("formula_ner")
def formula_ner(doc):
    # Regex pattern for chemical formulas (e.g., Bi2Te3, (Bi0.5Sb0.5)2Te3, Bi2Te3:Cu)
    formula_pattern = r'\b(?:[A-Z][a-z]?(?:\d*\.?\d*)?)+(?:[:][A-Z][a-z]?(?:\d*\.?\d*)?)?\b'
    invalid_terms = {'p-type', 'n-type', 'doping', 'doped', 'thermoelectric', 'material', 'the', 'and',
                    'is', 'exhibits', 'type', 'based', 'sample', 'compound', 'system', 'properties'}
    
    new_ents = []
    for match in re.finditer(formula_pattern, doc.text):
        start_char, end_char = match.span()
        formula_text = doc.text[start_char:end_char]
        
        # Skip if the text is in invalid terms
        if formula_text.lower() in invalid_terms:
            continue
        
        # Validate with pymatgen if available
        if PYMAGEN_AVAILABLE:
            try:
                comp = Composition(formula_text)
                if not comp.valid:
                    continue
            except Exception:
                continue
        
        # Find the corresponding span in the doc
        span = doc.char_span(start_char, end_char, label="FORMULA")
        if span is not None:
            new_ents.append(span)
    
    # Merge new entities, avoiding overlaps
    doc.ents = [ent for ent in doc.ents if ent.label_ != "FORMULA"] + new_ents
    return doc

nlp.add_pipe("custom_tokenizer", before="parser")
nlp.add_pipe("formula_ner", after="ner")
nlp.max_length = 500_000

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

def update_log(message):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_usage = psutil.Process().memory_info().rss / 1024**2  # MB
    log_message = f"[{timestamp}] {message} (Memory: {memory_usage:.2f} MB)"
    st.session_state.log_buffer.append(log_message)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(log_message)

def standardize_material_formula(formula, preserve_stoichiometry=False):
    """
    Standardize material formula using pymatgen if available, with enhanced validation.
    
    Args:
        formula (str): Input formula to standardize.
        preserve_stoichiometry (bool): If True, retain exact stoichiometry.
    
    Returns:
        str or None: Standardized formula or None if invalid.
    """
    if not formula or not isinstance(formula, str):
        update_log(f"Invalid input formula: {formula}")
        return None
    
    # Basic cleaning
    formula = re.sub(r'\s+', '', formula)  # Remove whitespace
    formula = re.sub(r'[\[\]\{\}]', '', formula)  # Remove brackets, keep parentheses
    
    # Pre-validation: Check for at least one valid element
    element_pattern = r'[A-Z][a-z]?\d*'
    if not re.search(element_pattern, formula):
        update_log(f"Skipped non-chemical term '{formula}': no valid elements")
        return None
    
    # List of invalid terms to skip
    invalid_terms = [
        'p-type', 'n-type', 'doping', 'doped', 'thermoelectric', 'material', 'the', 'and',
        'is', 'exhibits', 'type', 'based', 'sample', 'compound', 'system', 'properties'
    ]
    if any(term.lower() in formula.lower() for term in invalid_terms):
        update_log(f"Skipped non-chemical term '{formula}': contains invalid term")
        return None
    
    # Handle doped materials
    doping_pattern = r'(.+?)(?::|doped\s+)([A-Za-z0-9]+)'
    doping_match = re.match(doping_pattern, formula, re.IGNORECASE)
    dopant = None
    if doping_match:
        base_formula, dopant = doping_match.groups()
        formula = base_formula.strip()
        update_log(f"Detected doped material: base='{formula}', dopant='{dopant}'")
    
    # If pymatgen is available and enabled
    if PYMAGEN_AVAILABLE and st.session_state.get('enable_pymatgen', False):
        try:
            comp = Composition(formula)
            if not comp.valid:
                update_log(f"Invalid chemical formula '{formula}': not a valid composition")
                return None
            
            # Validate elements
            elements = comp.elements
            if not all(isinstance(el, Element) for el in elements):
                update_log(f"Invalid elements in formula '{formula}'")
                return None
            
            # Standardize formula
            if preserve_stoichiometry:
                el_amt_dict = comp.get_el_amt_dict()
                standardized_formula = ''.join(f"{el}{amt:.2f}" if amt != int(amt) else f"{el}{int(amt)}"
                                              for el, amt in sorted(el_amt_dict.items()))
            else:
                standardized_formula = comp.reduced_formula
            
            # Reattach dopant
            if dopant:
                try:
                    dopant_comp = Composition(dopant)
                    standardized_formula = f"{standardized_formula}:{dopant_comp.reduced_formula}"
                except Exception as e:
                    update_log(f"Failed to parse dopant '{dopant}' in '{formula}': {e}")
            
            update_log(f"Standardized formula '{formula}' to '{standardized_formula}' using pymatgen")
            return standardized_formula
        except Exception as e:
            update_log(f"pymatgen could not parse formula '{formula}': {str(e)}")
            # Fall back to basic normalization
    
    # Basic normalization
    formula = re.sub(r'([A-Z][a-z]?)(\d*\.?\d*)', lambda m: m.group(1) + (m.group(2) if m.group(2) else ""), formula)
    
    # Handle common thermoelectric materials
    common_materials = {
        'Bi2Te3': 'Bi₂Te₃', 'PbTe': 'PbTe', 'SnSe': 'SnSe', 'CoSb3': 'CoSb₃', 'SiGe': 'SiGe',
        'Zn4Sb3': 'Zn₄Sb₃', 'Mg2Si': 'Mg₂Si', 'Cu2Se': 'Cu₂Se'
    }
    for key, value in common_materials.items():
        if re.search(rf'\b{re.escape(key)}\b|-based\b', formula, re.IGNORECASE):
            formula = value
            update_log(f"Matched common material '{key}' to '{value}'")
            break
    
    # Reattach dopant
    if dopant:
        formula = f"{formula}:{dopant}"
    
    # Convert numbers to subscripts
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    formula = formula.translate(subscript_map)
    
    update_log(f"Normalized formula '{formula}' using basic standardization")
    return formula

def extract_material_classifications(db_file, preserve_stoichiometry=False):
    """
    Extract and classify materials as p-type or n-type using spaCy NER for formula detection.
    
    Args:
        db_file (str): Path to the SQLite database.
        preserve_stoichiometry (bool): If True, retain exact stoichiometry.
    
    Returns:
        pd.DataFrame: DataFrame with classified materials.
    """
    try:
        update_log("Starting p-type/n-type material classification with NER")
        conn = sqlite3.connect(db_file)
        query = "SELECT id, title, year, content FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%'"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            update_log("No valid papers found for material classification")
            return pd.DataFrame()
        
        material_classifications = []
        # Patterns to locate p-type/n-type contexts
        p_type_patterns = [
            r"p-type\s+([A-Za-z0-9\(\)\-\s,:]+?)(?=\s|,|\.|;|:|$)",
            r"p-type\s+material.*?([A-Za-z0-9\(\)\-\s,:]+?)(?=\s|,|\.|;|:|$)",
            r"([A-Za-z0-9\(\)\-\s,:]+?)\s+is\s+p-type",
            r"([A-Za-z0-9\(\)\-\s,:]+?)\s+exhibits\s+p-type",
            r"p-type\s+([A-Za-z0-9\(\)\-\s,:]+?)\s+thermoelectric",
            r"p-type\s+doped\s+([A-Za-z0-9\(\)\-\s,:]+?)",
            r"([A-Za-z0-9\(\)\-\s,:]+?)\s+doped\s+p-type"
        ]
        n_type_patterns = [
            r"n-type\s+([A-Za-z0-9\(\)\-\s,:]+?)(?=\s|,|\.|;|:|$)",
            r"n-type\s+material.*?([A-Za-z0-9\(\)\-\s,:]+?)(?=\s|,|\.|;|:|$)",
            r"([A-Za-z0-9\(\)\-\s,:]+?)\s+is\s+n-type",
            r"([A-Za-z0-9\(\)\-\s,:]+?)\s+exhibits\s+n-type",
            r"n-type\s+([A-Za-z0-9\(\)\-\s,:]+?)\s+thermoelectric",
            r"n-type\s+doped\s+([A-Za-z0-9\(\)\-\s,:]+?)",
            r"([A-Za-z0-9\(\)\-\s,:]+?)\s+doped\s+n-type"
        ]
        common_te_materials = [
            "Bi2Te3", "PbTe", "SnSe", "CoSb3", "SiGe", "Skutterudite",
            "Half-Heusler", "Clathrate", "Zn4Sb3", "Mg2Si", "Cu2Se"
        ]
        
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            content = row["content"]
            if len(content) > nlp.max_length:
                content = content[:nlp.max_length]
                update_log(f"Truncated content for paper {row['id']}")
            
            # Process content with spaCy for FORMULA entities
            doc = nlp(content)
            formula_entities = [ent.text for ent in doc.ents if ent.label_ == "FORMULA"]
            
            # Extract p-type materials
            p_type_materials = set()
            for pattern in p_type_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    material = match.group(1).strip()
                    if material and len(material) > 2:
                        # Check if material is a recognized FORMULA entity
                        if material in formula_entities:
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                p_type_materials.add(standardized_material)
                                update_log(f"Added p-type material '{standardized_material}' from regex match")
            
            # Extract n-type materials
            n_type_materials = set()
            for pattern in n_type_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    material = match.group(1).strip()
                    if material and len(material) > 2:
                        if material in formula_entities:
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                n_type_materials.add(standardized_material)
                                update_log(f"Added n-type material '{standardized_material}' from regex match")
            
            # Check common thermoelectric materials in context
            p_type_context = re.search(r"p-type[^\.]{0,500}", content, re.IGNORECASE)
            n_type_context = re.search(r"n-type[^\.]{0,500}", content, re.IGNORECASE)
            
            if p_type_context:
                context_doc = nlp(p_type_context.group(0))
                for ent in context_doc.ents:
                    if ent.label_ == "FORMULA":
                        standardized_material = standardize_material_formula(ent.text, preserve_stoichiometry)
                        if standardized_material:
                            p_type_materials.add(standardized_material)
                            update_log(f"Added p-type material '{standardized_material}' from context")
            
            if n_type_context:
                context_doc = nlp(n_type_context.group(0))
                for ent in context_doc.ents:
                    if ent.label_ == "FORMULA":
                        standardized_material = standardize_material_formula(ent.text, preserve_stoichiometry)
                        if standardized_material:
                            n_type_materials.add(standardized_material)
                            update_log(f"Added n-type material '{standardized_material}' from context")
            
            # Add common thermoelectric materials if detected as FORMULA
            for material in common_te_materials:
                if material.lower() in content.lower():
                    doc = nlp(material)
                    if any(ent.label_ == "FORMULA" for ent in doc.ents):
                        standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                        if standardized_material:
                            if p_type_context and material.lower() in p_type_context.group(0).lower():
                                p_type_materials.add(standardized_material)
                                update_log(f"Added common p-type material '{standardized_material}'")
                            if n_type_context and material.lower() in n_type_context.group(0).lower():
                                n_type_materials.add(standardized_material)
                                update_log(f"Added common n-type material '{standardized_material}'")
            
            # Add to results
            for material in p_type_materials:
                material_classifications.append({
                    "paper_id": row["id"],
                    "title": row["title"],
                    "year": row["year"],
                    "material": material,
                    "classification": "p-type",
                    "context": f"Found in context: {content[max(0, match.start()-50):min(len(content), match.end()+50)]}..."
                })
            
            for material in n_type_materials:
                material_classifications.append({
                    "paper_id": row["id"],
                    "title": row["title"],
                    "year": row["year"],
                    "material": material,
                    "classification": "n-type",
                    "context": f"Found in context: {content[max(0, match.start()-50):min(len(content), match.end()+50)]}..."
                })
            
            progress_value = min((i + 1) / len(df), 1.0)
            progress_bar.progress(progress_value)
        
        material_df = pd.DataFrame(material_classifications)
        
        # Sort and clean the material column
        if not material_df.empty:
            material_df = material_df.drop_duplicates(subset=["paper_id", "material", "classification"])
            material_df = material_df.sort_values(by=["material", "classification"])
            update_log(f"Cleaned and sorted DataFrame: {len(material_df)} unique classifications")
        
        update_log(f"Extracted {len(material_df)} material classifications")
        return material_df
    
    except Exception as e:
        update_log(f"Error in material classification: {str(e)}")
        return pd.DataFrame()

def classify_formula(formula):
    """
    Classify a user-input chemical formula as p-type or n-type based on material classifications.
    """
    try:
        if not formula.strip():
            update_log("Empty formula input provided")
            return None, "Please enter a valid chemical formula."
        
        # Normalize formula
        normalized_formula = standardize_material_formula(formula, preserve_stoichiometry=st.session_state.get('preserve_stoichiometry', False))
        if not normalized_formula:
            update_log(f"Invalid chemical formula: {formula}")
            return None, f"'{formula}' is not a valid chemical formula."
        
        update_log(f"Normalized formula '{formula}' to '{normalized_formula}'")
        
        # Check classifications
        if st.session_state.material_classifications is None:
            update_log("No material classifications available for formula lookup")
            return None, "Please run Material Classification Analysis first (in the Material Classification tab)."
        
        material_df = st.session_state.material_classifications
        formula_matches = material_df[material_df["material"].str.lower() == normalized_formula.lower()]
        
        if formula_matches.empty:
            update_log(f"No classification found for formula '{normalized_formula}'")
            return None, f"No p-type or n-type classification found for '{normalized_formula}'."
        
        classifications = formula_matches["classification"].unique()
        paper_ids = formula_matches["paper_id"].unique()
        
        if len(classifications) == 1:
            classification = classifications[0]
            update_log(f"Formula '{normalized_formula}' classified as {classification}")
            return {
                "formula": normalized_formula,
                "classification": classification,
                "paper_ids": paper_ids.tolist(),
                "count": len(formula_matches)
            }, None
        else:
            update_log(f"Formula '{normalized_formula}' has multiple classifications: {', '.join(classifications)}")
            return {
                "formula": normalized_formula,
                "classification": "Multiple (p-type and n-type)",
                "paper_ids": paper_ids.tolist(),
                "count": len(formula_matches)
            }, None
    
    except Exception as e:
        update_log(f"Error classifying formula '{formula}': {str(e)}")
        return None, f"Error classifying formula: {str(e)}"

def plot_material_classifications(df, top_n=20):
    """
    Create visualizations for p-type and n-type material classifications.
    """
    if df.empty:
        return None, None, None
    
    # Count materials by classification
    material_counts = df.groupby(["material", "classification"]).size().reset_index(name="count")
    
    # Get top materials
    top_materials = material_counts.groupby("material")["count"].sum().nlargest(top_n).index
    filtered_df = material_counts[material_counts["material"].isin(top_materials)]
    
    # Create bar chart
    fig_bar = px.bar(
        filtered_df, 
        x="material", 
        y="count", 
        color="classification",
        title=f"Top {top_n} Materials by p-type/n-type Classification",
        labels={"material": "Material", "count": "Frequency", "classification": "Type"}
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    
    # Create pie chart for classification distribution
    class_dist = df["classification"].value_counts()
    fig_pie = px.pie(
        values=class_dist.values,
        names=class_dist.index,
        title="Distribution of p-type vs n-type Classifications"
    )
    
    # Create timeline of classifications by year
    if "year" in df.columns and df["year"].notna().any():
        yearly_data = df.groupby(["year", "classification"]).size().reset_index(name="count")
        fig_timeline = px.line(
            yearly_data,
            x="year",
            y="count",
            color="classification",
            title="Trend of p-type and n-type Classifications Over Time",
            labels={"year": "Year", "count": "Number of Mentions", "classification": "Type"}
        )
    else:
        fig_timeline = None
    
    return fig_bar, fig_pie, fig_timeline

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Database Inspection", "Common Terms Analysis", "NER Analysis", 
                                             "Material Classification", "Formula Classification"])
    
    # Placeholder for other tabs
    with tab1:
        st.header("Database Inspection")
        st.markdown("Run database inspection to view tables, schema, and sample data.")
    
    with tab2:
        st.header("Common Terms Analysis")
        st.markdown("Extract and visualize common terms and phrases from the database.")
    
    with tab3:
        st.header("NER Analysis")
        st.markdown("Perform Named Entity Recognition to extract entities like Seebeck coefficient, thermal conductivity, etc.")
    
    with tab4:
        st.header("Material Classification Analysis (p-type vs n-type)")
        
        with st.sidebar:
            st.subheader("Material Classification Parameters")
            material_top_n = st.slider("Number of Top Materials to Show", min_value=5, max_value=30, value=10, key="material_top_n")
            enable_pymatgen = st.checkbox("Use pymatgen for formula standardization", value=PYMAGEN_AVAILABLE, 
                                         disabled=not PYMAGEN_AVAILABLE,
                                         help="Requires pymatgen installation", key="enable_pymatgen")
            preserve_stoichiometry = st.checkbox("Preserve Exact Stoichiometry", value=False, 
                                                help="Retain exact stoichiometry (e.g., Bi2Te2.7Se0.3) instead of reducing", 
                                                key="preserve_stoichiometry")
        
        if st.button("Extract Material Classifications", key="extract_materials"):
            with st.spinner("Extracting p-type and n-type material classifications..."):
                material_df = extract_material_classifications(st.session_state.db_file, preserve_stoichiometry)
                st.session_state.material_classifications = material_df
                
            if material_df.empty:
                st.warning("No material classifications found. Try adjusting extraction patterns.")
            else:
                st.success(f"Extracted {len(material_df)} unique material classifications!")
                
                # Show summary statistics
                st.subheader("Classification Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Classifications", len(material_df))
                with col2:
                    p_type_count = len(material_df[material_df["classification"] == "p-type"])
                    st.metric("p-type Materials", p_type_count)
                with col3:
                    n_type_count = len(material_df[material_df["classification"] == "n-type"])
                    st.metric("n-type Materials", n_type_count)
                
                # Show material formula standardization info
                if PYMAGEN_AVAILABLE and enable_pymatgen:
                    st.info(f"Material formulas standardized using pymatgen (exact stoichiometry {'preserved' if preserve_stoichiometry else 'reduced'})")
                else:
                    st.warning("pymatgen not available or disabled. Using basic formula standardization.")
                
                # Show visualizations
                st.subheader("Visualizations")
                fig_bar, fig_pie, fig_timeline = plot_material_classifications(material_df, material_top_n)
                
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if fig_pie:
                        st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    if fig_timeline:
                        st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Show data table
                st.subheader("Extracted Material Classifications")
                st.dataframe(
                    material_df[["paper_id", "title", "year", "material", "classification", "context"]].head(100),
                    use_container_width=True
                )
                
                # Download button
                material_csv = material_df.to_csv(index=False)
                st.download_button(
                    "Download Material Classifications CSV", 
                    material_csv, 
                    "material_classifications.csv", 
                    "text/csv", 
                    key="download_materials"
                )
    
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="material_logs")
    
    with tab5:
        st.header("Formula Classification")
        st.markdown("""
        Enter a chemical formula (e.g., `Bi2Te3`, `PbTe`, `SnSe`) to check if it is classified as p-type or n-type based on the database analysis.
        **Note**: You must run the Material Classification Analysis (in the Material Classification tab) first to populate the classification data.
        """)
        
        formula_input = st.text_input("Enter Chemical Formula", key="formula_input")
        if st.button("Classify Formula", key="classify_formula"):
            if not formula_input:
                st.error("Please enter a chemical formula.")
            else:
                with st.spinner(f"Classifying formula '{formula_input}'..."):
                    result, error = classify_formula(formula_input)
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Formula: **{result['formula']}**")
                        st.write(f"Classification: **{result['classification']}**")
                        st.write(f"Found in {result['count']} paper(s): {', '.join(result['paper_ids'])}")
                        update_log(f"Displayed classification for '{result['formula']}': {result['classification']}")
        
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="formula_logs")
else:
    st.warning("Select or upload a database file.")

# Placeholder for other functions (unchanged)
def inspect_database(db_path):
    pass  # Existing implementation

def extract_common_terms(db_file, min_freq=10, phrase_weight=1.5, pmi_threshold=2.0, ngram_range=(1, 3)):
    pass  # Existing implementation

def perform_ner_on_terms(db_file, selected_terms):
    pass  # Existing implementation

def plot_word_cloud(terms, top_n, font_size, font_type, colormap):
    pass  # Existing implementation

def plot_term_histogram(terms, top_n):
    pass  # Existing implementation

def plot_term_co_occurrence(terms, top_n, db_file, font_size, colormap):
    pass  # Existing implementation

def plot_ner_histogram(df, top_n, colormap):
    pass  # Existing implementation

def plot_ner_co_occurrence(df, top_n, font_size, colormap):
    pass  # Existing implementation

def plot_ner_value_histogram(df, top_n, colormap):
    pass  # Existing implementation

def plot_individual_ner_value_histograms(df, colormap):
    pass  # Existing implementation

def plot_ner_value_radial(df, top_n, colormap):
    pass  # Existing implementation

def plot_ner_value_boxplot(df, top_n, colormap):
    pass  # Existing implementation

def plot_term_frequency_chart(terms, top_n):
    pass  # Existing implementation

def get_scibert_embedding(text):
    pass  # Existing implementation

def calculate_pmi(phrase, word_counts, phrase_counts, total_words):
    pass  # Existing implementation
