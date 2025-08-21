import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
import re
import logging
import plotly.express as px
import plotly.graph_objects as go
import uuid
import psutil
from datetime import datetime
import numpy as np
from collections import Counter
import glob

# Try to import pymatgen for material formula parsing
try:
    from pymatgen.core.composition import Composition
    from pymatgen.core.periodic_table import Element
    PYMAGEN_AVAILABLE = True
except ImportError:
    PYMAGEN_AVAILABLE = False
    st.warning("pymatgen is not installed. Material formula standardization will be limited. Install with: `pip install pymatgen`")

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
This tool extracts p-type and n-type material classifications from SQLite databases and allows classification of user-input chemical formulas.

**Date and Time**: 08:30 PM CEST, Thursday, August 21, 2025

**Dependencies**:
- `pip install streamlit pandas sqlite3 spacy plotly psutil pymatgen`
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

# Rule-based NER for chemical formulas
@Language.component("formula_ner")
def formula_ner(doc):
    # Enhanced regex pattern for chemical formulas, including doped and fractional formulas
    formula_pattern = r'\b(?:[A-Z][a-z]?(?:\d*\.?\d*)?)+(?:[-:][A-Z][a-z]?(?:\d*\.?\d*)?)*\b'
    invalid_terms = {
        'p-type', 'n-type', 'doping', 'doped', 'thermoelectric', 'material', 'the', 'and',
        'is', 'exhibits', 'type', 'based', 'sample', 'compound', 'system', 'properties'
    }
    
    new_ents = []
    for match in re.finditer(formula_pattern, doc.text):
        start_char, end_char = match.span()
        formula_text = doc.text[start_char:end_char]
        
        # Skip invalid terms
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
        
        span = doc.char_span(start_char, end_char, label="FORMULA")
        if span is not None:
            new_ents.append(span)
    
    doc.ents = [ent for ent in doc.ents if ent.label_ != "FORMULA"] + new_ents
    return doc

nlp.add_pipe("custom_tokenizer", before="parser")
nlp.add_pipe("formula_ner", after="ner")
nlp.max_length = 500_000

# Initialize session state
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "material_classifications" not in st.session_state:
    st.session_state.material_classifications = None
if "db_file" not in st.session_state:
    st.session_state.db_file = None
if "error_summary" not in st.session_state:
    st.session_state.error_summary = []

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_usage = psutil.Process().memory_info().rss / 1024**2  # MB
    log_message = f"[{timestamp}] {message} (Memory: {memory_usage:.2f} MB)"
    st.session_state.log_buffer.append(log_message)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(log_message)

def standardize_material_formula(formula, preserve_stoichiometry=False, canonical_order=True):
    """
    Standardize material formula using pymatgen if available, with enhanced validation.
    
    Args:
        formula (str): Input formula to standardize.
        preserve_stoichiometry (bool): If True, retain exact stoichiometry.
        canonical_order (bool): If True, sort elements alphabetically.
    
    Returns:
        str or None: Standardized formula or None if invalid.
    """
    if not formula or not isinstance(formula, str):
        update_log(f"Invalid input formula: {formula}")
        st.session_state.error_summary.append(f"Invalid formula: {formula}")
        return None
    
    # Basic cleaning
    formula = re.sub(r'\s+', '', formula)  # Remove whitespace
    formula = re.sub(r'[\[\]\{\}]', '', formula)  # Remove brackets, keep parentheses
    
    # Pre-validation: Check for at least one valid element
    element_pattern = r'[A-Z][a-z]?\d*'
    if not re.search(element_pattern, formula):
        update_log(f"Skipped non-chemical term '{formula}': no valid elements")
        st.session_state.error_summary.append(f"Skipped '{formula}': no valid elements")
        return None
    
    # List of invalid terms
    invalid_terms = [
        'p-type', 'n-type', 'doping', 'doped', 'thermoelectric', 'material', 'the', 'and',
        'is', 'exhibits', 'type', 'based', 'sample', 'compound', 'system', 'properties'
    ]
    if any(term.lower() in formula.lower() for term in invalid_terms):
        update_log(f"Skipped non-chemical term '{formula}': contains invalid term")
        st.session_state.error_summary.append(f"Skipped '{formula}': contains invalid term")
        return None
    
    # Handle doped materials
    doping_pattern = r'(.+?)(?::|doped\s+)([A-Za-z0-9,\.]+)'
    doping_match = re.match(doping_pattern, formula, re.IGNORECASE)
    dopants = None
    if doping_match:
        base_formula, dopants = doping_match.groups()
        formula = base_formula.strip()
        dopants = dopants.split(',')
        update_log(f"Detected doped material: base='{formula}', dopants='{','.join(dopants)}'")
    
    # If pymatgen is available and enabled
    if PYMAGEN_AVAILABLE and st.session_state.get('enable_pymatgen', False):
        try:
            comp = Composition(formula)
            if not comp.valid:
                update_log(f"Invalid chemical formula '{formula}': not a valid composition")
                st.session_state.error_summary.append(f"Invalid formula '{formula}': not a valid composition")
                return None
            
            # Validate elements
            elements = comp.elements
            if not all(isinstance(el, Element) for el in elements):
                update_log(f"Invalid elements in formula '{formula}'")
                st.session_state.error_summary.append(f"Invalid elements in formula '{formula}'")
                return None
            
            # Standardize formula
            if preserve_stoichiometry:
                el_amt_dict = comp.get_el_amt_dict()
                standardized_formula = ''.join(
                    f"{el}{amt:.2f}" if amt != int(amt) else f"{el}{int(amt)}"
                    for el, amt in (sorted(el_amt_dict.items()) if canonical_order else el_amt_dict.items())
                )
            else:
                standardized_formula = comp.reduced_formula
            
            # Reattach dopants
            if dopants:
                valid_dopants = []
                for dopant in dopants:
                    try:
                        dopant_comp = Composition(dopant.strip())
                        valid_dopants.append(dopant_comp.reduced_formula)
                    except Exception as e:
                        update_log(f"Failed to parse dopant '{dopant}' in '{formula}': {e}")
                        st.session_state.error_summary.append(f"Failed to parse dopant '{dopant}' in '{formula}'")
                if valid_dopants:
                    standardized_formula = f"{standardized_formula}:{','.join(valid_dopants)}"
            
            update_log(f"Standardized formula '{formula}' to '{standardized_formula}' using pymatgen")
            return standardized_formula
        except Exception as e:
            update_log(f"pymatgen could not parse formula '{formula}': {str(e)}")
            st.session_state.error_summary.append(f"pymatgen failed for '{formula}': {str(e)}")
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
    
    # Reattach dopants
    if dopants:
        formula = f"{formula}:{','.join(dopants)}"
    
    # Convert numbers to subscripts
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    formula = formula.translate(subscript_map)
    
    update_log(f"Normalized formula '{formula}' using basic standardization")
    return formula

def extract_material_classifications(db_file, preserve_stoichiometry=False, year_range=None):
    """
    Extract and classify materials as p-type or n-type using spaCy NER.
    
    Args:
        db_file (str): Path to the SQLite database.
        preserve_stoichiometry (bool): If True, retain exact stoichiometry.
        year_range (tuple): Optional (start_year, end_year) to filter papers.
    
    Returns:
        pd.DataFrame: DataFrame with classified materials.
    """
    try:
        update_log("Starting p-type/n-type material classification with NER")
        conn = sqlite3.connect(db_file)
        query = "SELECT id, title, year, content FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%'"
        if year_range:
            query += f" AND year BETWEEN {year_range[0]} AND {year_range[1]}"
        df = pd.read_sql_query(query, conn)
        
        # Check for standardized formulas table
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='standardized_formulas'")
        if cursor.fetchone():
            cached_df = pd.read_sql_query("SELECT material, classification FROM standardized_formulas", conn)
            update_log("Loaded cached standardized formulas")
            conn.close()
            return cached_df
        
        conn.close()
        
        if df.empty:
            update_log("No valid papers found for material classification")
            st.session_state.error_summary.append("No valid papers found in database")
            return pd.DataFrame()
        
        material_classifications = []
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
        
        def chunk_text(text, max_length=400000):
            """Split text into chunks to handle spaCy max_length limitation."""
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + max_length, len(text))
                # Ensure chunk ends at a sentence boundary if possible
                if end < len(text):
                    last_period = text.rfind('.', start, end)
                    end = last_period + 1 if last_period > start else end
                chunks.append(text[start:end])
                start = end
            return chunks
        
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            content = row["content"]
            # Chunk large texts
            chunks = chunk_text(content)
            
            for chunk in chunks:
                doc = nlp(chunk)
                formula_entities = [ent.text for ent in doc.ents if ent.label_ == "FORMULA"]
                
                # Extract p-type materials
                p_type_materials = set()
                for pattern in p_type_patterns:
                    matches = re.finditer(pattern, chunk, re.IGNORECASE)
                    for match in matches:
                        material = match.group(1).strip()
                        if material and len(material) > 2 and material in formula_entities:
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                p_type_materials.add((standardized_material, match.start()))
                
                # Extract n-type materials
                n_type_materials = set()
                for pattern in n_type_patterns:
                    matches = re.finditer(pattern, chunk, re.IGNORECASE)
                    for match in matches:
                        material = match.group(1).strip()
                        if material and len(material) > 2 and material in formula_entities:
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                n_type_materials.add((standardized_material, match.start()))
                
                # Check common thermoelectric materials
                p_type_context = re.search(r"p-type[^\.]{0,500}", chunk, re.IGNORECASE)
                n_type_context = re.search(r"n-type[^\.]{0,500}", chunk, re.IGNORECASE)
                
                if p_type_context:
                    context_doc = nlp(p_type_context.group(0))
                    for ent in context_doc.ents:
                        if ent.label_ == "FORMULA":
                            standardized_material = standardize_material_formula(ent.text, preserve_stoichiometry)
                            if standardized_material:
                                p_type_materials.add((standardized_material, ent.start_char))
                
                if n_type_context:
                    context_doc = nlp(n_type_context.group(0))
                    for ent in context_doc.ents:
                        if ent.label_ == "FORMULA":
                            standardized_material = standardize_material_formula(ent.text, preserve_stoichiometry)
                            if standardized_material:
                                n_type_materials.add((standardized_material, ent.start_char))
                
                # Add common thermoelectric materials
                for material in common_te_materials:
                    if material.lower() in chunk.lower():
                        doc = nlp(material)
                        if any(ent.label_ == "FORMULA" for ent in doc.ents):
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                if p_type_context and material.lower() in p_type_context.group(0).lower():
                                    p_type_materials.add((standardized_material, 0))
                                if n_type_context and material.lower() in n_type_context.group(0).lower():
                                    n_type_materials.add((standardized_material, 0))
                
                # Add to results
                for material, start_pos in p_type_materials:
                    context = chunk[max(0, start_pos-50):min(len(chunk), start_pos+50)]
                    material_classifications.append({
                        "paper_id": row["id"],
                        "title": row["title"],
                        "year": row["year"],
                        "material": material,
                        "classification": "p-type",
                        "context": f"Found in context: {context}..."
                    })
                
                for material, start_pos in n_type_materials:
                    context = chunk[max(0, start_pos-50):min(len(chunk), start_pos+50)]
                    material_classifications.append({
                        "paper_id": row["id"],
                        "title": row["title"],
                        "year": row["year"],
                        "material": material,
                        "classification": "n-type",
                        "context": f"Found in context: {context}..."
                    })
            
            progress_value = min((i + 1) / len(df), 1.0)
            progress_bar.progress(progress_value)
        
        material_df = pd.DataFrame(material_classifications)
        
        # Sort and clean
        if not material_df.empty:
            material_df = material_df.drop_duplicates(subset=["paper_id", "material", "classification"])
            material_df = material_df.sort_values(by=["material", "classification"])
            update_log(f"Cleaned and sorted DataFrame: {len(material_df)} unique classifications")
            
            # Cache standardized formulas
            conn = sqlite3.connect(db_file)
            material_df[["material", "classification"]].to_sql("standardized_formulas", conn, if_exists="replace", index=False)
            conn.close()
            update_log("Cached standardized formulas in database")
        
        update_log(f"Extracted {len(material_df)} material classifications")
        return material_df
    
    except Exception as e:
        update_log(f"Error in material classification: {str(e)}")
        st.session_state.error_summary.append(f"Extraction error: {str(e)}")
        return pd.DataFrame()

def classify_formula(formula, material_df):
    """
    Classify a chemical formula as p-type or n-type with confidence score.
    """
    try:
        if not formula.strip():
            update_log("Empty formula input provided")
            return None, "Please enter a valid chemical formula."
        
        normalized_formula = standardize_material_formula(formula, 
                                                        preserve_stoichiometry=st.session_state.get('preserve_stoichiometry', False))
        if not normalized_formula:
            update_log(f"Invalid chemical formula: {formula}")
            return None, f"'{formula}' is not a valid chemical formula."
        
        update_log(f"Normalized formula '{formula}' to '{normalized_formula}'")
        
        if material_df is None or material_df.empty:
            update_log("No material classifications available for formula lookup")
            return None, "Please run Material Classification Analysis first."
        
        formula_matches = material_df[material_df["material"].str.lower() == normalized_formula.lower()]
        
        if formula_matches.empty:
            update_log(f"No classification found for formula '{normalized_formula}'")
            return None, f"No p-type or n-type classification found for '{normalized_formula}'."
        
        classifications = formula_matches["classification"].value_counts()
        total_matches = len(formula_matches)
        paper_ids = formula_matches["paper_id"].unique()
        contexts = formula_matches["context"].tolist()
        
        # Calculate confidence score
        confidence = {cls: count / total_matches for cls, count in classifications.items()}
        primary_classification = classifications.idxmax()
        confidence_score = confidence.get(primary_classification, 0.0)
        
        update_log(f"Formula '{normalized_formula}' classified as {primary_classification} (confidence: {confidence_score:.2%})")
        return {
            "formula": normalized_formula,
            "classification": primary_classification,
            "confidence": confidence_score,
            "paper_ids": paper_ids.tolist(),
            "count": total_matches,
            "contexts": contexts,
            "all_classifications": confidence
        }, None
    
    except Exception as e:
        update_log(f"Error classifying formula '{formula}': {str(e)}")
        return None, f"Error classifying formula: {str(e)}"

def batch_classify_formulas(formulas, material_df):
    """
    Classify a list of chemical formulas from a file or text input.
    """
    results = []
    errors = []
    for formula in formulas:
        result, error = classify_formula(formula.strip(), material_df)
        if error:
            errors.append(error)
        else:
            results.append(result)
    return results, errors

def plot_material_classifications(df, top_n=20, year_range=None):
    """
    Create visualizations for p-type and n-type material classifications.
    """
    if df.empty:
        return None, None, None, None
    
    # Filter by year range
    if year_range:
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    
    # Count materials by classification
    material_counts = df.groupby(["material", "classification"]).size().reset_index(name="count")
    
    # Get top materials
    top_materials = material_counts.groupby("material")["count"].sum().nlargest(top_n).index
    filtered_df = material_counts[material_counts["material"].isin(top_materials)]
    
    # Bar chart
    fig_bar = px.bar(
        filtered_df, 
        x="material", 
        y="count", 
        color="classification",
        title=f"Top {top_n} Materials by p-type/n-type Classification",
        labels={"material": "Material", "count": "Frequency", "classification": "Type"}
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    
    # Pie chart
    class_dist = df["classification"].value_counts()
    fig_pie = px.pie(
        values=class_dist.values,
        names=class_dist.index,
        title="Distribution of p-type vs n-type Classifications"
    )
    
    # Timeline
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
    
    # Heatmap of material co-occurrence
    material_papers = df.groupby(["material", "paper_id"]).size().unstack(fill_value=0)
    co_occurrence = material_papers.T.dot(material_papers)
    np.fill_diagonal(co_occurrence.values, 0)
    top_materials = material_counts.groupby("material")["count"].sum().nlargest(top_n).index
    co_occurrence = co_occurrence.loc[top_materials, top_materials]
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=co_occurrence.values,
        x=co_occurrence.columns,
        y=co_occurrence.index,
        colorscale="Viridis",
        text=co_occurrence.values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig_heatmap.update_layout(
        title="Material Co-occurrence Heatmap",
        xaxis_title="Material",
        yaxis_title="Material",
        xaxis_tickangle=-45
    )
    
    return fig_bar, fig_pie, fig_timeline, fig_heatmap

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
    # Database preview
    conn = sqlite3.connect(st.session_state.db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers WHERE content IS NOT NULL AND content NOT LIKE 'Error%'")
    paper_count = cursor.fetchone()[0]
    conn.close()
    st.info(f"Database contains {paper_count} valid papers.")
    
    tab1, tab2 = st.tabs(["Material Classification", "Formula Classification"])
    
    with tab1:
        st.header("Material Classification Analysis (p-type vs n-type)")
        
        with st.sidebar:
            st.subheader("Material Classification Parameters")
            material_top_n = st.slider("Number of Top Materials to Show", min_value=5, max_value=30, value=10, key="material_top_n")
            enable_pymatgen = st.checkbox("Use pymatgen for formula standardization", value=PYMAGEN_AVAILABLE, 
                                         disabled=not PYMAGEN_AVAILABLE, key="enable_pymatgen")
            preserve_stoichiometry = st.checkbox("Preserve Exact Stoichiometry", value=False, key="preserve_stoichiometry")
            year_range = st.slider("Year Range", min_value=1980, max_value=2025, value=(2000, 2025), key="year_range")
            material_filter = st.multiselect("Filter Materials", options=[], placeholder="Select materials after extraction", key="material_filter")
        
        if st.button("Extract Material Classifications", key="extract_materials"):
            st.session_state.error_summary = []  # Reset error summary
            with st.spinner("Extracting p-type and n-type material classifications..."):
                material_df = extract_material_classifications(st.session_state.db_file, preserve_stoichiometry, year_range)
                st.session_state.material_classifications = material_df
                
                # Update material filter options
                if not material_df.empty:
                    st.session_state.material_filter_options = sorted(material_df["material"].unique())
                    st.sidebar.multiselect("Filter Materials", options=st.session_state.material_filter_options, 
                                         default=st.session_state.material_filter, key="material_filter")
            
            if material_df.empty:
                st.warning("No material classifications found. Check logs for details.")
                if st.session_state.error_summary:
                    st.error("Errors encountered:\n- " + "\n- ".join(set(st.session_state.error_summary)))
            else:
                st.success(f"Extracted {len(material_df)} unique material classifications!")
                
                # Filter by selected materials
                filtered_df = material_df if not st.session_state.material_filter else \
                             material_df[material_df["material"].isin(st.session_state.material_filter)]
                
                # Summary statistics
                st.subheader("Classification Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Classifications", len(filtered_df))
                with col2:
                    p_type_count = len(filtered_df[filtered_df["classification"] == "p-type"])
                    st.metric("p-type Materials", p_type_count)
                with col3:
                    n_type_count = len(filtered_df[filtered_df["classification"] == "n-type"])
                    st.metric("n-type Materials", n_type_count)
                
                # Standardization info
                if PYMAGEN_AVAILABLE and enable_pymatgen:
                    st.info(f"Material formulas standardized using pymatgen (exact stoichiometry {'preserved' if preserve_stoichiometry else 'reduced'})")
                else:
                    st.warning("pymatgen not available or disabled. Using basic formula standardization.")
                
                # Visualizations
                st.subheader("Visualizations")
                fig_bar, fig_pie, fig_timeline, fig_heatmap = plot_material_classifications(filtered_df, material_top_n, year_range)
                
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if fig_pie:
                        st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    if fig_timeline:
                        st.plotly_chart(fig_timeline, use_container_width=True)
                
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Data table
                st.subheader("Extracted Material Classifications")
                st.dataframe(
                    filtered_df[["paper_id", "title", "year", "material", "classification", "context"]].head(100),
                    use_container_width=True
                )
                
                # Download button
                material_csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Download Material Classifications CSV", 
                    material_csv, 
                    "material_classifications.csv", 
                    "text/csv", 
                    key="download_materials"
                )
        
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="material_logs")
    
    with tab2:
        st.header("Formula Classification")
        st.markdown("""
        Enter a chemical formula or upload a CSV file with formulas to check their p-type or n-type classification.
        **Note**: Run Material Classification Analysis first to populate the classification data.
        """)
        
        with st.sidebar:
            st.subheader("Formula Classification Parameters")
            classification_mode = st.radio("Input Mode", ["Single Formula", "Batch CSV Upload"], key="classification_mode")
        
        if classification_mode == "Single Formula":
            formula_input = st.text_input("Enter Chemical Formula (e.g., Bi2Te3, PbTe)", key="formula_input")
            if st.button("Classify Formula", key="classify_formula"):
                if not formula_input:
                    st.error("Please enter a chemical formula.")
                else:
                    with st.spinner(f"Classifying formula '{formula_input}'..."):
                        result, error = classify_formula(formula_input, st.session_state.material_classifications)
                        if error:
                            st.error(error)
                        else:
                            st.success(f"Formula: **{result['formula']}**")
                            st.write(f"Classification: **{result['classification']}** (Confidence: {result['confidence']:.2%})")
                            st.write(f"Found in {result['count']} paper(s): {', '.join(result['paper_ids'])}")
                            st.write("Context Snippets:")
                            for i, context in enumerate(result['contexts'][:5], 1):
                                st.write(f"{i}. {context}")
                            if len(result['all_classifications']) > 1:
                                st.write("All Classifications:", {k: f"{v:.2%}" for k, v in result['all_classifications'].items()})
        
        else:
            uploaded_csv = st.file_uploader("Upload CSV with Formulas (column: 'formula')", type=["csv"], key="formula_csv")
            if uploaded_csv and st.button("Classify Batch Formulas", key="classify_batch"):
                with st.spinner("Classifying batch formulas..."):
                    formulas_df = pd.read_csv(uploaded_csv)
                    if 'formula' not in formulas_df.columns:
                        st.error("CSV must contain a 'formula' column.")
                    else:
                        formulas = formulas_df['formula'].dropna().tolist()
                        results, errors = batch_classify_formulas(formulas, st.session_state.material_classifications)
                        
                        if errors:
                            st.error("Errors encountered:\n- " + "\n- ".join(set(errors)))
                        
                        if results:
                            batch_df = pd.DataFrame([{
                                "Formula": r["formula"],
                                "Classification": r["classification"],
                                "Confidence": f"{r['confidence']:.2%}",
                                "Paper Count": r["count"],
                                "Paper IDs": ", ".join(r["paper_ids"])
                            } for r in results])
                            st.subheader("Batch Classification Results")
                            st.dataframe(batch_df, use_container_width=True)
                            
                            batch_csv = batch_df.to_csv(index=False)
                            st.download_button(
                                "Download Batch Classification Results", 
                                batch_csv, 
                                "batch_formula_classifications.csv", 
                                "text/csv", 
                                key="download_batch"
                            )
        
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="formula_logs")
else:
    st.warning("Select or upload a database file.")
