import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Span, Doc
from spacy.util import filter_spans
from spacy.matcher import PhraseMatcher
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
from difflib import SequenceMatcher
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import h5py

# Try to import pymatgen for material formula parsing
try:
    from pymatgen.core.composition import Composition
    from pymatgen.core.periodic_table import Element
    PYMAGEN_AVAILABLE = True
except ImportError:
    PYMAGEN_AVAILABLE = False
    st.error("pymatgen is required for formula standardization and featurization. Install with: `pip install pymatgen`")
    st.stop()

# Define valid chemical elements
VALID_ELEMENTS = set(Element.__members__.keys())

# Invalid terms to exclude from formula detection
INVALID_TERMS = {
    'p-type', 'n-type', 'doping', 'doped', 'thermoelectric', 'material', 'the', 'and',
    'is', 'exhibits', 'type', 'based', 'sample', 'compound', 'system', 'properties',
    'references', 'acknowledgments', 'data', 'matrix', 'experimental', 'note', 'level',
    'conflict', 'result', 'captions', 'average', 'teg', 'tegs', 'marco', 'skeaf',
    'equation', 'figure', 'table', 'section', 'method', 'results', 'discussion'
}

# Directory and logging setup
DB_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(
    filename=os.path.join(DB_DIR, 'thermoelectric_ner_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Streamlit configuration
st.set_page_config(page_title="Thermoelectric Material Classification Tool", layout="wide")
st.title("Thermoelectric Material Classification and Analysis Tool")
st.markdown("""
This tool extracts p-type and n-type material classifications from SQLite databases and allows classification of user-input chemical formulas using NLP and ANN.

**Date and Time**: 06:14 PM CEST, Saturday, August 23, 2025

**Dependencies**:
- `pip install streamlit pandas sqlite3 spacy plotly psutil pymatgen scikit-learn joblib torch h5py tensorflow`
- `python -m spacy download en_core_web_sm`
""")

# Initialize session state safely
def initialize_session_state():
    """Initialize session state variables with defaults."""
    defaults = {
        "log_buffer": [],
        "material_classifications": None,
        "db_file": None,
        "error_summary": [],
        "progress_log": [],
        "text_column": "content",
        "synonyms": {
            "p-type": ["p-type", "positive type", "positive thermoelectric", "hole conducting"],
            "n-type": ["n-type", "negative type", "negative thermoelectric", "electron conducting"]
        },
        "ann_model": None,
        "scaler": None,
        "save_formats": ["pkl", "db", "pt", "h5"],
        "model_files": {},
        "material_filter_options": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    update_log("Session state initialized")

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_usage = psutil.Process().memory_info().rss / 1024**2
    log_message = f"[{timestamp}] {message} (Memory: {memory_usage:.2f} MB)"
    st.session_state.log_buffer.append(log_message)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(log_message)

def update_progress(message):
    st.session_state.progress_log.append(message)
    if len(st.session_state.progress_log) > 10:
        st.session_state.progress_log.pop(0)

# Reintroduce missing functions
def detect_text_column(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(papers)")
    columns = {col[1].lower() for col in cursor.fetchall()}  # Case-insensitive
    possible_text_columns = ['content', 'text', 'abstract', 'body']
    for col in possible_text_columns:
        if col.lower() in columns:
            update_log(f"Detected text column: {col}")
            return col
    update_log("No text column found in 'papers' table")
    return None

def detect_year_column(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(papers)")
    columns = {col[1].lower() for col in cursor.fetchall()}  # Case-insensitive
    possible_year_columns = ['year', 'publication_year', 'date']
    for col in possible_year_columns:
        if col.lower() in columns:
            update_log(f"Detected year column: {col}")
            return col
    update_log("No year column found in 'papers' table")
    return None

# Call initialization at the start of the app
initialize_session_state()

# Main app
st.header("Select or Upload Database")
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
db_options = [os.path.basename(f) for f in db_files] + ["Upload a new .db file"]
db_selection = st.selectbox("Select Database", db_options, index=db_options.index("thermoelectric_universe.db") if "thermoelectric_universe.db" in db_options else 0, key="db_select")
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
    try:
        conn = sqlite3.connect(st.session_state.db_file)
        cursor = conn.cursor()
        
        # Inspect database schema
        cursor.execute("PRAGMA table_info(papers)")
        db_columns = [col[1].lower() for col in cursor.fetchall()]
        update_log(f"Database 'papers' table columns: {db_columns}")
        
        text_column = detect_text_column(conn)
        if not text_column:
            st.error("No text column (content, text, abstract, body) found in database. Please check the database schema.")
            conn.close()
            st.stop()
        st.session_state.text_column = text_column
        
        cursor.execute(f"SELECT COUNT(*) FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%'")
        paper_count = cursor.fetchone()[0]
        
        year_column = detect_year_column(conn)
        select_columns = f"id AS paper_id, title, {text_column}"
        if year_column:
            select_columns += f", {year_column} AS year"
        
        query = f"SELECT {select_columns} FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%' LIMIT 5"
        preview_data = pd.read_sql_query(query, conn)
        conn.close()
        
        st.info(f"Database contains {paper_count} valid papers.")
        
        st.subheader("Database Preview (First 5 Papers)")
        display_columns = [col for col in ["paper_id", "title", "year"] if col in preview_data.columns]
        update_log(f"Preview data columns: {preview_data.columns.tolist()}")
        
        if text_column in preview_data.columns:
            preview_data_display = preview_data[display_columns].copy()
            preview_data_display[f"{text_column}_preview"] = preview_data[text_column].str[:100] + "..."
            st.dataframe(preview_data_display, use_container_width=True)
        else:
            st.dataframe(preview_data[display_columns], use_container_width=True)
            st.warning(f"Text column '{text_column}' not found in preview data. Available columns: {', '.join(preview_data.columns)}")
        
        if st.button("Clear Cached Formulas", key="clear_cache"):
            conn = sqlite3.connect(st.session_state.db_file)
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS standardized_formulas")
            cursor.execute("DROP TABLE IF EXISTS models")
            conn.commit()
            conn.close()
            update_log("Cleared cached standardized formulas and models")
            st.success("Cached formulas and models cleared. Run extraction again to refresh.")
    
    except sqlite3.OperationalError as e:
        st.error(f"Database error: {str(e)}")
        st.session_state.error_summary.append(f"Database error: {str(e)}")
        st.stop()
    except Exception as e:
        update_log(f"Unexpected error during database access: {str(e)}")
        st.error(f"Unexpected error: {str(e)}. Check logs for details.")
        st.session_state.error_summary.append(f"Unexpected error: {str(e)}")
        st.stop()
    
    tab1, tab2 = st.tabs(["Material Classification", "Formula Classification"])
    
    with tab1:
        st.header("Material Classification Analysis (p-type vs n-type)")
        
        with st.sidebar:
            st.subheader("Material Classification Parameters")
            material_top_n = st.slider("Number of Top Materials to Show", min_value=5, max_value=30, value=10, key="material_top_n")
            preserve_stoichiometry = st.checkbox("Preserve Exact Stoichiometry", value=False, key="preserve_stoichiometry")
            year_range = st.slider("Year Range", min_value=1980, max_value=2025, value=(2000, 2025), key="year_range")
            
            st.subheader("Model Save Formats")
            save_formats = st.multiselect(
                "Select formats to save models",
                options=["db", "pkl", "pt", "h5"],
                default=st.session_state.get('save_formats', ["pkl", "db", "pt", "h5"]),
                key="save_formats_selector"
            )
            if save_formats != st.session_state.get('save_formats', []):
                st.session_state['save_formats'] = save_formats
                update_log(f"Updated save formats to: {save_formats}")
            st.write("Models will be saved in:", ", ".join(st.session_state.save_formats) if st.session_state.save_formats else "None")
            
            st.subheader("Synonym Settings")
            with st.form("add_synonym_form"):
                st.write("➕ Add new synonym")
                synonym_text = st.text_input("Phrase (e.g. 'hole transport'):", key="synonym_text")
                synonym_type = st.selectbox("Maps to:", ["p-type", "n-type"], key="synonym_type")
                submitted = st.form_submit_button("Add Synonym")
                if submitted and synonym_text.strip():
                    st.session_state.synonyms[synonym_type].append(synonym_text.strip())
                    st.success(f"Added '{synonym_text}' → {synonym_type}")
                    update_log(f"Added synonym '{synonym_text}' for {synonym_type}")
            
            st.subheader("Remove Synonym")
            with st.form("remove_synonym_form"):
                synonym_options = sum([[f"{syn} ({typ})" for syn in synonyms] for typ, synonyms in st.session_state.synonyms.items()], [])
                synonym_to_remove = st.selectbox(
                    "Select synonym to remove:",
                    options=synonym_options if synonym_options else ["No synonyms available"],
                    key="synonym_remove_select"
                )
                remove_submitted = st.form_submit_button("Remove Synonym")
                if remove_submitted and synonym_to_remove and synonym_to_remove != "No synonyms available":
                    syn, typ = synonym_to_remove.rsplit(" (", 1)
                    typ = typ.rstrip(")")
                    if syn in st.session_state.synonyms[typ]:
                        st.session_state.synonyms[typ].remove(syn)
                        st.success(f"Removed '{syn}' from {typ}")
                        update_log(f"Removed synonym '{syn}' from {typ}")
            
            st.write("### Current synonyms:")
            st.json(st.session_state.synonyms)
            
            material_filter_options = st.session_state.get("material_filter_options", [])
            material_filter = st.multiselect("Filter Materials", options=material_filter_options, 
                                           placeholder="Select materials after extraction", key="material_filter")
        
        if st.button("Extract Material Classifications", key="extract_materials"):
            st.session_state.error_summary = []
            st.session_state.progress_log = []
            with st.spinner("Extracting p-type and n-type material classifications..."):
                try:
                    material_df = extract_material_classifications(st.session_state.db_file, preserve_stoichiometry, year_range)
                    st.session_state.material_classifications = material_df
                    
                    if not material_df.empty:
                        st.session_state.material_filter_options = sorted(material_df["material"].unique())
                except Exception as e:
                    update_log(f"Error during material extraction: {str(e)}")
                    st.error(f"Failed to extract material classifications: {str(e)}")
                    st.session_state.error_summary.append(f"Extraction error: {str(e)}")
                    material_df = pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
                    st.session_state.material_classifications = material_df
            
            if material_df.empty:
                st.warning("No material classifications found. Check logs for details.")
                if st.session_state.error_summary:
                    st.error("Errors encountered:\n- " + "\n- ".join(set(st.session_state.error_summary)))
            else:
                st.success(f"Extracted {len(material_df)} unique material classifications!")
                
                filtered_df = material_df if not material_filter else material_df[material_df["material"].isin(material_filter)]
                
                # Check if material_filter resulted in empty DataFrame
                if material_filter and not material_df["material"].isin(material_filter).any():
                    update_log("Material filter resulted in empty DataFrame")
                    st.warning("Selected materials not found in extracted data. Showing all classifications.")
                    filtered_df = material_df
                
                # Validate display_columns
                display_columns = ["paper_id", "title", "material", "classification", "context"]
                if 'year' in filtered_df.columns:
                    display_columns.insert(2, "year")
                
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                update_log(f"Attempting to display columns: {available_columns}")
                if len(available_columns) < len(display_columns):
                    missing_columns = [col for col in display_columns if col not in filtered_df.columns]
                    update_log(f"Missing columns in filtered_df: {missing_columns}")
                    st.warning(f"Some expected columns are missing: {', '.join(missing_columns)}. Displaying available columns: {', '.join(available_columns)}")
                
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
                
                st.subheader("Visualizations")
                try:
                    fig_bar, fig_pie, fig_timeline, fig_heatmap, fig_sunburst = plot_material_classifications(filtered_df, material_top_n, year_range)
                except Exception as e:
                    update_log(f"Error generating visualizations: {str(e)}")
                    st.error(f"Failed to generate visualizations: {str(e)}")
                    fig_bar = fig_pie = fig_timeline = fig_heatmap = fig_sunburst = None
                
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning("No data available for bar chart.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if fig_pie:
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.warning("No data available for pie chart.")
                with col2:
                    if fig_timeline:
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    else:
                        st.warning("No data available for timeline chart.")
                
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.warning("No data available for co-occurrence heatmap.")
                
                if fig_sunburst:
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                else:
                    st.warning("No data available for sunburst chart.")
                
                st.subheader("Extracted Material Classifications")
                if available_columns:
                    st.dataframe(
                        filtered_df[available_columns].head(100),
                        use_container_width=True
                    )
                else:
                    st.error("No valid columns available to display classifications.")
                
                csv_df = filtered_df[["material", "classification"] + (["year"] if 'year' in filtered_df.columns else [])].rename(
                    columns={"material": "Formula", "classification": "Material Type", "year": "Year"}
                )
                material_csv = csv_df.to_csv(index=False)
                st.download_button(
                    "Download Formula Classifications CSV", 
                    material_csv, 
                    "formula_classifications_via_nlp.csv", 
                    "text/csv", 
                    key="download_materials"
                )
                
                if st.session_state.model_files:
                    st.subheader("Download Saved Models")
                    for model_file, file_path in st.session_state.model_files.items():
                        try:
                            with open(file_path, 'rb') as f:
                                st.download_button(
                                    f"Download {model_file}",
                                    f,
                                    model_file,
                                    key=f"download_{model_file}"
                                )
                        except Exception as e:
                            st.error(f"Failed to provide download for {model_file}: {str(e)}")
                            update_log(f"Model download error for {model_file}: {str(e)}")
                
                st.subheader("Extraction Progress")
                progress_log_display = "\n".join(st.session_state.progress_log) if st.session_state.progress_log else "No progress messages yet."
                st.text(progress_log_display)
        
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="material_logs")
    
    with tab2:
        st.header("Formula Classification")
        st.markdown("""
        Enter a chemical formula or upload a CSV file with formulas to check their p-type or n-type classification.
        Classifications are based on extracted data or ANN predictions for unseen formulas.
        **Note**: Run Material Classification Analysis first to populate the classification data and train the ANN.
        """)
        
        with st.sidebar:
            st.subheader("Formula Classification Parameters")
            classification_mode = st.radio("Input Mode", ["Single Formula", "Batch CSV Upload"], key="classification_mode")
            fuzzy_match = st.checkbox("Enable Fuzzy Matching", value=False, key="fuzzy_match")
        
        if classification_mode == "Single Formula":
            formula_input = st.text_input("Enter Chemical Formula (e.g., Bi2Te3, PbTe)", key="formula_input")
            corrected_formula = st.text_input("Corrected Formula (optional)", value=formula_input, key="corrected_formula")
            if st.button("Classify Formula", key="classify_formula"):
                if not formula_input:
                    st.error("Please enter a chemical formula.")
                else:
                    with st.spinner(f"Classifying formula '{corrected_formula}'..."):
                        try:
                            result, error, similar_formula = classify_formula(corrected_formula, st.session_state.material_classifications, fuzzy_match)
                            if error:
                                st.error(error)
                                if similar_formula:
                                    st.warning(f"Suggested similar formula: {similar_formula}")
                                    if st.button(f"Classify Suggested Formula: {similar_formula}", key="classify_similar"):
                                        result, error, _ = classify_formula(similar_formula, st.session_state.material_classifications, fuzzy_match)
                                        if error:
                                            st.error(error)
                                        else:
                                            st.success(f"Formula: **{result['formula']}**")
                                            st.write(f"Classification: **{result['classification']}** (Confidence: {result['confidence']:.2%})")
                                            if result['count'] > 0:
                                                st.write(f"Found in {result['count']} paper(s): {', '.join(result['paper_ids'])}")
                                                st.write("Context Snippets:")
                                                for i, context in enumerate(result['contexts'][:5], 1):
                                                    st.write(f"{i}. {context}")
                                            else:
                                                st.write("Classification based on ANN prediction.")
                                            st.write("All Classifications:", {k: f"{v:.2%}" for k, v in result['all_classifications'].items()})
                            else:
                                st.success(f"Formula: **{result['formula']}**")
                                st.write(f"Classification: **{result['classification']}** (Confidence: {result['confidence']:.2%})")
                                if result['count'] > 0:
                                    st.write(f"Found in {result['count']} paper(s): {', '.join(result['paper_ids'])}")
                                    st.write("Context Snippets:")
                                    for i, context in enumerate(result['contexts'][:5], 1):
                                        st.write(f"{i}. {context}")
                                else:
                                    st.write("Classification based on ANN prediction.")
                                st.write("All Classifications:", {k: f"{v:.2%}" for k, v in result['all_classifications'].items()})
                        except Exception as e:
                            update_log(f"Error classifying formula '{corrected_formula}': {str(e)}")
                            st.error(f"Failed to classify formula: {str(e)}")
        
        else:
            uploaded_csv = st.file_uploader("Upload CSV with Formulas (column: 'formula')", type=["csv"], key="formula_csv")
            if uploaded_csv and st.button("Classify Batch Formulas", key="classify_batch"):
                with st.spinner("Classifying batch formulas..."):
                    try:
                        formulas_df = pd.read_csv(uploaded_csv)
                        if 'formula' not in formulas_df.columns:
                            st.error("CSV must contain a 'formula' column.")
                        else:
                            formulas = formulas_df['formula'].dropna().tolist()
                            results, errors, suggestions = batch_classify_formulas(formulas, st.session_state.material_classifications, fuzzy_match)
                            
                            if errors:
                                st.error("Errors encountered:\n- " + "\n- ".join(set(errors)))
                                if suggestions:
                                    st.warning("Suggested corrections for some formulas:")
                                    for formula, suggestion in suggestions:
                                        st.write(f"{formula} -> {suggestion}")
                            
                            if results:
                                batch_df = pd.DataFrame([{
                                    "Formula": r["formula"],
                                    "Material Type": r["classification"],
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
                    except Exception as e:
                        update_log(f"Error during batch classification: {str(e)}")
                        st.error(f"Failed to classify batch formulas: {str(e)}")
        
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="formula_logs")
else:
    st.warning("Select or upload a database file.")

# Keep the extract_material_classifications from the previous fix
def extract_material_classifications(db_file, preserve_stoichiometry=False, year_range=None):
    try:
        update_log("Starting p-type/n-type material classification with NER")
        update_progress("Connecting to database...")
        
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
        if not cursor.fetchone():
            update_log("Database does not contain 'papers' table")
            st.session_state.error_summary.append("Database does not contain 'papers' table")
            conn.close()
            return pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
        
        cursor.execute("PRAGMA table_info(papers)")
        columns = {col[1].lower() for col in cursor.fetchall()}
        required_columns = {'id', 'title'}
        if not required_columns.issubset(columns):
            missing = required_columns - columns
            update_log(f"Missing required columns: {missing}")
            st.session_state.error_summary.append(f"Missing required columns: {missing}")
            conn.close()
            return pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
        
        text_column = detect_text_column(conn)
        if not text_column:
            st.session_state.error_summary.append("No text column (content, text, abstract, body) found in database")
            conn.close()
            return pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
        st.session_state.text_column = text_column
        
        year_column = detect_year_column(conn)
        select_columns = f"id AS paper_id, title, {text_column}"
        if year_column:
            select_columns += f", {year_column} AS year"
        
        query = f"SELECT {select_columns} FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%'"
        if year_column and year_range:
            query += f" AND {year_column} BETWEEN {year_range[0]} AND {year_range[1]}"
        df = pd.read_sql_query(query, conn)
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='standardized_formulas'")
        if cursor.fetchone():
            cached_df = pd.read_sql_query("SELECT material, classification FROM standardized_formulas", conn)
            if year_column:
                try:
                    cached_df['year'] = pd.read_sql_query("SELECT year FROM papers", conn)['year']
                except Exception as e:
                    update_log(f"Failed to load year from cached data: {str(e)}")
            if 'paper_id' not in cached_df.columns:
                cached_df['paper_id'] = pd.read_sql_query("SELECT id FROM papers", conn)['id']
            if 'title' not in cached_df.columns:
                cached_df['title'] = pd.read_sql_query("SELECT title FROM papers", conn)['title']
            if 'context' not in cached_df.columns:
                cached_df['context'] = ''
            update_log("Loaded cached standardized formulas")
            conn.close()
            return cached_df
        
        conn.close()
        
        if df.empty:
            update_log("No valid papers found for material classification")
            st.session_state.error_summary.append("No valid papers found in database")
            return pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
        
        nlp = load_spacy_model(st.session_state.synonyms)
        
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
        
        def chunk_text(text, max_length=200000):
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + max_length, len(text))
                if end < len(text):
                    last_period = text.rfind('.', start, end)
                    end = last_period + 1 if last_period > start else end
                chunks.append(text[start:end])
                start = end
            return chunks
        
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            update_progress(f"Processing paper {row['paper_id']} ({i+1}/{len(df)})")
            content = row[text_column]
            chunks = chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                doc = nlp(chunk)
                formula_entities = [ent.text for ent in doc.ents if ent.label_ == "FORMULA"]
                material_entities = [ent for ent in doc.ents if ent.label_ == "MATERIAL_TYPE"]
                
                linked_pairs = link_formula_to_material(doc)
                
                for pair in linked_pairs:
                    if pair["Material_Type"] in ["p-type", "n-type"]:
                        classification_entry = {
                            "paper_id": row["paper_id"],
                            "title": row["title"],
                            "material": pair["Formula"],
                            "classification": pair["Material_Type"],
                            "context": f"Found in context: {chunk[max(0, chunk.find(pair['Formula'])-50):min(len(chunk), chunk.find(pair['Formula'])+50)]}..."
                        }
                        if 'year' in row:
                            classification_entry['year'] = row['year']
                        material_classifications.append(classification_entry)
                
                p_type_materials = set()
                for pattern in p_type_patterns:
                    matches = re.finditer(pattern, chunk, re.IGNORECASE)
                    for match in matches:
                        material = match.group(1).strip()
                        if material and len(material) > 2 and material in formula_entities and validate_formula(material):
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                p_type_materials.add((standardized_material, match.start()))
                
                n_type_materials = set()
                for pattern in n_type_patterns:
                    matches = re.finditer(pattern, chunk, re.IGNORECASE)
                    for match in matches:
                        material = match.group(1).strip()
                        if material and len(material) > 2 and material in formula_entities and validate_formula(material):
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                n_type_materials.add((standardized_material, match.start()))
                
                p_type_context = re.search(r"p-type[^\.]{0,500}", chunk, re.IGNORECASE)
                n_type_context = re.search(r"n-type[^\.]{0,500}", chunk, re.IGNORECASE)
                
                if p_type_context:
                    context_doc = nlp(p_type_context.group(0))
                    for ent in context_doc.ents:
                        if ent.label_ == "FORMULA" and validate_formula(ent.text):
                            standardized_material = standardize_material_formula(ent.text, preserve_stoichiometry)
                            if standardized_material:
                                p_type_materials.add((standardized_material, ent.start_char))
                
                if n_type_context:
                    context_doc = nlp(n_type_context.group(0))
                    for ent in context_doc.ents:
                        if ent.label_ == "FORMULA" and validate_formula(ent.text):
                            standardized_material = standardize_material_formula(ent.text, preserve_stoichiometry)
                            if standardized_material:
                                n_type_materials.add((standardized_material, ent.start_char))
                
                for material in common_te_materials:
                    if material.lower() in chunk.lower():
                        doc = nlp(material)
                        if any(ent.label_ == "FORMULA" for ent in doc.ents) and validate_formula(material):
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                if p_type_context and material.lower() in p_type_context.group(0).lower():
                                    p_type_materials.add((standardized_material, 0))
                                if n_type_context and material.lower() in n_type_context.group(0).lower():
                                    n_type_materials.add((standardized_material, 0))
                
                for material, start_pos in p_type_materials:
                    context = chunk[max(0, start_pos-50):min(len(chunk), start_pos+50)]
                    classification_entry = {
                        "paper_id": row["paper_id"],
                        "title": row["title"],
                        "material": material,
                        "classification": "p-type",
                        "context": f"Found in context: {context}..."
                    }
                    if 'year' in row:
                        classification_entry['year'] = row['year']
                    material_classifications.append(classification_entry)
                
                for material, start_pos in n_type_materials:
                    context = chunk[max(0, start_pos-50):min(len(chunk), start_pos+50)]
                    classification_entry = {
                        "paper_id": row["paper_id"],
                        "title": row["title"],
                        "material": material,
                        "classification": "n-type",
                        "context": f"Found in context: {context}..."
                    }
                    if 'year' in row:
                        classification_entry['year'] = row['year']
                    material_classifications.append(classification_entry)
                
                doc = None
                import gc
                gc.collect()
            
            progress_value = min((i + 1) / len(df), 1.0)
            progress_bar.progress(progress_value)
        
        material_df = pd.DataFrame(material_classifications)
        
        if material_df.empty:
            update_log("No material classifications extracted")
            st.session_state.error_summary.append("No material classifications found")
            return pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
        
        material_df = material_df.drop_duplicates(subset=["paper_id", "material", "classification"])
        material_df = material_df.sort_values(by=["material", "classification"])
        update_log(f"Cleaned and sorted DataFrame: {len(material_df)} unique classifications")
        update_log(f"material_df columns: {material_df.columns.tolist()}")
        
        conn = sqlite3.connect(db_file)
        material_df[["material", "classification"] + (["year"] if 'year' in material_df.columns else [])].to_sql("standardized_formulas", conn, if_exists="replace", index=False)
        conn.close()
        update_log("Cached standardized formulas in database")
        
        formulas = material_df["material"].tolist()
        labels = material_df["classification"].tolist()
        model, scaler, model_files = train_ann(formulas, labels)
        st.session_state.ann_model = model
        st.session_state.scaler = scaler
        st.session_state.model_files = model_files
        
        update_log(f"Extracted {len(material_df)} material classifications")
        return material_df
    
    except sqlite3.OperationalError as e:
        update_log(f"SQLite error: {str(e)}")
        st.session_state.error_summary.append(f"SQLite error: {str(e)}")
        return pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
    except Exception as e:
        update_log(f"Error in material classification: {str(e)}")
        st.session_state.error_summary.append(f"Extraction error: {str(e)}")
        return pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
