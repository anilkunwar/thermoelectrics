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

# Try to import pymatgen for material formula parsing
try:
    from pymatgen.core.composition import Composition
    from pymatgen.core.periodic_table import Element
    PYMAGEN_AVAILABLE = True
except ImportError:
    PYMAGEN_AVAILABLE = False
    st.warning("pymatgen is not installed. Material formula standardization will be limited. Install with: `pip install pymatgen`")

# -----------------------------
# Regex NER for formulas
# -----------------------------
@Language.component("formula_ner")
def formula_ner(doc):
    formula_pattern = r'\b(?:[A-Z][a-z]?(?:\d*\.?\d*)?)+(?:-[A-Z][a-z]?(?:\d*\.?\d*)?)*\b'
    spans = []
    for match in re.finditer(formula_pattern, doc.text):
        span = doc.char_span(match.start(), match.end(), label="FORMULA")
        if span:
            spans.append(span)
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

# -----------------------------
# Material matcher with synonyms
# -----------------------------
def build_material_matcher(nlp, synonyms):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for canonical, variants in synonyms.items():
        patterns = [nlp.make_doc(v) for v in variants]
        matcher.add(canonical, patterns)
    return matcher

@Language.component("material_matcher")
def material_matcher(doc):
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

# -----------------------------
# Load spaCy model with improved NER
# -----------------------------
def load_spacy_model(synonyms):
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception as e:
        st.error(f"Failed to load spaCy: {e}. Install: `python -m spacy download en_core_web_sm`")
        st.stop()
    
    # Add custom components
    nlp.add_pipe("formula_ner", last=True)
    
    # Material matcher
    matcher = build_material_matcher(nlp, synonyms)
    nlp.add_pipe("material_matcher", last=True)
    
    # Attach matcher to doc
    if not Doc.has_extension("material_matcher"):
        Doc.set_extension("material_matcher", default=None)
    Doc.set_extension("material_matcher", default=matcher, force=True)
    
    if not Span.has_extension("norm"):
        Span.set_extension("norm", default=None)
    
    return nlp

# -----------------------------
# Link formulas to nearest material type
# -----------------------------
def link_formula_to_material(doc):
    formulas = [ent for ent in doc.ents if ent.label_ == "FORMULA"]
    materials = [ent for ent in doc.ents if ent.label_ == "MATERIAL_TYPE"]

    pairs = []
    for f in formulas:
        nearest_material = None
        min_distance = float("inf")
        for m in materials:
            distance = abs(f.start_char - m.start_char)
            if distance < min_distance:
                min_distance = distance
                nearest_material = m
        pairs.append({
            "Formula": f.text,
            "Material_Type": nearest_material._.norm if nearest_material else "-"
        })
    return pairs

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
This tool extracts p-type and n-type material classifications from SQLite databases (e.g., thermoelectric_universe.db) and allows classification of user-input chemical formulas.

**Date and Time**: 09:06 PM CEST, Thursday, August 21, 2025

**Dependencies**:
- `pip install streamlit pandas sqlite3 spacy plotly psutil pymatgen`
- `python -m spacy download en_core_web_sm`
""")

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

# Initialize session state
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "material_classifications" not in st.session_state:
    st.session_state.material_classifications = None
if "db_file" not in st.session_state:
    st.session_state.db_file = None
if "error_summary" not in st.session_state:
    st.session_state.error_summary = []
if "progress_log" not in st.session_state:
    st.session_state.progress_log = []
if "text_column" not in st.session_state:
    st.session_state.text_column = "content"
if "synonyms" not in st.session_state:
    st.session_state.synonyms = {
        "p-type": ["p-type", "positive type", "positive thermoelectric", "hole conducting"],
        "n-type": ["n-type", "negative type", "negative thermoelectric", "electron conducting"]
    }

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

def standardize_material_formula(formula, preserve_stoichiometry=False, canonical_order=True):
    if not formula or not isinstance(formula, str):
        update_log(f"Invalid input formula: {formula}")
        st.session_state.error_summary.append(f"Invalid formula: {formula}")
        return None
    
    formula = re.sub(r'\s+', '', formula)
    formula = re.sub(r'[\[\]\{\}]', '', formula)
    
    element_pattern = r'[A-Z][a-z]?\d*'
    if not re.search(element_pattern, formula):
        update_log(f"Skipped non-chemical term '{formula}': no valid elements")
        st.session_state.error_summary.append(f"Skipped '{formula}': no valid elements")
        return None
    
    invalid_terms = [
        'p-type', 'n-type', 'doping', 'doped', 'thermoelectric', 'material', 'the', 'and',
        'is', 'exhibits', 'type', 'based', 'sample', 'compound', 'system', 'properties'
    ]
    if any(term.lower() in formula.lower() for term in invalid_terms):
        update_log(f"Skipped non-chemical term '{formula}': contains invalid term")
        st.session_state.error_summary.append(f"Skipped '{formula}': contains invalid term")
        return None
    
    doping_pattern = r'(.+?)(?::|doped\s+)([A-Za-z0-9,\.]+)'
    doping_match = re.match(doping_pattern, formula, re.IGNORECASE)
    dopants = None
    if doping_match:
        base_formula, dopants = doping_match.groups()
        formula = base_formula.strip()
        dopants = dopants.split(',')
        update_log(f"Detected doped material: base='{formula}', dopants='{','.join(dopants)}'")
    
    if PYMAGEN_AVAILABLE and st.session_state.get('enable_pymatgen', False):
        try:
            comp = Composition(formula)
            if not comp.valid:
                update_log(f"Invalid chemical formula '{formula}': not a valid composition")
                st.session_state.error_summary.append(f"Invalid formula '{formula}': not a valid composition")
                return None
            
            elements = comp.elements
            if not all(isinstance(el, Element) for el in elements):
                update_log(f"Invalid elements in formula '{formula}'")
                st.session_state.error_summary.append(f"Invalid elements in formula '{formula}'")
                return None
            
            if preserve_stoichiometry:
                el_amt_dict = comp.get_el_amt_dict()
                standardized_formula = ''.join(
                    f"{el}{amt:.2f}" if amt != int(amt) else f"{el}{int(amt)}"
                    for el, amt in (sorted(el_amt_dict.items()) if canonical_order else el_amt_dict.items())
                )
            else:
                standardized_formula = comp.reduced_formula
            
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
    
    formula = re.sub(r'([A-Z][a-z]?)(\d*\.?\d*)', lambda m: m.group(1) + (m.group(2) if m.group(2) else ""), formula)
    
    common_materials = {
        'Bi2Te3': 'Bi₂Te₃', 'PbTe': 'PbTe', 'SnSe': 'SnSe', 'CoSb3': 'CoSb₃', 'SiGe': 'SiGe',
        'Zn4Sb3': 'Zn₄Sb₃', 'Mg2Si': 'Mg₂Si', 'Cu2Se': 'Cu₂Se'
    }
    for key, value in common_materials.items():
        if re.search(rf'\b{re.escape(key)}\b|-based\b', formula, re.IGNORECASE):
            formula = value
            update_log(f"Matched common material '{key}' to '{value}'")
            break
    
    if dopants:
        formula = f"{formula}:{','.join(dopants)}"
    
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    formula = formula.translate(subscript_map)
    
    update_log(f"Normalized formula '{formula}' using basic standardization")
    return formula

def detect_text_column(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(papers)")
    columns = {col[1] for col in cursor.fetchall()}
    possible_text_columns = ['content', 'text', 'abstract', 'body']
    for col in possible_text_columns:
        if col in columns:
            update_log(f"Detected text column: {col}")
            return col
    update_log("No text column found in 'papers' table")
    return None

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
            return pd.DataFrame()
        
        cursor.execute("PRAGMA table_info(papers)")
        columns = {col[1] for col in cursor.fetchall()}
        required_columns = {'id', 'title', 'year'}
        if not required_columns.issubset(columns):
            missing = required_columns - columns
            update_log(f"Missing required columns: {missing}")
            st.session_state.error_summary.append(f"Missing required columns: {missing}")
            conn.close()
            return pd.DataFrame()
        
        text_column = detect_text_column(conn)
        if not text_column:
            st.session_state.error_summary.append("No text column (content, text, abstract, body) found in database")
            conn.close()
            return pd.DataFrame()
        st.session_state.text_column = text_column
        
        query = f"SELECT id, title, year, {text_column} FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%'"
        if year_range:
            query += f" AND year BETWEEN {year_range[0]} AND {year_range[1]}"
        df = pd.read_sql_query(query, conn)
        
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
        
        # Load spaCy model with improved NER
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
            update_progress(f"Processing paper {row['id']} ({i+1}/{len(df)})")
            content = row[text_column]
            chunks = chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                doc = nlp(chunk)
                
                # Use the improved NER components
                formula_entities = [ent.text for ent in doc.ents if ent.label_ == "FORMULA"]
                material_entities = [ent for ent in doc.ents if ent.label_ == "MATERIAL_TYPE"]
                
                # Link formulas to materials
                linked_pairs = link_formula_to_material(doc)
                
                for pair in linked_pairs:
                    if pair["Material_Type"] in ["p-type", "n-type"]:
                        material_classifications.append({
                            "paper_id": row["id"],
                            "title": row["title"],
                            "year": row["year"],
                            "material": pair["Formula"],
                            "classification": pair["Material_Type"],
                            "context": f"Found in context: {chunk[max(0, chunk.find(pair['Formula'])-50):min(len(chunk), chunk.find(pair['Formula'])+50)]}..."
                        })
                
                # Also use the original pattern matching as fallback
                p_type_materials = set()
                for pattern in p_type_patterns:
                    matches = re.finditer(pattern, chunk, re.IGNORECASE)
                    for match in matches:
                        material = match.group(1).strip()
                        if material and len(material) > 2 and material in formula_entities:
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                p_type_materials.add((standardized_material, match.start()))
                
                n_type_materials = set()
                for pattern in n_type_patterns:
                    matches = re.finditer(pattern, chunk, re.IGNORECASE)
                    for match in matches:
                        material = match.group(1).strip()
                        if material and len(material) > 2 and material in formula_entities:
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                n_type_materials.add((standardized_material, match.start()))
                
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
                
                doc = None
                import gc
                gc.collect()
            
            progress_value = min((i + 1) / len(df), 1.0)
            progress_bar.progress(progress_value)
        
        material_df = pd.DataFrame(material_classifications)
        
        if not material_df.empty:
            material_df = material_df.drop_duplicates(subset=["paper_id", "material", "classification"])
            material_df = material_df.sort_values(by=["material", "classification"])
            update_log(f"Cleaned and sorted DataFrame: {len(material_df)} unique classifications")
            
            conn = sqlite3.connect(db_file)
            material_df[["material", "classification"]].to_sql("standardized_formulas", conn, if_exists="replace", index=False)
            conn.close()
            update_log("Cached standardized formulas in database")
        
        update_log(f"Extracted {len(material_df)} material classifications")
        return material_df
    
    except sqlite3.OperationalError as e:
        update_log(f"SQLite error: {str(e)}")
        st.session_state.error_summary.append(f"SQLite error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        update_log(f"Error in material classification: {str(e)}")
        st.session_state.error_summary.append(f"Extraction error: {str(e)}")
        return pd.DataFrame()

def classify_formula(formula, material_df, fuzzy_match=False):
    try:
        if not formula.strip():
            update_log("Empty formula input provided")
            return None, "Please enter a valid chemical formula.", None
        
        normalized_formula = standardize_material_formula(formula, 
                                                        preserve_stoichiometry=st.session_state.get('preserve_stoichiometry', False))
        if not normalized_formula:
            update_log(f"Invalid chemical formula: {formula}")
            return None, f"'{formula}' is not a valid chemical formula.", None
        
        update_log(f"Normalized formula '{formula}' to '{normalized_formula}'")
        
        if material_df is None or material_df.empty:
            update_log("No material classifications available for formula lookup")
            return None, "Please run Material Classification Analysis first.", None
        
        formula_matches = material_df[material_df["material"].str.lower() == normalized_formula.lower()]
        similar_formula = None
        
        if formula_matches.empty and fuzzy_match:
            materials = material_df["material"].unique()
            similarities = [(m, SequenceMatcher(None, normalized_formula.lower(), m.lower()).ratio()) for m in materials]
            best_match, similarity = max(similarities, key=lambda x: x[1]) if similarities else (None, 0)
            if similarity > 0.8:
                formula_matches = material_df[material_df["material"].str.lower() == best_match.lower()]
                similar_formula = best_match
                update_log(f"Fuzzy matched '{normalized_formula}' to '{best_match}' (similarity: {similarity:.2%})")
        
        if formula_matches.empty:
            update_log(f"No classification found for formula '{normalized_formula}'")
            return None, f"No p-type or n-type classification found for '{normalized_formula}'{' (similar: ' + similar_formula + ')' if similar_formula else ''}.", similar_formula
        
        classifications = formula_matches["classification"].value_counts()
        total_matches = len(formula_matches)
        paper_ids = formula_matches["paper_id"].unique()
        contexts = formula_matches["context"].tolist()
        
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
        }, None, similar_formula
    
    except Exception as e:
        update_log(f"Error classifying formula '{formula}': {str(e)}")
        return None, f"Error classifying formula: {str(e)}", None

def batch_classify_formulas(formulas, material_df, fuzzy_match=False):
    results = []
    errors = []
    suggestions = []
    for formula in formulas:
        result, error, similar_formula = classify_formula(formula.strip(), material_df, fuzzy_match)
        if error:
            errors.append(error)
            if similar_formula:
                suggestions.append((formula, similar_formula))
        else:
            results.append(result)
    return results, errors, suggestions

def plot_material_classifications(df, top_n=20, year_range=None):
    if df.empty:
        update_log("Empty DataFrame provided to plot_material_classifications")
        return None, None, None, None
    
    if year_range:
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
        if df.empty:
            update_log("No data after year range filtering")
            return None, None, None, None
    
    material_counts = df.groupby(["material", "classification"]).size().reset_index(name="count")
    
    # Get top N materials based on total count
    top_materials = material_counts.groupby("material")["count"].sum().nlargest(top_n).index
    filtered_df = material_counts[material_counts["material"].isin(top_materials)]
    
    # Bar chart
    fig_bar = px.bar(
        filtered_df, 
        x="material", 
        y="count", 
        color="classification",
        title=f"Top {top_n} Materials by p-type/n-type Classification",
        labels={"material": "Material", "count": "Frequency", "classification": "Type"},
        color_discrete_map={"p-type": "#636EFA", "n-type": "#EF553B"}
    )
    fig_bar.update_layout(xaxis_tickangle=-45, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    
    # Pie chart
    class_dist = df["classification"].value_counts()
    fig_pie = px.pie(
        values=class_dist.values,
        names=class_dist.index,
        title="Distribution of p-type vs n-type Classifications",
        color_discrete_map={"p-type": "#636EFA", "n-type": "#EF553B"}
    )
    fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    
    # Timeline chart
    if "year" in df.columns and df["year"].notna().any():
        yearly_data = df.groupby(["year", "classification"]).size().reset_index(name="count")
        fig_timeline = px.line(
            yearly_data,
            x="year",
            y="count",
            color="classification",
            title="Trend of p-type and n-type Classifications Over Time",
            labels={"year": "Year", "count": "Number of Mentions", "classification": "Type"},
            color_discrete_map={"p-type": "#636EFA", "n-type": "#EF553B"}
        )
        fig_timeline.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    else:
        fig_timeline = None
        update_log("No valid year data for timeline plot")
    
    # Heatmap: Co-occurrence matrix
    material_papers = df.groupby(["material", "paper_id"]).size().unstack(fill_value=0)
    co_occurrence = material_papers.T.dot(material_papers)
    np.fill_diagonal(co_occurrence.values, 0)
    
    # Filter top_materials to only those present in co_occurrence
    valid_materials = [m for m in top_materials if m in co_occurrence.index and m in co_occurrence.columns]
    update_log(f"Top materials: {list(top_materials)}")
    update_log(f"Valid materials for co-occurrence: {valid_materials}")
    update_log(f"Co-occurrence index: {list(co_occurrence.index)}")
    
    if not valid_materials:
        update_log("No valid materials for co-occurrence heatmap")
        fig_heatmap = None
    else:
        co_occurrence = co_occurrence.loc[valid_materials, valid_materials]
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
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
    
    return fig_bar, fig_pie, fig_timeline, fig_heatmap

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
        
        # Use detected text_column for paper count query
        text_column = detect_text_column(conn)
        if not text_column:
            st.error("No text column (content, text, abstract, body) found in database. Please check the database schema.")
            conn.close()
            st.stop()
        
        cursor.execute(f"SELECT COUNT(*) FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%'")
        paper_count = cursor.fetchone()[0]
        
        query = f"SELECT id, title, year, {text_column} FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%' LIMIT 5"
        preview_data = pd.read_sql_query(query, conn)
        conn.close()
        
        st.info(f"Database contains {paper_count} valid papers.")
        
        st.subheader("Database Preview (First 5 Papers)")
        display_columns = [col for col in ["id", "title", "year"] if col in preview_data.columns]
        update_log(f"Preview data columns: {preview_data.columns.tolist()}")
        
        # Check if text_column exists in preview_data
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
            conn.commit()
            conn.close()
            update_log("Cleared cached standardized formulas")
            st.success("Cached formulas cleared. Run extraction again to refresh.")
    
    except sqlite3.OperationalError as e:
        st.error(f"Database error: {str(e)}")
        st.session_state.error_summary.append(f"Database error: {str(e)}")
        st.stop()
    
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
            
            # Add synonym management UI
            st.subheader("Synonym Settings")
            with st.form("add_synonym_form"):
                st.write("➕ Add new synonym")
                synonym_text = st.text_input("Phrase (e.g. 'hole transport'):", key="synonym_text")
                synonym_type = st.selectbox("Maps to:", ["p-type", "n-type"], key="synonym_type")
                submitted = st.form_submit_button("Add Synonym")
                if submitted and synonym_text.strip():
                    st.session_state.synonyms[synonym_type].append(synonym_text.strip())
                    st.success(f"Added '{synonym_text}' → {synonym_type}")
            
            st.write("### Current synonyms:")
            st.json(st.session_state.synonyms)
            
            # Material filter - only show after extraction
            material_filter_options = st.session_state.get("material_filter_options", [])
            material_filter = st.multiselect("Filter Materials", options=material_filter_options, 
                                           placeholder="Select materials after extraction", key="material_filter")
        
        if st.button("Extract Material Classifications", key="extract_materials"):
            st.session_state.error_summary = []
            st.session_state.progress_log = []
            with st.spinner("Extracting p-type and n-type material classifications..."):
                material_df = extract_material_classifications(st.session_state.db_file, preserve_stoichiometry, year_range)
                st.session_state.material_classifications = material_df
                
                if not material_df.empty:
                    st.session_state.material_filter_options = sorted(material_df["material"].unique())
            
            if material_df.empty:
                st.warning("No material classifications found. Check logs for details.")
                if st.session_state.error_summary:
                    st.error("Errors encountered:\n- " + "\n- ".join(set(st.session_state.error_summary)))
            else:
                st.success(f"Extracted {len(material_df)} unique material classifications!")
                
                filtered_df = material_df if not material_filter else \
                             material_df[material_df["material"].isin(material_filter)]
                
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
                
                if PYMAGEN_AVAILABLE and enable_pymatgen:
                    st.info(f"Material formulas standardized using pymatgen (exact stoichiometry {'preserved' if preserve_stoichiometry else 'reduced'})")
                else:
                    st.warning("pymatgen not available or disabled. Using basic formula standardization.")
                
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
                
                st.subheader("Extracted Material Classifications")
                st.dataframe(
                    filtered_df[["paper_id", "title", "year", "material", "classification", "context"]].head(100),
                    use_container_width=True
                )
                
                material_csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Download Material Classifications CSV", 
                    material_csv, 
                    "material_classifications.csv", 
                    "text/csv", 
                    key="download_materials"
                )
                
                st.subheader("Extraction Progress")
                st.text_area("Progress Log", "\n".join(st.session_state.progress_log), height=100, key="progress_log")
        
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
            fuzzy_match = st.checkbox("Enable Fuzzy Matching", value=False, key="fuzzy_match")
        
        if classification_mode == "Single Formula":
            formula_input = st.text_input("Enter Chemical Formula (e.g., Bi2Te3, PbTe)", key="formula_input")
            corrected_formula = st.text_input("Corrected Formula (optional)", value=formula_input, key="corrected_formula")
            if st.button("Classify Formula", key="classify_formula"):
                if not formula_input:
                    st.error("Please enter a chemical formula.")
                else:
                    with st.spinner(f"Classifying formula '{corrected_formula}'..."):
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
                                        st.write(f"Found in {result['count']} paper(s): {', '.join(result['paper_ids'])}")
                                        st.write("Context Snippets:")
                                        for i, context in enumerate(result['contexts'][:5], 1):
                                            st.write(f"{i}. {context}")
                                        if len(result['all_classifications']) > 1:
                                            st.write("All Classifications:", {k: f"{v:.2%}" for k, v in result['all_classifications'].items()})
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
