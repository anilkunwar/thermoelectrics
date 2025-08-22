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
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(page_title="Thermoelectric Material Classification Tool", layout="wide")
st.title("Thermoelectric Material Classification and Analysis Tool")
st.markdown("""
This tool extracts p-type and n-type material classifications from SQLite databases and allows classification of user-input chemical formulas using an integrated classifier (informatics + ANN).

**Date and Time**: 05:40 AM CEST, Friday, August 22, 2025

**Dependencies**:
- `pip install streamlit pandas sqlite3 spacy plotly psutil pymatgen scikit-learn joblib torch h5py`
- `python -m spacy download en_core_web_sm`
""")

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
if "integrated_classifier" not in st.session_state:
    st.session_state.integrated_classifier = None
if "save_formats" not in st.session_state:
    st.session_state.save_formats = ["pkl", "db", "pt", "h5"]
if "model_files" not in st.session_state:
    st.session_state.model_files = {}

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_usage = psutil.Process().memory_info().rss / 1024**2
    log_message = f"[{timestamp}] {message} (Memory: {memory_usage:.2f} MB)"
    st.session_state.log_buffer.append(log_message)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logger.info(log_message)

def update_progress(message):
    st.session_state.progress_log.append(message)
    if len(st.session_state.progress_log) > 10:
        st.session_state.progress_log.pop(0)

# Detect text column in database
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

# Regex NER for formulas
@Language.component("formula_ner")
def formula_ner(doc):
    formula_pattern = r'\b(?:[A-Z][a-z]?[0-9]*\.?[0-9]*)+(?::[A-Z][a-z]?[0-9]*\.?[0-9]*)?\b'
    spans = []
    for match in re.finditer(formula_pattern, doc.text):
        formula = match.group(0)
        if validate_formula(formula):
            span = doc.char_span(match.start(), match.end(), label="FORMULA")
            if span:
                spans.append(span)
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

# Formula validation
def validate_formula(formula):
    if not formula or not isinstance(formula, str):
        return False
    base_formula = re.sub(r':.+', '', formula)
    if any(term.lower() in formula.lower() for term in INVALID_TERMS):
        return False
    if re.match(r'^[A-Z](?:-[A-Z]|\.\d+|)$', formula) or len(formula) <= 2:
        return False
    element_pattern = r'[A-Z][a-z]?[0-9]*\.?[0-9]*'
    elements = re.findall(element_pattern, base_formula)
    if not elements:
        return False
    for el in elements:
        el_symbol = re.match(r'[A-Z][a-z]?', el).group(0)
        if el_symbol not in VALID_ELEMENTS:
            return False
    if re.search(r'\b[X-Z][0-9]*\b', formula) and not re.match(r'^[A-Z][a-z]?[0-9]*$', formula):
        return False
    return True

# Formula context scoring
def score_formula_context(formula, text, synonyms):
    score = 0.0
    context_window = 100
    start_idx = max(0, text.lower().find(formula.lower()) - context_window)
    end_idx = min(len(text), text.lower().find(formula.lower()) + len(formula) + context_window)
    context = text[start_idx:end_idx].lower()
    positive_terms = ['thermoelectric', 'p-type', 'n-type', 'material', 'compound', 'semiconductor']
    positive_terms += [syn for syn_list in synonyms.values() for syn in syn_list]
    common_materials = ['Bi2Te3', 'PbTe', 'SnSe', 'CoSb3', 'SiGe', 'Skutterudite', 'Half-Heusler']
    for term in positive_terms + common_materials:
        if term.lower() in context:
            score += 0.2
    negative_terms = ['figure', 'table', 'references', 'acknowledgments', 'section', 'equation']
    for term in negative_terms:
        if term.lower() in context:
            score -= 0.3
    return max(0.0, min(score, 1.0))

# Material matcher with synonyms
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

# Load spaCy model
def load_spacy_model(synonyms):
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception as e:
        st.error(f"Failed to load spaCy: {e}. Install: `python -m spacy download en_core_web_sm`")
        st.stop()
    nlp.add_pipe("formula_ner", last=True)
    matcher = build_material_matcher(nlp, synonyms)
    nlp.add_pipe("material_matcher", last=True)
    if not Doc.has_extension("material_matcher"):
        Doc.set_extension("material_matcher", default=None)
    Doc.set_extension("material_matcher", default=matcher, force=True)
    if not Span.has_extension("norm"):
        Span.set_extension("norm", default=None)
    return nlp

# Link formulas to material type
def link_formula_to_material(doc):
    formulas = [(ent, score_formula_context(ent.text, doc.text, st.session_state.synonyms)) 
                for ent in doc.ents if ent.label_ == "FORMULA"]
    formulas = [f for f, score in formulas if score > 0.3]
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

# Formula standardization
def standardize_material_formula(formula, preserve_stoichiometry=False, canonical_order=True):
    if not formula or not isinstance(formula, str):
        update_log(f"Invalid input formula: {formula}")
        st.session_state.error_summary.append(f"Invalid formula: {formula}")
        return None
    formula = re.sub(r'\s+', '', formula)
    formula = re.sub(r'[\[\]\{\}]', '', formula)
    if not validate_formula(formula):
        update_log(f"Invalid formula '{formula}': failed validation")
        st.session_state.error_summary.append(f"Invalid formula '{formula}'")
        return None
    doping_pattern = r'(.+?)(?::|doped\s+)([A-Za-z0-9,\.]+)'
    doping_match = re.match(doping_pattern, formula, re.IGNORECASE)
    dopants = None
    if doping_match:
        base_formula, dopants = doping_match.groups()
        formula = base_formula.strip()
        dopants = dopants.split(',')
        update_log(f"Detected doped material: base='{formula}', dopants='{','.join(dopants)}'")
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
                if not validate_formula(dopant):
                    update_log(f"Invalid dopant '{dopant}' in '{formula}'")
                    st.session_state.error_summary.append(f"Invalid dopant '{dopant}' in '{formula}'")
                    continue
                try:
                    dopant_comp = Composition(dopant.strip())
                    valid_dopants.append(dopant_comp.reduced_formula)
                except Exception as e:
                    update_log(f"Failed to parse dopant '{dopant}' in '{formula}': {e}")
                    st.session_state.error_summary.append(f"Failed to parse dopant '{dopant}' in '{formula}'")
            if valid_dopants:
                standardized_formula = f"{standardized_formula}:{','.join(valid_dopants)}"
        update_log(f"Standardized formula '{formula}' to '{standardized_formula}'")
        return standardized_formula
    except Exception as e:
        update_log(f"pymatgen could not parse formula '{formula}': {str(e)}")
        st.session_state.error_summary.append(f"pymatgen failed for '{formula}': {str(e)}")
        return None

# Featurization for ANN
def featurize_formulas(formulas, labels=None):
    features = []
    valid_formulas = []
    valid_labels = []
    element_properties = {
        el.symbol: [
            float(el.Z or 0),
            float(el.X or 0),
            float(el.group or 0),
            float(el.row or 0),
            float(el.atomic_mass or 0)
        ] for el in Element
    }
    for i, formula in enumerate(formulas):
        if not validate_formula(formula):
            update_log(f"Skipped featurization for invalid formula '{formula}'")
            st.session_state.error_summary.append(f"Invalid formula '{formula}' for featurization")
            continue
        try:
            comp = Composition(formula)
            el_amt_dict = comp.get_el_amt_dict()
            total_atoms = sum(el_amt_dict.values())
            feature_vector = np.zeros(5)
            for el, amt in el_amt_dict.items():
                weight = amt / total_atoms
                props = element_properties.get(el, [0.0] * 5)
                feature_vector += np.array(props) * weight
            if np.any(np.isnan(feature_vector)):
                update_log(f"NaN features for formula '{formula}'")
                st.session_state.error_summary.append(f"NaN features for formula '{formula}'")
                continue
            features.append(feature_vector)
            valid_formulas.append(formula)
            if labels is not None:
                valid_labels.append(labels[i])
        except Exception as e:
            update_log(f"Failed to featurize formula '{formula}': {str(e)}")
            st.session_state.error_summary.append(f"Featurization failed for '{formula}': {str(e)}")
    if not features:
        update_log("No valid features generated for ANN training")
        return np.array([]), [], [] if labels is not None else None
    return np.array(features), valid_formulas, valid_labels if labels is not None else None

# Integrated formula classifier
def integrated_formula_classifier(formula, material_df, ann_model, scaler, fuzzy_match=False):
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
        
        # Database lookup
        if material_df is not None and not material_df.empty:
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
            if not formula_matches.empty:
                classifications = formula_matches["classification"].value_counts()
                total_matches = len(formula_matches)
                paper_ids = formula_matches["paper_id"].unique()
                contexts = formula_matches["context"].tolist()
                confidence = {cls: count / total_matches for cls, count in classifications.items()}
                primary_classification = classifications.idxmax()
                confidence_score = confidence.get(primary_classification, 0.0)
                update_log(f"Formula '{normalized_formula}' classified as {primary_classification} (confidence: {confidence_score:.2%}) via database")
                return {
                    "formula": normalized_formula,
                    "classification": primary_classification,
                    "confidence": confidence_score,
                    "paper_ids": paper_ids.tolist(),
                    "count": total_matches,
                    "contexts": contexts,
                    "all_classifications": confidence,
                    "source": "database"
                }, None, similar_formula
        
        # ANN prediction
        if ann_model is None or scaler is None:
            update_log("No ANN model or scaler available for prediction")
            return None, "ANN model not loaded. Please train or load the classifier.", None
        X, valid_formulas, _ = featurize_formulas([normalized_formula])
        if not X.size:
            update_log(f"Failed to featurize formula '{normalized_formula}' for ANN")
            return None, f"Could not featurize formula '{normalized_formula}' for prediction.", None
        X_scaled = scaler.transform(X)
        prob = ann_model.predict_proba(X_scaled)[0]
        prediction = "p-type" if prob[1] > prob[0] else "n-type"
        confidence = max(prob)
        update_log(f"ANN predicted '{normalized_formula}' as {prediction} (confidence: {confidence:.2%})")
        return {
            "formula": normalized_formula,
            "classification": prediction,
            "confidence": confidence,
            "paper_ids": [],
            "count": 0,
            "contexts": [],
            "all_classifications": {"p-type": prob[1], "n-type": prob[0]},
            "source": "ann"
        }, None, None
    except Exception as e:
        update_log(f"Error classifying formula '{formula}': {str(e)}")
        return None, f"Error classifying formula: {str(e)}", None

# Train integrated classifier
def train_ann(formulas, labels):
    if not formulas or not labels:
        update_log("No valid data for classifier training")
        return None, None, None, {}
    X, valid_formulas, valid_labels = featurize_formulas(formulas, labels)
    if len(X) < 2:
        update_log("Insufficient data for ANN training")
        return None, None, None, {}
    if np.any(np.isnan(X)):
        update_log("NaN values detected in feature matrix, skipping ANN training")
        return None, None, None, {}
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        update_log(f"Scaler error: {str(e)}")
        return None, None, None, {}
    label_map = {"p-type": 1, "n-type": 0}
    y = np.array([label_map[l] for l in valid_labels])
    ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    try:
        ann_model.fit(X_scaled, y)
    except ValueError as e:
        update_log(f"ANN training failed: {str(e)}")
        return None, None, None, {}
    
    # Create material_df
    material_df = pd.DataFrame({"material": valid_formulas, "classification": valid_labels})
    
    save_formats = st.session_state.get('save_formats', ["pkl", "db", "pt", "h5"])
    model_files = {}
    
    # SQLite Database (.db)
    if "db" in save_formats:
        try:
            conn = sqlite3.connect(st.session_state.db_file)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS integrated_classifier (
                    model_type TEXT,
                    format TEXT,
                    model_data BLOB
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS standardized_formulas (
                    material TEXT,
                    classification TEXT
                )
            """)
            model_bytes = joblib.dumps(ann_model)
            cursor.execute(
                "INSERT INTO integrated_classifier (model_type, format, model_data) VALUES (?, ?, ?)",
                ("ann_model", "pkl", model_bytes)
            )
            scaler_bytes = joblib.dumps(scaler)
            cursor.execute(
                "INSERT INTO integrated_classifier (model_type, format, model_data) VALUES (?, ?, ?)",
                ("scaler", "pkl", scaler_bytes)
            )
            material_df.to_sql("standardized_formulas", conn, if_exists="replace", index=False)
            conn.commit()
            conn.close()
            model_files["integrated_classifier.db"] = st.session_state.db_file
            update_log("Saved integrated classifier to SQLite database")
        except Exception as e:
            update_log(f"Failed to save integrated classifier to SQLite database: {str(e)}")
            st.session_state.error_summary.append(f"SQLite save error: {str(e)}")
    
    # Pickle (.pkl)
    if "pkl" in save_formats:
        try:
            model_path = os.path.join(DB_DIR, "integrated_classifier_model.pkl")
            scaler_path = os.path.join(DB_DIR, "integrated_classifier_scaler.pkl")
            material_df_path = os.path.join(DB_DIR, "integrated_classifier_material_df.pkl")
            joblib.dump(ann_model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(material_df, material_df_path)
            model_files["integrated_classifier_model.pkl"] = model_path
            model_files["integrated_classifier_scaler.pkl"] = scaler_path
            model_files["integrated_classifier_material_df.pkl"] = material_df_path
            update_log(f"Saved ANN model to {model_path}, scaler to {scaler_path}, and material_df to {material_df_path}")
        except Exception as e:
            update_log(f"Failed to save .pkl files: {str(e)}")
            st.session_state.error_summary.append(f"Pickle save error: {str(e)}")
    
    # PyTorch (.pt)
    if "pt" in save_formats:
        try:
            class MLP(nn.Module):
                def __init__(self, input_size, hidden_sizes, output_size):
                    super(MLP, self).__init__()
                    layers = []
                    prev_size = input_size
                    for size in hidden_sizes:
                        layers.extend([
                            nn.Linear(prev_size, size),
                            nn.ReLU(),
                        ])
                        prev_size = size
                    layers.append(nn.Linear(prev_size, output_size))
                    self.layers = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.layers(x)
            
            pytorch_model = MLP(input_size=5, hidden_sizes=[100, 50], output_size=2)
            state_dict = pytorch_model.state_dict()
            state_dict['layers.0.weight'] = torch.tensor(ann_model.coefs_[0].T)
            state_dict['layers.0.bias'] = torch.tensor(ann_model.intercepts_[0])
            state_dict['layers.2.weight'] = torch.tensor(ann_model.coefs_[1].T)
            state_dict['layers.2.bias'] = torch.tensor(ann_model.intercepts_[1])
            state_dict['layers.4.weight'] = torch.tensor(ann_model.coefs_[2].T)
            state_dict['layers.4.bias'] = torch.tensor(ann_model.intercepts_[2])
            pytorch_model.load_state_dict(state_dict)
            model_path = os.path.join(DB_DIR, "integrated_classifier_model.pt")
            torch.save(pytorch_model.state_dict(), model_path)
            scaler_params = {
                'mean': torch.tensor(scaler.mean_),
                'scale': torch.tensor(scaler.scale_)
            }
            scaler_path = os.path.join(DB_DIR, "integrated_classifier_scaler.pt")
            torch.save(scaler_params, scaler_path)
            material_df_path = os.path.join(DB_DIR, "integrated_classifier_material_df.pkl")
            joblib.dump(material_df, material_df_path)
            model_files["integrated_classifier_model.pt"] = model_path
            model_files["integrated_classifier_scaler.pt"] = scaler_path
            model_files["integrated_classifier_material_df.pkl"] = material_df_path
            update_log(f"Saved PyTorch model to {model_path}, scaler to {scaler_path}, and material_df to {material_df_path}")
        except Exception as e:
            update_log(f"Failed to save .pt files: {str(e)}")
            st.session_state.error_summary.append(f"PyTorch save error: {str(e)}")
    
    # HDF5 (.h5)
    if "h5" in save_formats:
        try:
            h5_path = os.path.join(DB_DIR, "integrated_classifier.h5")
            with h5py.File(h5_path, 'w') as f:
                model_group = f.create_group('ann_model')
                for i, (coef, intercept) in enumerate(zip(ann_model.coefs_, ann_model.intercepts_)):
                    model_group.create_dataset(f'coef_{i}', data=coef)
                    model_group.create_dataset(f'intercept_{i}', data=intercept)
                scaler_group = f.create_group('scaler')
                scaler_group.create_dataset('mean', data=scaler.mean_)
                scaler_group.create_dataset('scale', data=scaler.scale_)
                material_df_group = f.create_group('material_df')
                material_df_group.create_dataset('material', data=np.array(material_df['material'].astype(str), dtype='S'))
                material_df_group.create_dataset('classification', data=np.array(material_df['classification'].astype(str), dtype='S'))
            model_files["integrated_classifier.h5"] = h5_path
            update_log(f"Saved integrated classifier to HDF5 file {h5_path}")
        except Exception as e:
            update_log(f"Failed to save .h5 file: {str(e)}")
            st.session_state.error_summary.append(f"HDF5 save error: {str(e)}")
    
    update_log(f"Trained integrated classifier with {len(valid_formulas)} samples")
    return ann_model, scaler, material_df, model_files

# Load integrated classifier
def load_integrated_classifier(db_file, format_type="pkl"):
    ann_model = None
    scaler = None
    material_df = None
    try:
        if format_type == "db":
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT model_type, model_data FROM integrated_classifier WHERE format = 'pkl'")
            rows = cursor.fetchall()
            for model_type, model_data in rows:
                if model_type == "ann_model":
                    ann_model = joblib.loads(model_data)
                elif model_type == "scaler":
                    scaler = joblib.loads(model_data)
            material_df = pd.read_sql_query("SELECT material, classification FROM standardized_formulas", conn)
            conn.close()
            update_log("Loaded integrated classifier from SQLite database")
        
        elif format_type == "pkl":
            model_path = os.path.join(DB_DIR, "integrated_classifier_model.pkl")
            scaler_path = os.path.join(DB_DIR, "integrated_classifier_scaler.pkl")
            material_df_path = os.path.join(DB_DIR, "integrated_classifier_material_df.pkl")
            if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(material_df_path):
                ann_model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                material_df = joblib.load(material_df_path)
                update_log(f"Loaded integrated classifier from {model_path}, {scaler_path}, and {material_df_path}")
            else:
                update_log("One or more .pkl files not found")
                return None, None, None
        
        elif format_type == "pt":
            class MLP(nn.Module):
                def __init__(self, input_size, hidden_sizes, output_size):
                    super(MLP, self).__init__()
                    layers = []
                    prev_size = input_size
                    for size in hidden_sizes:
                        layers.extend([
                            nn.Linear(prev_size, size),
                            nn.ReLU(),
                        ])
                        prev_size = size
                    layers.append(nn.Linear(prev_size, output_size))
                    self.layers = nn.Sequential(*layers)
                def forward(self, x):
                    return self.layers(x)
            model_path = os.path.join(DB_DIR, "integrated_classifier_model.pt")
            scaler_path = os.path.join(DB_DIR, "integrated_classifier_scaler.pt")
            material_df_path = os.path.join(DB_DIR, "integrated_classifier_material_df.pkl")
            if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(material_df_path):
                pytorch_model = MLP(input_size=5, hidden_sizes=[100, 50], output_size=2)
                pytorch_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                scaler_params = torch.load(scaler_path, map_location=torch.device('cpu'))
                scaler = StandardScaler()
                scaler.mean_ = scaler_params['mean'].numpy()
                scaler.scale_ = scaler_params['scale'].numpy()
                material_df = joblib.load(material_df_path)
                class PyTorchClassifier:
                    def __init__(self, model):
                        self.model = model
                    def predict_proba(self, X):
                        self.model.eval()
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(X)
                            output = self.model(X_tensor)
                            prob = torch.softmax(output, dim=1).numpy()
                        return prob
                ann_model = PyTorchClassifier(pytorch_model)
                update_log(f"Loaded PyTorch integrated classifier from {model_path}, {scaler_path}, and {material_df_path}")
            else:
                update_log("One or more .pt or .pkl files not found")
                return None, None, None
        
        elif format_type == "h5":
            h5_path = os.path.join(DB_DIR, "integrated_classifier.h5")
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
                    coefs = [f['ann_model'][f'coef_{i}'][:] for i in range(3)]
                    intercepts = [f['ann_model'][f'intercept_{i}'][:] for i in range(3)]
                    ann_model.coefs_ = coefs
                    ann_model.intercepts_ = intercepts
                    scaler = StandardScaler()
                    scaler.mean_ = f['scaler']['mean'][:]
                    scaler.scale_ = f['scaler']['scale'][:]
                    material_df = pd.DataFrame({
                        'material': f['material_df']['material'][:].astype(str),
                        'classification': f['material_df']['classification'][:].astype(str)
                    })
                update_log(f"Loaded integrated classifier from {h5_path}")
            else:
                update_log(f"HDF5 file not found at {h5_path}")
                return None, None, None
        
        return ann_model, scaler, material_df
    
    except Exception as e:
        update_log(f"Failed to load integrated classifier ({format_type}): {str(e)}")
        return None, None, None

# Modified extract_material_classifications
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
        
        text_column = detect_text_column(conn)
        if not text_column:
            st.session_state.error_summary.append("No text column found in database")
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
            # Train integrated classifier
            formulas = cached_df["material"].tolist()
            labels = cached_df["classification"].tolist()
            ann_model, scaler, material_df, model_files = train_ann(formulas, labels)
            st.session_state.integrated_classifier = {"ann_model": ann_model, "scaler": scaler, "material_df": material_df}
            st.session_state.model_files = model_files
            conn.close()
            return cached_df
        
        conn.close()
        
        if df.empty:
            update_log("No valid papers found for material classification")
            st.session_state.error_summary.append("No valid papers found in database")
            return pd.DataFrame()
        
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
                formula_entities = [ent.text for ent in doc.ents if ent.label_ == "FORMULA"]
                material_entities = [ent for ent in doc.ents if ent.label_ == "MATERIAL_TYPE"]
                
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
            
            progress_bar.progress(min((i + 1) / len(df), 1.0))
        
        material_df = pd.DataFrame(material_classifications)
        
        if not material_df.empty:
            material_df = material_df.drop_duplicates(subset=["paper_id", "material", "classification"])
            material_df = material_df.sort_values(by=["material", "classification"])
            update_log(f"Cleaned and sorted DataFrame: {len(material_df)} unique classifications")
            
            conn = sqlite3.connect(db_file)
            material_df[["material", "classification"]].to_sql("standardized_formulas", conn, if_exists="replace", index=False)
            conn.close()
            update_log("Cached standardized formulas in database")
            
            formulas = material_df["material"].tolist()
            labels = material_df["classification"].tolist()
            ann_model, scaler, material_df, model_files = train_ann(formulas, labels)
            st.session_state.integrated_classifier = {"ann_model": ann_model, "scaler": scaler, "material_df": material_df}
            st.session_state.model_files = model_files
        
        update_log(f"Extracted {len(material_df)} material classifications")
        return material_df
    
    except Exception as e:
        update_log(f"Error in material classification: {str(e)}")
        st.session_state.error_summary.append(f"Extraction error: {str(e)}")
        return pd.DataFrame()

# Modified classify_formula
def classify_formula(formula, fuzzy_match=False):
    if not st.session_state.integrated_classifier:
        update_log("Integrated classifier not loaded")
        return None, "Please run Material Classification Analysis or load the classifier.", None
    return integrated_formula_classifier(
        formula,
        st.session_state.integrated_classifier["material_df"],
        st.session_state.integrated_classifier["ann_model"],
        st.session_state.integrated_classifier["scaler"],
        fuzzy_match
    )

# Modified batch_classify_formulas
def batch_classify_formulas(formulas, fuzzy_match=False):
    results = []
    errors = []
    suggestions = []
    for formula in formulas:
        result, error, similar_formula = classify_formula(formula.strip(), fuzzy_match)
        if error:
            errors.append(error)
            if similar_formula:
                suggestions.append((formula, similar_formula))
        else:
            results.append(result)
    return results, errors, suggestions

# Plot material classifications (unchanged)
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
    top_materials = material_counts.groupby("material")["count"].sum().nlargest(top_n).index
    filtered_df = material_counts[material_counts["material"].isin(top_materials)]
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
    class_dist = df["classification"].value_counts()
    fig_pie = px.pie(
        values=class_dist.values,
        names=class_dist.index,
        title="Distribution of p-type vs n-type Classifications",
        color_discrete_map={"p-type": "#636EFA", "n-type": "#EF553B"}
    )
    fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
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
    material_papers = df.groupby(["material", "paper_id"]).size().unstack(fill_value=0)
    co_occurrence = material_papers.T.dot(material_papers)
    np.fill_diagonal(co_occurrence.values, 0)
    valid_materials = [m for m in top_materials if m in co_occurrence.index and m in co_occurrence.columns]
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
        if text_column in preview_data.columns:
            preview_data_display = preview_data[display_columns].copy()
            preview_data_display[f"{text_column}_preview"] = preview_data[text_column].str[:100] + "..."
            st.dataframe(preview_data_display, use_container_width=True)
        else:
            st.dataframe(preview_data[display_columns], use_container_width=True)
        
        if st.button("Clear Cached Formulas and Models", key="clear_cache"):
            conn = sqlite3.connect(st.session_state.db_file)
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS standardized_formulas")
            cursor.execute("DROP TABLE IF EXISTS integrated_classifier")
            conn.commit()
            conn.close()
            st.session_state.integrated_classifier = None
            st.session_state.model_files = {}
            update_log("Cleared cached standardized formulas and models")
            st.success("Cached formulas and models cleared.")
        
        # Classifier loading UI
        st.subheader("Load Integrated Classifier")
        format_type = st.selectbox("Select Model Format to Load", ["pkl", "db", "pt", "h5"], key="classifier_format")
        if st.button("Load Integrated Classifier"):
            ann_model, scaler, material_df = load_integrated_classifier(st.session_state.db_file, format_type)
            if ann_model and scaler and material_df is not None:
                st.session_state.integrated_classifier = {"ann_model": ann_model, "scaler": scaler, "material_df": material_df}
                st.session_state.material_classifications = material_df
                st.session_state.material_filter_options = sorted(material_df["material"].unique())
                st.success(f"Loaded integrated classifier ({format_type})")
            else:
                st.error("Failed to load integrated classifier. Check logs for details.")
    
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
            
            st.write("### Current synonyms:")
            st.json(st.session_state.synonyms)
            
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
                
                st.subheader("Visualizations")
                fig_bar, fig_pie, fig_timeline, fig_heatmap = plot_material_classifications(filtered_df, material_top_n, year_range)
                
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
                
                st.subheader("Extracted Material Classifications")
                st.dataframe(
                    filtered_df[["paper_id", "title", "year", "material", "classification", "context"]].head(100),
                    use_container_width=True
                )
                
                csv_df = filtered_df[["material", "classification"]].rename(columns={"material": "formula", "classification": "material_type"})
                material_csv = csv_df.to_csv(index=False)
                st.download_button(
                    "Download Formula Classifications CSV", 
                    material_csv, 
                    "formula_classifications.csv", 
                    "text/csv", 
                    key="download_materials"
                )
                
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
                
                st.subheader("Extraction Progress")
                progress_log_display = "\n".join(st.session_state.progress_log) if st.session_state.progress_log else "No progress messages yet."
                st.text(progress_log_display)
        
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="material_logs")
    
    with tab2:
        st.header("Formula Classification")
        st.markdown("""
        Enter a chemical formula or upload a CSV file with formulas to check their p-type or n-type classification.
        Classifications use the integrated classifier (database lookup or ANN prediction).
        **Note**: Run Material Classification Analysis or load the classifier first.
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
                elif not st.session_state.integrated_classifier:
                    st.error("Please run Material Classification Analysis or load the classifier.")
                else:
                    with st.spinner(f"Classifying formula '{corrected_formula}'..."):
                        result, error, similar_formula = classify_formula(corrected_formula, fuzzy_match)
                        if error:
                            st.error(error)
                            if similar_formula:
                                st.warning(f"Suggested similar formula: {similar_formula}")
                                if st.button(f"Classify Suggested Formula: {similar_formula}", key="classify_similar"):
                                    result, error, _ = classify_formula(similar_formula, fuzzy_match)
                                    if error:
                                        st.error(error)
                                    else:
                                        st.success(f"Formula: **{result['formula']}**")
                                        st.write(f"Classification: **{result['classification']}** (Confidence: {result['confidence']:.2%})")
                                        st.write(f"Source: {result['source']}")
                                        if result['count'] > 0:
                                            st.write(f"Found in {result['count']} paper(s): {', '.join(result['paper_ids'])}")
                                            st.write("Context Snippets:")
                                            for i, context in enumerate(result['contexts'][:5], 1):
                                                st.write(f"{i}. {context}")
                                        st.write("All Classifications:", {k: f"{v:.2%}" for k, v in result['all_classifications'].items()})
                        else:
                            st.success(f"Formula: **{result['formula']}**")
                            st.write(f"Classification: **{result['classification']}** (Confidence: {result['confidence']:.2%})")
                            st.write(f"Source: {result['source']}")
                            if result['count'] > 0:
                                st.write(f"Found in {result['count']} paper(s): {', '.join(result['paper_ids'])}")
                                st.write("Context Snippets:")
                                for i, context in enumerate(result['contexts'][:5], 1):
                                    st.write(f"{i}. {context}")
                            st.write("All Classifications:", {k: f"{v:.2%}" for k, v in result['all_classifications'].items()})
        
        else:
            uploaded_csv = st.file_uploader("Upload CSV with Formulas (column: 'formula')", type=["csv"], key="formula_csv")
            if uploaded_csv and st.button("Classify Batch Formulas", key="classify_batch"):
                if not st.session_state.integrated_classifier:
                    st.error("Please run Material Classification Analysis or load the classifier.")
                else:
                    with st.spinner("Classifying batch formulas..."):
                        formulas_df = pd.read_csv(uploaded_csv)
                        if 'formula' not in formulas_df.columns:
                            st.error("CSV must contain a 'formula' column.")
                        else:
                            formulas = formulas_df['formula'].dropna().tolist()
                            results, errors, suggestions = batch_classify_formulas(formulas, fuzzy_match)
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
                                    "Source": r["source"],
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
