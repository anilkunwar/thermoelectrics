import streamlit as st
import sqlite3
import pandas as pd
import spacy
from spacy.matcher import Matcher
from collections import Counter
import re
import os
import logging
from datetime import datetime
import gc
import psutil
import pickle
import h5py
import torch
from cleantext import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log2
from pymatgen.core.composition import Composition
import numpy as np

# Set up logging
DB_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=os.path.join(DB_DIR, 'thermoelectric_ner_analysis.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit app
st.set_page_config(page_title="Thermoelectric NER Analysis", layout="wide")
st.title("Thermoelectric Material Parameter Extraction")
st.markdown("""
This tool extracts chemical formulas, material types (p-type, n-type, neutral), and Seebeck coefficients (μV/K) from `thermoelectric_universe.db` using lightweight NER with spaCy and regex. Results are saved as `.pkl`, `.h5`, and `.pt` files.

**Date and Time**: 02:12 AM CEST, Thursday, August 21, 2025

**Dependencies**:
- `pip install streamlit pandas sqlite3 spacy pymatgen scikit-learn clean-text psutil h5py torch`
- `python -m spacy download en_core_web_sm`
""")

# Initialize session state for default values
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "default_sample_size" not in st.session_state:
    st.session_state.default_sample_size = 50
if "default_relevance_threshold" not in st.session_state:
    st.session_state.default_relevance_threshold = 0.5
if "default_pmi_threshold" not in st.session_state:
    st.session_state.default_pmi_threshold = 1.0
if "default_strict_validation" not in st.session_state:
    st.session_state.default_strict_validation = False

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_usage = psutil.Process().memory_info().rss / 1024**2  # MB
    log_message = f"[{timestamp}] {message} (Memory: {memory_usage:.2f} MB)"
    st.session_state.log_buffer.append(log_message)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(log_message)

# Load database
def load_universe_db(db_file=os.path.join(DB_DIR, "thermoelectric_universe.db"), sample_size=50):
    try:
        conn = sqlite3.connect(db_file, timeout=5)
        query_sql = "SELECT id, title, authors, year, content FROM papers WHERE content IS NOT NULL"
        df = pd.read_sql_query(query_sql, conn)
        conn.close()
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        update_log(f"Loaded {len(df)} papers from {db_file}")
        return df
    except Exception as e:
        update_log(f"Error loading {db_file}: {str(e)}")
        st.error(f"Error loading {db_file}: {str(e)}")
        return pd.DataFrame()

# Filter relevant papers
def filter_relevant_papers(df, query, relevance_threshold):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df["content"].apply(preprocess_text))
        query_vec = vectorizer.transform([query])
        similarities = (X * query_vec.T).toarray().flatten()
        threshold = max(np.percentile(similarities, 75), relevance_threshold)  # Ensure at least top 25%
        filtered_df = df[similarities >= threshold]
        update_log(f"Filtered to {len(filtered_df)} relevant papers with threshold {threshold:.2f}")
        return filtered_df
    except Exception as e:
        update_log(f"Error filtering papers: {str(e)}")
        return df

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    return clean(text, no_urls=True, no_emails=True, no_punct=False, replace_with_url="", replace_with_email="", lowercase=True)[:1000]

# Load spaCy
@st.cache_resource
def load_spacy():
    update_log("Loading spaCy en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    matcher = Matcher(nlp.vocab)
    material_patterns = [
        [{"LOWER": {"IN": ["p-type", "p-doped", "hole-doped"]}}, {"LOWER": {"IN": ["material", "semiconductor"]}, "OP": "?"}],
        [{"LOWER": {"IN": ["n-type", "n-doped", "electron-doped"]}}, {"LOWER": {"IN": ["material", "semiconductor"]}, "OP": "?"}]
    ]
    matcher.add("P_TYPE", [material_patterns[0]])
    matcher.add("N_TYPE", [material_patterns[1]])
    update_log("Loaded spaCy and matcher")
    return nlp, matcher

# PMI calculation
def calculate_pmi(text, window_size=5, min_count=2, pmi_threshold=1.0):
    try:
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        bigram_counts = Counter()
        total_words = len(words)
        total_bigrams = 0
        for i in range(len(words) - 1):
            for j in range(i + 1, min(i + window_size + 1, len(words))):
                bigram = (words[i], words[j])
                bigram_counts[bigram] += 1
                total_bigrams += 1
        pmi_scores = {}
        for (w1, w2), count in bigram_counts.items():
            if count >= min_count:
                p_w1 = word_counts[w1] / total_words
                p_w2 = word_counts[w2] / total_words
                p_w1_w2 = count / total_bigrams
                if p_w1_w2 > 0 and p_w1 > 0 and p_w2 > 0:
                    pmi = log2(p_w1_w2 / (p_w1 * p_w2))
                    if pmi >= pmi_threshold:
                        pmi_scores[f"{w1} {w2}"] = pmi
        relevant_phrases = [
            "thermoelectric material", "seebeck coefficient", "p-type", "n-type", "power factor",
            "zt", "figure of merit", "thermal conductivity"
        ]
        filtered_pmi = {phrase: score for phrase, score in pmi_scores.items() if phrase in relevant_phrases}
        update_log(f"PMI phrases: {filtered_pmi}")
        return filtered_pmi
    except Exception as e:
        update_log(f"PMI calculation failed: {str(e)}")
        return {}

# Extract parameters
def extract_parameters(text, paper_id, title, year, nlp, matcher, pmi_threshold):
    try:
        pmi_phrases = calculate_pmi(text, pmi_threshold=pmi_threshold)
        context_terms = list(pmi_phrases.keys()) + [
            "thermoelectric", "seebeck coefficient", "p-type", "n-type", "zt", "power factor", "thermal conductivity"
        ]
        doc = nlp(preprocess_text(text))
        entities = []

        # Constants
        MATERIAL_FORMULA_PATTERN = r'\b[A-Z][a-z]?\d*\.?\d*(?:[A-Z][a-z]?\d*\.?\d*)*\b'
        P_TYPE_KEYWORDS = ["p-type", "p-doped", "hole-doped", "positive seebeck", "hole carrier", "hole conduction"]
        N_TYPE_KEYWORDS = ["n-type", "n-doped", "electron-doped", "negative seebeck", "electron carrier", "electron conduction"]
        P_TYPE_PATTERN = r'\b(?:' + '|'.join(P_TYPE_KEYWORDS) + r')\b'
        N_TYPE_PATTERN = r'\b(?:' + '|'.join(N_TYPE_KEYWORDS) + r')\b'

        # Material detection
        for ent in doc.ents:
            if any(fuzz.ratio(ent.text.lower(), term) > 75 for term in ["thermoelectric material", "semiconductor"]):
                entities.append({
                    "paper_id": paper_id,
                    "title": title,
                    "year": year,
                    "entity_text": ent.text,
                    "entity_label": "MATERIAL",
                    "value": None,
                    "unit": None,
                    "material_type": "neutral",
                    "context": text[max(0, ent.start_char - 50):min(len(text), ent.end_char + 50)].replace("\n", " ")
                })

        # Chemical formulas
        formula_matches = re.finditer(MATERIAL_FORMULA_PATTERN, text, re.IGNORECASE)
        for match in formula_matches:
            formula = match.group(0)
            try:
                if st.session_state["strict_validation"]:
                    Composition(formula)
                context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                if not any(term in context.lower() for term in context_terms):
                    continue
                material_type = "neutral"
                p_type_count = len(re.finditer(P_TYPE_PATTERN, context, re.IGNORECASE))
                n_type_count = len(re.finditer(N_TYPE_PATTERN, context, re.IGNORECASE))
                matches = matcher(doc)
                for match_id, start, end in matches:
                    rule_id = nlp.vocab.strings[match_id]
                    if rule_id == "P_TYPE":
                        p_type_count += 1
                    elif rule_id == "N_TYPE":
                        n_type_count += 1
                p_type_score = p_type_count * 2.0
                n_type_score = n_type_count * 2.0
                if p_type_score > n_type_score + 0.05:
                    material_type = "p-type"
                elif n_type_score > p_type_score + 0.05:
                    material_type = "n-type"
                entities.append({
                    "paper_id": paper_id,
                    "title": title,
                    "year": year,
                    "entity_text": formula,
                    "entity_label": "CHEMICAL_FORMULA",
                    "value": None,
                    "unit": None,
                    "material_type": material_type,
                    "context": context.replace("\n", " ")
                })
            except:
                continue

        # Seebeck coefficients
        patterns = [
            (r"(-?\d+\.?\d*)\s*(μV/K|μV K-1|microvolt per kelvin)", "SEEBECK_COEFFICIENT", "μV/K", lambda x: x),
            (r"(-?\d+\.?\d*)\s*(mV/K|mV K-1|millivolt per kelvin)", "SEEBECK_COEFFICIENT", "μV/K", lambda x: x * 1000),
            (r"(-?\d+\.?\d*)\s*to\s*(-?\d+\.?\d*)\s*(μV/K|μV K-1|microvolt per kelvin)", "SEEBECK_COEFFICIENT", "μV/K", lambda x: x),
            (r"(-?\d+\.?\d*)\s*to\s*(-?\d+\.?\d*)\s*(mV/K|mV K-1|millivolt per kelvin)", "SEEBECK_COEFFICIENT", "μV/K", lambda x: x * 1000),
            (r"(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(μV/K|μV K-1|microvolt per kelvin)", "SEEBECK_COEFFICIENT", "μV/K", lambda x: x),
            (r"(-?\d+\.?\d*)\s*±\s*(\d+\.?\d*)\s*(mV/K|mV K-1|millivolt per kelvin)", "SEEBECK_COEFFICIENT", "μV/K", lambda x: x * 1000)
        ]
        for pattern, label, unit, convert in patterns:
            for match in re.finditer(pattern, text):
                context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                if not any(term in context.lower() for term in context_terms):
                    continue
                if "to" in pattern:
                    start_val = convert(float(match.group(1)))
                    end_val = convert(float(match.group(2)))
                    if -1000 <= start_val <= 1000 and -1000 <= end_val <= 1000:
                        for val in np.linspace(start_val, end_val, 5):
                            entities.append({
                                "paper_id": paper_id,
                                "title": title,
                                "year": year,
                                "entity_text": f"{start_val} to {end_val}",
                                "entity_label": label,
                                "value": val,
                                "unit": unit,
                                "material_type": None,
                                "context": context.replace("\n", " ")
                            })
                elif "±" in pattern:
                    value = convert(float(match.group(1)))
                    uncertainty = convert(float(match.group(2)))
                    if -1000 <= value <= 1000:
                        for val in [value, value - uncertainty, value + uncertainty]:
                            if -1000 <= val <= 1000:
                                entities.append({
                                    "paper_id": paper_id,
                                    "title": title,
                                    "year": year,
                                    "entity_text": f"{value} ± {uncertainty}",
                                    "entity_label": label,
                                    "value": val,
                                    "unit": unit,
                                    "material_type": None,
                                    "context": context.replace("\n", " ")
                                })
                else:
                    value = convert(float(match.group(1)))
                    if -1000 <= value <= 1000:
                        entities.append({
                            "paper_id": paper_id,
                            "title": title,
                            "year": year,
                            "entity_text": match.group(0),
                            "entity_label": label,
                            "value": value,
                            "unit": unit,
                            "material_type": None,
                            "context": context.replace("\n", " ")
                        })

        return entities, pmi_phrases
    except Exception as e:
        update_log(f"NER failed for paper {paper_id}: {str(e)}")
        return [{"paper_id": paper_id, "title": title, "year": year, "entity_text": f"Error: {str(e)}", "entity_label": "ERROR", "value": None, "unit": None, "material_type": None, "context": ""}], {}

# Process database
def process_database(db_file, query, sample_size, relevance_threshold, pmi_threshold):
    try:
        nlp, matcher = load_spacy()
        df = load_universe_db(db_file, sample_size)
        if df.empty:
            update_log("No valid data in database")
            return pd.DataFrame(), {}
        df = filter_relevant_papers(df, query, relevance_threshold)
        if df.empty:
            update_log("No relevant papers after filtering")
            return pd.DataFrame(), {}
        results = []
        pmi_results = {}
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            if not any(term.lower() in row["content"].lower() for term in ["thermoelectric", "seebeck", "p-type", "n-type", "zt"]):
                continue
            entities, pmi_phrases = extract_parameters(row["content"], row["id"], row["title"], row["year"], nlp, matcher, pmi_threshold)
            results.extend(entities)
            pmi_results[row["id"]] = pmi_phrases
            progress_bar.progress((i + 1) / len(df))
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df[result_df["entity_label"] != "ERROR"]
            result_df = result_df[result_df.apply(lambda x: x["value"] is None or x["value"] >= -1000, axis=1)]
        update_log(f"Extracted {len(result_df)} entities from {len(df)} papers")
        del nlp, matcher
        gc.collect()
        return result_df, pmi_results
    except Exception as e:
        update_log(f"Error processing database: {str(e)}")
        st.error(f"Error processing database: {str(e)}")
        return pd.DataFrame(), {}

# Save results
def save_ner_results(df, base_name="thermoelectric_params"):
    try:
        h5_path = os.path.join(DB_DIR, f"{base_name}.h5")
        df.to_hdf(h5_path, key="ner_results", mode="w", format="table")
        pkl_path = os.path.join(DB_DIR, f"{base_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(df, f)
        pt_path = os.path.join(DB_DIR, f"{base_name}.pt")
        torch.save(df.to_dict(orient="records"), pt_path)
        update_log(f"Saved NER results to {h5_path}, {pkl_path}, {pt_path}")
        return h5_path, pkl_path, pt_path
    except Exception as e:
        update_log(f"Failed to save NER results: {str(e)}")
        st.error(f"Failed to save NER results: {str(e)}")
        return None, None, None

# Sidebar parameters
st.sidebar.header("NER Analysis Parameters")
query = st.sidebar.text_input("Query", "thermoelectric materials", key="query")
sample_size = st.sidebar.slider("Sample Size (papers)", 10, 100, st.session_state.default_sample_size, key="sample_size")
relevance_threshold = st.sidebar.slider("Relevance Threshold", 0.0, 1.0, st.session_state.default_relevance_threshold, key="relevance_threshold")
pmi_threshold = st.sidebar.slider("PMI Threshold", 0.0, 5.0, st.session_state.default_pmi_threshold, key="pmi_threshold")
strict_validation = st.sidebar.checkbox("Strict Formula Validation (pymatgen)", st.session_state.default_strict_validation, key="strict_validation")
analyze_button = st.sidebar.button("Run NER Analysis", key="analyze_button")

# Persist sample_size with a separate key
st.session_state.default_sample_size = sample_size

# Main analysis
if analyze_button:
    if not query.strip():
        st.error("Please provide a valid query.")
    else:
        db_file = os.path.join(DB_DIR, "thermoelectric_universe.db")
        if not os.path.exists(db_file):
            st.error(f"Database {db_file} not found.")
        else:
            with st.spinner("Processing thermoelectric_universe.db..."):
                df, pmi_results = process_database(
                    db_file,
                    query,
                    sample_size,
                    st.session_state["relevance_threshold"],
                    st.session_state["pmi_threshold"]
                )
                if not df.empty:
                    st.subheader("Extracted Parameters")
                    st.dataframe(
                        df[["paper_id", "title", "year", "entity_text", "entity_label", "value", "unit", "material_type", "context"]],
                        use_container_width=True,
                        column_config={
                            "context": st.column_config.TextColumn("Context", help="Surrounding text for the parameter"),
                            "value": st.column_config.NumberColumn("Value", help="Numerical value for Seebeck coefficient (μV/K)"),
                            "material_type": st.column_config.TextColumn("Material Type", help="p-type, n-type, or neutral")
                        }
                    )
                    st.success(f"Extracted {len(df)} parameters from {len(df['paper_id'].unique())} papers")
                    
                    # Save results
                    h5_path, pkl_path, pt_path = save_ner_results(df)
                    if h5_path:
                        st.info(f"Saved results to {h5_path}, {pkl_path}, and {pt_path}")
                        for path, mime in [(h5_path, "application/x-hdf"), (pkl_path, "application/octet-stream"), (pt_path, "application/octet-stream")]:
                            with open(path, "rb") as f:
                                st.download_button(
                                    label=f"Download {path}",
                                    data=f,
                                    file_name=os.path.basename(path),
                                    mime=mime
                                )
                    
                    # Save PMI results
                    pmi_df = pd.DataFrame([
                        {"paper_id": pid, "phrase": phrase, "PMI Score": score}
                        for pid, phrases in pmi_results.items()
                        for phrase, score in phrases.items()
                    ])
                    if not pmi_df.empty:
                        st.subheader("PMI Scores for Thermoelectric Context Phrases")
                        st.dataframe(pmi_df, use_container_width=True)
                        st.download_button(
                            label="Download PMI Scores CSV",
                            data=pmi_df.to_csv(index=False),
                            file_name="thermoelectric_pmi_scores.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("No parameters extracted. Try adjusting query, sample size, or thresholds.")
                
                # Display logs
                st.subheader("Logs")
                st.text_area("Analysis Logs", "\n".join(st.session_state.log_buffer), height=200)
