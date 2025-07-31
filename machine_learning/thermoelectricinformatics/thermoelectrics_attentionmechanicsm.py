import arxiv
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import urllib.request
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime
import numpy as np
import logging
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

# Define database directory and files
DB_DIR = os.path.dirname(__file__)
THERMOELECTRIC_DB_FILE = os.path.join(DB_DIR, "thermoelectric_knowledge.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "thermoelectrics_universe.db")  # Backup database

# Initialize logging
logging.basicConfig(filename='thermoelectric_download.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Streamlit app
st.set_page_config(page_title="Thermoelectric Papers Download Tool", layout="wide")
st.title("Thermoelectric Papers Download Tool with SciBERT")
st.markdown("""
This tool queries arXiv for papers on **thermoelectric materials**, using SciBERT to prioritize terms like **Seebeck coefficient**, **thermoelectric**, **ZT**, **thermal conductivity**, **electrical conductivity**, and **power factor**. It downloads PDFs for highly relevant papers (relevance > 50%) and saves metadata to `thermoelectric_knowledge.db`. A backup database `thermoelectrics_universe.db` stores full PDF text for fallback searches.

**Note**: For NER analysis (e.g., extracting Seebeck coefficients, ZT), extend this tool with spaCy and regex patterns in a separate module (to be added later).
""")

# Dependency check
st.sidebar.header("Setup")
st.sidebar.markdown("""
**Dependencies**:
- arxiv, pymupdf, pandas, streamlit, numpy, transformers, torch, scipy, matplotlib, networkx
- Install: `pip install arxiv pymupdf pandas streamlit numpy transformers torch scipy matplotlib networkx`
""")

# Load SciBERT model and tokenizer
try:
    logger.info("Loading SciBERT tokenizer and model: allenai/scibert_scivocab_uncased")
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model.eval()
    logger.info("SciBERT model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load SciBERT: {e}")
    st.error(f"Failed to load SciBERT: {e}. Install: `pip install transformers torch`")
    st.stop()

# Create PDFs directory
pdf_dir = "pdfs"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    logger.info(f"Created directory: {pdf_dir}")
    st.info(f"Created directory: {pdf_dir}")

# Initialize session state for logs
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []

def update_log(message):
    """Update the log buffer for display."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    logger.info(message)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)

# SciBERT scoring with attention mechanism
@st.cache_data
def score_abstract_with_scibert(abstract):
    prioritized_words = [
        "thermoelectric", "seebeck", "zt", "thermal conductivity", "electrical conductivity",
        "power factor", "figure of merit", "temperature", "material", "compound", "efficiency"
    ]
    secondary_words = ["semiconductor", "bandgap", "carrier", "doping", "lattice"]
    thermoelectric_terms = ["thermoelectric", "seebeck coefficient", "zt", "thermal conductivity"]
    
    try:
        inputs = scibert_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        logits = outputs.logits.numpy()
        attentions = outputs.attentions[-1][0].mean(dim=0).numpy()
        probs = softmax(logits, axis=1)
        relevance_prob = probs[0][1]
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        keyword_indices = []
        prioritized_indices = []
        for i, token in enumerate(tokens):
            token_lower = token.lower().replace("##", "")
            for kw in prioritized_words:
                if re.search(rf'\b{kw}\b', token_lower, re.IGNORECASE):
                    prioritized_indices.append(i)
                    keyword_indices.append(i)
            for kw in secondary_words:
                if re.search(rf'\b{kw}\b', token_lower, re.IGNORECASE):
                    keyword_indices.append(i)

        if prioritized_indices:
            attn_scores = attentions[prioritized_indices, :].sum(axis=1)
            avg_attn_score = attn_scores.mean()
            relevance_prob = min(relevance_prob + 0.4 * len(prioritized_indices) * avg_attn_score, 1.0)
            logger.info(f"Attention boost: {len(prioritized_indices)} prioritized tokens, avg attention: {avg_attn_score:.3f}")
            update_log(f"Attention boost: {len(prioritized_indices)} prioritized tokens, avg attention: {avg_attn_score:.3f}")

        abstract_lower = abstract.lower()
        for word in ["thermoelectric", "seebeck", "zt", "thermal conductivity", "electrical conductivity"]:
            if word in abstract_lower:
                word_pos = abstract_lower.find(word)
                context_window = abstract_lower[max(0, word_pos - 50):word_pos + len(word) + 50]
                if any(term in context_window for term in thermoelectric_terms):
                    relevance_prob = min(relevance_prob + 0.25, 1.0)
                    logger.info(f"Contextual boost: {word} near {', '.join([t for t in thermoelectric_terms if t in context_window])}")
                    update_log(f"Contextual boost: {word} near {', '.join([t for t in thermoelectric_terms if t in context_window])}")

        logger.info(f"SciBERT scored abstract: {relevance_prob:.3f}")
        update_log(f"SciBERT scored abstract: {relevance_prob:.3f}")
        return relevance_prob
    except Exception as e:
        logger.error(f"SciBERT scoring failed: {str(e)}")
        update_log(f"SciBERT scoring failed: {str(e)}")
        keywords = {
            "thermoelectric": 2.5, "seebeck": 2.5, "zt": 2.5, "thermal conductivity": 2.0,
            "electrical conductivity": 2.0, "power factor": 2.0, "figure of merit": 2.0,
            "temperature": 1.5, "material": 1.5, "compound": 1.5, "efficiency": 1.2,
            "semiconductor": 1.0, "bandgap": 1.0, "carrier": 1.0, "doping": 1.0, "lattice": 1.0
        }
        abstract_lower = abstract.lower()
        word_counts = Counter(re.findall(r'\b\w+\b', abstract_lower))
        total_words = sum(word_counts.values())
        score = 0.0
        matched_keywords = []
        for kw, weight in keywords.items():
            if kw in word_counts:
                score += weight * word_counts[kw] / (total_words + 1e-6)
                matched_keywords.append(kw)
        for word in ["thermoelectric", "seebeck", "zt", "thermal conductivity", "electrical conductivity"]:
            if word in abstract_lower:
                word_pos = abstract_lower.find(word)
                context_window = abstract_lower[max(0, word_pos - 50):word_pos + len(word) + 50]
                if any(term in context_window for term in thermoelectric_terms):
                    score += 1.0
                    logger.info(f"Fallback contextual boost: {word} near {', '.join([t for t in thermoelectric_terms if t in context_window])}")
                    update_log(f"Fallback contextual boost: {word} near {', '.join([t for t in thermoelectric_terms if t in context_window])}")
        if matched_keywords:
            score = max(score, 0.1)
        max_possible_score = sum(keywords.values()) / 10
        relevance_prob = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        logger.info(f"Fallback scoring: {relevance_prob:.3f} (matched: {', '.join(matched_keywords)})")
        update_log(f"Fallback scoring: {relevance_prob:.3f} (matched: {', '.join(matched_keywords)})")
        return relevance_prob

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        logger.info(f"Text extracted from {pdf_path} ({len(text)} characters)")
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {str(e)}")
        update_log(f"PDF extraction failed for {pdf_path}: {str(e)}")
        return f"Error: {str(e)}"

# Create thermoelectrics_universe.db
def create_universe_db(papers, db_file=UNIVERSE_DB_FILE):
    """Create or update thermoelectrics_universe.db with full PDF text and metadata."""
    try:
        logger.info(f"Creating/updating {db_file}")
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                content TEXT
            )
        """)
        cursor.execute("DELETE FROM papers")  # Clear existing data
        for paper in papers:
            cursor.execute("""
                INSERT OR REPLACE INTO papers (id, title, authors, year, content)
                VALUES (?, ?, ?, ?, ?)
            """, (
                paper["id"],
                paper["title"],
                paper.get("authors", "Unknown"),
                paper["year"],
                paper.get("content", "No text extracted")
            ))
        conn.commit()
        conn.close()
        logger.info(f"Created/updated {db_file} with {len(papers)} papers")
        update_log(f"Created/updated {db_file} with {len(papers)} papers")
        return db_file
    except Exception as e:
        logger.error(f"Error creating {db_file}: {str(e)}")
        update_log(f"Error creating {db_file}: {str(e)}")
        return None

# Database diagnostics
@st.cache_data
def diagnose_database(db_file):
    try:
        logger.info(f"Running diagnostics on {db_file}")
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
        papers_exists = cursor.fetchone() is not None
        papers_count = 0
        if papers_exists:
            papers_count = pd.read_sql_query("SELECT COUNT(*) AS count FROM papers", conn)["count"].iloc[0]
        conn.close()
        logger.info(f"Diagnostics: {db_file} contains {papers_count} papers")
        return {"papers_count": papers_count}
    except Exception as e:
        logger.error(f"Database diagnostics failed: {str(e)}")
        update_log(f"Database diagnostics failed: {str(e)}")
        return {"error": str(e), "papers_count": 0}

# Database validation
@st.cache_data
def validate_db(db_file):
    try:
        logger.info(f"Validating database: {db_file}")
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
        if not cursor.fetchone():
            conn.close()
            return False, "Database missing 'papers' table."
        df_papers = pd.read_sql_query("SELECT * FROM papers LIMIT 1", conn)
        required_columns = ["id", "title", "year"]
        missing_columns = [col for col in required_columns if col not in df_papers.columns]
        conn.close()
        if missing_columns:
            return False, f"Database 'papers' table missing columns: {', '.join(missing_columns)}"
        logger.info(f"Database {db_file} is valid")
        return True, "Database format is valid."
    except Exception as e:
        logger.error(f"Error validating database: {str(e)}")
        return False, f"Error reading database: {str(e)}"

# Save to SQLite
def save_to_sqlite(papers_df, db_file=THERMOELECTRIC_DB_FILE, universe_db_file=UNIVERSE_DB_FILE):
    try:
        logger.info(f"Saving to {db_file}")
        conn = sqlite3.connect(db_file)
        papers_df.to_sql("papers", conn, if_exists="replace", index=False)
        conn.close()
        if universe_db_file:
            universe_papers = papers_df[["id", "title", "authors", "year", "content"]].copy()
            create_universe_db(universe_papers.to_dict("records"), universe_db_file)
        logger.info(f"Saved to {db_file} and {universe_db_file}")
        update_log(f"Saved to {db_file} and {universe_db_file}")
        return f"Saved to {db_file} and {universe_db_file}"
    except Exception as e:
        logger.error(f"SQLite save failed: {str(e)}")
        update_log(f"SQLite save failed: {str(e)}")
        return f"Failed to save to SQLite: {str(e)}"

# arXiv query function
@st.cache_data
def query_arxiv(query, categories, max_results, start_year, end_year):
    try:
        logger.info(f"Querying arXiv: {query}, categories: {categories}, max_results: {max_results}, {start_year}-{end_year}")
        query_terms = query.strip().split()
        formatted_terms = []
        synonyms = {
            "thermoelectric": ["thermoelectric", "thermoelectrics"],
            "seebeck": ["seebeck", "seebeck coefficient"],
            "zt": ["zt", "figure of merit"],
            "thermal conductivity": ["thermal conductivity", "thermal conduction"],
            "electrical conductivity": ["electrical conductivity", "electric conductivity"],
            "power factor": ["power factor"],
            "temperature": ["temperature"],
            "material": ["material", "compound"],
            "efficiency": ["efficiency"],
            "semiconductor": ["semiconductor"],
            "bandgap": ["bandgap", "band gap"],
            "carrier": ["carrier", "charge carrier"],
            "doping": ["doping", "doped"],
            "lattice": ["lattice"]
        }
        for term in query_terms:
            term_clean = term.strip('"')
            formatted_terms.append(term_clean.replace(" ", "+"))
            for key, syn_list in synonyms.items():
                if term_clean.lower() == key:
                    formatted_terms.extend(syn.replace(" ", "+") for syn in syn_list)
        api_query = " ".join(formatted_terms)
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=api_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        for result in client.results(search):
            if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                abstract = result.summary.lower()
                title = result.title.lower()
                query_words = set(word.lower().strip('"') for word in re.split(r'\s+', query) if word)
                for key, syn_list in synonyms.items():
                    if key in query_words:
                        query_words.update(syn_list)
                matched_terms = [word for word in query_words if word in abstract or word in title]
                if not matched_terms:
                    continue
                relevance_prob = score_abstract_with_scibert(result.summary)
                abstract_highlighted = abstract
                for term in matched_terms:
                    abstract_highlighted = re.sub(r'\b{}\b'.format(term), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                papers.append({
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "authors": ", ".join([author.name for author in result.authors]),
                    "year": result.published.year,
                    "categories": ", ".join(result.categories),
                    "abstract": abstract,
                    "abstract_highlighted": abstract_highlighted,
                    "pdf_url": result.pdf_url,
                    "download_status": "Not downloaded",
                    "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                    "relevance_prob": round(relevance_prob * 100, 2),
                    "pdf_path": None,
                    "content": None
                })
                if len(papers) >= max_results:
                    break
        papers = sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
        logger.info(f"Found {len(papers)} papers")
        update_log(f"Found {len(papers)} papers")
        return papers
    except Exception as e:
        logger.error(f"arXiv query failed: {str(e)}")
        update_log(f"arXiv query failed: {str(e)}")
        st.error(f"Error querying arXiv: {str(e)}. Try simplifying the query.")
        return []

# Download PDF and extract text
@st.cache_data
def download_pdf_and_extract(pdf_url, paper_id):
    pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
    try:
        logger.info(f"Downloading PDF for {paper_id}: {pdf_url}")
        urllib.request.urlretrieve(pdf_url, pdf_path)
        file_size = os.path.getsize(pdf_path) / 1024
        text = extract_text_from_pdf(pdf_path)
        if not text.startswith("Error"):
            logger.info(f"Downloaded and extracted {paper_id} ({file_size:.2f} KB)")
            return f"Downloaded ({file_size:.2f} KB)", pdf_path, text
        else:
            logger.error(f"Failed to extract text for {paper_id}: {text}")
            return f"Failed: {text}", None, text
    except Exception as e:
        logger.error(f"PDF download failed for {paper_id}: {str(e)}")
        update_log(f"PDF download failed for {paper_id}: {str(e)}")
        return f"Failed: {str(e)}", None, f"Error: {str(e)}"

# Streamlit UI
st.header("arXiv Query for Thermoelectric Materials")
st.markdown("Search for abstracts on thermoelectric materials, prioritizing **Seebeck coefficient**, **thermoelectric**, **ZT**, **thermal conductivity**, **electrical conductivity**, and **power factor** using SciBERT's attention mechanism.")

log_container = st.empty()
def display_logs():
    log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)

with st.sidebar:
    st.subheader("Search Parameters")
    query = st.text_input("Query", value='thermoelectric "Seebeck coefficient" ZT "thermal conductivity" "electrical conductivity" "power factor" temperature material efficiency')
    default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph"]
    categories = st.multiselect("Categories", default_categories, default=default_categories)
    max_results = st.slider("Max Papers", min_value=1, max_value=200, value=100)
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=1990, max_value=current_year, value=2022)
    with col2:
        end_year = st.number_input("End Year", min_value=start_year, max_value=current_year, value=current_year)
    output_formats = st.multiselect("Output Formats", ["CSV", "SQLite (.db)", "JSON"], default=["SQLite (.db)"])
    search_button = st.button("Search arXiv")

if search_button:
    if not query.strip():
        st.error("Enter a valid query.")
    elif not categories:
        st.error("Select at least one category.")
    elif start_year > end_year:
        st.error("Start year must be ≤ end year.")
    else:
        with st.spinner("Querying arXiv..."):
            papers = query_arxiv(query, categories, max_results, start_year, end_year)
        
        if not papers:
            logger.warning("No papers found")
            st.warning("No papers found. Broaden query or categories.")
        else:
            st.success(f"Found **{len(papers)}** papers. Filtering for relevance > 50%...")
            relevant_papers = [p for p in papers if p["relevance_prob"] > 50.0]
            if not relevant_papers:
                logger.warning("No papers with relevance > 50%")
                st.warning("No papers with relevance > 50%. Broaden query or check 'thermoelectric_download.log'.")
            else:
                st.success(f"**{len(relevant_papers)}** papers with relevance > 50%. Downloading PDFs...")
                progress_bar = st.progress(0)
                for i, paper in enumerate(relevant_papers):
                    if paper["pdf_url"]:
                        status, pdf_path, content = download_pdf_and_extract(paper["pdf_url"], paper["id"])
                        paper["download_status"] = status
                        paper["pdf_path"] = pdf_path
                        paper["content"] = content
                    progress_bar.progress((i + 1) / len(relevant_papers))
                    time.sleep(0.1)
                    update_log(f"Processed paper {i+1}/{len(relevant_papers)}: {paper['title']}")
                
                df = pd.DataFrame(relevant_papers)
                st.subheader("Papers (Relevance > 50%)")
                st.dataframe(
                    df[["id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "relevance_prob", "download_status"]],
                    use_container_width=True
                )
                
                if "CSV" in output_formats:
                    csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                    st.download_button(
                        label="Download Paper Metadata CSV",
                        data=csv,
                        file_name="thermoelectric_papers.csv",
                        mime="text/csv"
                    )
                
                if "SQLite (.db)" in output_formats:
                    sqlite_status = save_to_sqlite(df.drop(columns=["abstract_highlighted"]))
                    st.info(sqlite_status)
                
                if "JSON" in output_formats:
                    json_data = df.drop(columns=["abstract_highlighted"]).to_json(orient="records", lines=True)
                    st.download_button(
                        label="Download Paper Metadata JSON",
                        data=json_data,
                        file_name="thermoelectric_papers.json",
                        mime="application/json"
                    )
                
                display_logs()

# Database diagnostics
st.subheader("Database Diagnostics")
diagnostics = diagnose_database(THERMOELECTRIC_DB_FILE)
if "error" in diagnostics:
    st.error(f"Diagnostics failed: {diagnostics['error']}")
else:
    st.write(f"Total papers: {diagnostics['papers_count']}")
    if diagnostics['papers_count'] == 0:
        st.warning("No papers found. Run a query to populate the database.")

