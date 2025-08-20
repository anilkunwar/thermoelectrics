import streamlit as st
import arxiv
import sqlite3
import os
from io import BytesIO
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from datetime import datetime
import pandas as pd
import yaml
import json
import urllib.request
import re
import fitz  # PyMuPDF

# Define database directory and files
DB_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_DB_FILE = os.path.join(DB_DIR, "thermoelectric_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "thermoelectric_universe.db")
DEFAULT_KEYTERMS_FILE = os.path.join(DB_DIR, "keyterms.yaml")

# Create PDFs directory
pdf_dir = os.path.join(DB_DIR, "pdfs")
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)
    st.info(f"Created directory: {pdf_dir}")

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'thermoelectric_query.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize session state
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "query" not in st.session_state:
    st.session_state.query = "thermoelectric materials"
if "max_results" not in st.session_state:
    st.session_state.max_results = 10
if "year_range" not in st.session_state:
    st.session_state.year_range = (2010, datetime.now().year)

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.log_buffer.append(f"[{timestamp}] {message}")
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# Load key terms from YAML or JSON
def load_keyterms(file_content, file_type="yaml"):
    try:
        if file_type == "yaml":
            keyterms = yaml.safe_load(file_content)
        elif file_type == "json":
            keyterms = json.loads(file_content.decode('utf-8'))
        update_log(f"Loaded key terms from {file_type} file")
        return keyterms
    except Exception as e:
        update_log(f"Failed to load key terms: {str(e)}")
        st.error(f"Failed to load key terms: {str(e)}")
        return None

# Load default key terms
def load_default_keyterms():
    try:
        with open(DEFAULT_KEYTERMS_FILE, 'r') as f:
            return load_keyterms(f, "yaml")
    except FileNotFoundError:
        update_log("Default keyterms.yaml not found. Using fallback terms.")
        return {
            "thermoelectric_keywords": [
                "seebeck coefficient", "thermopower", "seebeck", "power factor", "zt", "figure of merit",
                "thermoelectric", "thermoelectric material", "band gap", "electrical conductivity",
                "thermal conductivity", "carrier concentration", "carrier mobility", "p-type", "n-type"
            ],
            "synonym_mapping": {
                "seebeck": "seebeck coefficient", "thermopower": "seebeck coefficient",
                "figure of merit": "zt", "dimensionless figure of merit": "zt",
                "p-doped": "p-type", "n-doped": "n-type"
            },
            "categories": [
                "cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph", "cond-mat.soft"
            ]
        }

# Initialize databases
def initialize_metadata_db(db_file=METADATA_DB_FILE):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                categories TEXT,
                abstract TEXT,
                pdf_url TEXT,
                pdf_path TEXT,
                matched_terms TEXT
            )
        """)
        conn.commit()
        conn.close()
        update_log(f"Initialized metadata database schema for {db_file}")
    except Exception as e:
        update_log(f"Failed to initialize {db_file}: {str(e)}")
        st.error(f"Failed to initialize {db_file}: {str(e)}")

def initialize_universe_db(db_file=UNIVERSE_DB_FILE):
    try:
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
        conn.commit()
        conn.close()
        update_log(f"Initialized universe database schema for {db_file}")
    except Exception as e:
        update_log(f"Failed to initialize {db_file}: {str(e)}")
        st.error(f"Failed to initialize {db_file}: {str(e)}")

initialize_metadata_db()
initialize_universe_db()

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        update_log(f"Extracted text from {pdf_path}")
        return text
    except Exception as e:
        update_log(f"PDF text extraction failed for {pdf_path}: {str(e)}")
        return f"Error: {str(e)}"

# Retry logic for PDF download
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_pdf(url, path):
    urllib.request.urlretrieve(url, path)

# Download PDF and extract text
def download_pdf_and_save(paper_id, pdf_url, title):
    try:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:100]
        pdf_path = os.path.join(pdf_dir, f"{paper_id}_{safe_title}.pdf")
        download_pdf(pdf_url, pdf_path)
        file_size = os.path.getsize(pdf_path) / 1024
        text = extract_text_from_pdf(pdf_path)
        update_log(f"Downloaded PDF for {paper_id} to {pdf_path} ({file_size:.2f} KB)")
        return pdf_path, f"Downloaded ({file_size:.2f} KB)", text
    except Exception as e:
        update_log(f"PDF download failed for {paper_id}: {str(e)}")
        return None, f"Failed: {str(e)}", f"Error: {str(e)}"

# Fetch arXiv papers
@st.cache_data
def query_arxiv(query, keyterms, categories, max_results, start_year, end_year):
    try:
        query_terms = query.strip().split() + keyterms["thermoelectric_keywords"] + list(keyterms["synonym_mapping"].keys())
        formatted_terms = []
        for term in query_terms:
            term_clean = term.strip('"').replace(" ", "+")
            formatted_terms.append(term_clean)
            for key, value in keyterms["synonym_mapping"].items():
                if term_clean.lower() == key.replace(" ", "+").lower():
                    formatted_terms.append(value.replace(" ", "+"))
        api_query = " ".join(set(formatted_terms))
        update_log(f"Querying arXiv with: {api_query}")
        
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
                for key, value in keyterms["synonym_mapping"].items():
                    if key in query_words:
                        query_words.add(value)
                matched_terms = [word for word in query_words if word in abstract or word in title]
                for keyword in keyterms["thermoelectric_keywords"]:
                    if keyword in abstract or keyword in title:
                        matched_terms.append(keyword)
                matched_terms = list(set(matched_terms))
                if not matched_terms:
                    continue
                pdf_path, download_status, content = download_pdf_and_save(
                    result.entry_id.split('/')[-1],
                    result.pdf_url,
                    result.title
                )
                papers.append({
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "authors": ", ".join([author.name for author in result.authors]),
                    "year": result.published.year,
                    "categories": ", ".join(result.categories),
                    "abstract": result.summary,
                    "pdf_url": result.pdf_url,
                    "pdf_path": pdf_path if pdf_path else "Failed to download",
                    "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                    "content": content
                })
                if len(papers) >= max_results:
                    break
        update_log(f"Retrieved {len(papers)} papers")
        return papers
    except Exception as e:
        update_log(f"arXiv query failed: {str(e)}")
        st.error(f"Error querying arXiv: {str(e)}. Try simplifying the query.")
        return []

# Save to SQLite databases
def save_to_sqlite(papers_df, metadata_db_file=METADATA_DB_FILE, universe_db_file=UNIVERSE_DB_FILE):
    try:
        # Save metadata to thermoelectric_metadata.db
        initialize_metadata_db(metadata_db_file)
        metadata_df = papers_df[["id", "title", "authors", "year", "categories", "abstract", "pdf_url", "pdf_path", "matched_terms"]]
        conn = sqlite3.connect(metadata_db_file)
        metadata_df.to_sql("papers", conn, if_exists="replace", index=False)
        conn.close()
        update_log(f"Saved {len(metadata_df)} papers to {metadata_db_file}")

        # Save full text to thermoelectric_universe.db
        initialize_universe_db(universe_db_file)
        universe_df = papers_df[["id", "title", "authors", "year", "content"]]
        conn = sqlite3.connect(universe_db_file)
        universe_df.to_sql("papers", conn, if_exists="replace", index=False)
        conn.close()
        update_log(f"Saved {len(universe_df)} papers with full text to {universe_db_file}")

        return f"Saved to {metadata_db_file} and {universe_db_file}"
    except Exception as e:
        update_log(f"SQLite save failed: {str(e)}")
        st.error(f"SQLite save failed: {str(e)}")
        return f"Failed to save to SQLite: {str(e)}"

# Streamlit UI
st.title("arXiv Paper Downloader for Thermoelectric Topics")
st.markdown("""
This tool queries arXiv for papers related to thermoelectric topics (e.g., Seebeck coefficient, zt, thermoelectric materials), downloads their PDFs, and saves metadata to `thermoelectric_metadata.db` and full text to `thermoelectric_universe.db`. Key terms are loaded from `keyterms.yaml` or an uploaded YAML/JSON file.

**Date and Time**: 05:49 AM CEST, Wednesday, August 20, 2025
""")

st.sidebar.header("Search Parameters")
query = st.sidebar.text_input("Query", value=st.session_state.query)
categories = st.sidebar.multiselect("Categories", load_default_keyterms()["categories"], default=load_default_keyterms()["categories"])
max_results = st.sidebar.slider("Max Papers", min_value=1, max_value=200, value=st.session_state.max_results)
current_year = datetime.now().year
col1, col2 = st.sidebar.columns(2)
with col1:
    start_year = st.number_input("Start Year", min_value=1990, max_value=current_year, value=st.session_state.year_range[0])
with col2:
    end_year = st.number_input("End Year", min_value=start_year, max_value=current_year, value=st.session_state.year_range[1])
uploaded_file = st.sidebar.file_uploader("Upload Custom Key Terms (YAML/JSON)", type=["yaml", "yml", "json"])
output_formats = st.sidebar.multiselect("Output Formats", ["CSV", "SQLite (.db)", "JSON"], default=["SQLite (.db)"])
search_button = st.button("Search and Download Papers")

# Load key terms
keyterms = load_default_keyterms()
if uploaded_file is not None:
    file_type = "yaml" if uploaded_file.name.endswith(('.yaml', '.yml')) else "json"
    keyterms = load_keyterms(uploaded_file.read(), file_type) or keyterms

# Process inputs
try:
    st.session_state.query = query.strip() or st.session_state.query
    st.session_state.max_results = max_results
    st.session_state.year_range = (start_year, end_year)
except Exception as e:
    st.error(f"Invalid input: {str(e)}")
    query = st.session_state.query
    max_results = st.session_state.max_results
    start_year, end_year = st.session_state.year_range

# Process search
if search_button:
    if not query.strip():
        st.error("Enter a valid query.")
    elif not categories:
        st.error("Select at least one category.")
    elif start_year > end_year:
        st.error("Start year must be ≤ end year.")
    else:
        with st.spinner("Querying arXiv, downloading PDFs, and extracting text..."):
            papers = query_arxiv(query, keyterms, categories, max_results, start_year, end_year)
        
        if not papers:
            st.warning("No papers found. Broaden query or categories.")
            log_container = st.empty()
            log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)
        else:
            st.success(f"Found and downloaded **{len(papers)}** papers!")
            papers_df = pd.DataFrame(papers)
            
            # Display papers
            st.subheader("Retrieved Papers")
            st.dataframe(
                papers_df[["id", "title", "authors", "year", "categories", "abstract", "matched_terms", "pdf_path"]],
                use_container_width=True
            )
            
            # Save outputs
            if "CSV" in output_formats:
                csv = papers_df[["id", "title", "authors", "year", "categories", "abstract", "pdf_url", "pdf_path", "matched_terms"]].to_csv(index=False)
                st.download_button(
                    label="Download Paper Metadata CSV",
                    data=csv,
                    file_name="thermoelectric_papers.csv",
                    mime="text/csv"
                )
            
            if "SQLite (.db)" in output_formats:
                sqlite_status = save_to_sqlite(papers_df)
                st.info(sqlite_status)
                # Download thermoelectric_metadata.db
                try:
                    conn_source = sqlite3.connect(METADATA_DB_FILE)
                    conn_target = sqlite3.connect(":memory:")
                    conn_source.backup(conn_target)
                    sqlite_data_metadata = BytesIO()
                    for line in conn_target.iterdump():
                        sqlite_data_metadata.write(line.encode('utf-8'))
                    conn_source.close()
                    conn_target.close()
                    sqlite_data_metadata.seek(0)
                    st.download_button(
                        label="Download thermoelectric_metadata.db",
                        data=sqlite_data_metadata.getvalue(),
                        file_name="thermoelectric_metadata.db",
                        mime="application/x-sqlite3"
                    )
                except Exception as e:
                    st.error(f"Failed to prepare thermoelectric_metadata.db for download: {str(e)}")
                    update_log(f"Failed to prepare thermoelectric_metadata.db for download: {str(e)}")
                
                # Download thermoelectric_universe.db
                try:
                    conn_source = sqlite3.connect(UNIVERSE_DB_FILE)
                    conn_target = sqlite3.connect(":memory:")
                    conn_source.backup(conn_target)
                    sqlite_data_universe = BytesIO()
                    for line in conn_target.iterdump():
                        sqlite_data_universe.write(line.encode('utf-8'))
                    conn_source.close()
                    conn_target.close()
                    sqlite_data_universe.seek(0)
                    st.download_button(
                        label="Download thermoelectric_universe.db",
                        data=sqlite_data_universe.getvalue(),
                        file_name="thermoelectric_universe.db",
                        mime="application/x-sqlite3"
                    )
                except Exception as e:
                    st.error(f"Failed to prepare thermoelectric_universe.db for download: {str(e)}")
                    update_log(f"Failed to prepare thermoelectric_universe.db for download: {str(e)}")
            
            if "JSON" in output_formats:
                json_data = papers_df[["id", "title", "authors", "year", "categories", "abstract", "pdf_url", "pdf_path", "matched_terms"]].to_json(orient="records", lines=True)
                st.download_button(
                    label="Download Paper Metadata JSON",
                    data=json_data,
                    file_name="thermoelectric_papers.json",
                    mime="application/json"
                )
            
            log_container = st.empty()
            log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)