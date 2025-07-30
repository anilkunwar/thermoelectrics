import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core.composition import Composition
import re
import io
import sqlite3
import arxiv
import requests
from chemdataextractor import Document
from chemdataextractor.model import Compound, TemperatureModel, BaseModel
from chemdataextractor.parse import AutoSentenceParser
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.spatial.distance import cosine
import spacy
from spacy.language import Language
from spacy.tokens import Span
import nltk
from nltk.tokenize import sent_tokenize
import os
import matplotlib
from datetime import datetime
import mpi4py.MPI

# Download NLTK data
nltk.download('punkt')

# Set matplotlib to non-interactive backend
matplotlib.use('Agg')

# Electronegativity and thermoelectric weights
electronegativity = {
    'O': 3.44, 'Cl': 3.16, 'N': 3.04, 'Br': 2.96, 'I': 2.66, 'S': 2.58, 'Se': 2.55, 'Te': 2.1, 'P': 2.19, 'As': 2.18,
    'Sb': 2.05, 'Bi': 2.02, 'Si': 1.90, 'Ge': 2.01, 'Sn': 1.96, 'Pb': 2.33, 'B': 2.04, 'Al': 1.61, 'Ga': 1.81,
    'In': 1.78, 'Tl': 2.04, 'Mg': 1.31, 'Ca': 1.00, 'Sr': 0.95, 'Ba': 0.89, 'Li': 0.98, 'Na': 0.93, 'K': 0.82,
    'Rb': 0.82, 'Cs': 0.79, 'Sc': 1.36, 'Y': 1.22, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Sm': 1.17,
    'Eu': 1.2, 'Gd': 1.2, 'Tb': 1.1, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.1, 'Lu': 1.27,
    'Ti': 1.54, 'Zr': 1.33, 'Hf': 1.3, 'V': 1.63, 'Nb': 1.6, 'Ta': 1.5, 'Cr': 1.66, 'Mo': 2.16, 'Mn': 1.55,
    'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Cd': 1.69, 'Ag': 1.93, 'Au': 2.54, 'Pd': 2.20, 'Ru': 2.2
}

thermoelectric_weights = {
    'Bi': 2.0, 'Te': 2.0, 'Sb': 1.8, 'Pb': 1.8, 'Se': 1.5, 'Sn': 1.5, 'Ge': 1.3, 'Si': 1.3, 'Mg': 1.2
}

# List of 85 elements
all_elements = [
    'In', 'Tl', 'La', 'Sr', 'Mn', 'Ni', 'Ru', 'Pd', 'Hf', 'Cs', 'Sc', 'Co', 'Si', 'Fe', 'Li', 'Cl', 'Yb', 'Te',
    'N', 'Ti', 'Cd', 'Zr', 'Y', 'Ga', 'Cr', 'Pr', 'Tm', 'Br', 'Ca', 'Mg', 'Rb', 'Au', 'Nd', 'Ce', 'Ho', 'I', 'Ba',
    'Se', 'Pb', 'Ge', 'Gd', 'Tb', 'Dy', 'Cu', 'Na', 'Sb', 'Bi', 'P', 'As', 'Sm', 'Zn', 'Al', 'Sn', 'Ag', 'Nb', 'Mo',
    'V', 'S', 'K', 'Lu', 'O', 'Eu', 'Ta', 'B', 'Er', 'H', 'He', 'Be', 'C', 'F', 'Ne', 'Ar', 'Kr',
    'Xe', 'Tc', 'Rh', 'Pm', 'Re', 'Os', 'Ir', 'Pt', 'Hg', 'W'
]

# Color map for elements
default_color_list = (
    px.colors.qualitative.Plotly +
    px.colors.qualitative.Pastel1 +
    px.colors.qualitative.D3 +
    px.colors.qualitative.G10 +
    px.colors.qualitative.T10
)
default_element_color_map = dict(zip(all_elements, default_color_list[:len(all_elements)]))

# ChemDataExtractor models
class SeebeckCoefficient(BaseModel):
    value = FloatType()
    units = StringType(contextual=True)
    specifier = StringType()
    compound = ModelType(Compound, required=True)
    temperature = ModelType(TemperatureModel, required=True)

class PowerFactor(BaseModel):
    value = FloatType()
    units = StringType(contextual=True)
    specifier = StringType()
    compound = ModelType(Compound, required=True)
    temperature = ModelType(TemperatureModel, required=True)

class ThermalConductivity(BaseModel):
    value = FloatType()
    units = StringType(contextual=True)
    specifier = StringType()
    compound = ModelType(Compound, required=True)
    temperature = ModelType(TemperatureModel, required=True)

class ElectricalConductivity(BaseModel):
    value = FloatType()
    units = StringType(contextual=True)
    specifier = StringType()
    compound = ModelType(Compound, required=True)
    temperature = ModelType(TemperatureModel, required=True)

class ZT(BaseModel):
    value = FloatType()
    specifier = StringType()
    compound = ModelType(Compound, required=True)
    temperature = ModelType(TemperatureModel, required=True)

# SciBERT setup
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

# spaCy setup with custom NER
nlp = spacy.load("en_core_web_sm")
@Language.component("thermoelectric_ner")
def thermoelectric_ner(doc):
    entities = []
    for token in doc:
        if re.match(r'[A-Z][a-z]?[0-9]*\.?[0-9]*', token.text):
            entities.append(Span(doc, token.i, token.i+1, label="CHEMICAL"))
        if token.text.lower() in ["kelvin", "k"] and token.i > 0 and doc[token.i-1].like_num:
            entities.append(Span(doc, token.i-1, token.i+1, label="TEMPERATURE"))
        if token.text.lower() in ["μv/k", "v/k"] and token.i > 0 and doc[token.i-1].like_num:
            entities.append(Span(doc, token.i-1, token.i+1, label="SEEBECK"))
        if token.text.lower() in ["w/mk", "w/(mk)"] and token.i > 0 and doc[token.i-1].like_num:
            entities.append(Span(doc, token.i-1, token.i+1, label="THERMAL_CONDUCTIVITY"))
        if token.text.lower() in ["s/m"] and token.i > 0 and doc[token.i-1].like_num:
            entities.append(Span(doc, token.i-1, token.i+1, label="ELECTRICAL_CONDUCTIVITY"))
        if token.text.lower() in ["w/mk2", "w/(mk2)"] and token.i > 0 and doc[token.i-1].like_num:
            entities.append(Span(doc, token.i-1, token.i+1, label="POWER_FACTOR"))
    doc.ents = entities
    return doc
nlp.add_pipe("thermoelectric_ner", after="ner")

# Database setup
def init_databases():
    conn_meta = sqlite3.connect("thermoelectric_metadata.db")
    conn_full = sqlite3.connect("complete_knowledge.db")
    c_meta = conn_meta.cursor()
    c_full = conn_full.cursor()
    c_meta.execute('''CREATE TABLE IF NOT EXISTS metadata
                    (arxiv_id TEXT PRIMARY KEY, title TEXT, abstract TEXT, authors TEXT,
                     published TEXT, doi TEXT, pdf_url TEXT)''')
    c_full.execute('''CREATE TABLE IF NOT EXISTS full_text
                    (arxiv_id TEXT PRIMARY KEY, full_text TEXT)''')
    conn_meta.commit()
    conn_full.commit()
    return conn_meta, conn_full

# Functions from previous artifact
def parse_formula(formula):
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)?'
    elements = re.findall(pattern, formula)
    return list(set([element[0] for element in elements]))

def extract_multiplier_and_replace(input_formula):
    pattern = r'\)(\d*\.?\d*)?'
    match = re.search(pattern, input_formula)
    if match:
        multiplier = float(match.group(1)) if match.group(1) else 1.0
        parts = re.split(pattern, input_formula)
        formula_without_multiplier = parts[0]
        content_within_parentheses = formula_without_multiplier.split('(')[-1]
        elements_within_parentheses = re.findall(r'([A-Za-z]+)(\d*\.?\d*)', content_within_parentheses)
        modified_elements = [(element, str(float(stoichiometry) * multiplier) if stoichiometry else '0.0') for element, stoichiometry in elements_within_parentheses]
        modified_formula = formula_without_multiplier.split('(')[0]
        modified_formula += ''.join(element + stoichiometry for element, stoichiometry in modified_elements)
        return modified_formula
    return input_formula

def count_elements(df):
    elements = set()
    for formula in df['Formula']:
        try:
            elements.update(parse_formula(formula))
        except Exception as e:
            st.warning(f"Error parsing formula {formula}: {e}")
            continue
    return sorted(list(elements))

def featurize_materials(df, available_elements):
    features = []
    for _, row in df.iterrows():
        try:
            modified_formula = extract_multiplier_and_replace(row['Formula'])
            composition = Composition(modified_formula)
            composition_dict = composition.fractional_composition.as_dict()
            feature_vector = {element: composition_dict.get(element, 0) for element in available_elements}
            feature_vector['electrical_conductivity(S/m)'] = row.get('electrical_conductivity(S/m)', float('nan'))
            feature_vector['thermal_conductivity(W/mK)'] = row.get('thermal_conductivity(W/mK)', float('nan'))
            feature_vector['power_factor(W/mK2)'] = row.get('power_factor(W/mK2)', float('nan'))
            feature_vector['ZT'] = row.get('ZT', float('nan'))
            feature_vector['reference'] = row.get('reference', float('nan'))
            features.append(feature_vector)
        except Exception as e:
            st.warning(f"Error processing formula {row['Formula']}: {e}")
            continue
    return features

def plot_periodic_table(all_elements, present_elements, element_color_map, fontsize=12):
    periodic_table_positions = {
        'H': (1, 1), 'He': (1, 18),
        'Li': (2, 1), 'Be': (2, 2), 'B': (2, 13), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18),
        'Na': (3, 1), 'Mg': (3, 2), 'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
        'K': (4, 1), 'Ca': (4, 2), 'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6), 'Mn': (4, 7), 'Fe': (4, 8),
        'Co': (4, 9), 'Ni': (4, 10), 'Cu': (4, 11), 'Zn': (4, 12), 'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15),
        'Se': (4, 16), 'Br': (4, 17), 'Kr': (4, 18),
        'Rb': (5, 1), 'Sr': (5, 2), 'Y': (5, 3), 'Zr': (5, 4), 'Nb': (5, 5), 'Mo': (5, 6), 'Tc': (5, 7), 'Ru': (5, 8),
        'Rh': (5, 9), 'Pd': (5, 10), 'Ag': (5, 11), 'Cd': (5, 12), 'In': (5, 13), 'Sn': (5, 14), 'Sb': (5, 15),
        'Te': (5, 16), 'I': (5, 17), 'Xe': (5, 18),
        'Cs': (6, 1), 'Ba': (6, 2), 'La': (6, 3), 'Ce': (7, 3), 'Pr': (7, 4), 'Nd': (7, 5), 'Pm': (7, 6), 'Sm': (7, 7),
        'Eu': (7, 8), 'Gd': (7, 9), 'Tb': (7, 10), 'Dy': (7, 11), 'Ho': (7, 12), 'Er': (7, 13), 'Tm': (7, 14),
        'Yb': (7, 15), 'Lu': (7, 16), 'Hf': (6, 4), 'Ta': (6, 5), 'W': (6, 6), 'Re': (6, 7), 'Os': (6, 8),
        'Ir': (6, 9), 'Pt': (6, 10), 'Au': (6, 11), 'Hg': (6, 12), 'Tl': (6, 13), 'Pb': (6, 14), 'Bi': (6, 15)
    }
    fig = go.Figure()
    for element in all_elements:
        if element in periodic_table_positions:
            row, col = periodic_table_positions[element]
            en = electronegativity.get(element, 1.0)
            tw = thermoelectric_weights.get(element, 1.0)
            color = element_color_map.get(element, '#D3D3D3') if element in present_elements else '#D3D3D3'
            fig.add_trace(go.Scatter(
                x=[col], y=[-row],
                mode='markers+text',
                text=[element],
                textposition='middle center',
                marker=dict(size=40, color=color, line=dict(width=2, color='black')),
                hoverinfo='text',
                hovertext=[f"Element: {element}<br>Electronegativity: {en:.2f}<br>Thermoelectric Weight: {tw:.2f}"],
                customdata=[element],
                name=element,
                showlegend=False
            ))
    fig.update_layout(
        title=dict(text='Interactive Periodic Table', x=0.5, xanchor='center', font=dict(size=fontsize + 4, family='Arial')),
        xaxis=dict(range=[0, 19], showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(range=[-8, 0], showgrid=False, zeroline=False, showticklabels=False, title=''),
        plot_bgcolor='white', paper_bgcolor='white',
        width=900, height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# SciBERT relevance scoring
def compute_relevance_score(abstract, keyphrases=["thermoelectric", "Seebeck coefficient", "temperature"]):
    inputs = tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    abstract_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    phrase_embeddings = []
    for phrase in keyphrases:
        inputs_phrase = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True, max_length=10)
        with torch.no_grad():
            phrase_output = model(**inputs_phrase)
        phrase_embeddings.append(phrase_output.last_hidden_state.mean(dim=1).squeeze().numpy())
    scores = [1 - cosine(abstract_embedding, phrase_embedding) for phrase_embedding in phrase_embeddings]
    avg_score = np.mean(scores)
    return avg_score

# PMI calculation for co-occurrence
def calculate_pmi(sentences, entity1, entity2):
    count_e1 = sum(1 for sent in sentences if entity1 in sent)
    count_e2 = sum(1 for sent in sentences if entity2 in sent)
    count_both = sum(1 for sent in sentences if entity1 in sent and entity2 in sent)
    total_sentences = len(sentences)
    if count_both == 0 or count_e1 == 0 or count_e2 == 0:
        return 0
    p_e1 = count_e1 / total_sentences
    p_e2 = count_e2 / total_sentences
    p_both = count_both / total_sentences
    pmi = np.log2(p_both / (p_e1 * p_e2)) if p_both > 0 else 0
    return pmi

# Fetch arXiv articles
def fetch_arxiv_articles(start_date="2022-01-01", end_date="2025-07-31"):
    query = "thermoelectric \"Seebeck coefficient\" temperature cat:cond-mat* OR cat:physics.app-ph"
    search = arxiv.Search(
        query=query,
        max_results=1000,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    articles = []
    for result in search.results():
        pub_date = result.published.strftime("%Y-%m-%d")
        if start_date <= pub_date <= end_date:
            score = compute_relevance_score(result.summary)
            if score > 0.4:
                articles.append({
                    "arxiv_id": result.entry_id,
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": ", ".join(author.name for author in result.authors),
                    "published": pub_date,
                    "doi": result.doi or "N/A",
                    "pdf_url": result.pdf_url
                })
    return articles

# Extract data with ChemDataExtractor, spaCy, and SciBERT
def extract_thermoelectric_data(articles, conn_meta, conn_full):
    data_records = []
    c_meta = conn_meta.cursor()
    c_full = conn_full.cursor()
    for article in articles:
        try:
            # Download PDF
            response = requests.get(article["pdf_url"])
            if response.status_code != 200:
                continue
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            doc = Document.from_file("temp.pdf")
            text = doc.text
            sentences = sent_tokenize(text)

            # Store metadata
            c_meta.execute('''INSERT OR REPLACE INTO metadata
                            (arxiv_id, title, abstract, authors, published, doi, pdf_url)
                            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                            (article["arxiv_id"], article["title"], article["abstract"],
                             article["authors"], article["published"], article["doi"],
                             article["pdf_url"]))
            # Store full text
            c_full.execute('''INSERT OR REPLACE INTO full_text
                            (arxiv_id, full_text) VALUES (?, ?)''',
                            (article["arxiv_id"], text))
            
            # ChemDataExtractor extraction
            doc.models = [SeebeckCoefficient, PowerFactor, ThermalConductivity, ElectricalConductivity, ZT]
            for record in doc.records:
                try:
                    formula = record.compound.names[0] if record.compound.names else None
                    if not formula or len(parse_formula(formula)) < 2:
                        continue
                    temp = record.temperature.value if record.temperature else None
                    if temp is None or record.temperature.units.lower() not in ["kelvin", "k"]:
                        continue
                    entry = {
                        "Formula": formula,
                        "temperature(K)": float(temp),
                        "seebeck_coefficient(μV/K)": float(record.value) if isinstance(record, SeebeckCoefficient) else float('nan'),
                        "power_factor(W/mK2)": float(record.value) if isinstance(record, PowerFactor) else float('nan'),
                        "thermal_conductivity(W/mK)": float(record.value) if isinstance(record, ThermalConductivity) else float('nan'),
                        "electrical_conductivity(S/m)": float(record.value) if isinstance(record, ElectricalConductivity) else float('nan'),
                        "ZT": float(record.value) if isinstance(record, ZT) else float('nan'),
                        "reference": article["arxiv_id"]
                    }
                    data_records.append(entry)
                except Exception as e:
                    st.warning(f"Error processing record in {article['arxiv_id']}: {e}")

            # spaCy NER
            spacy_doc = nlp(text)
            for ent in spacy_doc.ents:
                if ent.label_ in ["CHEMICAL", "TEMPERATURE", "SEEBECK", "THERMAL_CONDUCTIVITY", "ELECTRICAL_CONDUCTIVITY", "POWER_FACTOR"]:
                    try:
                        # Find co-occurring entities in same sentence
                        sentence = ent.sent.text
                        sentence_doc = nlp(sentence)
                        entities = [e.text for e in sentence_doc.ents]
                        if "CHEMICAL" in [e.label_ for e in sentence_doc.ents] and any(label in ["SEEBECK", "TEMPERATURE", "POWER_FACTOR", "THERMAL_CONDUCTIVITY", "ELECTRICAL_CONDUCTIVITY"] for label in [e.label_ for e in sentence_doc.ents]):
                            formula = next((e.text for e in sentence_doc.ents if e.label_ == "CHEMICAL"), None)
                            if formula and len(parse_formula(formula)) >= 2:
                                entry = {
                                    "Formula": formula,
                                    "temperature(K)": float(next((e.text.split()[0] for e in sentence_doc.ents if e.label_ == "TEMPERATURE"), float('nan'))),
                                    "seebeck_coefficient(μV/K)": float(next((e.text.split()[0] for e in sentence_doc.ents if e.label_ == "SEEBECK"), float('nan'))),
                                    "power_factor(W/mK2)": float(next((e.text.split()[0] for e in sentence_doc.ents if e.label_ == "POWER_FACTOR"), float('nan'))),
                                    "thermal_conductivity(W/mK)": float(next((e.text.split()[0] for e in sentence_doc.ents if e.label_ == "THERMAL_CONDUCTIVITY"), float('nan'))),
                                    "electrical_conductivity(S/m)": float(next((e.text.split()[0] for e in sentence_doc.ents if e.label_ == "ELECTRICAL_CONDUCTIVITY"), float('nan'))),
                                    "ZT": float('nan'),
                                    "reference": article["arxiv_id"]
                                }
                                # PMI check
                                pmi_score = calculate_pmi(sentences, formula, "Seebeck coefficient")
                                if pmi_score > 0:
                                    data_records.append(entry)
                    except Exception as e:
                        st.warning(f"Error processing spaCy entity in {article['arxiv_id']}: {e}")

        except Exception as e:
            st.warning(f"Error processing article {article['arxiv_id']}: {e}")
        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")
    conn_meta.commit()
    conn_full.commit()
    return pd.DataFrame(data_records)

# Streamlit UI
st.title("Thermoelectric Material Extraction from arXiv")

# Initialize databases
conn_meta, conn_full = init_databases()

# Extraction
st.subheader("Extract Data from arXiv (2022-2025)")
if st.button("Start Extraction"):
    with st.spinner("Fetching and processing articles..."):
        articles = fetch_arxiv_articles()
        if not articles:
            st.error("No articles retrieved. Check connectivity or query.")
            st.stop()

        # Extract data
        df = extract_thermoelectric_data(articles, conn_meta, conn_full)
        if df.empty:
            st.error("No valid thermoelectric data extracted.")
            st.stop()

        # Process data
        df['modformula'] = df['Formula'].apply(extract_multiplier_and_replace)
        present_elements = count_elements(df)
        st.write("Number of elements present:", len(present_elements))
        st.write("Present elements:", present_elements)

        # Featurize
        features = featurize_materials(df, all_elements)
        if not features:
            st.error("No valid formulas processed.")
            st.stop()
        df_features = pd.DataFrame(features)

        # Combine and calculate sum_elements
        df_combined = pd.concat([df[['Formula', 'modformula', 'temperature(K)', 'seebeck_coefficient(μV/K)', 'power_factor(W/mK2)', 'thermal_conductivity(W/mK)', 'electrical_conductivity(S/m)', 'ZT', 'reference']], df_features], axis=1)
        df_combined['sum_elements'] = df_combined[all_elements].sum(axis=1)

        # Check for duplicate columns
        if df_combined.columns.duplicated().any():
            duplicate_columns = df_combined.columns[df_combined.columns.duplicated()].tolist()
            st.error(f"Duplicate columns found: {duplicate_columns}")
            st.stop()

        # Reorder columns
        output_columns = ['Formula', 'modformula'] + all_elements + [
            'temperature(K)', 'seebeck_coefficient(μV/K)', 'electrical_conductivity(S/m)',
            'thermal_conductivity(W/mK)', 'power_factor(W/mK2)', 'ZT', 'reference', 'sum_elements'
        ]
        for col in output_columns:
            if col not in df_combined.columns:
                df_combined[col] = float('nan')
        df_combined = df_combined[output_columns]

        # Display periodic table
        st.subheader("Interactive Periodic Table")
        st.write("Elements present in extracted data are colored; absent elements are gray.")
        fig_periodic = plot_periodic_table(all_elements, present_elements, default_element_color_map)
        st.plotly_chart(fig_periodic, use_container_width=True)

        # Display and download data
        st.subheader("Extracted and Featurized Data")
        st.write(df_combined)
        st.download_button(
            label="Download Combined CSV",
            data=df_combined.to_csv(index=False).encode('utf-8'),
            file_name='extracted_thermoelectric_data.csv',
            mime='text/csv'
        )

# Close database connections
conn_meta.close()
conn_full.close()
