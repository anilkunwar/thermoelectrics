import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition
import os
import joblib
import colorsys
from itertools import combinations
import logging
from transformers import AutoTokenizer, AutoModel
import arxiv
from datetime import datetime
from retrying import retry
import re
import spacy
from spacy.matcher import Matcher
from collections import Counter
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("arxiv").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

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

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim=66, latent_dim=8):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128, momentum=0.05), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64, momentum=0.05), nn.Dropout(0.4),
        )
        self.z_mean = nn.Linear(64, latent_dim)
        self.z_log_var = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.BatchNorm1d(64, momentum=0.05), nn.Dropout(0.4),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128, momentum=0.05), nn.Dropout(0.4),
            nn.Linear(128, input_dim), nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.z_mean(h)
        log_var = self.z_log_var(h)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

# Regressor Model
class Regressor(nn.Module):
    def __init__(self, latent_dim=8):
        super(Regressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(), nn.BatchNorm1d(16, momentum=0.05), nn.Dropout(0.4),
            nn.Linear(16, 8), nn.ReLU(), nn.BatchNorm1d(8, momentum=0.05), nn.Dropout(0.4),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Preprocessing for prediction
def featurize_composition(composition_dict, available_elements, temperature):
    feature_vector = {element: composition_dict.get(element, 0) for element in available_elements}
    feature_vector['temperature(K)'] = temperature
    return pd.DataFrame([feature_vector])

def preprocess_new_data(df, available_elements, scaler):
    features_df = df
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(features_df)
    X_scaled = scaler.transform(X_imputed)
    return X_scaled

# Compute z_mean statistics and bias vector
def compute_z_mean_stats_and_bias(elements, temperature, available_elements, _scaler, _vae, steps=30):
    z_means = []
    try:
        if len(elements) != 3:
            logger.error(f"Expected 3 elements, got {len(elements)}: {elements}")
            raise ValueError("Exactly 3 elements required")
        if not all(e in available_elements for e in elements):
            logger.error(f"Invalid elements: {elements}")
            raise ValueError("All elements must be in available_elements")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            logger.error(f"Invalid temperature: {temperature}")
            raise ValueError("Temperature must be a non-negative number")
        expected_features = len(available_elements) + 1
        if _scaler.n_features_in_ != expected_features:
            logger.error(f"Scaler expects {expected_features} features, got {_scaler.n_features_in_}")
            raise ValueError("Scaler feature mismatch")
        if _vae.input_dim != expected_features:
            logger.error(f"VAE expects {expected_features} input dimensions, got {_vae.input_dim}")
            raise ValueError("VAE input dimension mismatch")
        
        _vae.eval()
        with torch.no_grad():
            for a in np.linspace(0, 1, steps):
                for b in np.linspace(0, 1 - a, steps):
                    c = 1 - a - b
                    if c >= 0:
                        comp_dict = {elements[0]: a, elements[1]: b, elements[2]: c}
                        df = featurize_composition(comp_dict, available_elements, temperature)
                        X_scaled = preprocess_new_data(df, available_elements, _scaler)
                        if X_scaled.shape[1] != _vae.input_dim:
                            logger.error(f"Input shape mismatch: expected {_vae.input_dim}, got {X_scaled.shape[1]}")
                            raise ValueError("Input shape mismatch")
                        X_tensor = torch.FloatTensor(X_scaled).to(device)
                        logger.debug(f"Processing composition: {comp_dict}, X_tensor shape: {X_tensor.shape}")
                        _, z_mean, _ = _vae(X_tensor)
                        if z_mean.shape[1] != _vae.latent_dim:
                            logger.error(f"z_mean shape mismatch: expected {_vae.latent_dim}, got {z_mean.shape[1]}")
                            raise ValueError("z_mean shape mismatch")
                        z_means.append(z_mean.cpu().numpy())
        if not z_means:
            logger.error("No valid compositions generated")
            raise ValueError("No valid compositions generated")
        z_means = np.vstack(z_means)
        z_mean_avg = np.mean(z_means, axis=0)
        z_mean_std = np.std(z_means, axis=0)
        p_type_comp = {elements[0]: 0.0, elements[1]: 0.4, elements[2]: 0.6}
        n_type_comp = {elements[0]: 0.33, elements[1]: 0.33, elements[2]: 0.34}
        df_p = featurize_composition(p_type_comp, available_elements, temperature)
        df_n = featurize_composition(n_type_comp, available_elements, temperature)
        X_scaled_p = preprocess_new_data(df_p, available_elements, _scaler)
        X_scaled_n = preprocess_new_data(df_n, available_elements, _scaler)
        if X_scaled_p.shape[1] != _vae.input_dim or X_scaled_n.shape[1] != _vae.input_dim:
            logger.error(f"p-type/n-type input shape mismatch: expected {_vae.input_dim}, got {X_scaled_p.shape[1]}/{X_scaled_n.shape[1]}")
            raise ValueError("p-type/n-type input shape mismatch")
        X_tensor_p = torch.FloatTensor(X_scaled_p).to(device)
        X_tensor_n = torch.FloatTensor(X_scaled_n).to(device)
        _, z_mean_p, _ = _vae(X_tensor_p)
        _, z_mean_n, _ = _vae(X_tensor_n)
        bias_vector = (z_mean_p - z_mean_n).cpu().numpy()
        bias_norm = np.linalg.norm(bias_vector)
        if bias_norm > 0:
            bias_vector = bias_vector / bias_norm
        else:
            logger.warning("Bias vector has zero norm, using uniform vector")
            bias_vector = np.ones(_vae.latent_dim) / np.sqrt(_vae.latent_dim)
        bias_magnitude = 0.5 * np.mean(z_mean_std)
        logger.info(f"Computed z_mean_avg: {z_mean_avg.tolist()}, z_mean_std: {z_mean_std.tolist()}, bias_vector: {bias_vector.tolist()}, bias_magnitude: {bias_magnitude}")
        return z_mean_avg, z_mean_std, bias_vector, bias_magnitude
    except Exception as e:
        logger.error(f"Failed to compute z_mean statistics and bias: {e}")
        fallback_z_mean_avg = np.array([-0.0003, -0.0000, 0.0004, 0.0003, 0.0003, -0.0006, 0.0009, -0.0001])
        fallback_z_mean_std = np.array([0.0003, 0.0007, 0.0003, 0.0005, 0.0005, 0.0010, 0.0011, 0.0003])
        fallback_bias_vector = np.ones(8) / np.sqrt(8)
        fallback_bias_magnitude = 0.5 * np.mean(fallback_z_mean_std)
        logger.warning(f"Using fallback statistics: z_mean_avg={fallback_z_mean_avg.tolist()}, z_mean_std={fallback_z_mean_std.tolist()}, bias_vector={fallback_bias_vector.tolist()}, bias_magnitude={fallback_bias_magnitude}")
        return fallback_z_mean_avg, fallback_z_mean_std, fallback_bias_vector, fallback_bias_magnitude

# Plot z_mean bar chart (Plotly)
def plot_z_mean_bar_chart(z_mean_avg, z_mean_std, font_size):
    dimensions = [f"Dim {i+1}" for i in range(len(z_mean_avg))]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dimensions,
        y=z_mean_avg,
        error_y=dict(type='data', array=z_mean_std, visible=True),
        marker=dict(color='blue', opacity=0.7),
        name='z_mean'
    ))
    fig.update_layout(
        title=dict(text='Latent Space z_mean: Mean with SD Error Bars', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Latent Dimension',
        yaxis_title='z_mean Value',
        xaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# Plot z_mean bar chart (Matplotlib)
def plot_z_mean_bar_chart_matplotlib(z_mean_avg, z_mean_std, output_path='z_mean_bar_chart.pdf'):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    dimensions = [f'Dim {i+1}' for i in range(len(z_mean_avg))]
    x = np.arange(len(z_mean_avg))
    ax.bar(x, z_mean_avg, yerr=z_mean_std, capsize=5, color='#1f77b4', edgecolor='black', alpha=0.7, label='z_mean')
    ax.set_xlabel('Latent Dimension', fontsize=14, fontfamily='Arial')
    ax.set_ylabel('z_mean Value', fontsize=14, fontfamily='Arial')
    ax.set_title('Latent Space z_mean: Mean with SD Error Bars', fontsize=16, fontfamily='Arial', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=12, fontfamily='Arial')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    try:
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        logger.info(f"Saved Matplotlib figure to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Matplotlib figure: {e}")
    return fig

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    logger.warning(f"Failed to load 'en_core_web_lg': {e}. Using 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e2:
        logger.error(f"Failed to load spaCy: {e2}")
        st.error(f"Failed to load spaCy: {e2}. Install: `python -m spacy download en_core_web_sm`")
        st.stop()

# Initialize spaCy matcher
matcher = Matcher(nlp.vocab)
material_patterns = [
    [{"LOWER": "p-type"}, {"LOWER": {"IN": ["material", "semiconductor", "thermoelectric", "compound"]}, "OP": "?"}],
    [{"LOWER": "n-type"}, {"LOWER": {"IN": ["material", "semiconductor", "thermoelectric", "compound"]}, "OP": "?"}]
]
matcher.add("P_TYPE", [material_patterns[0]])
matcher.add("N_TYPE", [material_patterns[1]])

# Synonym mapping and weights
THERMOELECTRIC_SYNONYMS = {
    "seebeck coefficient": ["thermopower", "seebeck", "thermoelectric voltage"],
    "power factor": ["power factor", "thermoelectric power factor"],
    "zt": ["figure of merit", "zt", "thermoelectric figure of merit", "dimensionless figure of merit"],
    "thermoelectric": ["thermoelectric", "thermoelectric material", "thermoelectrics"],
    "p-type": ["p-type", "p-type semiconductor", "p-type material"],
    "n-type": ["n-type", "n-type semiconductor", "n-type material"]
}

TERM_WEIGHTS = {
    "seebeck coefficient": 2.5, "thermopower": 2.5, "seebeck": 2.0,
    "power factor": 2.0, "zt": 2.5, "figure of merit": 2.5,
    "thermoelectric": 1.5, "thermoelectric material": 1.5,
    "p-type": 2.0, "n-type": 2.0, "p-type semiconductor": 2.0, "n-type semiconductor": 2.0
}

UNIT_VARIANTS = [
    "microvolt/K", "μV/K", "mV/K", "V/K", "microvolts per Kelvin",
    "microvolts/Kelvin", "microvolt per Kelvin", "μV per Kelvin"
]

# Fetch arXiv abstracts
@st.cache_data(hash_funcs={dict: lambda x: tuple(sorted(x.items()))})
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_arxiv_abstracts(elements, composition_dict):
    try:
        comp = Composition({el: composition_dict.get(el, 0) for el in elements})
        formula = comp.reduced_formula
        query_terms = [formula]
        for key, synonyms in THERMOELECTRIC_SYNONYMS.items():
            query_terms.extend(synonyms)
        query_terms.extend(UNIT_VARIANTS)
        query = f"{formula} ({' OR '.join(query_terms)})"
        logger.info(f"Querying arXiv with: {query}")
        results = arxiv.Client().results(arxiv.Search(query=query, max_results=50))
        abstracts = [
            {
                "title": r.title,
                "abstract": r.summary,
                "arxiv_id": r.entry_id,
                "published": r.published,
                "authors": ", ".join([author.name for author in r.authors])
            }
            for r in results if r.published.year >= 2020
        ]
        logger.info(f"Retrieved {len(abstracts)} abstracts for {formula}")
        return abstracts
    except Exception as e:
        logger.error(f"Failed to fetch arXiv abstracts: {str(e)}")
        return []

# Score abstracts with SciBERT
@st.cache_data
def score_abstract_with_scibert(abstract, formula):
    try:
        inputs = scibert_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        attentions = outputs.attentions[-1][0].mean(dim=0).numpy()
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        relevance_prob = 0.5
        keyword_indices = []
        formula_indices = []
        abstract_lower = abstract.lower()
        formula_lower = formula.lower()
        
        for i, token in enumerate(tokens):
            token_lower = token.lower().replace("##", "")
            for key, synonyms in THERMOELECTRIC_SYNONYMS.items():
                for term in [key] + synonyms:
                    if re.search(rf'\b{re.escape(term)}\b', token_lower):
                        keyword_indices.append(i)
            if re.search(rf'\b{re.escape(formula_lower)}\b', token_lower):
                formula_indices.append(i)
        
        if keyword_indices:
            attn_scores = attentions[keyword_indices].sum()
            avg_attn_score = attn_scores / len(keyword_indices)
            relevance_prob = min(relevance_prob + 0.4 * len(keyword_indices) * avg_attn_score, 1.0)
            logger.info(f"Attention boost: {len(keyword_indices)} tokens, avg attention: {avg_attn_score:.3f}")
        
        for term in ["seebeck coefficient", "power factor", "zt", "thermoelectric"] + UNIT_VARIANTS:
            if term in abstract_lower:
                term_pos = abstract_lower.find(term)
                formula_pos = abstract_lower.find(formula_lower)
                if formula_pos != -1 and abs(term_pos - formula_pos) < 100:
                    relevance_prob = min(relevance_prob + 0.25, 1.0)
                    logger.info(f"Contextual boost: {term} near {formula}")
        
        if not keyword_indices:
            word_counts = Counter(re.findall(r'\b\w+\b', abstract_lower))
            total_words = sum(word_counts.values())
            score = 0.0
            for term, weight in TERM_WEIGHTS.items():
                if term in word_counts:
                    score += weight * word_counts[term] / (total_words + 1e-6)
            max_possible_score = sum(TERM_WEIGHTS.values()) / 10
            relevance_prob = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
            logger.info(f"Fallback scoring: {relevance_prob:.3f}")
        
        return relevance_prob
    except Exception as e:
        logger.error(f"SciBERT scoring failed: {str(e)}")
        return 0.5

# Extract material type and compute probabilities
@st.cache_data(hash_funcs={dict: lambda x: tuple(sorted(x.items()))})
def extract_material_type(elements, composition_dict):
    try:
        global scibert_tokenizer, scibert_model
        if 'scibert_tokenizer' not in globals() or 'scibert_model' not in globals():
            scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            scibert_model.eval()
            scibert_model.to(device)
        
        comp = Composition({el: composition_dict.get(el, 0) for el in elements})
        formula = comp.reduced_formula
        formula_variants = [
            formula,
            formula.replace('2', '₂'),
            f"{elements[0]}{elements[1]}{elements[2]}",
            re.compile(f"{re.escape(elements[0])}[0-9.]*{re.escape(elements[1])}[0-9.]*{re.escape(elements[2])}[0-9.]*")
        ]
        
        abstracts = fetch_arxiv_abstracts(elements, composition_dict)
        total_abstracts = len(abstracts)
        if not abstracts:
            logger.warning(f"No abstracts found for {formula}")
            summary_dict = {
                "total_abstracts": 0,
                "formula_matches": 0,
                "p_type_count": 0,
                "n_type_count": 0,
                "neutral_count": 0,
                "relevance_scores": [],
                "matched_terms": [],
                "p_type_prob": 0.0,
                "n_type_prob": 0.0
            }
            verbatim_matches = [{"arxiv_id": "", "title": "", "snippet": "No abstracts found.", "label": "Neutral", "score": 0.0}]
            return "Neutral", summary_dict, verbatim_matches
        
        classifications = []
        verbatim_matches = []
        formula_matches = 0
        relevance_scores = []
        p_scores = []
        n_scores = []
        
        for abstract_data in abstracts:
            abstract = abstract_data['abstract'].lower()
            title = abstract_data['title']
            arxiv_id = abstract_data['arxiv_id']
            
            formula_present = any(variant.lower() in abstract for variant in formula_variants[:-1]) or formula_variants[-1].search(abstract)
            if not formula_present:
                continue
            formula_matches += 1
            
            relevance_score = score_abstract_with_scibert(abstract, formula)
            if relevance_score < 0.3:
                continue
            relevance_scores.append(relevance_score)
            
            doc = nlp(abstract_data['abstract'])
            matches = matcher(doc)
            for match_id, start, end in matches:
                label = nlp.vocab.strings[match_id].lower()
                span = doc[start:end]
                context_start = max(0, start - 50)
                context_end = min(len(doc), end + 50)
                context_text = doc[context_start:context_end].text
                formula_in_context = any(variant.lower() in context_text.lower() for variant in formula_variants[:-1]) or formula_variants[-1].search(context_text)
                if not formula_in_context:
                    continue
                thermoelectric_context = any(term in context_text.lower() for term in sum(THERMOELECTRIC_SYNONYMS.values(), []) + UNIT_VARIANTS)
                score = relevance_score * TERM_WEIGHTS.get(label, 2.0) * (1.2 if thermoelectric_context else 1.0)
                classifications.append(label)
                verbatim_matches.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "snippet": context_text,
                    "label": label,
                    "score": score
                })
                if label == "p-type":
                    p_scores.append(score)
                elif label == "n-type":
                    n_scores.append(score)
        
        p_count = classifications.count('p-type')
        n_count = classifications.count('n-type')
        neutral_count = formula_matches - p_count - n_count
        p_type_prob = sum(p_scores) / (sum(p_scores) + sum(n_scores) + 1e-6) if p_scores or n_scores else 0.0
        n_type_prob = sum(n_scores) / (sum(p_scores) + sum(n_scores) + 1e-6) if p_scores or n_scores else 0.0
        
        summary_dict = {
            "total_abstracts": total_abstracts,
            "formula_matches": formula_matches,
            "p_type_count": p_count,
            "n_type_count": n_count,
            "neutral_count": neutral_count,
            "relevance_scores": relevance_scores,
            "matched_terms": list(set(sum([[term for term in sum(THERMOELECTRIC_SYNONYMS.values(), []) + UNIT_VARIANTS if term in abstract.lower()] for abstract in [a['abstract'].lower() for a in abstracts]], []))),
            "p_type_prob": p_type_prob,
            "n_type_prob": n_type_prob
        }
        
        if classifications:
            p_score = sum(p_scores)
            n_score = sum(n_scores)
            material_type = 'p-type' if p_score > n_score else 'n-type' if n_score > p_score else 'Neutral'
        else:
            material_type = 'Neutral'
            if formula_matches > 0:
                verbatim_matches.append({
                    "arxiv_id": abstracts[0]['arxiv_id'],
                    "title": abstracts[0]['title'],
                    "snippet": f"Formula {formula} found but no p-type or n-type mentions.",
                    "label": "Neutral",
                    "score": 0.0
                })
        
        logger.info(f"Classifications for {formula}: p-type={p_count}, n-type={n_count}, Neutral={neutral_count}, p-prob={p_type_prob:.3f}, n-prob={n_type_prob:.3f}, Selected: {material_type}")
        return material_type, summary_dict, verbatim_matches
    except Exception as e:
        logger.error(f"Failed to extract material type: {str(e)}")
        summary_dict = {
            "total_abstracts": 0,
            "formula_matches": 0,
            "p_type_count": 0,
            "n_type_count": 0,
            "neutral_count": 0,
            "relevance_scores": [],
            "matched_terms": [],
            "p_type_prob": 0.0,
            "n_type_prob": 0.0
        }
        verbatim_matches = [{"arxiv_id": "", "title": "", "snippet": f"Error: {str(e)}", "label": "Neutral", "score": 0.0}]
        return "Neutral", summary_dict, verbatim_matches

# Plot histograms for p-type and n-type counts
def plot_material_type_histogram(summary_dict, font_size):
    labels = ['p-type', 'n-type', 'Neutral']
    counts = [summary_dict['p_type_count'], summary_dict['n_type_count'], summary_dict['neutral_count']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c'], opacity=0.7),
        text=counts,
        textposition='auto'
    ))
    fig.update_layout(
        title=dict(text='Material Type Classifications from arXiv Abstracts', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Material Type',
        yaxis_title='Count',
        xaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# Plot probability bar chart
def plot_material_probabilities(summary_dict, font_size):
    labels = ['p-type', 'n-type']
    probs = [summary_dict['p_type_prob'], summary_dict['n_type_prob']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=probs,
        marker=dict(color=['#1f77b4', '#ff7f0e'], opacity=0.7),
        text=[f"{p:.3f}" for p in probs],
        textposition='auto'
    ))
    fig.update_layout(
        title=dict(text='Attention-Based Probabilities for Material Types', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Material Type',
        yaxis_title='Probability',
        xaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size)), range=[0, 1]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# Plot relevance score box plot
def plot_relevance_box_plot(summary_dict, font_size):
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=summary_dict['relevance_scores'],
        name='Relevance Scores',
        marker_color='#1f77b4',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
    ))
    fig.update_layout(
        title=dict(text='Relevance Scores of arXiv Abstracts', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        yaxis_title='Relevance Score',
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        xaxis=dict(tickfont=dict(size=font_size), showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# Plot term co-occurrence network
def plot_term_cooccurrence_network(abstracts, font_size):
    terms = sum(THERMOELECTRIC_SYNONYMS.values(), []) + UNIT_VARIANTS
    G = nx.Graph()
    for term in terms:
        G.add_node(term)
    
    for abstract_data in abstracts:
        abstract = abstract_data['abstract'].lower()
        present_terms = [term for term in terms if term in abstract]
        for term1, term2 in combinations(present_terms, 2):
            if G.has_edge(term1, term2):
                G[term1][term2]['weight'] += 1
            else:
                G.add_edge(term1, term2, weight=1)
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_x = []
    edge_y = []
    edge_weights = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=[w * 0.5 for w in edge_weights], color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        textfont=dict(size=font_size - 2, family='Arial'),
        marker=dict(
            size=10,
            color='#1f77b4',
            line=dict(width=2, color='black')
        ),
        hoverinfo='text'
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(text='Term Co-occurrence Network in Abstracts', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

def predict_seebeck(composition_dict, temperature, available_elements, _scaler, _vae, _regressor, _y_scaler, sign_bias=None, bias_vector=None, bias_magnitude=0.0003):
    try:
        df = featurize_composition(composition_dict, available_elements, temperature)
        X_scaled = preprocess_new_data(df, available_elements, _scaler)
        if X_scaled.shape[1] != _vae.input_dim:
            logger.error(f"Input shape mismatch in predict_seebeck: expected {_vae.input_dim}, got {X_scaled.shape[1]}")
            raise ValueError("Input shape mismatch")
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        _vae.eval()
        _regressor.eval()
        with torch.no_grad():
            _, z_mean, _ = _vae(X_tensor)
            z_mean_original = z_mean.clone()
            y_scaled_pred_unbiased = _regressor(z_mean_original)
            y_pred_unbiased = _y_scaler.inverse_transform(y_scaled_pred_unbiased.cpu().numpy().reshape(-1, 1)).ravel()
            y_pred_unbiased = np.clip(y_pred_unbiased, -300, 300)
            logger.debug(f"Unbiased z_mean: {z_mean_original.cpu().numpy().tolist()}, y_pred_unbiased: {y_pred_unbiased.tolist()}")
            if sign_bias is not None and bias_vector is not None:
                bias_vector = torch.FloatTensor(bias_vector).to(device) * bias_magnitude
                if sign_bias == 'p-type':
                    z_mean = z_mean + bias_vector
                    logger.info(f"Applied p-type bias: {bias_vector.tolist()}")
                elif sign_bias == 'n-type':
                    z_mean = z_mean - bias_vector
                    logger.info(f"Applied n-type bias: {bias_vector.tolist()}")
                y_scaled_pred = _regressor(z_mean)
                y_pred = _y_scaler.inverse_transform(y_scaled_pred.cpu().numpy().reshape(-1, 1)).ravel()
                y_pred = np.clip(y_pred, -300, 300)
                if sign_bias == 'n-type' and y_pred[0] > 0:
                    y_pred = -y_pred
                    logger.warning(f"N-type bias produced positive Seebeck {y_pred[0]:.2f}, enforced negative sign")
                if abs(y_pred[0]) > 0:
                    y_pred = y_pred * (abs(y_pred_unbiased[0]) / abs(y_pred[0]))
                logger.debug(f"Biased z_mean: {z_mean.cpu().numpy().tolist()}, y_pred: {y_pred.tolist()}")
            else:
                y_pred = y_pred_unbiased
        return y_pred[0], y_pred_unbiased[0]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if sign_bias is not None:
            logger.warning(f"Retrying prediction without sign bias due to error with {sign_bias} bias.")
            return predict_seebeck(composition_dict, temperature, available_elements, _scaler, _vae, _regressor, _y_scaler, sign_bias=None, bias_vector=None, bias_magnitude=bias_magnitude)
        return None, None

# Load models and scalers
script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    vae = VAE().to(device)
    regressor = Regressor().to(device)
    vae.load_state_dict(torch.load(os.path.join(script_dir, 'vae_model.pt'), map_location=device))
    regressor.load_state_dict(torch.load(os.path.join(script_dir, 'regressor_model.pt'), map_location=device))
    scaler = joblib.load(os.path.join(script_dir, 'scaler.pkl'))
    y_scaler = joblib.load(os.path.join(script_dir, 'y_scaler.pkl'))
except FileNotFoundError as e:
    st.error(f"Required files not found: {e}")
    st.stop()
except RuntimeError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Available elements
available_elements = [
    'Mg', 'Cs', 'Co', 'Zr', 'Se', 'Dy', 'Pb', 'Ga', 'O', 'Sn', 'Yb', 'B', 'La', 'Si', 'V', 'Fe', 'S', 'Sc', 'Tl', 'Zn',
    'Cl', 'Ce', 'Er', 'Nd', 'Pd', 'Y', 'P', 'Ta', 'In', 'Te', 'Ru', 'Rb', 'Tm', 'Tb', 'Sb', 'Al', 'Lu', 'Bi', 'Pr', 'Eu',
    'Sm', 'Ba', 'Cr', 'Sr', 'Ni', 'Ca', 'As', 'Mn', 'Mo', 'Cd', 'Ti', 'Nb', 'Hf', 'Gd', 'Ag', 'Ge', 'Li', 'Br', 'Au', 'I',
    'N', 'Na', 'Cu', 'Ho', 'K'
]

# All elements for full periodic table
all_elements = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'
]

# Enhanced color map for all elements
base_color_list = (
    px.colors.qualitative.Plotly +
    px.colors.qualitative.Pastel1 +
    px.colors.qualitative.D3 +
    px.colors.qualitative.G10 +
    px.colors.qualitative.T10 +
    px.colors.qualitative.Set1 +
    px.colors.qualitative.Set2 +
    px.colors.qualitative.Set3 +
    px.colors.qualitative.Pastel2 +
    px.colors.qualitative.Dark2
)
num_additional_colors = len(all_elements) - len(base_color_list)
additional_colors = []
for i in range(num_additional_colors):
    hue = i / num_additional_colors
    rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    additional_colors.append(hex_color)
default_color_list = base_color_list + additional_colors
default_element_color_map = dict(zip(all_elements, default_color_list[:len(all_elements)]))

# Streamlit UI
st.title("Ternary Seebeck Coefficient Predictor with Informatics-Aided Classification")
st.markdown("""
This application predicts the Seebeck coefficient for a ternary composition using a two-tab interface:
- **Tab 1: Material Type Classification**: Select elements and compositions, fetch arXiv abstracts (2020-2025), and classify the material as p-type, n-type, or Neutral using SciBERT and spaCy NER. View histograms, probabilities, relevance score box plots, and term co-occurrence networks. Confirm or override the material type.
- **Tab 2: Seebeck Prediction**: Use the confirmed material type to bias the Seebeck coefficient prediction, visualized in a ternary diagram and temperature variance plot.

**Informatics Classification**:
- **SciBERT Attention**: Scores abstracts based on thermoelectric terms and formula proximity.
- **spaCy NER**: Detects p-type/n-type mentions in context.
- **Visualizations**: Histograms for p-type/n-type counts, probability bars, relevance score box plots, and term co-occurrence networks.
- **User Confirmation**: Allows manual selection of material type based on informatics guidance.

**Maximum Seebeck Calculation**: Computes the maximum |S(x)| from 496 ternary compositions at the specified temperature.

**Date and Time**: 07:22 AM CEST, Tuesday, August 19, 2025
""")

# Sidebar for figure customization
st.sidebar.header("Figure Customization")
color_scales = [
    'aggrnyl', 'agsunset', 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brwnyl',
    'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'darkmint', 'electric', 'emrld', 'gnbu',
    'greens', 'greys', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magma', 'mint',
    'orrd', 'oranges', 'oryel', 'peach', 'pinkyl', 'plasma', 'plotly3', 'pubu', 'pubugn',
    'purp', 'purples', 'purpor', 'rainbow', 'rdpu', 'reds', 'sunset', 'sunsetdark', 'teal',
    'tealgrn', 'turbo', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
]
color_scale = st.sidebar.selectbox("Ternary Color Scale", color_scales, index=color_scales.index('viridis'))
legend_font_size = st.sidebar.slider("Legend Font Size", 8, 20, 12)
axes_line_width = st.sidebar.slider("Axes Line Width", 1, 5, 2)
font_size = st.sidebar.slider("Font Size (Axes/Title)", 8, 20, 16)
grid_width = st.sidebar.slider("Grid Width", 0.5, 3.0, 1.0, step=0.5)
user_point_color = st.sidebar.color_picker("User Composition Point Color", '#FF0000')
max_point_color = st.sidebar.color_picker("Max |Seebeck| Point Color", '#00FF00')
ternary_grid_color = st.sidebar.color_picker("Ternary Grid Color", '#000000')
ternary_axes_color = st.sidebar.color_picker("Ternary Axes Color", '#000000')
point_size = st.sidebar.slider("Point Size (Ternary/Temperature)", 5, 20, 10)
axes_box_thickness = st.sidebar.slider("Axes Box Thickness", 1, 5, 2)
legend_spacing = st.sidebar.slider("Legend Spacing (Point Legend to Ternary)", 0.0, 0.5, 0.3, step=0.05)

# Initialize session state
try:
    if 'selected_elements' not in st.session_state:
        st.session_state.selected_elements = []
    if 'proportions' not in st.session_state:
        st.session_state.proportions = {}
    if 'compositions' not in st.session_state:
        st.session_state.compositions = {}
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 800
    if 'material_type' not in st.session_state:
        st.session_state.material_type = 'Neutral'
    if 'summary_dict' not in st.session_state:
        st.session_state.summary_dict = {}
    if 'verbatim_matches' not in st.session_state:
        st.session_state.verbatim_matches = []
    if 'abstracts' not in st.session_state:
        st.session_state.abstracts = []
except Exception as e:
    st.warning(f"Session state initialization failed: {e}. Resetting to defaults.")
    st.session_state.selected_elements = []
    st.session_state.proportions = {}
    st.session_state.compositions = {}
    st.session_state.temperature = 800
    st.session_state.material_type = 'Neutral'
    st.session_state.summary_dict = {}
    st.session_state.verbatim_matches = []
    st.session_state.abstracts = []

# Tabs for material type classification and Seebeck prediction
tab1, tab2 = st.tabs(["Material Type Classification", "Seebeck Coefficient Prediction"])

# Tab 1: Material Type Classification
with tab1:
    st.header("Material Type Classification")
    
    # Periodic Table for Reference
    st.subheader("Periodic Table Reference")
    st.write("Below are two periodic tables: one showing only available elements in color, and another showing the full periodic table with unavailable elements in gray. Selected elements have bold outlines in both.")
    
    def plot_periodic_table(available_elements, selected_elements, element_color_map, show_all_elements=False, fontsize=14):
        periodic_table_positions = {
            'H': (1, 1), 'He': (1, 18),
            'Li': (2, 1), 'Be': (2, 2), 'B': (2, 13), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18),
            'Na': (3, 1), 'Mg': (3, 2), 'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
            'K': (4, 1), 'Ca': (4, 2), 'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6), 'Mn': (4, 7), 'Fe': (4, 8), 'Co': (4, 9), 'Ni': (4, 10), 'Cu': (4, 11), 'Zn': (4, 12), 'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15), 'Se': (4, 16), 'Br': (4, 17), 'Kr': (4, 18),
            'Rb': (5, 1), 'Sr': (5, 2), 'Y': (5, 3), 'Zr': (5, 4), 'Nb': (5, 5), 'Mo': (5, 6), 'Tc': (5, 7), 'Ru': (5, 8), 'Rh': (5, 9), 'Pd': (5, 10), 'Ag': (5, 11), 'Cd': (5, 12), 'In': (5, 13), 'Sn': (5, 14), 'Sb': (5, 15), 'Te': (5, 16), 'I': (5, 17), 'Xe': (5, 18),
            'Cs': (6, 1), 'Ba': (6, 2), 'La': (6, 3), 'Ce': (7, 3), 'Pr': (7, 4), 'Nd': (7, 5), 'Pm': (7, 6), 'Sm': (7, 7), 'Eu': (7, 8), 'Gd': (7, 9), 'Tb': (7, 10), 'Dy': (7, 11), 'Ho': (7, 12), 'Er': (7, 13), 'Tm': (7, 14), 'Yb': (7, 15), 'Lu': (7, 16), 'Hf': (6, 4), 'Ta': (6, 5), 'W': (6, 6), 'Re': (6, 7), 'Os': (6, 8), 'Ir': (6, 9), 'Pt': (6, 10), 'Au': (6, 11), 'Hg': (6, 12), 'Tl': (6, 13), 'Pb': (6, 14), 'Bi': (6, 15)
        }
        elements_to_plot = all_elements if show_all_elements else [e for e in all_elements if e in available_elements or e in selected_elements]
        fig = go.Figure()
        for element in elements_to_plot:
            if element in periodic_table_positions:
                row, col = periodic_table_positions[element]
                color = element_color_map.get(element, '#D3D3D3') if element in available_elements else '#D3D3D3'
                opacity = 1.0 if element in selected_elements else (0.7 if element in available_elements else 0.3)
                line_width = 4 if element in selected_elements else 2
                fig.add_trace(go.Scatter(
                    x=[col], y=[-row],
                    mode='markers+text',
                    text=[element],
                    textposition='middle center',
                    textfont=dict(size=fontsize, family='Arial'),
                    marker=dict(size=40, color=color, opacity=opacity, line=dict(width=line_width, color='black')),
                    hoverinfo='text',
                    hovertext=[f"Element: {element}<br>Electronegativity: {electronegativity.get(element, 1.0):.2f}<br>Thermoelectric Weight: {thermoelectric_weights.get(element, 1.0):.2f}"],
                    name=element,
                    showlegend=False
                ))
        title_text = 'Periodic Table: Available Elements' if not show_all_elements else 'Periodic Table: Full (Unavailable in Gray)'
        fig.update_layout(
            title=dict(text=f"{title_text} (Selected Elements with Bold Outline)", x=0.5, xanchor='center', font=dict(size=fontsize + 4, family='Arial')),
            xaxis=dict(range=[0, 19], showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(range=[-8, 0], showgrid=False, zeroline=False, showticklabels=False, title=''),
            plot_bgcolor='white', paper_bgcolor='white',
            width=900, height=450,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    st.subheader("Available Elements Only")
    fig_present = plot_periodic_table(available_elements, st.session_state.selected_elements, default_element_color_map, show_all_elements=False)
    st.plotly_chart(fig_present, use_container_width=True)

    st.subheader("Full Periodic Table")
    fig_full = plot_periodic_table(available_elements, st.session_state.selected_elements, default_element_color_map, show_all_elements=True)
    st.plotly_chart(fig_full, use_container_width=True)

    # Element selection
    st.header("Select Elements")
    st.session_state.selected_elements = st.multiselect(
        "Select up to three elements",
        options=available_elements,
        default=st.session_state.selected_elements,
        max_selections=3,
        key='element_selector'
    )

    # Update proportions and compositions
    for element in st.session_state.selected_elements:
        if element not in st.session_state.proportions:
            st.session_state.proportions[element] = 0.0
        if element not in st.session_state.compositions:
            st.session_state.compositions[element] = 0.0
    st.session_state.proportions = {k: v for k, v in st.session_state.proportions.items() if k in st.session_state.selected_elements}
    st.session_state.compositions = {k: v for k, v in st.session_state.compositions.items() if k in st.session_state.selected_elements}

    # Proportion and Composition input
    st.header("Input Proportions and View Normalized Compositions")
    if st.session_state.selected_elements:
        st.write(f"Selected Elements: {', '.join(st.session_state.selected_elements)}")
        
        st.subheader("Proportions")
        cols = st.columns(len(st.session_state.selected_elements))
        for idx, element in enumerate(st.session_state.selected_elements):
            with cols[idx]:
                st.session_state.proportions[element] = st.number_input(
                    f"Proportion for {element}", min_value=0.0, value=st.session_state.proportions.get(element, 0.0), step=0.1, key=f"prop_{element}"
                )
        
        if st.button("Normalize Proportions"):
            total = sum(st.session_state.proportions.values())
            if total > 0:
                for element in st.session_state.proportions:
                    st.session_state.compositions[element] = st.session_state.proportions[element] / total
                st.rerun()
            else:
                st.error("Please provide non-zero proportions for at least one element.")
        
        st.subheader("Normalized Compositions")
        cols = st.columns(len(st.session_state.selected_elements))
        for idx, element in enumerate(st.session_state.selected_elements):
            with cols[idx]:
                st.number_input(
                    f"Composition for {element}", min_value=0.0, max_value=1.0,
                    value=st.session_state.compositions.get(element, 0.0), step=0.1, key=f"comp_{element}", disabled=True
                )
    else:
        st.write("Please select up to three elements from the dropdown.")

    # Temperature input
    st.session_state.temperature = st.number_input("Enter Temperature (K):", min_value=0, max_value=5000, value=st.session_state.temperature, step=10)

    # Material type classification
    if st.button("Classify Material Type"):
        if len(st.session_state.selected_elements) > 0:
            elements, proportions, compositions = complete_to_three_elements(
                st.session_state.selected_elements.copy(),
                st.session_state.proportions.copy(),
                st.session_state.compositions.copy(),
                available_elements
            )
            total = sum(proportions.values())
            if total == 0:
                st.error("Please provide non-zero proportions for at least one element.")
            else:
                user_composition_dict = {elements[i]: compositions.get(elements[i], 0) for i in range(3)}
                material_type, summary_dict, verbatim_matches = extract_material_type(elements, user_composition_dict)
                st.session_state.material_type = material_type
                st.session_state.summary_dict = summary_dict
                st.session_state.verbatim_matches = verbatim_matches
                st.session_state.abstracts = fetch_arxiv_abstracts(elements, user_composition_dict)
                
                st.write("### Material Type Classification Results")
                st.write(f"**Chemical Formula**: {Composition(user_composition_dict).reduced_formula}")
                st.write(f"**SciBERT Material Type**: {material_type}")
                st.write(f"**p-type Probability**: {summary_dict['p_type_prob']:.3f}")
                st.write(f"**n-type Probability**: {summary_dict['n_type_prob']:.3f}")
                st.write("**Classification Summary**:")
                st.write(f"- Total abstracts retrieved: {summary_dict['total_abstracts']}")
                st.write(f"- Abstracts mentioning formula: {summary_dict['formula_matches']}")
                st.write(f"- p-type classifications: {summary_dict['p_type_count']}")
                st.write(f"- n-type classifications: {summary_dict['n_type_count']}")
                st.write(f"- Neutral classifications: {summary_dict['neutral_count']}")
                st.write(f"- Average relevance score: {np.mean(summary_dict['relevance_scores']):.3f}" if summary_dict['relevance_scores'] else "- Average relevance score: N/A")
                st.write(f"- Matched thermoelectric terms: {', '.join(summary_dict['matched_terms']) if summary_dict['matched_terms'] else 'None'}")
                
                st.write("### Visualizations")
                # Histogram
                fig_hist = plot_material_type_histogram(summary_dict, font_size)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Probability bar chart
                fig_prob = plot_material_probabilities(summary_dict, font_size)
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Relevance score box plot
                if summary_dict['relevance_scores']:
                    fig_box = plot_relevance_box_plot(summary_dict, font_size)
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Term co-occurrence network
                if st.session_state.abstracts:
                    fig_network = plot_term_cooccurrence_network(st.session_state.abstracts, font_size)
                    st.plotly_chart(fig_network, use_container_width=True)
                
                st.write("**Verbatim Matches**:")
                for vm in verbatim_matches:
                    st.write(f"- **{vm['label'].capitalize()}** (Score: {vm['score']:.3f}) in [{vm['arxiv_id']}]({vm['arxiv_id']}): *{vm['snippet']}*")
                
                st.session_state.material_type = st.selectbox(
                    "Confirm or Select Material Type",
                    options=['p-type', 'n-type', 'Neutral'],
                    index=['p-type', 'n-type', 'Neutral'].index(material_type),
                    key='material_type_selector'
                )
        else:
            st.error("Please select at least one element.")

# Tab 2: Seebeck Coefficient Prediction
with tab2:
    st.header("Seebeck Coefficient Prediction")
    
    if st.button("Generate Seebeck Prediction"):
        if len(st.session_state.selected_elements) > 0:
            elements, proportions, compositions = complete_to_three_elements(
                st.session_state.selected_elements.copy(),
                st.session_state.proportions.copy(),
                st.session_state.compositions.copy(),
                available_elements
            )
            total = sum(proportions.values())
            if total == 0:
                st.error("Please provide non-zero proportions for at least one element.")
            else:
                z_mean_avg, z_mean_std, bias_vector, bias_magnitude = compute_z_mean_stats_and_bias(elements, st.session_state.temperature, available_elements, scaler, vae)
                st.write("### Latent Space Statistics (z_mean)")
                st.write(f"**Mean per dimension**: {[f'{x:.4f}' for x in z_mean_avg]}")
                st.write(f"**Std per dimension**: {[f'{x:.4f}' for x in z_mean_std]}")
                st.write(f"**Bias vector (p-type to n-type)**: {[f'{x:.4f}' for x in bias_vector]}")
                st.write(f"**Applied bias magnitude**: {bias_magnitude:.4f}")
                fig_z_mean = plot_z_mean_bar_chart(z_mean_avg, z_mean_std, font_size)
                st.plotly_chart(fig_z_mean, use_container_width=True)
                fig_z_mean_matplotlib = plot_z_mean_bar_chart_matplotlib(z_mean_avg, z_mean_std, os.path.join(script_dir, 'z_mean_bar_chart.pdf'))
                st.pyplot(fig_z_mean_matplotlib)
                st.download_button(
                    label="Download z_mean Bar Chart as PDF",
                    data=open(os.path.join(script_dir, 'z_mean_bar_chart.pdf'), 'rb').read(),
                    file_name="z_mean_bar_chart.pdf",
                    mime="application/pdf"
                )
                
                user_composition = [compositions.get(elements[i], 0) for i in range(3)]
                user_composition_dict = {elements[i]: user_composition[i] for i in range(3)}
                sign_bias = st.session_state.material_type if st.session_state.material_type != 'Neutral' else None
                user_seebeck, user_seebeck_unbiased = predict_seebeck(
                    user_composition_dict,
                    st.session_state.temperature,
                    available_elements,
                    scaler,
                    vae,
                    regressor,
                    y_scaler,
                    sign_bias=sign_bias,
                    bias_vector=bias_vector,
                    bias_magnitude=bias_magnitude
                )
                if user_seebeck is None:
                    st.warning("Failed to predict Seebeck coefficient with sign bias, using unbiased prediction.")
                    user_seebeck, user_seebeck_unbiased = predict_seebeck(
                        user_composition_dict,
                        st.session_state.temperature,
                        available_elements,
                        scaler,
                        vae,
                        regressor,
                        y_scaler,
                        sign_bias=None,
                        bias_vector=None,
                        bias_magnitude=bias_magnitude
                    )
                if user_seebeck is None:
                    st.error("Failed to predict Seebeck coefficient even without bias. Please check inputs or model files.")
                    user_seebeck = 0.0
                    user_seebeck_unbiased = 0.0
                
                st.write("### Composition and Seebeck Coefficient")
                st.write(f"**User Composition**: {elements[0]}: {user_composition[0]:.2f}, {elements[1]}: {user_composition[1]:.2f}, {elements[2]}: {user_composition[2]:.2f}")
                st.write(f"**Chemical Formula**: {Composition(user_composition_dict).reduced_formula}")
                st.write(f"**Selected Material Type**: {st.session_state.material_type}")
                st.write(f"**User |Seebeck Coefficient| (Biased)**: {abs(user_seebeck):.2f} μV/K")
                st.write(f"**User Signed Seebeck Coefficient (Biased)**: {user_seebeck:.2f} μV/K ({'p-type' if user_seebeck > 0 else 'n-type' if user_seebeck < 0 else 'neutral'})")
                st.write(f"**User |Seebeck Coefficient| (Unbiased)**: {abs(user_seebeck_unbiased):.2f} μV/K")
                st.write(f"**User Signed Seebeck Coefficient (Unbiased)**: {user_seebeck_unbiased:.2f} μV/K ({'p-type' if user_seebeck_unbiased > 0 else 'n-type' if user_seebeck_unbiased < 0 else 'neutral'})")
                
                try:
                    compositions_array, seebeck_values = generate_ternary_data(
                        vae, regressor, scaler, y_scaler, elements, st.session_state.temperature, available_elements, sign_bias=sign_bias, bias_vector=bias_vector, bias_magnitude=bias_magnitude
                    )
                except Exception as e:
                    st.error(f"Failed to generate ternary data due to computation error: {e}")
                    compositions_array, seebeck_values = [], []
                
                if len(compositions_array) == 0:
                    st.error("No valid ternary data generated. Using user composition as fallback.")
                    max_comp, max_seebeck_abs, max_seebeck_signed = user_composition, abs(user_seebeck), user_seebeck
                else:
                    ternary_df = pd.DataFrame(compositions_array, columns=[elements[0], elements[1], elements[2]])
                    ternary_df['|Seebeck| (μV/K)'] = seebeck_values
                    max_row = ternary_df.loc[ternary_df['|Seebeck| (μV/K)'].idxmax()]
                    max_comp = [max_row[elements[0]], max_row[elements[1]], max_row[elements[2]]]
                    max_seebeck_abs = max_row['|Seebeck| (μV/K)']
                    max_seebeck_signed, _ = predict_seebeck(
                        {elements[i]: max_comp[i] for i in range(3)},
                        st.session_state.temperature,
                        available_elements,
                        scaler,
                        vae,
                        regressor,
                        y_scaler,
                        sign_bias=sign_bias,
                        bias_vector=bias_vector,
                        bias_magnitude=bias_magnitude
                    )
                    if max_seebeck_signed is None:
                        max_seebeck_signed, _ = predict_seebeck(
                            {elements[i]: max_comp[i] for i in range(3)},
                            st.session_state.temperature,
                            available_elements,
                            scaler,
                            vae,
                            regressor,
                            y_scaler,
                            sign_bias=None,
                            bias_vector=None,
                            bias_magnitude=bias_magnitude
                        )
                        if max_seebeck_signed is None:
                            max_seebeck_signed = user_seebeck
                            max_comp = user_composition
                            max_seebeck_abs = abs(user_seebeck)
                
                st.write(f"**Maximum |Seebeck| Composition**: {elements[0]}: {max_comp[0]:.2f}, {elements[1]}: {max_comp[1]:.2f}, {elements[2]}: {max_comp[2]:.2f}")
                st.write(f"**Maximum |Seebeck Coefficient|**: {max_seebeck_abs:.2f} μV/K")
                st.write(f"**Maximum Signed Seebeck Coefficient**: {max_seebeck_signed:.2f} μV/K ({'p-type' if max_seebeck_signed > 0 else 'n-type' if max_seebeck_signed < 0 else 'neutral'})")
                
                st.write("### Ternary Diagram")
                fig_ternary = plot_ternary_diagram(
                    compositions_array, seebeck_values, elements, user_composition, user_seebeck,
                    max_comp, max_seebeck_abs, color_scale, font_size, axes_line_width, point_size,
                    axes_box_thickness, legend_spacing, user_point_color, max_point_color,
                    ternary_grid_color, ternary_axes_color
                )
                if fig_ternary:
                    st.plotly_chart(fig_ternary, use_container_width=True)
                    try:
                        fig_ternary.write_html(os.path.join(script_dir, 'ternary_diagram.html'))
                    except Exception as e:
                        st.warning(f"Failed to save ternary diagram: {e}")
                    ternary_df = pd.DataFrame(compositions_array, columns=[elements[0], elements[1], elements[2]])
                    ternary_df['|Seebeck| (μV/K)'] = seebeck_values
                    csv = ternary_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Ternary Data as CSV",
                        data=csv,
                        file_name="ternary_data.csv",
                        mime="text/csv"
                    )
                
                st.write("### |Seebeck Coefficient| vs Temperature")
                fig_temp, temps, user_seebeck_vals, max_seebeck_vals = plot_temperature_variance(
                    elements, user_composition, max_comp, [100, 1000], available_elements,
                    scaler, vae, regressor, y_scaler, sign_bias, bias_vector, bias_magnitude, font_size, axes_line_width, grid_width,
                    user_point_color, max_point_color, point_size, axes_box_thickness
                )
                if fig_temp:
                    st.plotly_chart(fig_temp, use_container_width=True)
                    try:
                        fig_temp.write_html(os.path.join(script_dir, 'temperature_variance.html'))
                    except Exception as e:
                        st.warning(f"Failed to save temperature variance plot: {e}")
                    temp_df = pd.DataFrame({
                        'Temperature (K)': temps,
                        'User |Seebeck| (μV/K)': user_seebeck_vals,
                        'Max |Seebeck| (μV/K)': max_seebeck_vals
                    })
                    csv = temp_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Temperature Variance Data as CSV",
                        data=csv,
                        file_name="temperature_variance_data.csv",
                        mime="text/csv"
                    )
        else:
            st.error("Please select at least one element.")

# Complete to three elements if fewer are selected
def complete_to_three_elements(selected_elements, proportions, compositions, available_elements):
    while len(selected_elements) < 3:
        remaining_elements = [e for e in available_elements if e in ['Ag', 'Bi', 'Te'] and e not in selected_elements]
        if not remaining_elements:
            remaining_elements = [e for e in available_elements if e not in selected_elements]
        if remaining_elements:
            random_element = np.random.choice(remaining_elements)
            selected_elements.append(random_element)
            proportions[random_element] = 0.0
            compositions[random_element] = 0.0
        else:
            st.error("Not enough available elements to complete the ternary composition.")
            return selected_elements, proportions, compositions
    return selected_elements, proportions, compositions

# Generate ternary data with caching
@st.cache_resource
def generate_ternary_data(_vae, _regressor, _scaler, _y_scaler, elements, temperature, available_elements, sign_bias, bias_vector, bias_magnitude, steps=30):
    compositions = []
    seebeck_values = []
    for a in np.linspace(0, 1, steps):
        for b in np.linspace(0, 1 - a, steps):
            c = 1 - a - b
            if c >= 0:
                comp_dict = {elements[0]: a, elements[1]: b, elements[2]: c}
                seebeck, _ = predict_seebeck(comp_dict, temperature, available_elements, _scaler, _vae, _regressor, _y_scaler, sign_bias=sign_bias, bias_vector=bias_vector, bias_magnitude=bias_magnitude)
                if seebeck is not None:
                    compositions.append([a, b, c])
                    seebeck_values.append(abs(seebeck))
                else:
                    logger.warning(f"Prediction failed for composition {comp_dict}, skipping.")
    if not compositions:
        logger.error("No valid compositions generated.")
    return np.array(compositions), np.array(seebeck_values)

def plot_ternary_diagram(compositions, seebeck_values, elements, user_composition, user_seebeck, max_comp, max_seebeck, color_scale, font_size, axes_line_width, point_size, axes_box_thickness, legend_spacing, user_point_color, max_point_color, ternary_grid_color, ternary_axes_color):
    if len(compositions) == 0:
        st.warning("No valid ternary data to plot. Returning empty figure.")
        return None

    try:
        ternary_df = pd.DataFrame(compositions, columns=[elements[0], elements[1], elements[2]])
        ternary_df['|Seebeck| (μV/K)'] = seebeck_values

        fig = go.Figure()
        fig.add_trace(go.Scatterternary(
            a=ternary_df[elements[0]],
            b=ternary_df[elements[1]],
            c=ternary_df[elements[2]],
            mode='markers',
            marker=dict(
                size=point_size,
                color=ternary_df['|Seebeck| (μV/K)'],
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(
                    title='|Seebeck| (μV/K)',
                    titleside='right',
                    titlefont=dict(size=font_size, family='Arial'),
                    tickfont=dict(size=font_size - 2, family='Arial')
                ),
                opacity=0.7
            ),
            text=ternary_df['|Seebeck| (μV/K)'].apply(lambda x: f'{x:.2f} μV/K'),
            hoverinfo='text'
        ))

        # User composition point
        fig.add_trace(go.Scatterternary(
            a=[user_composition[0]],
            b=[user_composition[1]],
            c=[user_composition[2]],
            mode='markers+text',
            marker=dict(size=point_size * 1.5, color=user_point_color, symbol='circle', line=dict(width=2, color='black')),
            text=['User'],
            textposition='top center',
            textfont=dict(size=font_size - 2, family='Arial'),
            hoverinfo='text',
            hovertext=[f'User |Seebeck|: {abs(user_seebeck):.2f} μV/K'],
            name='User Composition'
        ))

        # Maximum Seebeck point
        fig.add_trace(go.Scatterternary(
            a=[max_comp[0]],
            b=[max_comp[1]],
            c=[max_comp[2]],
            mode='markers+text',
            marker=dict(size=point_size * 1.5, color=max_point_color, symbol='diamond', line=dict(width=2, color='black')),
            text=['Max'],
            textposition='bottom center',
            textfont=dict(size=font_size - 2, family='Arial'),
            hoverinfo='text',
            hovertext=[f'Max |Seebeck|: {max_seebeck:.2f} μV/K'],
            name='Max |Seebeck|'
        ))

        fig.update_layout(
            title=dict(
                text=f'Ternary Diagram of |Seebeck Coefficient| at {st.session_state.temperature} K',
                x=0.5, xanchor='center',
                font=dict(size=font_size + 4, family='Arial')
            ),
            ternary=dict(
                sum=1,
                aaxis=dict(
                    title=elements[0],
                    titlefont=dict(size=font_size, family='Arial'),
                    tickfont=dict(size=font_size - 2, family='Arial'),
                    gridcolor=ternary_grid_color,
                    linecolor=ternary_axes_color,
                    linewidth=axes_line_width,
                    ticks='outside',
                    ticklen=5
                ),
                baxis=dict
