import streamlit as st
import torch
import numpy as np
from pymatgen.core.composition import Composition
import plotly.graph_objects as go
import networkx as nx
from itertools import combinations
import logging
from transformers import AutoTokenizer, AutoModel
import arxiv
from retrying import retry
import re
import spacy
from spacy.matcher import Matcher
from collections import Counter
from math import log2

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("arxiv").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

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

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    logger.warning(f"Failed to load 'en_core_web_lg': {e}. Attempting 'en_core_web_sm'.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e2:
        st.error(f"Failed to load spaCy model: {e2}. Please install with: `python -m spacy download en_core_web_sm`")
        logger.error(f"spaCy loading error: {e2}")
        st.stop()

# Initialize spaCy matcher
try:
    matcher = Matcher(nlp.vocab)
    material_patterns = [
        [{"LOWER": "p-type"}, {"LOWER": {"IN": ["material", "semiconductor", "thermoelectric", "compound"]}, "OP": "?"}],
        [{"LOWER": "n-type"}, {"LOWER": {"IN": ["material", "semiconductor", "thermoelectric", "compound"]}, "OP": "?"}]
    ]
    matcher.add("P_TYPE", [material_patterns[0]])
    matcher.add("N_TYPE", [material_patterns[1]])
except Exception as e:
    st.error(f"Failed to initialize spaCy matcher: {str(e)}")
    logger.error(f"spaCy matcher error: {str(e)}")
    st.stop()

# PMI calculation
def calculate_pmi(abstracts, term1, term2, min_count=1):
    try:
        total_abstracts = len(abstracts)
        if total_abstracts == 0:
            return 0.0
        count_term1 = sum(1 for a in abstracts if term1.lower() in a['abstract'].lower())
        count_term2 = sum(1 for a in abstracts if term2.lower() in a['abstract'].lower())
        count_both = sum(1 for a in abstracts if term1.lower() in a['abstract'].lower() and term2.lower() in a['abstract'].lower())
        if count_term1 < min_count or count_term2 < min_count or count_both == 0:
            return 0.0
        p_term1 = count_term1 / total_abstracts
        p_term2 = count_term2 / total_abstracts
        p_both = count_both / total_abstracts
        pmi = log2(p_both / (p_term1 * p_term2 + 1e-6))
        return max(pmi, 0.0)  # Ensure non-negative PMI
    except Exception as e:
        logger.error(f"PMI calculation error: {str(e)}")
        return 0.0

# Fetch arXiv abstracts with customizable keywords and year range
@st.cache_data(hash_funcs={dict: lambda x: tuple(sorted(x.items()))})
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_arxiv_abstracts(elements, composition_dict, custom_keywords, year_range):
    try:
        comp = Composition({el: composition_dict.get(el, 0) for el in elements})
        formula = comp.reduced_formula
        query_terms = [formula] + custom_keywords + UNIT_VARIANTS
        for key, synonyms in THERMOELECTRIC_SYNONYMS.items():
            query_terms.extend(synonyms)
        query = f"{formula} ({' OR '.join(query_terms)})"
        logger.info(f"Querying arXiv with: {query}")
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=50, sort_by=arxiv.SortCriterion.Relevance)
        results = client.results(search)
        abstracts = [
            {
                "title": r.title,
                "abstract": r.summary,
                "arxiv_id": r.entry_id,
                "published": r.published,
                "authors": ", ".join([author.name for author in r.authors])
            }
            for r in results if year_range[0] <= r.published.year <= year_range[1]
        ]
        logger.info(f"Retrieved {len(abstracts)} abstracts for {formula}")
        return abstracts
    except Exception as e:
        st.error(f"Failed to fetch arXiv abstracts: {str(e)}")
        logger.error(f"arXiv fetch error: {str(e)}")
        return []

# Score abstracts with SciBERT
@st.cache_data
def score_abstract_with_scibert(abstract, formula, custom_keywords):
    try:
        global scibert_tokenizer, scibert_model
        if 'scibert_tokenizer' not in globals() or 'scibert_model' not in globals():
            scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            scibert_model.eval()
            scibert_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        inputs = scibert_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        attentions = outputs.attentions[-1][0].mean(dim=0).numpy()
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        relevance_score = 0.0
        matched_terms = []
        formula_lower = formula.lower()
        abstract_lower = abstract.lower()
        
        # Calculate base relevance based on attention weights
        for i, token in enumerate(tokens):
            token_lower = token.lower().replace("##", "")
            token_attention = attentions[i].mean() if i < len(attentions) else 0.0
            
            # Check for formula match
            if formula_lower in token_lower or token_lower in formula_lower:
                relevance_score += 3.0 * token_attention
                matched_terms.append(formula)
            
            # Check for custom keywords
            for keyword in custom_keywords:
                if keyword.lower() in token_lower or token_lower in keyword.lower():
                    relevance_score += 1.5 * token_attention
                    matched_terms.append(keyword)
            
            # Check for thermoelectric terms and synonyms
            for key, synonyms in THERMOELECTRIC_SYNONYMS.items():
                if any(syn.lower() in token_lower or token_lower in syn.lower() for syn in [key] + synonyms):
                    weight = TERM_WEIGHTS.get(key, 1.0)
                    relevance_score += weight * token_attention
                    matched_terms.append(key)
            
            # Check for unit variants
            for unit in UNIT_VARIANTS:
                if unit.lower() in token_lower or token_lower in unit.lower():
                    relevance_score += 1.0 * token_attention
                    matched_terms.append(unit)
        
        # Normalize relevance score
        relevance_score = min(relevance_score / (len(tokens) + 1e-6), 1.0)
        matched_terms = list(set(matched_terms))  # Remove duplicates
        return relevance_score, matched_terms
    except Exception as e:
        logger.error(f"SciBERT scoring error: {str(e)}")
        return 0.0, []

# Extract material type with PMI analysis
def extract_material_type(elements, composition_dict, custom_keywords, year_range, pmi_threshold):
    try:
        comp = Composition({el: composition_dict.get(el, 0) for el in elements})
        formula = comp.reduced_formula
        abstracts = fetch_arxiv_abstracts(elements, composition_dict, custom_keywords, year_range)
        
        if not abstracts:
            logger.warning("No abstracts retrieved, returning neutral type.")
            return "Neutral", {"p_type_prob": 0.5, "n_type_prob": 0.5, "total_abstracts": 0, "formula_matches": 0, "matched_terms": [], "pmi_scores": {}}, []
        
        p_type_scores = []
        n_type_scores = []
        verbatim_matches = []
        matched_terms_counter = Counter()
        formula_matches = 0
        
        for abstract_data in abstracts:
            abstract = abstract_data["abstract"]
            doc = nlp(abstract)
            matches = matcher(doc)
            p_type_count = 0
            n_type_count = 0
            for match_id, start, end in matches:
                rule_id = nlp.vocab.strings[match_id]
                if rule_id == "P_TYPE":
                    p_type_count += 1
                elif rule_id == "N_TYPE":
                    n_type_count += 1
            
            # SciBERT scoring
            relevance_score, matched_terms = score_abstract_with_scibert(abstract, formula, custom_keywords)
            matched_terms_counter.update(matched_terms)
            
            # Check for formula in abstract
            if re.search(r'\b' + re.escape(formula) + r'\b', abstract, re.IGNORECASE):
                formula_matches += 1
                snippet = " ".join([token.text for token in doc[max(0, start-10):min(len(doc), end+10)]])
                verbatim_matches.append({
                    "title": abstract_data["title"],
                    "arxiv_id": abstract_data["arxiv_id"],
                    "snippet": snippet,
                    "label": "p-type" if p_type_count > n_type_count else "n-type" if n_type_count > p_type_count else "neutral",
                    "score": relevance_score
                })
            
            # Weight scores by relevance
            if p_type_count > n_type_count:
                p_type_scores.append(relevance_score)
                n_type_scores.append(0.0)
            elif n_type_count > p_type_count:
                n_type_scores.append(relevance_score)
                p_type_scores.append(0.0)
            else:
                p_type_scores.append(relevance_score * 0.5)
                n_type_scores.append(relevance_score * 0.5)
        
        # Compute probabilities
        total_score = sum(p_type_scores) + sum(n_type_scores) + 1e-6
        p_type_prob = sum(p_type_scores) / total_score if total_score > 0 else 0.5
        n_type_prob = sum(n_type_scores) / total_score if total_score > 0 else 0.5
        
        # PMI analysis for key terms
        all_terms = list(matched_terms_counter.keys()) + ["p-type", "n-type"] + custom_keywords
        pmi_scores = {}
        for term1, term2 in combinations(all_terms, 2):
            pmi = calculate_pmi(abstracts, term1, term2, min_count=2)
            if pmi >= pmi_threshold:
                pmi_scores[f"{term1} - {term2}"] = pmi
        
        # Determine material type
        material_type = "p-type" if p_type_prob > n_type_prob + 0.1 else "n-type" if n_type_prob > p_type_prob + 0.1 else "Neutral"
        
        summary_dict = {
            "p_type_prob": p_type_prob,
            "n_type_prob": n_type_prob,
            "total_abstracts": len(abstracts),
            "formula_matches": formula_matches,
            "matched_terms": list(matched_terms_counter.keys()),
            "pmi_scores": pmi_scores
        }
        
        return material_type, summary_dict, verbatim_matches
    except Exception as e:
        logger.error(f"Material type extraction error: {str(e)}")
        return "Neutral", {"p_type_prob": 0.5, "n_type_prob": 0.5, "total_abstracts": 0, "formula_matches": 0, "matched_terms": [], "pmi_scores": {}}, []

# Plot material type histogram
def plot_material_type_histogram(summary_dict, font_size):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['p-type', 'n-type'],
        y=[summary_dict['p_type_prob'], summary_dict['n_type_prob']],
        marker=dict(color=['#1f77b4', '#ff7f0e'], opacity=0.7),
        text=[f"{summary_dict['p_type_prob']:.3f}", f"{summary_dict['n_type_prob']:.3f}"],
        textposition='auto'
    ))
    fig.update_layout(
        title=dict(text='Material Type Probabilities', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Material Type',
        yaxis_title='Probability',
        xaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        template='seaborn'
    )
    return fig

# Plot material type probabilities
def plot_material_probabilities(summary_dict, font_size):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=['p-type', 'n-type'],
        y=[summary_dict['p_type_prob'], summary_dict['n_type_prob']],
        mode='markers+lines',
        marker=dict(size=15, color=['#1f77b4', '#ff7f0e']),
        line=dict(width=2),
        text=[f"{summary_dict['p_type_prob']:.3f}", f"{summary_dict['n_type_prob']:.3f}"],
        textposition='top center'
    ))
    fig.update_layout(
        title=dict(text='Material Type Probability Trend', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Material Type',
        yaxis_title='Probability',
        xaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size)), range=[0, 1]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        template='seaborn'
    )
    return fig

# Plot relevance box plot
def plot_relevance_box_plot(summary_dict, font_size):
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=[summary_dict['p_type_prob'], summary_dict['n_type_prob']],
        name='Material Type Scores',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker=dict(color='#1f77b4'),
        line=dict(color='#1f77b4')
    ))
    fig.update_layout(
        title=dict(text='Relevance Score Distribution', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        yaxis_title='Score',
        xaxis=dict(showticklabels=False),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        template='seaborn'
    )
    return fig

# Plot PMI network
def plot_pmi_network(summary_dict, font_size):
    G = nx.Graph()
    for term_pair, pmi in summary_dict['pmi_scores'].items():
        term1, term2 = term_pair.split(" - ")
        G.add_edge(term1, term2, weight=pmi)
    
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
    
    # Normalize edge weights for visualization
    max_weight = max(edge_weights, default=1.0)
    edge_widths = [2 + 5 * (w / max_weight) for w in edge_weights]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=edge_widths, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(size=20, color='#1f77b4', line=dict(width=2, color='black')),
        hoverinfo='text',
        textfont=dict(size=font_size, family='Arial')
    ))
    
    fig.update_layout(
        title=dict(text='PMI Term Network', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        template='seaborn'
    )
    return fig
