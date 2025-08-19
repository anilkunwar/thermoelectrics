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
    "seebeck coefficient": ["thermopower", "seebeck", "thermoelectric voltage", "seebeck effect"],
    "power factor": ["power factor", "thermoelectric power factor"],
    "zt": ["figure of merit", "zt", "thermoelectric figure of merit", "dimensionless figure of merit"],
    "thermoelectric": ["thermoelectric", "thermoelectric material", "thermoelectrics"],
    "p-type": ["p-type", "p-type semiconductor", "p-type material", "positive seebeck"],
    "n-type": ["n-type", "n-type semiconductor", "n-type material", "negative seebeck"]
}

TERM_WEIGHTS = {
    "seebeck coefficient": 2.5, "thermopower": 2.5, "seebeck": 2.0, "seebeck effect": 2.0,
    "power factor": 2.0, "zt": 2.5, "figure of merit": 2.5,
    "thermoelectric": 1.5, "thermoelectric material": 1.5,
    "p-type": 2.0, "n-type": 2.0, "p-type semiconductor": 2.0, "n-type semiconductor": 2.0,
    "positive seebeck": 2.0, "negative seebeck": 2.0
}

# Common thermoelectric material formulas
THERMOELECTRIC_MATERIALS = [
    "Bi2Te3", "PbTe", "Mg2Si", "SnSe", "Sb2Te3", "CoSb3", "Mg2Sn", "SiGe", "FeSi2",
    "Bi2Se3", "Ca3Co4O9", "NaCo2O4", "Zn4Sb3", "Yb14MnSb11", "Mg2Ge", "SrTiO3",
    "La3Te4", "Ba8Ga16Ge30", "CsBi4Te6", "AgSbTe2", "Cu2Se", "SnTe", "PbSe", "PbS",
    "BiCuSeO", "CaMnO3", "In2Se3", "Ag2Te", "CuCrO2", "NiO", "ZnO", "TiO2"
]

UNIT_VARIANTS = [
    r"\b[mμ]?V\s*/\s*K\b", r"\bmicrovolt[s]?\s*/\s*Kelvin\b", r"\bmicrovolt[s]?\s*per\s*Kelvin\b",
    r"\bmV\s*per\s*Kelvin\b", r"\bV\s*/\s*K\b", r"\bvolt[s]?\s*/\s*Kelvin\b"
]

# Regex patterns
SEEBECK_VALUE_PATTERN = r'([-+]?\d*\.?\d+)\s*(?:μV/K|mV/K|V/K|microvolt[s]?/Kelvin|microvolt[s]? per Kelvin|volt[s]?/Kelvin)'
MATERIAL_FORMULA_PATTERN = r'\b(?:' + '|'.join([re.escape(mat) for mat in THERMOELECTRIC_MATERIALS]) + r'|[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b'
P_TYPE_PATTERN = r'\b(?:p-type|positive\s+seebeck)\b'
N_TYPE_PATTERN = r'\b(?:n-type|negative\s+seebeck)\b'

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
        [{"LOWER": {"IN": ["p-type", "positive"]}}, {"LOWER": {"IN": ["material", "semiconductor", "thermoelectric", "compound", "seebeck"]}, "OP": "?"}],
        [{"LOWER": {"IN": ["n-type", "negative"]}}, {"LOWER": {"IN": ["material", "semiconductor", "thermoelectric", "compound", "seebeck"]}, "OP": "?"}]
    ]
    matcher.add("P_TYPE", [material_patterns[0]])
    matcher.add("N_TYPE", [material_patterns[1]])
except Exception as e:
    st.error(f"Failed to initialize spaCy matcher: {str(e)}")
    logger.error(f"spaCy matcher error: {str(e)}")
    st.stop()

# Reference sentence for syntactic similarity
REFERENCE_SENTENCE = "This material exhibits a high Seebeck coefficient and is suitable for thermoelectric applications."

# PMI calculation
def calculate_pmi(abstracts, term1, term2, min_count=1):
    try:
        total_abstracts = len(abstracts)
        if total_abstracts == 0:
            return 0.0
        count_term1 = sum(1 for a in abstracts if re.search(r'\b' + re.escape(term1) + r'\b', a['abstract'], re.IGNORECASE))
        count_term2 = sum(1 for a in abstracts if re.search(r'\b' + re.escape(term2) + r'\b', a['abstract'], re.IGNORECASE))
        count_both = sum(1 for a in abstracts if re.search(r'\b' + re.escape(term1) + r'\b', a['abstract'], re.IGNORECASE) and 
                        re.search(r'\b' + re.escape(term2) + r'\b', a['abstract'], re.IGNORECASE))
        if count_term1 < min_count or count_term2 < min_count or count_both == 0:
            return 0.0
        p_term1 = count_term1 / total_abstracts
        p_term2 = count_term2 / total_abstracts
        p_both = count_both / total_abstracts
        pmi = log2(p_both / (p_term1 * p_term2 + 1e-6))
        return max(pmi, 0.0)
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
        query_terms = [formula] + custom_keywords + THERMOELECTRIC_MATERIALS + UNIT_VARIANTS
        for key, synonyms in THERMOELECTRIC_SYNONYMS.items():
            query_terms.extend(synonyms)
        query = ' '.join([f'"{term}"' if ' ' in term else term for term in query_terms])
        logger.info(f"Querying arXiv with: {query}")
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=100, sort_by=arxiv.SortCriterion.Relevance)
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

# Score abstracts with SciBERT at sentence level
@st.cache_data
def score_abstract_with_scibert(abstract, formula, custom_keywords):
    try:
        global scibert_tokenizer, scibert_model
        if 'scibert_tokenizer' not in globals() or 'scibert_model' not in globals():
            scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            scibert_model.eval()
            scibert_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Sentence-level tokenization
        doc = nlp(abstract)
        sentences = [sent.text for sent in doc.sents]
        ref_doc = nlp(REFERENCE_SENTENCE)
        
        relevance_scores = []
        matched_terms = []
        matched_formulas = []
        seebeck_values = []
        
        for sentence in sentences:
            # Syntactic similarity
            sent_doc = nlp(sentence)
            similarity = sent_doc.similarity(ref_doc) if sent_doc.has_vector and ref_doc.has_vector else 0.0
            
            # Tokenize sentence for SciBERT
            inputs = scibert_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = scibert_model(**inputs, output_attentions=True)
            attentions = outputs.attentions[-1][0].mean(dim=0).numpy()
            tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            sentence_score = 0.0
            sentence_terms = []
            sentence_formulas = []
            
            # Regex for Seebeck values
            seebeck_matches = re.finditer(SEEBECK_VALUE_PATTERN, sentence, re.IGNORECASE)
            for match in seebeck_matches:
                value = float(match.group(1))
                unit = match.group(0).split()[1]
                # Convert to μV/K
                if 'mV/K' in unit:
                    value *= 1000
                elif 'V/K' in unit:
                    value *= 1e6
                seebeck_values.append(value)
                sentence_terms.append(match.group(0))
            
            # Regex for chemical formulas
            formula_matches = re.finditer(MATERIAL_FORMULA_PATTERN, sentence, re.IGNORECASE)
            for match in formula_matches:
                matched_formula = match.group(0)
                try:
                    Composition(matched_formula)  # Validate formula
                    sentence_formulas.append(matched_formula)
                    sentence_terms.append(matched_formula)
                except:
                    continue
            
            # SciBERT token scoring
            for i, token in enumerate(tokens):
                token_lower = token.lower().replace("##", "")
                token_attention = attentions[i].mean() if i < len(attentions) else 0.0
                
                # Formula match
                if formula.lower() in token_lower or token_lower in formula.lower():
                    sentence_score += 3.0 * token_attention
                    sentence_terms.append(formula)
                
                # Custom keywords
                for keyword in custom_keywords:
                    if keyword.lower() in token_lower or token_lower in keyword.lower():
                        sentence_score += 1.5 * token_attention
                        sentence_terms.append(keyword)
                
                # Thermoelectric terms
                for key, synonyms in THERMOELECTRIC_SYNONYMS.items():
                    if any(syn.lower() in token_lower or token_lower in syn.lower() for syn in [key] + synonyms):
                        weight = TERM_WEIGHTS.get(key, 1.0)
                        sentence_score += weight * token_attention
                        sentence_terms.append(key)
                
                # Unit variants
                for unit in UNIT_VARIANTS:
                    if re.search(unit, token_lower, re.IGNORECASE):
                        sentence_score += 1.0 * token_attention
                        sentence_terms.append(unit)
            
            # Weight by syntactic similarity
            sentence_score *= (similarity + 0.5)  # Boost by similarity (0 to 1.5)
            relevance_scores.append(sentence_score)
            matched_terms.extend(sentence_terms)
            matched_formulas.extend(sentence_formulas)
        
        # Aggregate scores
        relevance_score = min(sum(relevance_scores) / (len(sentences) + 1e-6), 1.0)
        matched_terms = list(set(matched_terms))
        matched_formulas = list(set(matched_formulas))
        return relevance_score, matched_terms, matched_formulas, seebeck_values
    except Exception as e:
        logger.error(f"SciBERT scoring error: {str(e)}")
        return 0.0, [], [], []

# Extract material type with PMI analysis
def extract_material_type(elements, composition_dict, custom_keywords, year_range, pmi_threshold):
    try:
        comp = Composition({el: composition_dict.get(el, 0) for el in elements})
        formula = comp.reduced_formula
        abstracts = fetch_arxiv_abstracts(elements, composition_dict, custom_keywords, year_range)
        
        if not abstracts:
            logger.warning("No abstracts retrieved, returning neutral type.")
            return "Neutral", {"p_type_prob": 0.5, "n_type_prob": 0.5, "total_abstracts": 0, "formula_matches": 0, "matched_terms": [], "pmi_scores": {}, "seebeck_values": []}, []
        
        p_type_scores = []
        n_type_scores = []
        verbatim_matches = []
        matched_terms_counter = Counter()
        matched_formulas = []
        seebeck_values = []
        formula_matches = 0
        
        for abstract_data in abstracts:
            abstract = abstract_data["abstract"]
            doc = nlp(abstract)
            matches = matcher(doc)
            p_type_count = len(re.finditer(P_TYPE_PATTERN, abstract, re.IGNORECASE))
            n_type_count = len(re.finditer(N_TYPE_PATTERN, abstract, re.IGNORECASE))
            
            # SpaCy matcher for additional context
            for match_id, start, end in matches:
                rule_id = nlp.vocab.strings[match_id]
                if rule_id == "P_TYPE":
                    p_type_count += 1
                elif rule_id == "N_TYPE":
                    n_type_count += 1
            
            # SciBERT scoring at sentence level
            relevance_score, matched_terms, formulas, sent_seebeck = score_abstract_with_scibert(abstract, formula, custom_keywords)
            matched_terms_counter.update(matched_terms)
            matched_formulas.extend(formulas)
            seebeck_values.extend(sent_seebeck)
            
            # Check for formula in abstract
            formula_match = re.search(r'\b' + re.escape(formula) + r'\b', abstract, re.IGNORECASE)
            if formula_match:
                formula_matches += 1
                snippet_start = max(0, formula_match.start() - 100)
                snippet_end = min(len(abstract), formula_match.end() + 100)
                snippet = abstract[snippet_start:snippet_end]
                matched_formula = formulas[0] if formulas else formula
                verbatim_matches.append({
                    "title": abstract_data["title"],
                    "arxiv_id": abstract_data["arxiv_id"],
                    "snippet": snippet,
                    "label": "p-type" if p_type_count > n_type_count else "n-type" if n_type_count > p_type_count else "neutral",
                    "score": relevance_score,
                    "matched_formula": matched_formula
                })
            
            # Weight scores by Seebeck values
            seebeck_weight = sum(abs(v) for v in sent_seebeck) / 1000.0 if sent_seebeck else 1.0
            if p_type_count > n_type_count or sum(1 for v in sent_seebeck if v > 0) > sum(1 for v in sent_seebeck if v < 0):
                p_type_scores.append(relevance_score * seebeck_weight)
                n_type_scores.append(0.0)
            elif n_type_count > p_type_count or sum(1 for v in sent_seebeck if v < 0) > sum(1 for v in sent_seebeck if v > 0):
                n_type_scores.append(relevance_score * seebeck_weight)
                p_type_scores.append(0.0)
            else:
                p_type_scores.append(relevance_score * seebeck_weight * 0.5)
                n_type_scores.append(relevance_score * seebeck_weight * 0.5)
        
        # Compute probabilities
        total_score = sum(p_type_scores) + sum(n_type_scores) + 1e-6
        p_type_prob = sum(p_type_scores) / total_score if total_score > 0 else 0.5
        n_type_prob = sum(n_type_scores) / total_score if total_score > 0 else 0.5
        
        # PMI analysis
        all_terms = list(matched_terms_counter.keys()) + ["p-type", "n-type"] + custom_keywords + matched_formulas
        pmi_scores = {}
        for term1, term2 in combinations(set(all_terms), 2):
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
            "pmi_scores": pmi_scores,
            "seebeck_values": seebeck_values
        }
        
        return material_type, summary_dict, verbatim_matches
    except Exception as e:
        logger.error(f"Material type extraction error: {str(e)}")
        return "Neutral", {"p_type_prob": 0.5, "n_type_prob": 0.5, "total_abstracts": 0, "formula_matches": 0, "matched_terms": [], "pmi_scores": {}, "seebeck_values": []}, []

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

# Plot relevance box plot with Seebeck values
def plot_relevance_box_plot(summary_dict, font_size):
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=summary_dict['seebeck_values'],
        name='Seebeck Values (μV/K)',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker=dict(color='#1f77b4'),
        line=dict(color='#1f77b4')
    ))
    fig.update_layout(
        title=dict(text='Seebeck Coefficient Distribution', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        yaxis_title='Seebeck Coefficient (μV/K)',
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
