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
            scibert_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
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
            "total_abstract
