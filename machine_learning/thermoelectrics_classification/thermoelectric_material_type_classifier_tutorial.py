# This app intelligently classifies whether a thermoelectric material is p-type or n-type
import streamlit as st
import re
import spacy
from spacy.language import Language
from spacy.tokens import Span
from spacy.util import filter_spans
from spacy.matcher import PhraseMatcher

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
MATERIAL_SYNONYMS = {
    "p-type": [
        "p-type", "positive type", "positive type thermoelectric", "hole conducting"
    ],
    "n-type": [
        "n-type", "negative type", "negative type thermoelectric", "electron conducting"
    ]
}

def build_material_matcher(nlp):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for canonical, variants in MATERIAL_SYNONYMS.items():
        patterns = [nlp.make_doc(v) for v in variants]
        matcher.add(canonical, patterns)
    return matcher

@Language.component("material_matcher")
def material_matcher(doc):
    matches = doc._.material_matcher(doc)
    spans = []
    for match_id, start, end in matches:
        canonical = doc.vocab.strings[match_id]
        span = doc[start:end].as_span(label="MATERIAL_TYPE")
        if span:
            span._.norm = canonical
            spans.append(span)
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

# -----------------------------
# Loader
# -----------------------------
def load_spacy_model():
    nlp = spacy.blank("en")

    # add regex-based formula NER
    nlp.add_pipe("formula_ner", last=True)

    # add material matcher
    matcher = build_material_matcher(nlp)
    nlp.add_pipe("material_matcher", last=True)

    # attach matcher to doc
    from spacy.tokens import Doc
    if not Doc.has_extension("material_matcher"):
        Doc.set_extension("material_matcher", default=matcher)

    # attach normalized form extension
    if not Span.has_extension("norm"):
        Span.set_extension("norm", default=None)

    return nlp

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔬 Custom NER for Thermoelectric Materials")

st.write("This demo extracts **chemical formulas** and **material types** (p-type / n-type) with normalization.")

text = st.text_area("Enter text:", "This Bi2Te3 sample is a positive type thermoelectric material, while Sb2Te3 is n-type.")

if st.button("Analyze"):
    nlp = load_spacy_model()
    doc = nlp(text)

    # Highlight entities
    colors = {"FORMULA": "#FFD700", "MATERIAL_TYPE": "#7FFFD4"}
    html = ""
    last_end = 0
    for ent in doc.ents:
        html += text[last_end:ent.start_char]
        color = colors.get(ent.label_, "#DDDDDD")
        html += f"<span style='background-color:{color}; padding:2px; border-radius:4px;'>{ent.text} ({ent.label_})</span>"
        last_end = ent.end_char
    html += text[last_end:]

    st.markdown(html, unsafe_allow_html=True)

    # Show structured output
    st.subheader("📋 Extracted Entities")
    rows = []
    for ent in doc.ents:
        rows.append({
            "Text": ent.text,
            "Label": ent.label_,
            "Normalized": ent._.norm if ent._.norm else "-"
        })
    st.dataframe(rows)
