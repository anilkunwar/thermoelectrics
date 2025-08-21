# streamlit_app.py
import streamlit as st
import re
import spacy
from spacy.language import Language
from spacy.tokens import Span, Doc
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
def build_material_matcher(nlp, synonyms):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for canonical, variants in synonyms.items():
        patterns = [nlp.make_doc(v) for v in variants]
        matcher.add(canonical, patterns)
    return matcher

@Language.component("material_matcher")
def material_matcher(doc):
    matches = doc._.material_matcher(doc)
    spans = []
    for match_id, start, end in matches:
        canonical = doc.vocab.strings[match_id]
        span = Span(doc, start, end, label="MATERIAL_TYPE")  # FIXED
        span._.norm = canonical
        spans.append(span)
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

def load_spacy_model(synonyms):
    nlp = spacy.blank("en")

    # regex-based formula detector
    nlp.add_pipe("formula_ner", last=True)

    # material matcher
    matcher = build_material_matcher(nlp, synonyms)
    nlp.add_pipe("material_matcher", last=True)

    # attach matcher to doc
    if not Doc.has_extension("material_matcher"):
        Doc.set_extension("material_matcher", default=None)
    Doc.set_extension("material_matcher", default=matcher, force=True)

    if not Span.has_extension("norm"):
        Span.set_extension("norm", default=None)

    return nlp

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔬 Custom NER for Thermoelectric Materials")

st.sidebar.header("⚙️ Synonym Settings")

# Default synonyms
if "synonyms" not in st.session_state:
    st.session_state.synonyms = {
        "p-type": ["p-type", "positive type", "positive thermoelectric", "hole conducting"],
        "n-type": ["n-type", "negative type", "negative thermoelectric", "electron conducting"]
    }

# Allow user to add new synonyms
with st.sidebar.form("add_synonym"):
    st.write("➕ Add new synonym")
    synonym_text = st.text_input("Phrase (e.g. 'hole transport'):")
    synonym_type = st.selectbox("Maps to:", ["p-type", "n-type"])
    submitted = st.form_submit_button("Add")
    if submitted and synonym_text.strip():
        st.session_state.synonyms[synonym_type].append(synonym_text.strip())
        st.success(f"Added '{synonym_text}' → {synonym_type}")

st.sidebar.write("### Current synonyms:")
st.sidebar.json(st.session_state.synonyms)

# Main text input
text = st.text_area("Enter text:", 
    "This Bi2Te3 sample is a positive type thermoelectric material, "
    "while Sb2Te3 is n-type. Another hole transport example is Cu2Se."
)

if st.button("Analyze"):
    nlp = load_spacy_model(st.session_state.synonyms)
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

    # Show structured table
    st.subheader("📋 Extracted Entities")
    rows = []
    for ent in doc.ents:
        rows.append({
            "Text": ent.text,
            "Label": ent.label_,
            "Normalized": ent._.norm if ent._.norm else "-"
        })
    st.dataframe(rows)
