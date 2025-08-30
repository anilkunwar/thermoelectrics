import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from graphviz import Digraph
import tempfile

# ===========================
# Load your dataset
# ===========================
st.set_page_config(layout="wide")
st.title("📊 Formula Classification Graph Explorer")

df = pd.read_csv("formula_classifications_via_nlp.csv")

# ===========================
# Build a bipartite graph
# ===========================
G = nx.Graph()
formulas = df["formula"].unique()

for _, row in df.iterrows():
    f = row["formula"]
    t = row["material_type"]
    G.add_node(f, bipartite=0)
    G.add_node(t, bipartite=1)
    G.add_edge(f, t)

# ===========================
# Tabs for visualization
# ===========================
tab1, tab2, tab3 = st.tabs(["📷 NetworkX (matplotlib)", "🌐 PyVis (interactive)", "📦 Graphviz"])

# ---------------------------
# TAB 1: NetworkX with matplotlib
# ---------------------------
with tab1:
    st.subheader("NetworkX (static matplotlib view)")
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.3)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color="skyblue",
        font_size=8,
        node_size=600,
        edge_color="gray",
        ax=ax
    )
    st.pyplot(fig)

# ---------------------------
# TAB 2: PyVis interactive
# ---------------------------
with tab2:
    st.subheader("PyVis (interactive graph)")
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)

    # Save to temp HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_code = open(tmp_file.name, "r", encoding="utf-8").read()
        st.components.v1.html(html_code, height=600, scrolling=True)

# ---------------------------
# TAB 3: Graphviz
# ---------------------------
with tab3:
    st.subheader("Graphviz (DOT view)")
    dot = Digraph(comment="Formula Graph", format="svg")
    dot.attr(rankdir="LR")

    for _, row in df.iterrows():
        f = row["formula"]
        t = row["material_type"]
        dot.node(f, f, shape="ellipse", color="lightblue", style="filled")
        dot.node(t, t, shape="box", color="orange", style="filled")
        dot.edge(f, t)

    st.graphviz_chart(dot.source)
