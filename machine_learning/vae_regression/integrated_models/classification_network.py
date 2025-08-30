import os
import tempfile
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from graphviz import Digraph

# ===========================
# App config
# ===========================
st.set_page_config(layout="wide", page_title="Formula Classification Explorer")
st.title("🔬 Formula Classification Explorer")

# ===========================
# Load dataset safely
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "formula_classifications_via_nlp.csv")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(CSV_PATH)

# ===========================
# Sidebar filters
# ===========================
st.sidebar.header("Filter by Category")
show_p = st.sidebar.checkbox("Show p-type", True)
show_n = st.sidebar.checkbox("Show n-type", True)

categories_to_show = []
if show_p:
    categories_to_show.append("p-type")
if show_n:
    categories_to_show.append("n-type")

df_filtered = df[df["material_type"].isin(categories_to_show)]

st.sidebar.write(f"Showing {len(df_filtered)} formulas")

# ===========================
# Build bipartite graph
# ===========================
G = nx.Graph()
for cat in df_filtered["material_type"].unique():
    G.add_node(cat, bipartite=1, color="orange")  # category hubs

for _, row in df_filtered.iterrows():
    f = row["formula"]
    t = row["material_type"]
    G.add_node(f, bipartite=0, color="skyblue")
    G.add_edge(f, t)

# ===========================
# Tabs for visualization
# ===========================
tab1, tab2, tab3 = st.tabs(["📷 NetworkX", "🌐 PyVis", "📦 Graphviz"])

# ---------------------------
# TAB 1: NetworkX static
# ---------------------------
with tab1:
    st.subheader("NetworkX (static matplotlib)")
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.3)
    node_colors = [G.nodes[n].get("color", "gray") for n in G.nodes()]
    nx.draw(
        G, pos, with_labels=True,
        node_color=node_colors, node_size=500,
        font_size=8, edge_color="gray", ax=ax
    )
    st.pyplot(fig)

# ---------------------------
# TAB 2: PyVis interactive
# ---------------------------
with tab2:
    st.subheader("PyVis (interactive)")
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    for n, data in G.nodes(data=True):
        net.add_node(n, label=n, color=data.get("color", "lightblue"), title=f"Category: {n}" if n in ["p-type", "n-type"] else None)
    for u, v in G.edges():
        net.add_edge(u, v)
    
    # Save to temporary HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_code = open(tmp_file.name, "r", encoding="utf-8").read()
        st.components.v1.html(html_code, height=600, scrolling=True)

# ---------------------------
# TAB 3: Graphviz
# ---------------------------
with tab3:
    st.subheader("Graphviz")
    dot = Digraph(comment="Formula Graph", format="svg")
    dot.attr(rankdir="LR")
    
    # Category hubs
    for cat in df_filtered["material_type"].unique():
        dot.node(cat, shape="box", style="filled", color="orange")
    
    # Formulas
    for _, row in df_filtered.iterrows():
        f = row["formula"]
        t = row["material_type"]
        dot.node(f, shape="ellipse", style="filled", color="skyblue")
        dot.edge(f, t)
    
    st.graphviz_chart(dot.source)

# ---------------------------
# Summary counts
# ---------------------------
st.write("### Summary Counts")
st.write(df_filtered["material_type"].value_counts())
