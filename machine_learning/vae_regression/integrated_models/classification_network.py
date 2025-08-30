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
st.title("🔬 Formula Classification Graph Explorer")

# ===========================
# Load CSV safely
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
# Build weighted bipartite graph
# ===========================
# Count duplicate formulas
edge_counts = df_filtered.groupby(["formula", "material_type"]).size().reset_index(name="weight")

G = nx.Graph()
# Add category hubs
for cat in df_filtered["material_type"].unique():
    G.add_node(cat, bipartite=1, color="orange")

# Add formulas and edges with weight
for _, row in edge_counts.iterrows():
    f = row["formula"]
    t = row["material_type"]
    w = row["weight"]
    G.add_node(f, bipartite=0, color="skyblue")
    G.add_edge(f, t, weight=w)

# ===========================
# Tabs
# ===========================
tab1, tab2, tab3, tab4 = st.tabs(["📷 NetworkX", "🌐 PyVis", "📦 Graphviz", "📊 Adjacency Matrix"])

# ---------------------------
# TAB 1: NetworkX weighted
# ---------------------------
with tab1:
    st.subheader("NetworkX (weighted visualization)")
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.3)

    # Node size proportional to degree
    node_sizes = [200 + 50*G.degree(n) for n in G.nodes()]
    node_colors = [G.nodes[n].get("color", "gray") for n in G.nodes()]

    # Edge width proportional to weight
    edge_widths = [G.edges[e].get("weight", 1) for e in G.edges()]

    nx.draw(
        G, pos, with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        width=edge_widths,
        font_size=8,
        edge_color="gray",
        ax=ax
    )
    st.pyplot(fig)

# ---------------------------
# TAB 2: PyVis interactive
# ---------------------------
with tab2:
    st.subheader("PyVis (interactive weighted)")
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    for n, data in G.nodes(data=True):
        net.add_node(n, label=n, color=data.get("color", "lightblue"),
                     size=200 + 10*G.degree(n),  # scale node size
                     title=f"Degree: {G.degree(n)}")

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, value=data.get("weight", 1), width=data.get("weight", 1))

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
    for cat in df_filtered["material_type"].unique():
        dot.node(cat, shape="box", style="filled", color="orange")
    for _, row in edge_counts.iterrows():
        f = row["formula"]
        t = row["material_type"]
        w = row["weight"]
        dot.node(f, shape="ellipse", style="filled", color="skyblue")
        dot.edge(f, t, label=str(w))
    st.graphviz_chart(dot.source)

# ---------------------------
# TAB 4: Adjacency matrix
# ---------------------------
with tab4:
    st.subheader("Adjacency Matrix (formulas × categories)")
    formulas = df_filtered["formula"].unique()
    categories = df_filtered["material_type"].unique()
    adj = pd.DataFrame(0, index=formulas, columns=categories)

    for _, row in edge_counts.iterrows():
        adj.loc[row["formula"], row["material_type"]] = row["weight"]

    st.dataframe(adj)
