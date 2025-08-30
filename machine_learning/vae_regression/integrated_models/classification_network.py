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

# Add some explanation
st.markdown("""
This application visualizes the relationships between chemical formulas and their material classifications.
Use the sidebar to filter by material type and adjust visualization parameters.
""")

# ===========================
# Load dataset safely
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "formula_classifications_via_nlp.csv")

@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        if not all(col in df.columns for col in ["formula", "material_type"]):
            st.error("CSV file must contain 'formula' and 'material_type' columns")
            return None
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at: {CSV_PATH}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data(CSV_PATH)

if df is None:
    st.stop()

# ===========================
# Sidebar filters
# ===========================
st.sidebar.header("Filter options")
show_p = st.sidebar.checkbox("Show p-type", True)
show_n = st.sidebar.checkbox("Show n-type", True)
top_n = st.sidebar.slider("Top N formulas for PyVis", 10, 500, 100)

categories_to_show = []
if show_p:
    categories_to_show.append("p-type")
if show_n:
    categories_to_show.append("n-type")

df_filtered = df[df["material_type"].isin(categories_to_show)]

st.sidebar.write(f"Showing {len(df_filtered)} formulas after category filter")

# Add some statistics
if st.sidebar.checkbox("Show statistics"):
    st.sidebar.subheader("Dataset Statistics")
    st.sidebar.write(f"Total formulas: {df['formula'].nunique()}")
    st.sidebar.write(f"Total p-type: {len(df[df['material_type'] == 'p-type'])}")
    st.sidebar.write(f"Total n-type: {len(df[df['material_type'] == 'n-type'])}")
    st.sidebar.write(f"Filtered formulas: {df_filtered['formula'].nunique()}")

# ===========================
# Build full weighted bipartite graph
# ===========================
@st.cache_resource
def build_graph(dataframe):
    edge_counts = dataframe.groupby(["formula", "material_type"]).size().reset_index(name="weight")
    G = nx.Graph()
    # Category nodes
    for cat in dataframe["material_type"].unique():
        G.add_node(cat, bipartite=1, color="orange", size=30)
    # Formula nodes and edges
    for _, row in edge_counts.iterrows():
        f = row["formula"]
        t = row["material_type"]
        w = row["weight"]
        G.add_node(f, bipartite=0, color="skyblue", size=10 + min(w * 5, 50))
        G.add_edge(f, t, weight=w, width=w)
    return G, edge_counts

G_full, edge_counts = build_graph(df_filtered)

# ===========================
# Tabs
# ===========================
tab1, tab2, tab3, tab4 = st.tabs(["📷 NetworkX", "🌐 PyVis", "📦 Graphviz", "📊 Adjacency Matrix"])

# ---------------------------
# TAB 1: NetworkX static
# ---------------------------
with tab1:
    st.subheader("NetworkX (static weighted)")
    st.markdown("This visualization shows formulas connected to their material types with edge weights representing frequency.")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G_full, seed=42, k=0.3)
    node_sizes = [G_full.nodes[n].get('size', 50) for n in G_full.nodes()]
    node_colors = [G_full.nodes[n].get("color", "gray") for n in G_full.nodes()]
    edge_widths = [G_full.edges[e].get("weight", 1) for e in G_full.edges()]

    nx.draw(
        G_full, pos, with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        width=[0.5 + w/5 for w in edge_widths],
        font_size=8,
        edge_color="gray",
        ax=ax
    )
    st.pyplot(fig)

# ---------------------------
# TAB 2: PyVis interactive (filtered subset)
# ---------------------------
with tab2:
    st.subheader(f"PyVis Interactive (Top {top_n} formulas)")
    st.markdown("Interactive visualization showing the most frequent formulas. Hover over nodes to see connections.")

    # Select top N formulas by occurrence count
    top_formulas = df_filtered.groupby("formula").size().nlargest(top_n).index
    df_pyvis = df_filtered[df_filtered["formula"].isin(top_formulas)]
    
    # Build the PyVis graph
    edge_counts_pyvis = df_pyvis.groupby(["formula", "material_type"]).size().reset_index(name="weight")
    G_pyvis = nx.Graph()
    for cat in df_pyvis["material_type"].unique():
        G_pyvis.add_node(cat, bipartite=1, color="orange", size=30)
    for _, row in edge_counts_pyvis.iterrows():
        f = row["formula"]
        t = row["material_type"]
        w = row["weight"]
        G_pyvis.add_node(f, bipartite=0, color="skyblue", size=10 + min(w * 5, 50))
        G_pyvis.add_edge(f, t, weight=w, width=w)

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.force_atlas_2based()
    
    for n, data in G_pyvis.nodes(data=True):
        net.add_node(n, label=n, color=data.get("color", "lightblue"),
                     size=data.get("size", 20),
                     title=f"Connections: {G_pyvis.degree(n)}")
    
    for u, v, data in G_pyvis.edges(data=True):
        net.add_edge(u, v, value=data.get("weight", 1), width=0.5 + data.get("weight", 1)/5)

    # Generate HTML content
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            html_code = open(tmp_file.name, "r", encoding="utf-8").read()
            st.components.v1.html(html_code, height=600, scrolling=True)
    except Exception as e:
        st.error(f"Error generating interactive visualization: {str(e)}")

# ---------------------------
# TAB 3: Graphviz
# ---------------------------
with tab3:
    st.subheader("Graphviz (full graph)")
    st.markdown("Hierarchical layout showing formulas connected to material types.")
    
    dot = Digraph(comment="Formula Graph", format="svg")
    dot.attr(rankdir="LR")
    
    # Category hubs
    for cat in df_filtered["material_type"].unique():
        dot.node(cat, shape="box", style="filled", color="orange")
    
    for _, row in edge_counts.iterrows():
        f = row["formula"]
        t = row["material_type"]
        w = row["weight"]
        dot.node(f, shape="ellipse", style="filled", color="skyblue")
        dot.edge(f, t, label=str(w), penwidth=str(0.5 + w/5))
    
    st.graphviz_chart(dot.source)

# ---------------------------
# TAB 4: Adjacency matrix
# ---------------------------
with tab4:
    st.subheader("Adjacency Matrix (formulas × categories)")
    st.markdown("Matrix view showing the connection strength between formulas and material types.")
    
    formulas = df_filtered["formula"].unique()
    categories = df_filtered["material_type"].unique()
    adj = pd.DataFrame(0, index=formulas, columns=categories)
    
    for _, row in edge_counts.iterrows():
        adj.loc[row["formula"], row["material_type"]] = row["weight"]
    
    # Style the dataframe for better visualization
    styled_adj = adj.style.background_gradient(cmap='Blues')
    st.dataframe(styled_adj)
    
    # Add download button
    csv = adj.to_csv().encode('utf-8')
    st.download_button(
        label="Download adjacency matrix as CSV",
        data=csv,
        file_name="adjacency_matrix.csv",
        mime="text/csv",
    )
