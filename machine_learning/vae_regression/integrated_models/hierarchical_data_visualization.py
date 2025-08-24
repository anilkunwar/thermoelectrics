import os
import pandas as pd
import plotly.express as px
import streamlit as st

# Directory setup
DB_DIR = os.path.dirname(os.path.abspath(__file__))

def create_sunburst_chart(csv_path, colormap_choice, show_labels, label_fontsize, include_year, excluded_labels):
    try:
        if not os.path.exists(csv_path):
            st.error(f"CSV file not found at {csv_path}. Please ensure 'formula_classifications_via_nlp.csv' is in the same directory.")
            return None
        
        df = pd.read_csv(csv_path)

        required_columns = ["Formula", "Material Type"]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error(f"CSV missing required columns: {', '.join(missing_cols)}")
            return None

        # Filter excluded labels
        if excluded_labels:
            df = df[~df["Material Type"].isin(excluded_labels)]
            df = df[~df["Formula"].isin(excluded_labels)]
            if "Year" in df.columns:
                df = df[~df["Year"].astype(str).isin(excluded_labels)]

        # Create hierarchy
        if include_year and 'Year' in df.columns and df["Year"].notna().any():
            sunburst_data = df.groupby(['Year', 'Formula', 'Material Type']).size().reset_index(name='count')
            path = ['Year', 'Formula', 'Material Type']
            title = "Hierarchical Distribution of Material Classifications by Year"
        else:
            sunburst_data = df.groupby(['Formula', 'Material Type']).size().reset_index(name='count')
            path = ['Formula', 'Material Type']
            title = "Hierarchical Distribution of Material Classifications (No Year Data)"

        # Build chart
        fig_sunburst = px.sunburst(
            sunburst_data,
            path=path,
            values='count',
            title=title,
            color='Material Type',
            color_continuous_scale=colormap_choice if colormap_choice else None,
            labels={
                "Year": "Year",
                "Formula": "Formula",
                "Material Type": "Material Type",
                "count": "Frequency"
            }
        )

        # Label visibility & styling
        fig_sunburst.update_traces(
            textinfo="label+percent entry+value" if show_labels else "none",
            textfont_size=label_fontsize,
        )

        fig_sunburst.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, l=25, r=25, b=25)
        )

        return fig_sunburst

    except Exception as e:
        st.error(f"Failed to generate sunburst chart: {str(e)}")
        return None


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Enhanced Sunburst Chart", layout="wide")
st.title("🌞 Enhanced Sunburst Chart for Material Classifications")

csv_path = os.path.join(DB_DIR, "formula_classifications_via_nlp.csv")

# Sidebar controls
st.sidebar.header("Chart Customization")

colormap_choice = st.sidebar.selectbox(
    "Choose Color Map",
    ["Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo", None],
    index=0
)

show_labels = st.sidebar.checkbox("Show Labels", value=True)
label_fontsize = st.sidebar.slider("Label Font Size", 8, 24, 12)
include_year = st.sidebar.checkbox("Include Year in Hierarchy (if available)", value=True)

# Exclude certain labels
excluded_labels = st.sidebar.multiselect(
    "Exclude Labels",
    options=[],
    help="Hide specific Formula, Material Type, or Year values"
)

# Generate chart
fig_sunburst = create_sunburst_chart(
    csv_path,
    colormap_choice,
    show_labels,
    label_fontsize,
    include_year,
    excluded_labels
)

if fig_sunburst:
    st.plotly_chart(fig_sunburst, use_container_width=True)
else:
    st.warning("Unable to display sunburst chart. Check error messages above.")
