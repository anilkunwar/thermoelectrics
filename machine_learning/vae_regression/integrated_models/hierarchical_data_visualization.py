import os
import pandas as pd
import plotly.express as px
import streamlit as st

# Directory setup
DB_DIR = os.path.dirname(os.path.abspath(__file__))

def create_sunburst_chart(csv_path, colormap_choice, discrete_mode, show_labels, label_fontsize, excluded_labels):
    try:
        if not os.path.exists(csv_path):
            st.error(f"CSV file not found at {csv_path}. Please ensure it's in the same directory.")
            return None

        df = pd.read_csv(csv_path)

        required_columns = ["formula", "material_type"]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error(f"CSV missing required columns: {', '.join(missing_cols)}")
            return None

        # Filter excluded labels
        if excluded_labels:
            df = df[~df["material_type"].isin(excluded_labels)]
            df = df[~df["formula"].isin(excluded_labels)]

        # Aggregate counts
        sunburst_data = df.groupby(['formula', 'material_type']).size().reset_index(name='count')

        # Color logic: discrete vs continuous
        if discrete_mode:
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=['formula', 'material_type'],
                values='count',
                color='material_type',
                title="Hierarchical Distribution of Materials",
                color_discrete_map={"p-type": "#EF553B", "n-type": "#636EFA"}
            )
        else:
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=['formula', 'material_type'],
                values='count',
                color='count',
                color_continuous_scale=colormap_choice,
                title="Hierarchical Distribution of Materials"
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
st.title("🌞 Enhanced Sunburst Chart for Materials")

csv_path = os.path.join(DB_DIR, "formula_classifications_via_nlp.csv")

# Sidebar controls
st.sidebar.header("Chart Customization")

discrete_mode = st.sidebar.radio("Color Mode", ["Discrete (by type)", "Continuous (by count)"]) == "Discrete (by type)"

colormap_choice = st.sidebar.selectbox(
    "Choose Continuous Color Map",
    ["Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo"],
    index=0
)

show_labels = st.sidebar.checkbox("Show Labels", value=True)
label_fontsize = st.sidebar.slider("Label Font Size", 8, 24, 12)

# Exclude certain formulas or types
all_labels = pd.read_csv(csv_path)[["formula", "material_type"]].stack().unique()
excluded_labels = st.sidebar.multiselect("Exclude Labels", options=all_labels)

# Generate chart
fig_sunburst = create_sunburst_chart(
    csv_path,
    colormap_choice,
    discrete_mode,
    show_labels,
    label_fontsize,
    excluded_labels
)

if fig_sunburst:
    st.plotly_chart(fig_sunburst, use_container_width=True)
else:
    st.warning("Unable to display sunburst chart. Check error messages above.")
