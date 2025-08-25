import os
import pandas as pd
import plotly.express as px
import streamlit as st
import sqlite3
from datetime import datetime
import logging
import base64
from io import BytesIO

# Directory setup
DB_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize logging
logging.basicConfig(
    filename=os.path.join(DB_DIR, 'thermoelectric_sunburst_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    logging.info(log_message)
    if 'log_buffer' not in st.session_state:
        st.session_state.log_buffer = []
    st.session_state.log_buffer.append(log_message)
    if len(st.session_state.log_buffer) > 20:
        st.session_state.log_buffer.pop(0)

def detect_year_column(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(papers)")
    columns = {col[1].lower() for col in cursor.fetchall()}
    possible_year_columns = ['year', 'publication_year', 'date']
    for col in possible_year_columns:
        if col.lower() in columns:
            update_log(f"Detected year column: {col}")
            return col
    update_log("No year column found in 'papers' table")
    return None

def load_data_from_db(db_file, year_range=None):
    """Load material classification data from database"""
    try:
        conn = sqlite3.connect(db_file)
        
        # Check if standardized_formulas table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='standardized_formulas'")
        if not cursor.fetchone():
            update_log("No standardized_formulas table found in database")
            st.error("Please run Material Classification Analysis first to extract and standardize formulas")
            return None
        
        # Load data
        query = "SELECT * FROM standardized_formulas"
        df = pd.read_sql_query(query, conn)
        
        # Try to get year data if available
        year_column = detect_year_column(conn)
        if year_column and 'paper_id' in df.columns:
            year_query = f"SELECT id, {year_column} FROM papers"
            year_df = pd.read_sql_query(year_query, conn)
            df = df.merge(year_df, left_on='paper_id', right_on='id', how='left')
            df.rename(columns={year_column: 'year'}, inplace=True)
        
        conn.close()
        
        # Apply year filter if specified
        if year_range and 'year' in df.columns:
            df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
        
        update_log(f"Loaded {len(df)} records from database")
        return df
    
    except Exception as e:
        update_log(f"Error loading data from database: {str(e)}")
        st.error(f"Database error: {str(e)}")
        return None

def create_sunburst_chart(df, colormap_choice, discrete_mode, show_labels, label_fontsize, excluded_labels, year_range=None):
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            return None
        
        # Filter excluded labels
        if excluded_labels:
            if 'material_type' in df.columns:
                df = df[~df["material_type"].isin(excluded_labels)]
            if 'material' in df.columns:
                df = df[~df["material"].isin(excluded_labels)]
            if 'classification' in df.columns:
                df = df[~df["classification"].isin(excluded_labels)]
        
        # Standardize column names
        if 'material' in df.columns and 'classification' in df.columns:
            df = df.rename(columns={'material': 'formula', 'classification': 'material_type'})
        
        # Aggregate counts
        if 'year' in df.columns:
            sunburst_data = df.groupby(['year', 'formula', 'material_type']).size().reset_index(name='count')
            path = ['year', 'formula', 'material_type']
        else:
            sunburst_data = df.groupby(['formula', 'material_type']).size().reset_index(name='count')
            path = ['formula', 'material_type']
        
        if sunburst_data.empty:
            st.warning("No data available after filtering")
            return None

        # Color logic: discrete vs continuous
        if discrete_mode:
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=path,
                values='count',
                color='material_type',
                title="Hierarchical Distribution of Materials",
                color_discrete_map={"p-type": "#EF553B", "n-type": "#636EFA", "unknown": "#7F7F7F"}
            )
        else:
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=path,
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
        update_log(f"Sunburst chart error: {str(e)}")
        return None

def get_chart_download_link(fig, filename="chart.png"):
    """Generate a download link for the chart"""
    try:
        # Try to export as PNG
        img_bytes = fig.to_image(format="png")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download PNG</a>'
        return href
    except Exception as e:
        update_log(f"PNG export failed: {str(e)}")
        
        # Fallback to HTML export
        try:
            html = fig.to_html()
            b64 = base64.b64encode(html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="{filename.replace(".png", ".html")}">Download HTML</a>'
            return href
        except Exception as e2:
            update_log(f"HTML export also failed: {str(e2)}")
            return None

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Enhanced Thermoelectric Material Sunburst Analysis", layout="wide")
st.title("🌞 Enhanced Sunburst Analysis for Thermoelectric Materials")

# Initialize session state
if "db_file" not in st.session_state:
    # Look for database files
    db_files = [f for f in os.listdir(DB_DIR) if f.endswith('.db')]
    if db_files:
        st.session_state.db_file = os.path.join(DB_DIR, db_files[0])
    else:
        st.session_state.db_file = None

if "data_df" not in st.session_state:
    st.session_state.data_df = None

# Database selection
st.sidebar.header("Data Source")
db_files = [f for f in os.listdir(DB_DIR) if f.endswith('.db')]
if db_files:
    selected_db = st.sidebar.selectbox("Select Database", db_files, index=0)
    st.session_state.db_file = os.path.join(DB_DIR, selected_db)
else:
    st.sidebar.warning("No database files found in directory")

# Load data button
if st.sidebar.button("Load Data from Database") and st.session_state.db_file:
    with st.spinner("Loading data from database..."):
        st.session_state.data_df = load_data_from_db(st.session_state.db_file)
        if st.session_state.data_df is not None:
            st.sidebar.success(f"Loaded {len(st.session_state.data_df)} records")

# Year range filter
year_range = None
if st.session_state.data_df is not None and 'year' in st.session_state.data_df.columns:
    min_year = int(st.session_state.data_df['year'].min()) if st.session_state.data_df['year'].notna().any() else 2000
    max_year = int(st.session_state.data_df['year'].max()) if st.session_state.data_df['year'].notna().any() else 2023
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

# Reload data with year filter if needed
if year_range and st.sidebar.button("Apply Year Filter") and st.session_state.db_file:
    with st.spinner("Loading filtered data..."):
        st.session_state.data_df = load_data_from_db(st.session_state.db_file, year_range)

# Chart customization
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
if st.session_state.data_df is not None:
    all_labels = []
    if 'material_type' in st.session_state.data_df.columns:
        all_labels.extend(st.session_state.data_df["material_type"].unique().tolist())
    if 'classification' in st.session_state.data_df.columns:
        all_labels.extend(st.session_state.data_df["classification"].unique().tolist())
    if 'formula' in st.session_state.data_df.columns:
        all_labels.extend(st.session_state.data_df["formula"].unique().tolist())
    if 'material' in st.session_state.data_df.columns:
        all_labels.extend(st.session_state.data_df["material"].unique().tolist())
    
    excluded_labels = st.sidebar.multiselect("Exclude Labels", options=list(set(all_labels)))
else:
    excluded_labels = []
    st.sidebar.info("Load data first to see exclusion options")

# Data summary
if st.session_state.data_df is not None:
    st.header("Data Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(st.session_state.data_df))
    
    with col2:
        if 'material_type' in st.session_state.data_df.columns:
            p_type_count = len(st.session_state.data_df[st.session_state.data_df["material_type"] == "p-type"])
            st.metric("p-type Materials", p_type_count)
        elif 'classification' in st.session_state.data_df.columns:
            p_type_count = len(st.session_state.data_df[st.session_state.data_df["classification"] == "p-type"])
            st.metric("p-type Materials", p_type_count)
        else:
            st.metric("p-type Materials", "N/A")
    
    with col3:
        if 'material_type' in st.session_state.data_df.columns:
            n_type_count = len(st.session_state.data_df[st.session_state.data_df["material_type"] == "n-type"])
            st.metric("n-type Materials", n_type_count)
        elif 'classification' in st.session_state.data_df.columns:
            n_type_count = len(st.session_state.data_df[st.session_state.data_df["classification"] == "n-type"])
            st.metric("n-type Materials", n_type_count)
        else:
            st.metric("n-type Materials", "N/A")
    
    # Show data preview
    with st.expander("View Data Preview"):
        st.dataframe(st.session_state.data_df.head(100))

# Generate chart
if st.session_state.data_df is not None:
    fig_sunburst = create_sunburst_chart(
        st.session_state.data_df,
        colormap_choice,
        discrete_mode,
        show_labels,
        label_fontsize,
        excluded_labels,
        year_range
    )

    if fig_sunburst:
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # Add download buttons with error handling
        st.subheader("Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as CSV
            csv = st.session_state.data_df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="thermoelectric_materials_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Try to download as PNG with fallback
            try:
                img_bytes = fig_sunburst.to_image(format="png")
                st.download_button(
                    label="Download Chart as PNG",
                    data=img_bytes,
                    file_name="thermoelectric_materials_sunburst.png",
                    mime="image/png"
                )
            except Exception as e:
                st.warning("PNG export not available. Please use the built-in Plotly export tools.")
                update_log(f"PNG export error: {str(e)}")
        
        with col3:
            # Download as HTML
            try:
                html = fig_sunburst.to_html()
                st.download_button(
                    label="Download Chart as HTML",
                    data=html,
                    file_name="thermoelectric_materials_sunburst.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error("HTML export failed")
                update_log(f"HTML export error: {str(e)}")
    else:
        st.warning("Unable to generate sunburst chart. Check data format.")
else:
    st.info("Please load data from the database to generate visualizations")

# Logs section
st.sidebar.header("Logs")
if 'log_buffer' in st.session_state:
    st.sidebar.text_area("Recent Logs", "\n".join(st.session_state.log_buffer[-10:]), height=200)
else:
    st.sidebar.info("No logs yet")

# Footer with information
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Thermoelectric Material Analysis Tool**  
    This tool visualizes p-type and n-type material classifications
    extracted from scientific literature using NLP techniques.
    """
)

# Add installation instructions for Kaleido if needed
with st.expander("Troubleshooting"):
    st.markdown("""
    **If you encounter issues with chart export:**
    
    1. Install the Kaleido library for better export functionality:
    ```bash
    pip install kaleido
    ```
    
    2. For Streamlit Cloud deployment, add `kaleido` to your requirements.txt file
    
    3. Alternatively, use the built-in Plotly export tools (camera icon in the chart)
    """)
