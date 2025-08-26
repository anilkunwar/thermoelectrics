import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sqlite3
from datetime import datetime
import logging
import base64
from io import BytesIO
import json
import numpy as np

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

# Comprehensive color scale options
COLOR_SCALES = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'blackbody', 'bluered',
    'blues', 'earth', 'electric', 'greens', 'greys', 'oranges', 'picnic', 'portland',
    'rainbow', 'rdbu', 'reds', 'ylgnbu', 'ylorrd', 'deep', 'dense'
]

# Secondary color scale for "Other" categories
OTHER_COLOR_SCALES = {
    'p-type': 'blues',  # Light blue shades for Other p-type
    'n-type': 'reds'    # Light red shades for Other n-type
}

def create_three_tier_sunburst(df, colormap_choice, discrete_mode, show_labels, label_fontsize, 
                              excluded_labels, year_range=None, chart_height=800, branchvalues='total',
                              label_threshold=1.0, show_values=True, show_percentages=True):
    """
    Create a three-tier sunburst chart with Year -> Material Type -> Formula hierarchy
    """
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            return None, None
        
        # Validate colormap_choice
        if colormap_choice.lower() not in COLOR_SCALES:
            update_log(f"Invalid colormap '{colormap_choice}' selected. Falling back to 'viridis'.")
            colormap_choice = 'viridis'

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
        
        # Ensure we have the required columns
        required_cols = []
        if 'year' in df.columns:
            required_cols.append('year')
        required_cols.extend(['formula', 'material_type'])
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None, None
        
        # Create three-tier hierarchy data
        if 'year' in df.columns:
            # Group by year, material type, and formula
            sunburst_data = df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
            
            # Create IDs for each level
            sunburst_data['year_id'] = sunburst_data['year'].astype(str)
            sunburst_data['type_id'] = sunburst_data['year_id'] + '_' + sunburst_data['material_type']
            sunburst_data['formula_id'] = sunburst_data['type_id'] + '_' + sunburst_data['formula']
            
            # Create parent-child relationships
            years = sunburst_data[['year_id', 'year']].drop_duplicates()
            years['parent'] = ''
            years['id'] = years['year_id']
            
            types = sunburst_data[['type_id', 'material_type', 'year_id']].drop_duplicates()
            types['parent'] = types['year_id']
            types['id'] = types['type_id']
            
            formulas = sunburst_data[['formula_id', 'formula', 'count', 'type_id']].copy()
            formulas['parent'] = formulas['type_id']
            formulas['id'] = formulas['formula_id']
            
            # Combine all levels
            hierarchy_data = pd.concat([
                years[['id', 'parent', 'year']].rename(columns={'year': 'label'}),
                types[['id', 'parent', 'material_type']].rename(columns={'material_type': 'label'}),
                formulas[['id', 'parent', 'formula', 'count']].rename(columns={'formula': 'label'})
            ], ignore_index=True)
            
            # Add values for intermediate levels (sum of children)
            for type_id in types['id']:
                type_sum = formulas[formulas['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum
                
            for year_id in years['id']:
                year_sum = types[types['parent'] == year_id]['id'].apply(
                    lambda x: hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].values[0] if not hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].empty else 0
                ).sum()
                hierarchy_data.loc[hierarchy_data['id'] == year_id, 'count'] = year_sum
            
            # Calculate percentages for label visibility
            total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
            hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100
            
            # Create the sunburst chart
            if discrete_mode:
                # Create a color mapping for material types
                unique_types = hierarchy_data[hierarchy_data['id'].str.contains('_') & 
                                            ~hierarchy_data['id'].str.contains('_', regex=False).str.count('_').gt(1)]['label'].unique()
                color_map = {}
                colors = px.colors.qualitative.Plotly
                for i, t in enumerate(unique_types):
                    color_map[t] = colors[i % len(colors)]
                
                # Apply colors based on material type
                hierarchy_data['color'] = hierarchy_data['label'].map(color_map)
                hierarchy_data.loc[hierarchy_data['parent'] == '', 'color'] = '#E5ECF6'  # Light blue for years
                
                # Create custom text based on threshold
                hierarchy_data['display_text'] = hierarchy_data['label']
                if label_threshold > 0:
                    hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
                
                fig = go.Figure(go.Sunburst(
                    ids=hierarchy_data['id'],
                    labels=hierarchy_data['display_text'],
                    parents=hierarchy_data['parent'],
                    values=hierarchy_data['count'],
                    branchvalues=branchvalues,
                    marker=dict(colors=hierarchy_data['color']),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
                    textinfo="label+text" if show_labels else "none",
                    texttemplate=(
                        "%{label}" + 
                        ("<br>%{value}" if show_values else "") + 
                        ("<br>%{percentParent:.1%}" if show_percentages else "")
                    ),
                    textfont=dict(size=label_fontsize),
                    insidetextorientation='horizontal'
                ))
            else:
                # Create custom text based on threshold
                hierarchy_data['display_text'] = hierarchy_data['label']
                if label_threshold > 0:
                    hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
                
                fig = go.Figure(go.Sunburst(
                    ids=hierarchy_data['id'],
                    labels=hierarchy_data['display_text'],
                    parents=hierarchy_data['parent'],
                    values=hierarchy_data['count'],
                    branchvalues=branchvalues,
                    marker=dict(
                        colors=hierarchy_data['count'],
                        colorscale=colormap_choice.lower(),
                        showscale=True,
                        colorbar=dict(title="Count"),
                        cmin=hierarchy_data['count'].min(),
                        cmax=hierarchy_data['count'].max()
                    ),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
                    textinfo="label+text" if show_labels else "none",
                    texttemplate=(
                        "%{label}" + 
                        ("<br>%{value}" if show_values else "") + 
                        ("<br>%{percentParent:.1%}" if show_percentages else "")
                    ),
                    textfont=dict(size=label_fontsize),
                    insidetextorientation='horizontal'
                ))
        else:
            # Fallback to two-tier if no year data
            sunburst_data = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
            
            # Create IDs for each level
            sunburst_data['type_id'] = sunburst_data['material_type']
            sunburst_data['formula_id'] = sunburst_data['type_id'] + '_' + sunburst_data['formula']
            
            # Create parent-child relationships
            types = sunburst_data[['type_id', 'material_type']].drop_duplicates()
            types['parent'] = ''
            types['id'] = types['type_id']
            
            formulas = sunburst_data[['formula_id', 'formula', 'count', 'type_id']].copy()
            formulas['parent'] = formulas['type_id']
            formulas['id'] = formulas['formula_id']
            
            # Combine all levels
            hierarchy_data = pd.concat([
                types[['id', 'parent', 'material_type']].rename(columns={'material_type': 'label'}),
                formulas[['id', 'parent', 'formula', 'count']].rename(columns={'formula': 'label'})
            ], ignore_index=True)
            
            # Add values for intermediate levels (sum of children)
            for type_id in types['id']:
                type_sum = formulas[formulas['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum
            
            # Calculate percentages for label visibility
            total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
            hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100
            
            # Create the sunburst chart
            if discrete_mode:
                # Create a color mapping for material types
                unique_types = hierarchy_data[hierarchy_data['parent'] == '']['label'].unique()
                color_map = {}
                colors = px.colors.qualitative.Plotly
                for i, t in enumerate(unique_types):
                    color_map[t] = colors[i % len(colors)]
                
                # Apply colors based on material type
                hierarchy_data['color'] = hierarchy_data['label'].map(color_map)
                
                # Create custom text based on threshold
                hierarchy_data['display_text'] = hierarchy_data['label']
                if label_threshold > 0:
                    hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
                
                fig = go.Figure(go.Sunburst(
                    ids=hierarchy_data['id'],
                    labels=hierarchy_data['display_text'],
                    parents=hierarchy_data['parent'],
                    values=hierarchy_data['count'],
                    branchvalues=branchvalues,
                    marker=dict(colors=hierarchy_data['color']),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
                    textinfo="label+text" if show_labels else "none",
                    texttemplate=(
                        "%{label}" + 
                        ("<br>%{value}" if show_values else "") + 
                        ("<br>%{percentParent:.1%}" if show_percentages else "")
                    ),
                    textfont=dict(size=label_fontsize),
                    insidetextorientation='horizontal'
                ))
            else:
                # Create custom text based on threshold
                hierarchy_data['display_text'] = hierarchy_data['label']
                if label_threshold > 0:
                    hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
                
                fig = go.Figure(go.Sunburst(
                    ids=hierarchy_data['id'],
                    labels=hierarchy_data['display_text'],
                    parents=hierarchy_data['parent'],
                    values=hierarchy_data['count'],
                    branchvalues=branchvalues,
                    marker=dict(
                        colors=hierarchy_data['count'],
                        colorscale=colormap_choice.lower(),
                        showscale=True,
                        colorbar=dict(title="Count"),
                        cmin=hierarchy_data['count'].min(),
                        cmax=hierarchy_data['count'].max()
                    ),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
                    textinfo="label+text" if show_labels else "none",
                    texttemplate=(
                        "%{label}" + 
                        ("<br>%{value}" if show_values else "") + 
                        ("<br>%{percentParent:.1%}" if show_percentages else "")
                    ),
                    textfont=dict(size=label_fontsize),
                    insidetextorientation='horizontal'
                ))

        # Update layout for publication quality
        fig.update_layout(
            height=chart_height,
            plot_bgcolor="rgba(255,255,255,1)",
            paper_bgcolor="rgba(255,255,255,1)",
            margin=dict(t=80, l=20, r=20, b=20),
            font=dict(family="Arial, sans-serif", size=12, color="#000000"),
            title=dict(
                text="Three-Tier Hierarchy of Thermoelectric Materials",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, family="Arial, sans-serif")
            )
        )

        return fig, hierarchy_data

    except Exception as e:
        st.error(f"Failed to generate sunburst chart: {str(e)}")
        update_log(f"Sunburst chart error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None, None

def create_top_n_sunburst(df, top_n, colormap_choice, discrete_mode, show_labels, label_fontsize, 
                         excluded_labels, year_range=None, chart_height=800, branchvalues='total',
                         label_threshold=1.0, show_values=True, show_percentages=True):
    """
    Create a sunburst chart showing top N materials per type, grouping others
    """
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            return None, None
        
        # Validate colormap_choice
        if colormap_choice.lower() not in COLOR_SCALES:
            update_log(f"Invalid colormap '{colormap_choice}' selected. Falling back to 'viridis'.")
            colormap_choice = 'viridis'

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
        
        # Ensure we have the required columns
        required_cols = ['formula', 'material_type']
        if 'year' in df.columns:
            required_cols.append('year')
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None, None
        
        # Create hierarchy data with top N materials
        if 'year' in df.columns:
            # Group by year, material type, and formula
            sunburst_data = df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
            
            # Get top N formulas for each material type per year
            grouped_data = []
            for year in sunburst_data['year'].unique():
                for mtype in sunburst_data['material_type'].unique():
                    type_data = sunburst_data[(sunburst_data['year'] == year) & 
                                            (sunburst_data['material_type'] == mtype)]
                    # Sort by count and select top N
                    top_formulas = type_data.nlargest(top_n, 'count')
                    # Group remaining as "Other"
                    other_data = type_data[~type_data['formula'].isin(top_formulas['formula'])]
                    if not other_data.empty:
                        other_count = other_data['count'].sum()
                        other_row = pd.DataFrame({
                            'year': [year],
                            'material_type': [mtype],
                            'formula': [f'Other {mtype}'],
                            'count': [other_count]
                        })
                        type_data = pd.concat([top_formulas, other_row], ignore_index=True)
                    else:
                        type_data = top_formulas
                    grouped_data.append(type_data)
            
            sunburst_data = pd.concat(grouped_data, ignore_index=True)
            
            # Create IDs for each level
            sunburst_data['year_id'] = sunburst_data['year'].astype(str)
            sunburst_data['type_id'] = sunburst_data['year_id'] + '_' + sunburst_data['material_type']
            sunburst_data['formula_id'] = sunburst_data['type_id'] + '_' + sunburst_data['formula']
            
            # Create parent-child relationships
            years = sunburst_data[['year_id', 'year']].drop_duplicates()
            years['parent'] = ''
            years['id'] = years['year_id']
            
            types = sunburst_data[['type_id', 'material_type', 'year_id']].drop_duplicates()
            types['parent'] = types['year_id']
            types['id'] = types['type_id']
            
            formulas = sunburst_data[['formula_id', 'formula', 'count', 'type_id']].copy()
            formulas['parent'] = formulas['type_id']
            formulas['id'] = formulas['formula_id']
            
            # Combine all levels
            hierarchy_data = pd.concat([
                years[['id', 'parent', 'year']].rename(columns={'year': 'label'}),
                types[['id', 'parent', 'material_type']].rename(columns={'material_type': 'label'}),
                formulas[['id', 'parent', 'formula', 'count']].rename(columns={'formula': 'label'})
            ], ignore_index=True)
            
            # Add values for intermediate levels
            for type_id in types['id']:
                type_sum = formulas[formulas['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum
                
            for year_id in years['id']:
                year_sum = types[types['parent'] == year_id]['id'].apply(
                    lambda x: hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].values[0] if not hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].empty else 0
                ).sum()
                hierarchy_data.loc[hierarchy_data['id'] == year_id, 'count'] = year_sum
            
            # Calculate percentages
            total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
            hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100
            
            # Create the sunburst chart
            if discrete_mode:
                # Create a color mapping for material types and "Other"
                unique_types = hierarchy_data[hierarchy_data['id'].str.contains('_') & 
                                            ~hierarchy_data['id'].str.contains('_', regex=False).str.count('_').gt(1)]['label'].unique()
                color_map = {}
                colors = px.colors.qualitative.Plotly
                for i, t in enumerate(unique_types):
                    if t.startswith('Other'):
                        # Use specific color scale for "Other" based on material type
                        mtype = t.split(' ')[1]
                        other_colors = px.colors.sequential.__dict__[OTHER_COLOR_SCALES.get(mtype, 'greys')]
                        color_map[t] = other_colors[min(i, len(other_colors)-1)]
                    else:
                        color_map[t] = colors[i % len(colors)]
                
                hierarchy_data['color'] = hierarchy_data['label'].map(color_map)
                hierarchy_data.loc[hierarchy_data['parent'] == '', 'color'] = '#E5ECF6'
                
                hierarchy_data['display_text'] = hierarchy_data['label']
                if label_threshold > 0:
                    hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
                
                fig = go.Figure(go.Sunburst(
                    ids=hierarchy_data['id'],
                    labels=hierarchy_data['display_text'],
                    parents=hierarchy_data['parent'],
                    values=hierarchy_data['count'],
                    branchvalues=branchvalues,
                    marker=dict(colors=hierarchy_data['color']),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
                    textinfo="label+text" if show_labels else "none",
                    texttemplate=(
                        "%{label}" + 
                        ("<br>%{value}" if show_values else "") + 
                        ("<br>%{percentParent:.1%}" if show_percentages else "")
                    ),
                    textfont=dict(size=label_fontsize),
                    insidetextorientation='horizontal'
                ))
            else:
                hierarchy_data['display_text'] = hierarchy_data['label']
                if label_threshold > 0:
                    hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
                
                # Adjust colors for "Other" categories
                colors = hierarchy_data['count'].copy()
                other_mask = hierarchy_data['label'].str.startswith('Other')
                if other_mask.any():
                    for mtype in ['p-type', 'n-type']:
                        mask = hierarchy_data['label'] == f'Other {mtype}'
                        if mask.any():
                            other_count = hierarchy_data.loc[mask, 'count'].iloc[0]
                            other_colors = px.colors.sequential.__dict__[OTHER_COLOR_SCALES.get(mtype, 'greys')]
                            # Map count to a color in the secondary scale
                            color_idx = min(int((other_count / hierarchy_data['count'].max()) * (len(other_colors) - 1)), len(other_colors) - 1)
                            colors[mask] = other_colors[color_idx]
                
                fig = go.Figure(go.Sunburst(
                    ids=hierarchy_data['id'],
                    labels=hierarchy_data['display_text'],
                    parents=hierarchy_data['parent'],
                    values=hierarchy_data['count'],
                    branchvalues=branchvalues,
                    marker=dict(
                        colors=colors,
                        colorscale=colormap_choice.lower(),
                        showscale=True,
                        colorbar=dict(title="Count"),
                        cmin=hierarchy_data['count'].min(),
                        cmax=hierarchy_data['count'].max()
                    ),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
                    textinfo="label+text" if show_labels else "none",
                    texttemplate=(
                        "%{label}" + 
                        ("<br>%{value}" if show_values else "") + 
                        ("<br>%{percentParent:.1%}" if show_percentages else "")
                    ),
                    textfont=dict(size=label_fontsize),
                    insidetextorientation='horizontal'
                ))
        else:
            # Two-tier hierarchy (no year)
            sunburst_data = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
            
            # Get top N formulas for each material type
            grouped_data = []
            for mtype in sunburst_data['material_type'].unique():
                type_data = sunburst_data[sunburst_data['material_type'] == mtype]
                top_formulas = type_data.nlargest(top_n, 'count')
                other_data = type_data[~type_data['formula'].isin(top_formulas['formula'])]
                if not other_data.empty:
                    other_count = other_data['count'].sum()
                    other_row = pd.DataFrame({
                        'material_type': [mtype],
                        'formula': [f'Other {mtype}'],
                        'count': [other_count]
                    })
                    type_data = pd.concat([top_formulas, other_row], ignore_index=True)
                else:
                    type_data = top_formulas
                grouped_data.append(type_data)
            
            sunburst_data = pd.concat(grouped_data, ignore_index=True)
            
            # Create IDs
            sunburst_data['type_id'] = sunburst_data['material_type']
            sunburst_data['formula_id'] = sunburst_data['type_id'] + '_' + sunburst_data['formula']
            
            types = sunburst_data[['type_id', 'material_type']].drop_duplicates()
            types['parent'] = ''
            types['id'] = types['type_id']
            
            formulas = sunburst_data[['formula_id', 'formula', 'count', 'type_id']].copy()
            formulas['parent'] = formulas['type_id']
            formulas['id'] = formulas['formula_id']
            
            hierarchy_data = pd.concat([
                types[['id', 'parent', 'material_type']].rename(columns={'material_type': 'label'}),
                formulas[['id', 'parent', 'formula', 'count']].rename(columns={'formula': 'label'})
            ], ignore_index=True)
            
            for type_id in types['id']:
                type_sum = formulas[formulas['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum
            
            total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
            hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100
            
            if discrete_mode:
                unique_types = hierarchy_data[hierarchy_data['parent'] == '']['label'].unique()
                color_map = {}
                colors = px.colors.qualitative.Plotly
                for i, t in enumerate(unique_types):
                    if t.startswith('Other'):
                        mtype = t.split(' ')[1]
                        other_colors = px.colors.sequential.__dict__[OTHER_COLOR_SCALES.get(mtype, 'greys')]
                        color_map[t] = other_colors[min(i, len(other_colors)-1)]
                    else:
                        color_map[t] = colors[i % len(colors)]
                
                hierarchy_data['color'] = hierarchy_data['label'].map(color_map)
                
                hierarchy_data['display_text'] = hierarchy_data['label']
                if label_threshold > 0:
                    hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
                
                fig = go.Figure(go.Sunburst(
                    ids=hierarchy_data['id'],
                    labels=hierarchy_data['display_text'],
                    parents=hierarchy_data['parent'],
                    values=hierarchy_data['count'],
                    branchvalues=branchvalues,
                    marker=dict(colors=hierarchy_data['color']),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
                    textinfo="label+text" if show_labels else "none",
                    texttemplate=(
                        "%{label}" + 
                        ("<br>%{value}" if show_values else "") + 
                        ("<br>%{percentParent:.1%}" if show_percentages else "")
                    ),
                    textfont=dict(size=label_fontsize),
                    insidetextorientation='horizontal'
                ))
            else:
                hierarchy_data['display_text'] = hierarchy_data['label']
                if label_threshold > 0:
                    hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
                
                colors = hierarchy_data['count'].copy()
                other_mask = hierarchy_data['label'].str.startswith('Other')
                if other_mask.any():
                    for mtype in ['p-type', 'n-type']:
                        mask = hierarchy_data['label'] == f'Other {mtype}'
                        if mask.any():
                            other_count = hierarchy_data.loc[mask, 'count'].iloc[0]
                            other_colors = px.colors.sequential.__dict__[OTHER_COLOR_SCALES.get(mtype, 'greys')]
                            color_idx = min(int((other_count / hierarchy_data['count'].max()) * (len(other_colors) - 1)), len(other_colors) - 1)
                            colors[mask] = other_colors[color_idx]
                
                fig = go.Figure(go.Sunburst(
                    ids=hierarchy_data['id'],
                    labels=hierarchy_data['display_text'],
                    parents=hierarchy_data['parent'],
                    values=hierarchy_data['count'],
                    branchvalues=branchvalues,
                    marker=dict(
                        colors=colors,
                        colorscale=colormap_choice.lower(),
                        showscale=True,
                        colorbar=dict(title="Count"),
                        cmin=hierarchy_data['count'].min(),
                        cmax=hierarchy_data['count'].max()
                    ),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
                    textinfo="label+text" if show_labels else "none",
                    texttemplate=(
                        "%{label}" + 
                        ("<br>%{value}" if show_values else "") + 
                        ("<br>%{percentParent:.1%}" if show_percentages else "")
                    ),
                    textfont=dict(size=label_fontsize),
                    insidetextorientation='horizontal'
                ))

        # Update layout
        fig.update_layout(
            height=chart_height,
            plot_bgcolor="rgba(255,255,255,1)",
            paper_bgcolor="rgba(255,255,255,1)",
            margin=dict(t=80, l=20, r=20, b=20),
            font=dict(family="Arial, sans-serif", size=12, color="#000000"),
            title=dict(
                text=f"Top {top_n} Thermoelectric Materials per Type",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, family="Arial, sans-serif")
            )
        )

        return fig, hierarchy_data

    except Exception as e:
        st.error(f"Failed to generate top N sunburst chart: {str(e)}")
        update_log(f"Top N sunburst chart error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None, None

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Three-Tier Thermoelectric Material Analysis", layout="wide")
st.title("🌞 Three-Tier Hierarchy Analysis for Thermoelectric Materials")

# Initialize session state
if "db_file" not in st.session_state:
    db_files = [f for f in os.listdir(DB_DIR) if f.endswith('.db')]
    if db_files:
        st.session_state.db_file = os.path.join(DB_DIR, db_files[0])
    else:
        st.session_state.db_file = None

if "data_df" not in st.session_state:
    st.session_state.data_df = None

if "hierarchy_data" not in st.session_state:
    st.session_state.hierarchy_data = None

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
if discrete_mode:
    st.sidebar.info("In Discrete mode, colors are assigned by material type using Plotly's qualitative colors. Color map selection is used only in Continuous mode.")
colormap_choice = st.sidebar.selectbox(
    "Choose Color Map (Continuous mode)",
    COLOR_SCALES,
    index=0
)

show_labels = st.sidebar.checkbox("Show Labels", value=True)
label_fontsize = st.sidebar.slider("Label Font Size", 8, 24, 12)
chart_height = st.sidebar.slider("Chart Height", 400, 1200, 800)
branchvalues = st.sidebar.selectbox("Branch Values", ["total", "remainder"], index=0)

# Label visibility options
st.sidebar.header("Label Visibility")
label_threshold = st.sidebar.slider("Label Threshold (%)", 0.0, 10.0, 1.0, 0.1,
                                   help="Only show labels for segments larger than this percentage of the total")
show_values = st.sidebar.checkbox("Show Values in Labels", value=True)
show_percentages = st.sidebar.checkbox("Show Percentages in Labels", value=True)

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

# Top N materials selection
st.sidebar.header("Top N Materials Chart")
if st.session_state.data_df is not None:
    max_formulas = max(50, len(st.session_state.data_df['formula'].unique()) if 'formula' in st.session_state.data_df.columns else 50)
    top_n = st.sidebar.slider("Number of Top Materials per Type", 1, max_formulas, 5,
                              help="Select the number of top materials to display individually. Remaining materials are grouped as 'Other'.")
else:
    top_n = 5
    st.sidebar.info("Load data to adjust top N materials")

# Data summary
if st.session_state.data_df is not None:
    st.header("Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    
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
            
    with col4:
        if 'year' in st.session_state.data_df.columns:
            year_range_str = f"{int(st.session_state.data_df['year'].min())}-{int(st.session_state.data_df['year'].max())}"
            st.metric("Year Range", year_range_str)
        else:
            st.metric("Year Data", "Not Available")
    
    with st.expander("View Data Preview"):
        st.dataframe(st.session_state.data_df.head(100))

# Generate charts
if st.session_state.data_df is not None:
    # First chart (original sunburst)
    st.subheader("Full Hierarchy Sunburst Chart")
    fig_sunburst, hierarchy_data = create_three_tier_sunburst(
        st.session_state.data_df,
        colormap_choice,
        discrete_mode,
        show_labels,
        label_fontsize,
        excluded_labels,
        year_range,
        chart_height,
        branchvalues,
        label_threshold,
        show_values,
        show_percentages
    )
    
    st.session_state.hierarchy_data = hierarchy_data

    if fig_sunburst:
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    # Second chart (top N materials)
    st.subheader(f"Top {top_n} Materials Sunburst Chart")
    if st.sidebar.button("Generate Top N Materials Chart"):
        with st.spinner("Generating top N materials chart..."):
            fig_top_n, top_n_hierarchy_data = create_top_n_sunburst(
                st.session_state.data_df,
                top_n,
                colormap_choice,
                discrete_mode,
                show_labels,
                label_fontsize,
                excluded_labels,
                year_range,
                chart_height,
                branchvalues,
                label_threshold,
                show_values,
                show_percentages
            )
            
            if fig_top_n:
                st.plotly_chart(fig_top_n, use_container_width=True)
            else:
                st.warning("Unable to generate top N materials chart. Check data format.")
    
    # Export options
    if fig_sunburst:
        st.subheader("Export Options")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            csv = st.session_state.data_df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="thermoelectric_materials_data.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.session_state.hierarchy_data is not None:
                hierarchy_csv = st.session_state.hierarchy_data.to_csv(index=False)
                st.download_button(
                    label="Download Hierarchy as CSV",
                    data=hierarchy_csv,
                    file_name="thermoelectric_materials_hierarchy.csv",
                    mime="text/csv"
                )
        
        with col3:
            try:
                if st.session_state.hierarchy_data is not None:
                    simplified_data = st.session_state.hierarchy_data[['id', 'parent', 'label', 'count', 'percentage']].to_dict('records')
                    hierarchy_json = json.dumps(simplified_data, indent=2)
                    st.download_button(
                        label="Download Hierarchy as JSON",
                        data=hierarchy_json,
                        file_name="thermoelectric_materials_hierarchy.json",
                        mime="application/json"
                    )
            except Exception as e:
                st.error("JSON export failed")
                update_log(f"JSON export error: {str(e)}")
        
        with col4:
            st.info("Use the camera icon 📷 in the chart to export as PNG")
        
        with st.expander("Publication Quality Tips"):
            st.markdown("""
            **For publication-quality figures:**
            
            1. **Adjust the label threshold** to reduce clutter from small segments
            2. **Use the camera icon** in the chart to export as high-resolution PNG
            3. **Adjust the chart height** for better proportions
            4. **Consider using discrete colors** for clearer material type differentiation
            5. **For best results**, use the exported data in dedicated visualization software like:
               - Adobe Illustrator
               - Inkscape
               - Python with Matplotlib/Seaborn for complete control
            """)
        
        with st.expander("Hierarchy Statistics"):
            if st.session_state.hierarchy_data is not None:
                st.write(f"Total segments: {len(st.session_state.hierarchy_data)}")
                if 'percentage' in st.session_state.hierarchy_data.columns:
                    st.write(f"Segments above {label_threshold}% threshold: {len(st.session_state.hierarchy_data[st.session_state.hierarchy_data['percentage'] >= label_threshold])}")
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
    **Three-Tier Thermoelectric Material Analysis**  
    This tool visualizes the hierarchy of p-type and n-type material classifications
    across years, material types, and specific formulas.
    """
)

