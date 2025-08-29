import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sqlite3
from datetime import datetime
import logging
import json
import numpy as np
import uuid

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

def detect_year_column(conn, table_name='papers'):
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {col[1].lower() for col in cursor.fetchall()}
        possible_year_columns = ['year', 'publication_year', 'date']
        for col in possible_year_columns:
            if col.lower() in columns:
                update_log(f"Detected year column: {col} in table {table_name}")
                return col
        update_log(f"No year column found in table {table_name}")
        return None
    except Exception as e:
        update_log(f"Error detecting year column: {str(e)}")
        return None

def load_data_from_db(db_file, year_range=None):
    """Load material classification data from database with robust validation."""
    try:
        if not os.path.isfile(db_file):
            raise FileNotFoundError(f"Database file '{db_file}' does not exist")
        
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='standardized_formulas'")
        if not cursor.fetchone():
            update_log("No standardized_formulas table found in database")
            st.error("Please run Material Classification Analysis first to extract and standardize formulas")
            conn.close()
            return None
        
        query = "SELECT * FROM standardized_formulas"
        df = pd.read_sql_query(query, conn)
        
        year_column = detect_year_column(conn)
        if year_column and 'paper_id' in df.columns:
            cursor.execute(f"PRAGMA table_info(papers)")
            if cursor.fetchone():
                year_query = f"SELECT id, {year_column} FROM papers"
                year_df = pd.read_sql_query(year_query, conn)
                df = df.merge(year_df, left_on='paper_id', right_on='id', how='left')
                df.rename(columns={year_column: 'year'}, inplace=True)
        
        conn.close()
        
        if df.empty:
            update_log("No data loaded from standardized_formulas table")
            st.error("No data found in standardized_formulas table")
            return None
        
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            if year_range:
                df = df[df['year'].notna() & (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
        if 'count' in df.columns:
            df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
            update_log(f"Validated count column, dtype: {df['count'].dtype}")
        
        update_log(f"Loaded {len(df)} records from database")
        return df
    
    except Exception as e:
        update_log(f"Error loading data from database: {str(e)}")
        st.error(f"Database error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None

# Color scale options
COLOR_SCALES = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'blackbody', 'bluered',
    'blues', 'earth', 'electric', 'greens', 'greys', 'oranges', 'picnic', 'portland',
    'rainbow', 'rdbu', 'reds', 'ylgnbu', 'ylorrd', 'deep', 'dense'
]
DISCRETE_COLORS = px.colors.qualitative.D3 + px.colors.qualitative.Set3
COLORBLIND_DISCRETE = px.colors.qualitative.Safe
HIGHLIGHT_COLORS = px.colors.qualitative.Plotly
OTHER_COLOR_SCALES = {'p-type': 'Blues', 'n-type': 'Reds'}

def validate_color_scale(scale_name):
    try:
        colors = getattr(px.colors.sequential, scale_name)
        if not isinstance(colors, (list, tuple)) or not all(isinstance(c, str) for c in colors):
            raise ValueError(f"Color scale '{scale_name}' is not a valid list of color strings")
        update_log(f"Validated color scale: {scale_name}")
        return colors
    except (AttributeError, ValueError) as e:
        update_log(f"Invalid color scale '{scale_name}': {str(e)}. Falling back to Greys.")
        return px.colors.sequential.Greys

def create_color_map(labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe, highlight_materials=None):
    color_map = custom_colors.copy()
    highlight_materials = highlight_materials or []
    
    if discrete_mode:
        colors = COLORBLIND_DISCRETE if colorblind_safe else DISCRETE_COLORS
        highlight_colors = HIGHLIGHT_COLORS
        for i, label in enumerate(labels):
            if label in highlight_materials:
                color_map[label] = highlight_colors[i % len(highlight_colors)]
            elif label not in color_map:
                if label.startswith('Other') or label.startswith('Sub-Other'):
                    mtype = label.split(' ')[1].split('L')[0] if 'L' in label else label.split(' ')[1]
                    mtype = mtype if mtype in ['p-type', 'n-type'] else 'p-type'
                    other_colors = validate_color_scale(OTHER_COLOR_SCALES.get(mtype, 'Greys'))
                    color_map[label] = other_colors[min(i % len(other_colors), len(other_colors)-1)]
                else:
                    color_map[label] = colors[i % len(colors)]
    else:
        colors = validate_color_scale(colormap_choice if colormap_choice in COLOR_SCALES else 'cividis')
        for i, label in enumerate(labels):
            if label in highlight_materials:
                color_map[label] = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            elif label not in color_map:
                if label.startswith('Other') or label.startswith('Sub-Other'):
                    mtype = label.split(' ')[1].split('L')[0] if 'L' in label else label.split(' ')[1]
                    mtype = mtype if mtype in ['p-type', 'n-type'] else 'p-type'
                    other_colors = validate_color_scale(OTHER_COLOR_SCALES.get(mtype, 'Greys'))
                    color_map[label] = other_colors[-1]
                else:
                    color_map[label] = colors[i % len(colors)]
    return color_map

def filter_excluded_labels(df, excluded_labels, update_log=None):
    """Filter out rows where any of the relevant columns match excluded_labels."""
    if not excluded_labels:
        return df
    
    mask = False
    if 'material_type' in df.columns:
        mask |= df["material_type"].isin(excluded_labels)
    if 'material' in df.columns:
        mask |= df["material"].isin(excluded_labels)
    if 'classification' in df.columns:
        mask |= df["classification"].isin(excluded_labels)

    if isinstance(mask, pd.Series) and mask.any():
        df = df[~mask]
        if update_log:
            update_log(f"Excluded {mask.sum()} rows based on excluded_labels")
    else:
        if update_log:
            update_log("No valid rows matched for exclusion filtering")

    return df

def create_three_tier_sunburst(df, colormap_choice, discrete_mode, show_labels, label_fontsize, 
                              excluded_labels, year_range=None, chart_height=800, branchvalues='total',
                              label_threshold=1.0, show_values=True, show_percentages=True, custom_colors=None, colorblind_safe=False):
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            return None, None
        
        if colormap_choice.lower() not in COLOR_SCALES:
            update_log(f"Invalid colormap '{colormap_choice}' selected. Falling back to 'cividis'.")
            colormap_choice = 'cividis'

        # Apply exclusion filtering
        df = filter_excluded_labels(df, excluded_labels, update_log)
        
        if 'material' in df.columns and 'classification' in df.columns:
            df = df.rename(columns={'material': 'formula', 'classification': 'material_type'})
        
        required_cols = ['formula', 'material_type']
        if 'year' in df.columns:
            required_cols.append('year')
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None, None
        
        if 'year' in df.columns:
            sunburst_data = df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            
            sunburst_data['year_id'] = sunburst_data['year'].astype(str)
            sunburst_data['type_id'] = sunburst_data['year_id'] + '_' + sunburst_data['material_type']
            sunburst_data['formula_id'] = sunburst_data['type_id'] + '_' + sunburst_data['formula']
            
            years = sunburst_data[['year_id', 'year']].drop_duplicates()
            years['parent'] = ''
            years['id'] = years['year_id']
            
            types = sunburst_data[['type_id', 'material_type', 'year_id']].drop_duplicates()
            types['parent'] = types['year_id']
            types['id'] = types['type_id']
            
            formulas = sunburst_data[['formula_id', 'formula', 'count', 'type_id']].copy()
            formulas['parent'] = formulas['type_id']
            formulas['id'] = formulas['formula_id']
            
            hierarchy_data = pd.concat([
                years[['id', 'parent', 'year']].rename(columns={'year': 'label'}),
                types[['id', 'parent', 'material_type']].rename(columns={'material_type': 'label'}),
                formulas[['id', 'parent', 'formula', 'count']].rename(columns={'formula': 'label'})
            ], ignore_index=True)
            
            for type_id in types['id']:
                type_sum = formulas[formulas['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum
                
            for year_id in years['id']:
                year_sum = types[types['parent'] == year_id]['id'].apply(
                    lambda x: hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].values[0] if not hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].empty else 0
                ).sum()
                hierarchy_data.loc[hierarchy_data['id'] == year_id, 'count'] = year_sum
            
            total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
            hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100
            
            custom_colors = custom_colors or {}
            unique_labels = hierarchy_data['label'].unique()
            color_map = st.session_state.get('color_map', create_color_map(unique_labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe))
            st.session_state.color_map = color_map
            
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
            sunburst_data = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            
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
            
            custom_colors = custom_colors or {}
            unique_labels = hierarchy_data['label'].unique()
            color_map = st.session_state.get('color_map', create_color_map(unique_labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe))
            st.session_state.color_map = color_map
            
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
                         label_threshold=1.0, show_values=True, show_percentages=True, custom_colors=None, colorblind_safe=False):
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            return None, None, None
        
        if colormap_choice.lower() not in COLOR_SCALES:
            update_log(f"Invalid colormap '{colormap_choice}' selected. Falling back to 'cividis'.")
            colormap_choice = 'cividis'

        # Apply exclusion filtering
        df = filter_excluded_labels(df, excluded_labels, update_log)
        
        if 'material' in df.columns and 'classification' in df.columns:
            df = df.rename(columns={'material': 'formula', 'classification': 'material_type'})
        
        required_cols = ['formula', 'material_type']
        if 'year' in df.columns:
            required_cols.append('year')
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None, None, None
        
        if 'count' in df.columns:
            df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
        
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            if year_range:
                df = df[df['year'].notna() & (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
            sunburst_data = df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            
            grouped_data = []
            for year in sunburst_data['year'].dropna().unique():
                for mtype in sunburst_data['material_type'].unique():
                    type_data = sunburst_data[(sunburst_data['year'] == year) & 
                                            (sunburst_data['material_type'] == mtype)]
                    if type_data.empty:
                        continue
                    top_formulas = type_data.nlargest(top_n, 'count')
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
            
            if not grouped_data:
                st.error("No data available after filtering for top N materials")
                update_log("No data available after filtering for top N materials")
                return None, None, None
            
            sunburst_data = pd.concat(grouped_data, ignore_index=True)
            
            sunburst_data['year_id'] = sunburst_data['year'].astype(str)
            sunburst_data['type_id'] = sunburst_data['year_id'] + '_' + sunburst_data['material_type']
            sunburst_data['formula_id'] = sunburst_data['type_id'] + '_' + sunburst_data['formula']
            
            years = sunburst_data[['year_id', 'year']].drop_duplicates()
            years['parent'] = ''
            years['id'] = years['year_id']
            
            types = sunburst_data[['type_id', 'material_type', 'year_id']].drop_duplicates()
            types['parent'] = types['year_id']
            types['id'] = types['type_id']
            
            formulas = sunburst_data[['formula_id', 'formula', 'count', 'type_id']].copy()
            formulas['parent'] = formulas['type_id']
            formulas['id'] = formulas['formula_id']
            
            hierarchy_data = pd.concat([
                years[['id', 'parent', 'year']].rename(columns={'year': 'label'}),
                types[['id', 'parent', 'material_type']].rename(columns={'material_type': 'label'}),
                formulas[['id', 'parent', 'formula', 'count']].rename(columns={'formula': 'label'})
            ], ignore_index=True)
            
            for type_id in types['id']:
                type_sum = formulas[formulas['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum
                
            for year_id in years['id']:
                year_sum = types[types['parent'] == year_id]['id'].apply(
                    lambda x: hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].values[0] if not hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].empty else 0
                ).sum()
                hierarchy_data.loc[hierarchy_data['id'] == year_id, 'count'] = year_sum
            
            total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
            hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100
            
            custom_colors = custom_colors or {}
            unique_labels = hierarchy_data['label'].unique()
            color_map = st.session_state.get('color_map', create_color_map(unique_labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe))
            st.session_state.color_map = color_map
            
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
            sunburst_data = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            
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
            
            custom_colors = custom_colors or {}
            unique_labels = hierarchy_data['label'].unique()
            color_map = st.session_state.get('color_map', create_color_map(unique_labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe))
            st.session_state.color_map = color_map
            
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

        return fig, hierarchy_data, sunburst_data

    except Exception as e:
        st.error(f"Failed to generate top N sunburst chart: {str(e)}")
        update_log(f"Top N sunburst chart error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None, None, None
