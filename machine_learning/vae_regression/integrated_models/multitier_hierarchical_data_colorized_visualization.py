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
        
        if year_range and 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
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

def create_color_map(labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe):
    color_map = custom_colors.copy()
    if discrete_mode:
        colors = COLORBLIND_DISCRETE if colorblind_safe else DISCRETE_COLORS
        for i, label in enumerate(labels):
            if label not in color_map:
                if label.startswith('Other') or label.startswith('Sub-Other'):
                    mtype = label.split(' ')[1].split('L')[0] if 'L' in label else label.split(' ')[1]
                    mtype = mtype if mtype in ['p-type', 'n-type'] else 'p-type'
                    other_colors = validate_color_scale(OTHER_COLOR_SCALES.get(mtype, 'Greys'))
                    color_map[label] = other_colors[min(i % len(other_colors), len(other_colors)-1)]
                else:
                    color_map[label] = colors[i % len(colors)]
    else:
        colors = validate_color_scale(colormap_choice if colormap_choice in COLOR_SCALES else 'cividis')
        for label in labels:
            if label not in color_map and (label.startswith('Other') or label.startswith('Sub-Other')):
                mtype = label.split(' ')[1].split('L')[0] if 'L' in label else label.split(' ')[1]
                mtype = mtype if mtype in ['p-type', 'n-type'] else 'p-type'
                other_colors = validate_color_scale(OTHER_COLOR_SCALES.get(mtype, 'Greys'))
                color_map[label] = other_colors[-1]
    return color_map

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

        if excluded_labels:
            if 'material_type' in df.columns:
                df = df[~df["material_type"].isin(excluded_labels)]
            if 'material' in df.columns:
                df = df[~df["material"].isin(excluded_labels)]
            if 'classification' in df.columns:
                df = df[~df["classification"].isin(excluded_labels)]
        
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
            color_map = create_color_map(unique_labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe)
            st.session_state.color_map = color_map
            
            if discrete_mode:
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
                
                colors = np.log1p(hierarchy_data['count'].astype(float))
                other_mask = hierarchy_data['label'].str.contains('Other|Sub-Other', na=False)
                if other_mask.any():
                    for mtype in ['p-type', 'n-type']:
                        mask = hierarchy_data['label'].str.contains(f'Other {mtype}|Sub-Other {mtype}', na=False)
                        if mask.any():
                            other_count = hierarchy_data.loc[mask, 'count'].astype(float).iloc[0]
                            colorscale = validate_color_scale(colormap_choice)
                            color_idx = min(int((np.log1p(other_count) / colors.max()) * (len(colorscale) - 1)), len(colorscale) - 1)
                            colors[mask] = colorscale[color_idx]
                
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
                        colorbar=dict(title="Log(Count+1)"),
                        cmin=colors.min(),
                        cmax=colors.max()
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
            color_map = create_color_map(unique_labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe)
            st.session_state.color_map = color_map
            
            if discrete_mode:
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
                
                colors = np.log1p(hierarchy_data['count'].astype(float))
                other_mask = hierarchy_data['label'].str.contains('Other|Sub-Other', na=False)
                if other_mask.any():
                    for mtype in ['p-type', 'n-type']:
                        mask = hierarchy_data['label'].str.contains(f'Other {mtype}|Sub-Other {mtype}', na=False)
                        if mask.any():
                            other_count = hierarchy_data.loc[mask, 'count'].astype(float).iloc[0]
                            colorscale = validate_color_scale(colormap_choice)
                            color_idx = min(int((np.log1p(other_count) / colors.max()) * (len(colorscale) - 1)), len(colorscale) - 1)
                            colors[mask] = colorscale[color_idx]
                
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
                        colorbar=dict(title="Log(Count+1)"),
                        cmin=colors.min(),
                        cmax=colors.max()
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

        if excluded_labels:
            if 'material_type' in df.columns:
                df = df[~df["material_type"].isin(excluded_labels)]
            if 'material' in df.columns:
                df = df[~df["material"].isin(excluded_labels)]
            if 'classification' in df.columns:
                df = df[~df["classification"].isin(excluded_labels)]
        
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
            sunburst_data = df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            
            grouped_data = []
            for year in sunburst_data['year'].unique():
                for mtype in sunburst_data['material_type'].unique():
                    type_data = sunburst_data[(sunburst_data['year'] == year) & 
                                            (sunburst_data['material_type'] == mtype)]
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
            
            if discrete_mode:
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
                
                colors = np.log1p(hierarchy_data['count'].astype(float))
                other_mask = hierarchy_data['label'].str.contains('Other|Sub-Other', na=False)
                if other_mask.any():
                    for mtype in ['p-type', 'n-type']:
                        mask = hierarchy_data['label'].str.contains(f'Other {mtype}|Sub-Other {mtype}', na=False)
                        if mask.any():
                            other_count = hierarchy_data.loc[mask, 'count'].astype(float).iloc[0]
                            colorscale = validate_color_scale(colormap_choice)
                            color_idx = min(int((np.log1p(other_count) / colors.max()) * (len(colorscale) - 1)), len(colorscale) - 1)
                            colors[mask] = colorscale[color_idx]
                
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
                        colorbar=dict(title="Log(Count+1)"),
                        cmin=colors.min(),
                        cmax=colors.max()
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
            
            if discrete_mode:
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
                
                colors = np.log1p(hierarchy_data['count'].astype(float))
                other_mask = hierarchy_data['label'].str.contains('Other|Sub-Other', na=False)
                if other_mask.any():
                    for mtype in ['p-type', 'n-type']:
                        mask = hierarchy_data['label'].str.contains(f'Other {mtype}|Sub-Other {mtype}', na=False)
                        if mask.any():
                            other_count = hierarchy_data.loc[mask, 'count'].astype(float).iloc[0]
                            colorscale = validate_color_scale(colormap_choice)
                            color_idx = min(int((np.log1p(other_count) / colors.max()) * (len(colorscale) - 1)), len(colorscale) - 1)
                            colors[mask] = colorscale[color_idx]
                
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
                        colorbar=dict(title="Log(Count+1)"),
                        cmin=colors.min(),
                        cmax=colors.max()
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

def create_expanded_sunburst(df, top_ns, colormap_choice, discrete_mode, show_labels, label_fontsize, 
                            excluded_labels, year_range=None, chart_height=800, branchvalues='total',
                            label_threshold=1.0, show_values=True, show_percentages=True, custom_colors=None, colorblind_safe=False):
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            update_log("No data available for expanded sunburst chart")
            return None, None
        
        if colormap_choice.lower() not in COLOR_SCALES:
            update_log(f"Invalid colormap '{colormap_choice}' selected. Falling back to 'cividis'.")
            colormap_choice = 'cividis'

        if excluded_labels:
            if 'material_type' in df.columns:
                df = df[~df["material_type"].isin(excluded_labels)]
            if 'material' in df.columns:
                df = df[~df["material"].isin(excluded_labels)]
            if 'classification' in df.columns:
                df = df[~df["classification"].isin(excluded_labels)]
        
        if 'material' in df.columns and 'classification' in df.columns:
            df = df.rename(columns={'material': 'formula', 'classification': 'material_type'})
        
        required_cols = ['formula', 'material_type']
        if 'year' in df.columns:
            required_cols.append('year')
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            update_log(f"Missing required columns for expanded sunburst: {', '.join(missing)}")
            return None, None
        
        if 'count' in df.columns:
            df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
        
        unique_formulas = df.groupby(['material_type', 'year'] if 'year' in df.columns else ['material_type'])['formula'].nunique()
        max_formulas = unique_formulas.min()
        if max_formulas < max(top_ns):
            st.warning(f"Insufficient unique formulas ({max_formulas}) for requested top {max(top_ns)} in some categories. Adjusting layers.")
            update_log(f"Insufficient unique formulas ({max_formulas}) for top {max(top_ns)}")
            top_ns = [min(n, max_formulas) for n in top_ns]
        
        hierarchy_data = []
        all_sunburst_data = []
        current_df = df.copy()
        
        if 'year' in df.columns:
            sunburst_data = current_df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            
            for layer_idx, top_n in enumerate(top_ns):
                grouped_data = []
                parent_level = 'type_id' if layer_idx == 0 else f'other_{layer_idx-1}_id'
                
                for year in sunburst_data['year'].unique():
                    for mtype in sunburst_data['material_type'].unique():
                        type_data = sunburst_data[(sunburst_data['year'] == year) & 
                                                (sunburst_data['material_type'] == mtype)]
                        if type_data.empty:
                            continue
                        
                        top_formulas = type_data.nlargest(top_n, 'count')
                        other_data = type_data[~type_data['formula'].isin(top_formulas['formula'])]
                        
                        if layer_idx == 0:
                            type_id = f"{year}_{mtype}"
                        else:
                            type_id = f"{year}_{mtype}_other_{layer_idx-1}"
                        
                        type_data = top_formulas.copy()
                        type_data['parent'] = type_id
                        type_data['id'] = type_data['formula'].apply(lambda x: f"{type_id}_{x}")
                        
                        if not other_data.empty and layer_idx < len(top_ns) - 1:
                            other_count = other_data['count'].sum()
                            other_row = pd.DataFrame({
                                'year': [year],
                                'material_type': [mtype],
                                'formula': [f'Other {mtype} L{layer_idx+1}'],
                                'count': [other_count],
                                'parent': [type_id],
                                'id': [f"{type_id}_other_{layer_idx}"]
                            })
                            type_data = pd.concat([type_data, other_row], ignore_index=True)
                        
                        grouped_data.append(type_data)
                
                if grouped_data:
                    sunburst_data = pd.concat(grouped_data, ignore_index=True)
                    all_sunburst_data.append(sunburst_data)
                else:
                    update_log(f"No data for layer {layer_idx + 1}")
                    break
                
                if layer_idx < len(top_ns) - 1:
                    other_data = sunburst_data[sunburst_data['formula'].str.startswith('Other')]
                    if other_data.empty:
                        update_log(f"No 'Other' categories to expand in layer {layer_idx + 1}")
                        break
                    other_formulas = df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
                    for idx, row in other_data.iterrows():
                        year, mtype = row['year'], row['material_type']
                        prev_formulas = sunburst_data[sunburst_data['id'].str.startswith(f"{year}_{mtype}") & 
                                                     ~sunburst_data['formula'].str.startswith('Other')]['formula'].tolist()
                        other_formulas = other_formulas[
                            (other_formulas['year'] == year) & 
                            (other_formulas['material_type'] == mtype) & 
                            (~other_formulas['formula'].isin(prev_formulas))
                        ]
                        other_formulas['id'] = other_formulas['formula'].apply(lambda x: f"{row['id']}_{x}")
                        other_formulas['parent'] = row['id']
                        grouped_data.append(other_formulas)
                    if grouped_data:
                        sunburst_data = pd.concat(grouped_data, ignore_index=True)
                    else:
                        update_log(f"No remaining formulas for layer {layer_idx + 2}")
                        break
            
            years = sunburst_data[['year']].drop_duplicates()
            years['parent'] = ''
            years['id'] = years['year'].astype(str)
            years['label'] = years['year']
            
            types = sunburst_data[['year', 'material_type']].drop_duplicates()
            types['parent'] = types['year'].astype(str)
            types['id'] = types['year'].astype(str) + '_' + types['material_type']
            types['label'] = types['material_type']
            
            hierarchy_data = [years, types]
            
            for layer_idx, data in enumerate(all_sunburst_data):
                formulas = data[['id', 'parent', 'formula', 'count']].copy()
                formulas['label'] = formulas['formula']
                hierarchy_data.append(formulas)
            
            hierarchy_data = pd.concat(hierarchy_data, ignore_index=True)
            
            for type_id in types['id']:
                type_sum = hierarchy_data[hierarchy_data['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum
                
            for year_id in years['id']:
                year_sum = types[types['parent'] == year_id]['id'].apply(
                    lambda x: hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].values[0] if not hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].empty else 0
                ).sum()
                hierarchy_data.loc[hierarchy_data['id'] == year_id, 'count'] = year_sum
            
            total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
            hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100
            
            hierarchy_data['count'] = pd.to_numeric(hierarchy_data['count'], errors='coerce').fillna(0)
            
            custom_colors = custom_colors or {}
            unique_labels = hierarchy_data['label'].unique()
            color_map = st.session_state.get('color_map', create_color_map(unique_labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe))
            st.session_state.color_map = color_map
            
            if discrete_mode:
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
                
                colors = np.log1p(hierarchy_data['count'].astype(float))
                other_mask = hierarchy_data['label'].str.contains('Other|Sub-Other', na=False)
                if other_mask.any():
                    for mtype in ['p-type', 'n-type']:
                        mask = hierarchy_data['label'].str.contains(f'Other {mtype}|Sub-Other {mtype}', na=False)
                        if mask.any():
                            other_count = hierarchy_data.loc[mask, 'count'].astype(float).iloc[0]
                            colorscale = validate_color_scale(colormap_choice)
                            color_idx = min(int((np.log1p(other_count) / colors.max()) * (len(colorscale) - 1)), len(colorscale) - 1)
                            colors[mask] = colorscale[color_idx]
                
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
                        colorbar=dict(title="Log(Count+1)"),
                        cmin=colors.min(),
                        cmax=colors.max()
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
            sunburst_data = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            
            for layer_idx, top_n in enumerate(top_ns):
                grouped_data = []
                for mtype in sunburst_data['material_type'].unique():
                    type_data = sunburst_data[sunburst_data['material_type'] == mtype]
                    top_formulas = type_data.nlargest(top_n, 'count')
                    other_data = type_data[~type_data['formula'].isin(top_formulas['formula'])]
                    
                    type_id = mtype if layer_idx == 0 else f"{mtype}_other_{layer_idx-1}"
                    
                    type_data = top_formulas.copy()
                    type_data['parent'] = type_id
                    type_data['id'] = type_data['formula'].apply(lambda x: f"{type_id}_{x}")
                    
                    if not other_data.empty and layer_idx < len(top_ns) - 1:
                        other_count = other_data['count'].sum()
                        other_row = pd.DataFrame({
                            'material_type': [mtype],
                            'formula': [f'Other {mtype} L{layer_idx+1}'],
                            'count': [other_count],
                            'parent': [type_id],
                            'id': [f"{type_id}_other_{layer_idx}"]
                        })
                        type_data = pd.concat([type_data, other_row], ignore_index=True)
                    
                    grouped_data.append(type_data)
                
                if grouped_data:
                    sunburst_data = pd.concat(grouped_data, ignore_index=True)
                    all_sunburst_data.append(sunburst_data)
                else:
                    update_log(f"No data for layer {layer_idx + 1} (no year)")
                    break
                
                if layer_idx < len(top_ns) - 1:
                    other_data = sunburst_data[sunburst_data['formula'].str.startswith('Other')]
                    if other_data.empty:
                        update_log(f"No 'Other' categories to expand in layer {layer_idx + 1} (no year)")
                        break
                    other_formulas = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
                    for idx, row in other_data.iterrows():
                        mtype = row['material_type']
                        prev_formulas = sunburst_data[sunburst_data['material_type'] == mtype & 
                                                     ~sunburst_data['formula'].str.startswith('Other')]['formula'].tolist()
                        other_formulas = other_formulas[
                            (other_formulas['material_type'] == mtype) & 
                            (~other_formulas['formula'].isin(prev_formulas))
                        ]
                        other_formulas['id'] = other_formulas['formula'].apply(lambda x: f"{row['id']}_{x}")
                        other_formulas['parent'] = row['id']
                        grouped_data.append(other_formulas)
                    if grouped_data:
                        sunburst_data = pd.concat(grouped_data, ignore_index=True)
                    else:
                        update_log(f"No remaining formulas for layer {layer_idx + 2} (no year)")
                        break
            
            types = sunburst_data[['material_type']].drop_duplicates()
            types['parent'] = ''
            types['id'] = types['material_type']
            types['label'] = types['material_type']
            
            hierarchy_data = [types]
            
            for layer_idx, data in enumerate(all_sunburst_data):
                formulas = data[['id', 'parent', 'formula', 'count']].copy()
                formulas['label'] = formulas['formula']
                hierarchy_data.append(formulas)
            
            hierarchy_data = pd.concat(hierarchy_data, ignore_index=True)
            
            for type_id in types['id']:
                type_sum = hierarchy_data[hierarchy_data['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum
            
            total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
            hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100
            
            hierarchy_data['count'] = pd.to_numeric(hierarchy_data['count'], errors='coerce').fillna(0)
            
            custom_colors = custom_colors or {}
            unique_labels = hierarchy_data['label'].unique()
            color_map = st.session_state.get('color_map', create_color_map(unique_labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe))
            st.session_state.color_map = color_map
            
            if discrete_mode:
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
                
                colors = np.log1p(hierarchy_data['count'].astype(float))
                other_mask = hierarchy_data['label'].str.contains('Other|Sub-Other', na=False)
                if other_mask.any():
                    for mtype in ['p-type', 'n-type']:
                        mask = hierarchy_data['label'].str.contains(f'Other {mtype}|Sub-Other {mtype}', na=False)
                        if mask.any():
                            other_count = hierarchy_data.loc[mask, 'count'].astype(float).iloc[0]
                            colorscale = validate_color_scale(colormap_choice)
                            color_idx = min(int((np.log1p(other_count) / colors.max()) * (len(colorscale) - 1)), len(colorscale) - 1)
                            colors[mask] = colorscale[color_idx]
                
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
                        colorbar=dict(title="Log(Count+1)"),
                        cmin=colors.min(),
                        cmax=colors.max()
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

        fig.update_layout(
            height=chart_height,
            plot_bgcolor="rgba(255,255,255,1)",
            paper_bgcolor="rgba(255,255,255,1)",
            margin=dict(t=80, l=20, r=20, b=20),
            font=dict(family="Arial, sans-serif", size=12, color="#000000"),
            title=dict(
                text=f"Expanded Sunburst with {len(top_ns)} Layers (Top {', '.join(map(str, top_ns))} Materials)",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, family="Arial, sans-serif")
            )
        )

        return fig, hierarchy_data

    except Exception as e:
        st.error(f"Failed to generate expanded sunburst chart: {str(e)}")
        update_log(f"Expanded sunburst chart error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None, None

# Streamlit UI
st.set_page_config(page_title="Three-Tier Thermoelectric Material Analysis", layout="wide")
st.title("🌞 Three-Tier Hierarchy Analysis for Thermoelectric Materials")

# Initialize session state
if "db_file" not in st.session_state:
    db_files = [f for f in os.listdir(DB_DIR) if f.endswith('.db')]
    st.session_state.db_file = os.path.join(DB_DIR, db_files[0]) if db_files else None

if "data_df" not in st.session_state:
    st.session_state.data_df = None

if "hierarchy_data" not in st.session_state:
    st.session_state.hierarchy_data = None

if "expanded_layers" not in st.session_state:
    st.session_state.expanded_layers = [5, 5, 5]

if "color_map" not in st.session_state:
    st.session_state.color_map = {}

if "custom_colors" not in st.session_state:
    st.session_state.custom_colors = {}

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

if year_range and st.sidebar.button("Apply Year Filter") and st.session_state.db_file:
    with st.spinner("Loading filtered data..."):
        st.session_state.data_df = load_data_from_db(st.session_state.db_file, year_range)

# Chart customization
st.sidebar.header("Chart Customization")
discrete_mode = st.sidebar.radio("Color Mode", ["Discrete (by type)", "Continuous (by count)"]) == "Discrete (by type)"
colorblind_safe = st.sidebar.checkbox("Use Colorblind-Safe Palette", value=True)
if discrete_mode:
    st.sidebar.info("In Discrete mode, colors are assigned by material type using a qualitative palette.")
colormap_choice = st.sidebar.selectbox("Choose Color Map (Continuous mode)", COLOR_SCALES, index=COLOR_SCALES.index('cividis'))

# Custom color selection
st.sidebar.header("Custom Colors")
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
    
    selected_label = st.sidebar.selectbox("Select Label for Custom Color", options=list(set(all_labels)))
    custom_color = st.sidebar.color_picker("Pick a Color", value="#000000", key=f"color_{selected_label}")
    if st.sidebar.button("Apply Custom Color"):
        st.session_state.custom_colors[selected_label] = custom_color
        update_log(f"Applied custom color {custom_color} to label {selected_label}")

show_labels = st.sidebar.checkbox("Show Labels", value=True)
label_fontsize = st.sidebar.slider("Label Font Size", 8, 100, 12)
chart_height = st.sidebar.slider("Chart Height", 400, 1200, 800)
branchvalues = st.sidebar.selectbox("Branch Values", ["total", "remainder"], index=0)

# Label visibility options
st.sidebar.header("Label Visibility")
label_threshold = st.sidebar.slider("Label Threshold (%)", 0.0, 10.0, 1.0, 0.1,
                                   help="Only show labels for segments larger than this percentage of the total")
show_values = st.sidebar.checkbox("Show Values in Labels", value=True)
show_percentages = st.sidebar.checkbox("Show Percentages in Labels", value=True)

# Exclude labels
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

# Expanded sunburst layers
st.sidebar.header("Expanded Sunburst Chart")
if st.session_state.data_df is not None:
    max_formulas = max(50, len(st.session_state.data_df['formula'].unique()) if 'formula' in st.session_state.data_df.columns else 50)
    for layer_idx in range(len(st.session_state.expanded_layers)):
        st.sidebar.slider(
            f"Top Materials for Layer {layer_idx + 1}",
            1, max_formulas, st.session_state.expanded_layers[layer_idx],
            key=f"top_n_layer_{layer_idx}",
            help=f"Number of top materials to display in layer {layer_idx + 1}. Remaining materials are grouped as 'Other'."
        )
        st.session_state.expanded_layers[layer_idx] = st.session_state[f"top_n_layer_{layer_idx}"]

    if st.sidebar.button("Add Another Layer"):
        if len(st.session_state.expanded_layers) < 5:
            st.session_state.expanded_layers.append(5)
            update_log(f"Added new layer, total layers: {len(st.session_state.expanded_layers)}")
        else:
            st.sidebar.warning("Maximum number of layers (5) reached.")
            update_log("Attempted to add layer beyond maximum (5)")

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
        show_percentages,
        custom_colors=st.session_state.custom_colors,
        colorblind_safe=colorblind_safe
    )
    
    st.session_state.hierarchy_data = hierarchy_data

    if fig_sunburst:
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    st.subheader(f"Top {top_n} Materials Sunburst Chart")
    if st.sidebar.button("Generate Top N Materials Chart"):
        with st.spinner("Generating top N materials chart..."):
            fig_top_n, top_n_hierarchy_data, _ = create_top_n_sunburst(
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
                show_percentages,
                custom_colors=st.session_state.custom_colors,
                colorblind_safe=colorblind_safe
            )
            
            if fig_top_n:
                st.plotly_chart(fig_top_n, use_container_width=True)
            else:
                st.warning("Unable to generate top N materials chart. Check data format.")
    
    st.subheader(f"Expanded Sunburst Chart with {len(st.session_state.expanded_layers)} Layers")
    if st.sidebar.button("Generate Expanded Sunburst Chart"):
        with st.spinner("Generating expanded sunburst chart..."):
            fig_expanded, expanded_hierarchy_data = create_expanded_sunburst(
                st.session_state.data_df,
                st.session_state.expanded_layers,
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
                show_percentages,
                custom_colors=st.session_state.custom_colors,
                colorblind_safe=colorblind_safe
            )
            
            if fig_expanded:
                st.plotly_chart(fig_expanded, use_container_width=True)
            else:
                st.warning("Unable to generate expanded sunburst chart. Check data format or logs for details.")
    
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
            4. **Use colorblind-safe palette** for accessibility
            5. **Use Discrete mode** for clear differentiation of material types
            6. **Assign custom colors** to highlight specific materials or types
            7. **Export hierarchy data** as CSV or JSON for further analysis in other tools
            8. **Check logs** below for any data or rendering issues
            """)
        
        with st.expander("View Application Logs"):
            if 'log_buffer' in st.session_state and st.session_state.log_buffer:
                st.text_area("Logs (last 20 entries)", "\n".join(st.session_state.log_buffer), height=200)
            else:
                st.info("No logs available yet.")
        
        with st.expander("View Color Map"):
            if 'color_map' in st.session_state and st.session_state.color_map:
                st.write("Current color assignments:")
                for label, color in st.session_state.color_map.items():
                    st.markdown(f"<span style='color:{color};font-weight:bold'>{label}</span>: {color}", unsafe_allow_html=True)
            else:
                st.info("No color map available. Generate a chart to see color assignments.")
