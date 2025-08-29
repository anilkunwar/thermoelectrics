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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from plotly.subplots import make_subplots
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

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
                elif label in ['p-type', 'n-type']:
                    color_map[label] = validate_color_scale(OTHER_COLOR_SCALES.get(label, 'Greys'))[4]
                else:
                    color_map[label] = colors[i % len(colors)]
    else:
        colors = validate_color_scale(colormap_choice if colormap_choice in COLOR_SCALES else 'cividis')
        for label in labels:
            if label in highlight_materials:
                color_map[label] = HIGHLIGHT_COLORS[labels.tolist().index(label) % len(HIGHLIGHT_COLORS)]
            elif label not in color_map and (label.startswith('Other') or label.startswith('Sub-Other')):
                mtype = label.split(' ')[1].split('L')[0] if 'L' in label else label.split(' ')[1]
                mtype = mtype if mtype in ['p-type', 'n-type'] else 'p-type'
                other_colors = validate_color_scale(OTHER_COLOR_SCALES.get(mtype, 'Greys'))
                color_map[label] = other_colors[-1]
            elif label not in color_map and label in ['p-type', 'n-type']:
                color_map[label] = validate_color_scale(OTHER_COLOR_SCALES.get(label, 'Greys'))[4]
            elif label not in color_map:
                color_map[label] = '#808080'
    return color_map

def filter_excluded_labels(df, excluded_labels, update_log=None):
    if not excluded_labels:
        return df
    
    mask = False
    if 'material_type' in df.columns:
        mask |= df["material_type"].isin(excluded_labels)
    if 'material' in df.columns:
        mask |= df["material"].isin(excluded_labels)
    if 'classification' in df.columns:
        mask |= df["classification"].isin(excluded_labels)
    if 'formula' in df.columns:
        mask |= df["formula"].isin(excluded_labels)

    if isinstance(mask, pd.Series) and mask.any():
        df = df[~mask]
        if update_log:
            update_log(f"Excluded {mask.sum()} rows based on excluded_labels")
    else:
        if update_log:
            update_log("No valid rows matched for exclusion filtering")
    return df

def create_highlighted_sunburst(df, highlight_materials, colormap_choice, discrete_mode, show_labels, label_fontsize, 
                               excluded_labels, year_range=None, chart_height=800, branchvalues='total',
                               label_threshold=1.0, show_values=True, show_percentages=True, custom_colors=None, 
                               colorblind_safe=False, min_count_scale=10.0, outline_thickness=1):
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            update_log("No data available for highlighted sunburst chart")
            return None, None
        
        if colormap_choice.lower() not in COLOR_SCALES:
            update_log(f"Invalid colormap '{colormap_choice}' selected. Falling back to 'cividis'.")
            colormap_choice = 'cividis'

        df = filter_excluded_labels(df, excluded_labels, update_log)

        if 'material' in df.columns and 'classification' in df.columns:
            df = df.rename(columns={'material': 'formula', 'classification': 'material_type'})
        
        required_cols = ['formula', 'material_type']
        if 'year' in df.columns:
            required_cols.append('year')
        
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            update_log(f"Missing required columns for highlighted sunburst: {', '.join(missing)}")
            return None, None
        
        if highlight_materials:
            available_formulas = df['formula'].unique().tolist()
            valid_highlights = [m for m in highlight_materials if m in available_formulas]
            invalid_highlights = [m for m in highlight_materials if m not in valid_highlights]
            if invalid_highlights:
                st.warning(f"The following highlighted materials are not in the dataset: {', '.join(invalid_highlights)}")
                update_log(f"Invalid highlighted materials: {invalid_highlights}")
            if not valid_highlights:
                st.error("No valid materials selected for highlighting. Please choose materials present in the dataset.")
                update_log("No valid materials selected for highlighting")
                return None, None
            highlight_materials = valid_highlights
            update_log(f"Generating highlighted sunburst for materials: {highlight_materials}")
            df = df[df['formula'].isin(highlight_materials) | df['material_type'].isin(['p-type', 'n-type'])]
            if df.empty:
                st.error("No data remains after filtering for highlighted materials. Check if materials exist in the selected year range.")
                update_log("No data after filtering for highlighted materials")
                return None, None
        else:
            st.info("No materials selected for highlighting. Generating standard sunburst chart.")
            update_log("No materials selected for highlighting, using all data")

        if 'count' in df.columns:
            df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0)
        
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            if year_range:
                df = df[df['year'].notna() & (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
        if 'year' in df.columns:
            sunburst_data = df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            sunburst_data['scaled_count'] = sunburst_data['count'].apply(lambda x: max(x, min_count_scale) if x > 0 and x <= 2 else x)
            
            sunburst_data['year_id'] = sunburst_data['year'].astype(str)
            sunburst_data['type_id'] = sunburst_data['year_id'] + '_' + sunburst_data['material_type']
            sunburst_data['formula_id'] = sunburst_data['type_id'] + '_' + sunburst_data['formula']
            
            years = sunburst_data[['year_id', 'year']].drop_duplicates()
            years['parent'] = ''
            years['id'] = years['year_id']
            
            types = sunburst_data[['type_id', 'material_type', 'year_id']].drop_duplicates()
            types['parent'] = types['year_id']
            types['id'] = types['type_id']
            
            formulas = sunburst_data[['formula_id', 'formula', 'count', 'scaled_count', 'type_id']].copy()
            formulas['parent'] = formulas['type_id']
            formulas['id'] = formulas['formula_id']
            
            hierarchy_data = pd.concat([
                years[['id', 'parent', 'year']].rename(columns={'year': 'label'}),
                types[['id', 'parent', 'material_type']].rename(columns={'material_type': 'label'}),
                formulas[['id', 'parent', 'formula', 'count', 'scaled_count']].rename(columns={'formula': 'label'})
            ], ignore_index=True)
            
            for type_id in types['id']:
                type_sum = formulas[formulas['parent'] == type_id]['count'].sum()
                type_scaled_sum = formulas[formulas['parent'] == type_id]['scaled_count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'scaled_count'] = type_scaled_sum
                
            for year_id in years['id']:
                year_sum = types[types['parent'] == year_id]['id'].apply(
                    lambda x: hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].values[0] if not hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].empty else 0
                ).sum()
                year_scaled_sum = types[types['parent'] == year_id]['id'].apply(
                    lambda x: hierarchy_data.loc[hierarchy_data['id'] == x, 'scaled_count'].values[0] if not hierarchy_data.loc[hierarchy_data['id'] == x, 'scaled_count'].empty else 0
                ).sum()
                hierarchy_data.loc[hierarchy_data['id'] == year_id, 'count'] = year_sum
                hierarchy_data.loc[hierarchy_data['id'] == year_id, 'scaled_count'] = year_scaled_sum
        else:
            sunburst_data = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            sunburst_data['scaled_count'] = sunburst_data['count'].apply(lambda x: max(x, min_count_scale) if x > 0 and x <= 2 else x)
            
            sunburst_data['type_id'] = sunburst_data['material_type']
            sunburst_data['formula_id'] = sunburst_data['type_id'] + '_' + sunburst_data['formula']
            
            types = sunburst_data[['type_id', 'material_type']].drop_duplicates()
            types['parent'] = ''
            types['id'] = types['type_id']
            
            formulas = sunburst_data[['formula_id', 'formula', 'count', 'scaled_count', 'type_id']].copy()
            formulas['parent'] = formulas['type_id']
            formulas['id'] = formulas['formula_id']
            
            hierarchy_data = pd.concat([
                types[['id', 'parent', 'material_type']].rename(columns={'material_type': 'label'}),
                formulas[['id', 'parent', 'formula', 'count', 'scaled_count']].rename(columns={'formula': 'label'})
            ], ignore_index=True)
            
            for type_id in types['id']:
                type_sum = formulas[formulas['parent'] == type_id]['count'].sum()
                type_scaled_sum = formulas[formulas['parent'] == type_id]['scaled_count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'scaled_count'] = type_scaled_sum
        
        total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
        hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100
        
        custom_colors = custom_colors or {}
        unique_labels = hierarchy_data['label'].unique()
        color_map = st.session_state.get('color_map', create_color_map(unique_labels, discrete_mode, colormap_choice, custom_colors, colorblind_safe, highlight_materials))
        st.session_state.color_map = color_map
        
        hierarchy_data['font_size'] = hierarchy_data['label'].apply(lambda x: int(label_fontsize * 1.5) if x in highlight_materials else label_fontsize)
        hierarchy_data['font_weight'] = hierarchy_data['label'].apply(lambda x: 'bold' if x in highlight_materials else 'normal')
        
        if highlight_materials:
            line_widths = hierarchy_data['label'].apply(lambda x: outline_thickness if x in highlight_materials else 0).tolist()
            line_colors = hierarchy_data['label'].apply(lambda x: '#000000' if x in highlight_materials else '#FFFFFF').tolist()
            update_log(f"Line colors assigned: {line_colors[:10]}...")
        else:
            line_widths = [0] * len(hierarchy_data)
            line_colors = ['#FFFFFF'] * len(hierarchy_data)
            update_log("No highlighted materials, using default line colors (white)")

        hierarchy_data['display_text'] = hierarchy_data['label']
        if label_threshold > 0:
            hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
        
        hierarchy_data['text_template'] = hierarchy_data.apply(
            lambda row: (
                f"{row['label']}<br>Count: {row['count']}<br>Type: {row['parent'].split('_')[-1] if '_' in row['parent'] else row['parent']}"
                if row['label'] in highlight_materials else row['display_text']
            ), axis=1
        )

        if discrete_mode:
            hierarchy_data['color'] = hierarchy_data['label'].map(color_map)
            hierarchy_data.loc[hierarchy_data['parent'] == '', 'color'] = '#E5ECF6'
            
            fig = go.Figure(go.Sunburst(
                ids=hierarchy_data['id'],
                labels=hierarchy_data['display_text'],
                parents=hierarchy_data['parent'],
                values=hierarchy_data['scaled_count'],
                branchvalues=branchvalues,
                marker=dict(
                    colors=hierarchy_data['color'],
                    line=dict(width=line_widths, color=line_colors)
                ),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
                textinfo="label+text" if show_labels else "none",
                texttemplate=hierarchy_data['text_template'],
                textfont=dict(
                    size=[hierarchy_data['font_size'].iloc[i] for i in range(len(hierarchy_data))],
                    family="Arial, sans-serif",
                    weight=[hierarchy_data['font_weight'].iloc[i] for i in range(len(hierarchy_data))]
                ),
                insidetextorientation='radial',
                textangle=0,
                sort=False
            ))
        else:
            colors = np.log1p(hierarchy_data['scaled_count'].astype(float))
            hierarchy_data['color'] = hierarchy_data['label'].map(color_map)
            
            fig = go.Figure(go.Sunburst(
                ids=hierarchy_data['id'],
                labels=hierarchy_data['display_text'],
                parents=hierarchy_data['parent'],
                values=hierarchy_data['scaled_count'],
                branchvalues=branchvalues,
                marker=dict(
                    colors=hierarchy_data['color'],
                    colorscale=colormap_choice.lower(),
                    showscale=True,
                    colorbar=dict(title="Log(Scaled Count+1)"),
                    cmin=colors.min(),
                    cmax=colors.max(),
                    line=dict(width=line_widths, color=line_colors)
                ),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
                textinfo="label+text" if show_labels else "none",
                texttemplate=hierarchy_data['text_template'],
                textfont=dict(
                    size=[hierarchy_data['font_size'].iloc[i] for i in range(len(hierarchy_data))],
                    family="Arial, sans-serif",
                    weight=[hierarchy_data['font_weight'].iloc[i] for i in range(len(hierarchy_data))]
                ),
                insidetextorientation='radial',
                textangle=0,
                sort=False
            ))

        fig.update_layout(
            height=chart_height,
            plot_bgcolor="rgba(255,255,255,1)",
            paper_bgcolor="rgba(255,255,255,1)",
            margin=dict(t=80, l=20, r=20, b=20),
            font=dict(family="Arial, sans-serif", size=12, color="#000000"),
            title=dict(
                text=f"Sunburst with Highlighted Materials: {', '.join(highlight_materials) if highlight_materials else 'None'}",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, family="Arial, sans-serif")
            )
        )

        return fig, hierarchy_data

    except Exception as e:
        st.error(f"Failed to generate highlighted sunburst chart: {str(e)}. Check if highlighted materials exist in the dataset and review logs for details.")
        update_log(f"Highlighted sunburst chart error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None, None

def create_top_n_sunburst(df, top_n, colormap_choice, discrete_mode, show_labels, label_fontsize, 
                         excluded_labels, year_range=None, chart_height=800, branchvalues='total',
                         label_threshold=1.0, show_values=True, show_percentages=True, custom_colors=None, 
                         colorblind_safe=False, outline_thickness=1):
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            return None, None, None
        
        if colormap_choice.lower() not in COLOR_SCALES:
            update_log(f"Invalid colormap '{colormap_choice}' selected. Falling back to 'cividis'.")
            colormap_choice = 'cividis'

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
        
        grouped_data = []
        if 'year' in df.columns:
            sunburst_data = df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            
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
        else:
            sunburst_data = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            
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
        
        if not grouped_data:
            st.error("No data available after filtering for top N materials")
            update_log("No data available after filtering for top N materials")
            return None, None, None
        
        sunburst_data = pd.concat(grouped_data, ignore_index=True)
        
        if 'year' in sunburst_data.columns:
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
        else:
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
        color_map = {'p-type': '#1f77b4', 'n-type': '#d62728'}  # Blue for p-type, Red for n-type
        for label in hierarchy_data['label']:
            if label.startswith('Other p-type'):
                color_map[label] = '#6baed6'
            elif label.startswith('Other n-type'):
                color_map[label] = '#fd8d3c'
            elif label not in color_map and label not in ['p-type', 'n-type']:
                mtype = hierarchy_data[hierarchy_data['label'] == label]['parent'].iloc[0].split('_')[-1] if '_' in hierarchy_data[hierarchy_data['label'] == label]['parent'].iloc[0] else hierarchy_data[hierarchy_data['label'] == label]['parent'].iloc[0]
                color_map[label] = '#1f77b4' if mtype == 'p-type' else '#d62728'
        st.session_state.color_map = color_map
        
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
                colors=hierarchy_data['label'].map(color_map),
                line=dict(width=[outline_thickness if label not in ['p-type', 'n-type'] else 0 for label in hierarchy_data['label']], color=['#FFFFFF'] * len(hierarchy_data))
            ),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.2%}<extra></extra>',
            textinfo="label+text" if show_labels else "none",
            texttemplate=(
                "%{label}" + 
                ("<br>%{value}" if show_values else "") + 
                ("<br>%{percentParent:.1%}" if show_percentages else "")
            ),
            textfont=dict(
                size=label_fontsize,
                family="Arial, sans-serif"
            ),
            insidetextorientation='radial',
            textangle=0,
            sort=False
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

def create_word_cloud(df, top_n, material_type=None, year_range=None, excluded_labels=None, color_map=None):
    try:
        if df is None or df.empty:
            st.error("No data available for word cloud")
            return None
        
        df = filter_excluded_labels(df, excluded_labels, update_log)
        
        if 'material' in df.columns and 'classification' in df.columns:
            df = df.rename(columns={'material': 'formula', 'classification': 'material_type'})
        
        if 'year' in df.columns and year_range:
            df = df[df['year'].notna() & (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
        if material_type:
            df = df[df['material_type'] == material_type]
        
        word_data = df.groupby('formula').size().reset_index(name='count')
        top_words = word_data.nlargest(top_n, 'count')
        word_freq = dict(zip(top_words['formula'], top_words['count']))
        
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return color_map.get(word, '#808080') if color_map else '#808080'
        
        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            min_font_size=10, max_font_size=100, random_state=42,
            color_func=color_func
        ).generate_from_frequencies(word_freq)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return buf
    
    except Exception as e:
        st.error(f"Failed to generate word cloud: {str(e)}")
        update_log(f"Word cloud error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None

def create_radar_chart(df, top_n, material_type=None, year_range=None, excluded_labels=None, color_map=None, curve_thickness=2):
    try:
        if df is None or df.empty:
            st.error("No data available for radar chart")
            return None
        
        df = filter_excluded_labels(df, excluded_labels, update_log)
        
        if 'material' in df.columns and 'classification' in df.columns:
            df = df.rename(columns={'material': 'formula', 'classification': 'material_type'})
        
        if 'year' in df.columns and year_range:
            df = df[df['year'].notna() & (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
        if material_type:
            df = df[df['material_type'] == material_type]
        
        data = df.groupby('formula').size().reset_index(name='count')
        top_data = data.nlargest(top_n, 'count')
        
        categories = top_data['formula'].tolist()
        values = top_data['count'].tolist()
        
        if len(values) < 3:
            st.error("Need at least 3 materials for a radar chart")
            return None
        
        max_value = max(values)
        values = [v / max_value * 100 if max_value > 0 else 0 for v in values]
        
        color = color_map.get(material_type, '#808080') if material_type and color_map else '#808080'
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the radar chart
            theta=categories + [categories[0]],
            fill='toself',
            name=material_type if material_type else 'All Materials',
            line=dict(width=curve_thickness, color=color)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
                angularaxis=dict(rotation=90, direction="clockwise")
            ),
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            title=dict(
                text=f"Top {top_n} {'Materials' if not material_type else material_type} Radar Chart",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Failed to generate radar chart: {str(e)}")
        update_log(f"Radar chart error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None

def create_histogram(df, top_n, material_type=None, year_range=None, excluded_labels=None, color_map=None, outline_thickness=1):
    try:
        if df is None or df.empty:
            st.error("No data available for histogram")
            return None
        
        df = filter_excluded_labels(df, excluded_labels, update_log)
        
        if 'material' in df.columns and 'classification' in df.columns:
            df = df.rename(columns={'material': 'formula', 'classification': 'material_type'})
        
        if 'year' in df.columns and year_range:
            df = df[df['year'].notna() & (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
        if material_type:
            df = df[df['material_type'] == material_type]
        
        data = df.groupby('formula').size().reset_index(name='count')
        top_data = data.nlargest(top_n, 'count')
        
        # Assign colors based on material type
        top_data['color'] = top_data['formula'].apply(
            lambda x: color_map.get(x, '#1f77b4' if material_type == 'p-type' else '#d62728' if material_type == 'n-type' else '#808080')
        )
        
        fig = go.Figure()
        for formula, row in top_data.iterrows():
            fig.add_trace(go.Bar(
                x=[formula],
                y=[row['count']],
                marker=dict(
                    color=row['color'],
                    line=dict(width=outline_thickness, color='#000000')
                ),
                name=formula
            ))
        
        fig.update_layout(
            title=f"Top {top_n} {'Materials' if not material_type else material_type} Histogram",
            xaxis_title="Material",
            yaxis_title="Count",
            xaxis_tickangle=45,
            height=400,
            margin=dict(t=50, b=100, l=50, r=50),
            showlegend=False
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Failed to generate histogram: {str(e)}")
        update_log(f"Histogram error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None

def create_network(df, top_n, year_range=None, excluded_labels=None, color_map=None):
    try:
        if df is None or df.empty:
            st.error("No data available for network visualization")
            return None
        
        df = filter_excluded_labels(df, excluded_labels, update_log)
        
        if 'material' in df.columns and 'classification' in df.columns:
            df = df.rename(columns={'material': 'formula', 'classification': 'material_type'})
        
        if 'year' in df.columns and year_range:
            df = df[df['year'].notna() & (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
        
        data = df.groupby(['formula', 'material_type']).size().reset_index(name='count')
        top_data = data.groupby('formula')['count'].sum().nlargest(top_n).reset_index()
        top_formulas = top_data['formula'].tolist()
        data = data[data['formula'].isin(top_formulas)]
        
        G = nx.Graph()
        
        # Add nodes
        for formula in top_formulas:
            total_count = data[data['formula'] == formula]['count'].sum()
            p_count = data[(data['formula'] == formula) & (data['material_type'] == 'p-type')]['count'].sum()
            n_count = data[(data['formula'] == formula) & (data['material_type'] == 'n-type')]['count'].sum()
            p_prop = p_count / total_count if total_count > 0 else 0
            n_prop = n_count / total_count if total_count > 0 else 0
            G.add_node(formula, size=total_count * 10, color=color_map.get(formula, '#808080'),
                       label=f"{formula}\n{p_prop:.1%} p-type, {n_prop:.1%} n-type")
        G.add_node('p-type', size=1000, color='#1f77b4', label='p-type')
        G.add_node('n-type', size=1000, color='#d62728', label='n-type')
        
        # Add edges
        for _, row in data.iterrows():
            G.add_edge(row['formula'], row['material_type'], weight=row['count'])
        
        pos = graphviz_layout(G, prog='neato')
        
        edge_x = []
        edge_y = []
        edge_weights = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        max_weight = max(edge_weights) if edge_weights else 1
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=[w/max_weight*5 for w in edge_weights], color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_sizes = []
        node_colors = []
        node_text = []
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_sizes.append(node[1]['size'])
            node_colors.append(node[1]['color'])
            node_text.append(node[1]['label'])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='#000000')
            ),
            hoverinfo='text'
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Network of Top Materials and Types',
                            title_x=0.5,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        
        return fig
    
    except Exception as e:
        st.error(f"Failed to generate network visualization: {str(e)}")
        update_log(f"Network visualization error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None

# Streamlit UI
st.set_page_config(page_title="Advanced Thermoelectric Material Analysis", layout="wide")
st.title("🌞 Advanced Thermoelectric Material Analysis")

# Initialize session state
if "db_file" not in st.session_state:
    db_files = [f for f in os.listdir(DB_DIR) if f.endswith('.db')]
    st.session_state.db_file = os.path.join(DB_DIR, db_files[0]) if db_files else None

if "data_df" not in st.session_state:
    st.session_state.data_df = None

if "hierarchy_data" not in st.session_state:
    st.session_state.hierarchy_data = None

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
    max_year = int(st.session_state.data_df['year'].max()) if st.session_state.data_df['year'].notna().any() else 2025
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

if year_range and st.sidebar.button("Apply Year Filter") and st.session_state.db_file:
    with st.spinner("Loading filtered data..."):
        st.session_state.data_df = load_data_from_db(st.session_state.db_file, year_range)

# Chart customization
st.sidebar.header("Chart Customization")
discrete_mode = st.sidebar.radio("Color Mode", ["Discrete (by type)", "Continuous (by count)"]) == "Discrete (by type)"
colorblind_safe = st.sidebar.checkbox("Use Colorblind-Safe Palette", value=True)
colormap_choice = st.sidebar.selectbox("Choose Color Map (Continuous mode)", COLOR_SCALES, index=COLOR_SCALES.index('cividis'))
branchvalues = st.sidebar.selectbox("Branch Values", ["total", "remainder"], index=0)

# Publication-quality settings
st.sidebar.header("Publication-Quality Settings")
chart_title_fontsize = st.sidebar.slider("Chart Title Font Size", 12, 30, 20)
label_fontsize = st.sidebar.slider("Label Font Size", 8, 24, 12)
chart_height = st.sidebar.slider("Chart Height", 400, 1200, 800)
chart_width = st.sidebar.slider("Chart Width", 400, 1200, 800)
background_color = st.sidebar.color_picker("Background Color", "#FFFFFF")
font_family = st.sidebar.selectbox("Font Family", ["Arial, sans-serif", "Times New Roman, serif", "Helvetica, sans-serif"], index=0)
show_grid = st.sidebar.checkbox("Show Gridlines", value=False)
curve_thickness = st.sidebar.slider("Radar Curve Thickness", 1, 5, 2)
grid_thickness = st.sidebar.slider("Grid Thickness", 0.5, 3.0, 1.0)
outline_thickness = st.sidebar.slider("Outline Thickness", 0.5, 5.0, 1.0)
export_dpi = st.sidebar.slider("Export DPI", 100, 600, 300)

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
label_threshold = st.sidebar.slider("Label Threshold (%)", 0.0, 10.0, 1.0, 0.1)
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
st.sidebar.header("Top N Materials")
if st.session_state.data_df is not None:
    max_formulas = max(50, len(st.session_state.data_df['formula'].unique()) if 'formula' in st.session_state.data_df.columns else 50)
    top_n = st.sidebar.slider("Number of Top Materials per Type", 1, max_formulas, 5)
else:
    top_n = 5
    st.sidebar.info("Load data to adjust top N materials")

# Highlighted materials selection
st.sidebar.header("Highlighted Materials")
if st.session_state.data_df is not None:
    formula_options = st.session_state.data_df['formula'].unique().tolist() if 'formula' in st.session_state.data_df.columns else []
    if 'material' in st.session_state.data_df.columns:
        formula_options.extend(st.session_state.data_df['material'].unique().tolist())
    formula_options = list(set(formula_options))
    highlight_materials = st.sidebar.multiselect(
        "Select Materials to Highlight",
        options=formula_options,
        help="Select materials to highlight with distinct colors, bold labels, and outlines."
    )
    min_count_scale = st.sidebar.slider("Minimum Count Scale for Small Segments", 1.0, 20.0, 10.0)
else:
    highlight_materials = []
    min_count_scale = 10.0
    st.sidebar.info("Load data to select materials for highlighting")

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
    layout_settings = dict(
        height=chart_height,
        width=chart_width,
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font=dict(family=font_family, size=12, color="#000000"),
        showlegend=True,
        margin=dict(t=80, l=50, r=50, b=50),
        xaxis=dict(showgrid=show_grid, gridwidth=grid_thickness),
        yaxis=dict(showgrid=show_grid, gridwidth=grid_thickness),
        polar=dict(radialaxis=dict(showgrid=show_grid, gridwidth=grid_thickness))
    )

    st.subheader("Highlighted Materials Sunburst Chart")
    if st.sidebar.button("Generate Highlighted Materials Chart"):
        with st.spinner("Generating highlighted materials chart..."):
            fig_highlighted, highlighted_hierarchy_data = create_highlighted_sunburst(
                st.session_state.data_df,
                highlight_materials,
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
                colorblind_safe=colorblind_safe,
                min_count_scale=min_count_scale,
                outline_thickness=outline_thickness
            )
            
            if fig_highlighted:
                fig_highlighted.update_layout(
                    title_font_size=chart_title_fontsize,
                    **layout_settings
                )
                st.plotly_chart(fig_highlighted, use_container_width=True)
            else:
                st.warning("Unable to generate highlighted sunburst chart. Check if selected materials exist in the dataset or review logs for details.")
    
    st.subheader(f"Top {top_n} Materials Sunburst Chart")
    if st.sidebar.button("Generate Top N Materials Chart"):
        with st.spinner("Generating top N materials chart..."):
            fig_top_n, top_n_hierarchy_data, top_n_data = create_top_n_sunburst(
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
                colorblind_safe=colorblind_safe,
                outline_thickness=outline_thickness
            )
            
            if fig_top_n:
                fig_top_n.update_layout(
                    title_font_size=chart_title_fontsize,
                    **layout_settings
                )
                st.plotly_chart(fig_top_n, use_container_width=True)
            else:
                st.warning("Unable to generate top N sunburst chart. Check data format.")
    
    st.subheader("Word Clouds")
    col1, col2, col3 = st.columns(3)
    if st.sidebar.button("Generate Word Clouds"):
        with st.spinner("Generating word clouds..."):
            with col1:
                st.write("All Materials")
                wordcloud_all = create_word_cloud(
                    st.session_state.data_df, top_n, year_range=year_range, 
                    excluded_labels=excluded_labels, color_map=st.session_state.color_map
                )
                if wordcloud_all:
                    st.image(wordcloud_all, use_column_width=True)
            
            with col2:
                st.write("p-type Materials")
                wordcloud_p = create_word_cloud(
                    st.session_state.data_df, top_n, material_type='p-type', year_range=year_range,
                    excluded_labels=excluded_labels, color_map=st.session_state.color_map
                )
                if wordcloud_p:
                    st.image(wordcloud_p, use_column_width=True)
            
            with col3:
                st.write("n-type Materials")
                wordcloud_n = create_word_cloud(
                    st.session_state.data_df, top_n, material_type='n-type', year_range=year_range,
                    excluded_labels=excluded_labels, color_map=st.session_state.color_map
                )
                if wordcloud_n:
                    st.image(wordcloud_n, use_column_width=True)
    
    st.subheader("Radar Charts")
    col1, col2, col3 = st.columns(3)
    if st.sidebar.button("Generate Radar Charts"):
        with st.spinner("Generating radar charts..."):
            with col1:
                st.write("All Materials")
                radar_all = create_radar_chart(
                    st.session_state.data_df, top_n, year_range=year_range,
                    excluded_labels=excluded_labels, color_map=st.session_state.color_map,
                    curve_thickness=curve_thickness
                )
                if radar_all:
                    radar_all.update_layout(
                        title_font_size=chart_title_fontsize,
                        **layout_settings
                    )
                    st.plotly_chart(radar_all, use_container_width=True)
            
            with col2:
                st.write("p-type Materials")
                radar_p = create_radar_chart(
                    st.session_state.data_df, top_n, material_type='p-type', year_range=year_range,
                    excluded_labels=excluded_labels, color_map=st.session_state.color_map,
                    curve_thickness=curve_thickness
                )
                if radar_p:
                    radar_p.update_layout(
                        title_font_size=chart_title_fontsize,
                        **layout_settings
                    )
                    st.plotly_chart(radar_p, use_container_width=True)
            
            with col3:
                st.write("n-type Materials")
                radar_n = create_radar_chart(
                    st.session_state.data_df, top_n, material_type='n-type', year_range=year_range,
                    excluded_labels=excluded_labels, color_map=st.session_state.color_map,
                    curve_thickness=curve_thickness
                )
                if radar_n:
                    radar_n.update_layout(
                        title_font_size=chart_title_fontsize,
                        **layout_settings
                    )
                    st.plotly_chart(radar_n, use_container_width=True)
    
    st.subheader("Histograms")
    col1, col2, col3 = st.columns(3)
    if st.sidebar.button("Generate Histograms"):
        with st.spinner("Generating histograms..."):
            with col1:
                st.write("All Materials")
                hist_all = create_histogram(
                    st.session_state.data_df, top_n, year_range=year_range,
                    excluded_labels=excluded_labels, color_map=st.session_state.color_map,
                    outline_thickness=outline_thickness
                )
                if hist_all:
                    hist_all.update_layout(
                        title_font_size=chart_title_fontsize,
                        **layout_settings
                    )
                    st.plotly_chart(hist_all, use_container_width=True)
            
            with col2:
                st.write("p-type Materials")
                hist_p = create_histogram(
                    st.session_state.data_df, top_n, material_type='p-type', year_range=year_range,
                    excluded_labels=excluded_labels, color_map=st.session_state.color_map,
                    outline_thickness=outline_thickness
                )
                if hist_p:
                    hist_p.update_layout(
                        title_font_size=chart_title_fontsize,
                        **layout_settings
                    )
                    st.plotly_chart(hist_p, use_container_width=True)
            
            with col3:
                st.write("n-type Materials")
                hist_n = create_histogram(
                    st.session_state.data_df, top_n, material_type='n-type', year_range=year_range,
                    excluded_labels=excluded_labels, color_map=st.session_state.color_map,
                    outline_thickness=outline_thickness
                )
                if hist_n:
                    hist_n.update_layout(
                        title_font_size=chart_title_fontsize,
                        **layout_settings
                    )
                    st.plotly_chart(hist_n, use_container_width=True)
    
    st.subheader("Network Visualization")
    if st.sidebar.button("Generate Network Visualization"):
        with st.spinner("Generating network visualization..."):
            network_fig = create_network(
                st.session_state.data_df, top_n, year_range=year_range,
                excluded_labels=excluded_labels, color_map=st.session_state.color_map
            )
            if network_fig:
                network_fig.update_layout(
                    title_font_size=chart_title_fontsize,
                    **layout_settings
                )
                st.plotly_chart(network_fig, use_container_width=True)
    
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
        if 'fig_highlighted' in locals() and fig_highlighted:
            st.download_button(
                label="Download Charts as HTML",
                data=fig_highlighted.to_html(),
                file_name="thermoelectric_charts.html",
                mime="text/html"
            )
    
    with st.expander("Publication Quality Tips"):
        st.markdown("""
        **For publication-quality figures:**
        
        1. Adjust **Chart Title Font Size**, **Label Font Size**, **Chart Height**, and **Chart Width** for optimal proportions.
        2. Choose a **Font Family** suitable for your publication (e.g., Arial for clarity).
        3. Use the **Background Color** picker to match your publication's style.
        4. Set **Export DPI** to 300 or higher for high-resolution outputs.
        5. Adjust **Curve Thickness**, **Grid Thickness**, and **Outline Thickness** for visual clarity.
        6. Use the **camera icon** in Plotly charts to export as high-resolution PNG.
        7. Enable **Colorblind-Safe Palette** for accessibility.
        8. Adjust **Label Threshold** to reduce clutter from small segments.
        9. Use **Custom Colors** to highlight specific materials or types.
        10. Export data as **CSV** or **JSON** for further analysis in other tools.
        11. Check **logs** below for any data or rendering issues.
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
