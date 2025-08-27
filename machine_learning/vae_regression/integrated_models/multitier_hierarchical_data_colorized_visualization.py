import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import logging
import json
import os
from datetime import datetime

# Initialize logging
logging.basicConfig(
    filename='thermoelectric_sunburst_analysis.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def update_log(message):
    """Helper function to log messages to both file and console."""
    logging.info(message)
    print(message)

# Define color scales and mappings
COLOR_SCALES = ['viridis', 'inferno', 'magma', 'plasma', 'hot', 'blues', 'greens', 'reds']
OTHER_COLOR_SCALES = {'p-type': 'Blues', 'n-type': 'Reds'}
QUALITATIVE_COLOR_SCALES = ['Plotly', 'D3', 'G10', 'T10', 'Alphabet', 'Dark24', 'Light24']

def validate_color_scale(scale_name):
    """Return a list of color strings for the given scale name, or fallback to Greys."""
    try:
        colors = getattr(px.colors.sequential, scale_name)
        if not isinstance(colors, (list, tuple)) or not all(isinstance(c, str) for c in colors):
            raise ValueError(f"Color scale '{scale_name}' is not a valid list of color strings")
        update_log(f"Validated color scale: {scale_name}")
        return colors
    except (AttributeError, ValueError) as e:
        update_log(f"Invalid color scale '{scale_name}': {str(e)}. Falling back to Greys.")
        return px.colors.sequential.Greys

def load_data_from_db(db_path, table_name, year_range=None):
    """Load data from SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        if year_range:
            query += f" WHERE year BETWEEN {year_range[0]} AND {year_range[1]}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        update_log(f"Data loaded successfully from {db_path}, table {table_name}. Rows: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        update_log(f"Data loading error: {str(e)}")
        return None

def create_three_tier_sunburst(df, colormap_choice, discrete_mode, show_labels, label_fontsize, 
                              excluded_labels, year_range=None, chart_height=600, branchvalues='total',
                              label_threshold=1.0, show_values=True, show_percentages=True):
    """Create a three-tier sunburst chart with year, material_type, and formula."""
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            update_log("No data available for three-tier sunburst chart")
            return None, None

        if colormap_choice.lower() not in COLOR_SCALES:
            update_log(f"Invalid colormap '{colormap_choice}' selected. Falling back to 'viridis'.")
            colormap_choice = 'viridis'

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
            update_log(f"Missing required columns for three-tier sunburst: {', '.join(missing)}")
            return None, None

        if 'count' in df.columns:
            df['count'] = pd.to_numeric(df['count'], errors='coerce')
            if df['count'].isna().any():
                st.warning("NaN values found in 'count' column. Filling with 0.")
                update_log("NaN values in 'count' column. Filling with 0.")
                df['count'] = df['count'].fillna(0)

        if 'year' in df.columns:
            sunburst_data = df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)

            years = sunburst_data[['year']].drop_duplicates()
            years['parent'] = ''
            years['id'] = years['year'].astype(str)
            years['label'] = years['year']

            types = sunburst_data[['year', 'material_type']].drop_duplicates()
            types['parent'] = types['year'].astype(str)
            types['id'] = types['year'].astype(str) + '_' + types['material_type']
            types['label'] = types['material_type']

            formulas = sunburst_data[['year', 'material_type', 'formula', 'count']].copy()
            formulas['parent'] = formulas['year'].astype(str) + '_' + formulas['material_type']
            formulas['id'] = formulas['parent'] + '_' + formulas['formula']
            formulas['label'] = formulas['formula']

            hierarchy_data = pd.concat([years, types, formulas], ignore_index=True)

            for type_id in types['id']:
                type_sum = hierarchy_data[hierarchy_data['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum

            for year_id in years['id']:
                year_sum = types[types['parent'] == year_id]['id'].apply(
                    lambda x: hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].values[0] if not hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].empty else 0
                ).sum()
                hierarchy_data.loc[hierarchy_data['id'] == year_id, 'count'] = year_sum
        else:
            sunburst_data = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)

            types = sunburst_data[['material_type']].drop_duplicates()
            types['parent'] = ''
            types['id'] = types['material_type']
            types['label'] = types['material_type']

            formulas = sunburst_data[['material_type', 'formula', 'count']].copy()
            formulas['parent'] = formulas['material_type']
            formulas['id'] = formulas['material_type'] + '_' + formulas['formula']
            formulas['label'] = formulas['formula']

            hierarchy_data = pd.concat([types, formulas], ignore_index=True)

            for type_id in types['id']:
                type_sum = hierarchy_data[hierarchy_data['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum

        total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
        hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100

        hierarchy_data['count'] = pd.to_numeric(hierarchy_data['count'], errors='coerce').fillna(0)
        update_log(f"Three-tier count column dtype: {hierarchy_data['count'].dtype}")
        if hierarchy_data['count'].isna().any():
            update_log("Warning: NaN values in hierarchy_data['count'] after conversion")

        if discrete_mode:
            unique_labels = hierarchy_data[hierarchy_data['parent'] != '']['label'].unique()
            color_map = {t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, t in enumerate(unique_labels)}
            hierarchy_data['color'] = hierarchy_data['label'].map(color_map)
            hierarchy_data.loc[hierarchy_data['parent'] == '', 'color'] = '#E5ECF6'

            hierarchy_data['display_text'] = hierarchy_data['label']
            if label_threshold > 0:
                hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''

            update_log(f"Three-tier discrete colors dtype: {hierarchy_data['color'].dtype}, sample: {hierarchy_data['color'].head().tolist()}")

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

            colors = hierarchy_data['count'].astype(float).copy()
            update_log(f"Three-tier continuous colors dtype: {colors.dtype}, sample: {colors.head().tolist()}")

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
                text="Three-Tier Sunburst Chart (Year, Material Type, Formula)",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=20, family="Arial, sans-serif")
            )
        )

        return fig, hierarchy_data

    except Exception as e:
        st.error(f"Failed to generate three-tier sunburst chart: {str(e)}")
        update_log(f"Three-tier sunburst chart error: {str(e)}")
        import traceback
        update_log(traceback.format_exc())
        return None, None

def create_top_n_sunburst(df, top_n, colormap_choice, discrete_mode, show_labels, label_fontsize, 
                         excluded_labels, year_range=None, chart_height=600, branchvalues='total',
                         label_threshold=1.0, show_values=True, show_percentages=True):
    """Create a sunburst chart showing only top N formulas per material type."""
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            update_log("No data available for top N sunburst chart")
            return None, None

        if colormap_choice.lower() not in COLOR_SCALES:
            update_log(f"Invalid colormap '{colormap_choice}' selected. Falling back to 'viridis'.")
            colormap_choice = 'viridis'

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
            update_log(f"Missing required columns for top N sunburst: {', '.join(missing)}")
            return None, None

        if 'count' in df.columns:
            df['count'] = pd.to_numeric(df['count'], errors='coerce')
            if df['count'].isna().any():
                st.warning("NaN values found in 'count' column. Filling with 0.")
                update_log("NaN values in 'count' column. Filling with 0.")
                df['count'] = df['count'].fillna(0)

        grouped_data = []
        if 'year' in df.columns:
            sunburst_data = df.groupby(['year', 'material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)

            for year in sunburst_data['year'].unique():
                for mtype in sunburst_data['material_type'].unique():
                    type_data = sunburst_data[(sunburst_data['year'] == year) & 
                                            (sunburst_data['material_type'] == mtype)]
                    if type_data.empty:
                        continue
                    top_formulas = type_data.nlargest(top_n, 'count')
                    grouped_data.append(top_formulas)
        else:
            sunburst_data = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)

            for mtype in sunburst_data['material_type'].unique():
                type_data = sunburst_data[sunburst_data['material_type'] == mtype]
                if type_data.empty:
                    continue
                top_formulas = type_data.nlargest(top_n, 'count')
                grouped_data.append(top_formulas)

        if not grouped_data:
            st.error("No data available after filtering for top N")
            update_log("No data available after filtering for top N")
            return None, None

        sunburst_data = pd.concat(grouped_data, ignore_index=True)

        if 'year' in df.columns:
            years = sunburst_data[['year']].drop_duplicates()
            years['parent'] = ''
            years['id'] = years['year'].astype(str)
            years['label'] = years['year']

            types = sunburst_data[['year', 'material_type']].drop_duplicates()
            types['parent'] = types['year'].astype(str)
            types['id'] = types['year'].astype(str) + '_' + types['material_type']
            types['label'] = types['material_type']

            formulas = sunburst_data[['year', 'material_type', 'formula', 'count']].copy()
            formulas['parent'] = formulas['year'].astype(str) + '_' + formulas['material_type']
            formulas['id'] = formulas['parent'] + '_' + formulas['formula']
            formulas['label'] = formulas['formula']

            hierarchy_data = pd.concat([years, types, formulas], ignore_index=True)

            for type_id in types['id']:
                type_sum = hierarchy_data[hierarchy_data['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum

            for year_id in years['id']:
                year_sum = types[types['parent'] == year_id]['id'].apply(
                    lambda x: hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].values[0] if not hierarchy_data.loc[hierarchy_data['id'] == x, 'count'].empty else 0
                ).sum()
                hierarchy_data.loc[hierarchy_data['id'] == year_id, 'count'] = year_sum
        else:
            types = sunburst_data[['material_type']].drop_duplicates()
            types['parent'] = ''
            types['id'] = types['material_type']
            types['label'] = types['material_type']

            formulas = sunburst_data[['material_type', 'formula', 'count']].copy()
            formulas['parent'] = formulas['material_type']
            formulas['id'] = formulas['material_type'] + '_' + formulas['formula']
            formulas['label'] = formulas['formula']

            hierarchy_data = pd.concat([types, formulas], ignore_index=True)

            for type_id in types['id']:
                type_sum = hierarchy_data[hierarchy_data['parent'] == type_id]['count'].sum()
                hierarchy_data.loc[hierarchy_data['id'] == type_id, 'count'] = type_sum

        total_count = hierarchy_data[hierarchy_data['parent'] == '']['count'].sum()
        hierarchy_data['percentage'] = (hierarchy_data['count'] / total_count) * 100

        hierarchy_data['count'] = pd.to_numeric(hierarchy_data['count'], errors='coerce').fillna(0)
        update_log(f"Top N count column dtype: {hierarchy_data['count'].dtype}")
        if hierarchy_data['count'].isna().any():
            update_log("Warning: NaN values in hierarchy_data['count'] after conversion")

        if discrete_mode:
            unique_labels = hierarchy_data[hierarchy_data['parent'] != '']['label'].unique()
            color_map = {t: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, t in enumerate(unique_labels)}
            hierarchy_data['color'] = hierarchy_data['label'].map(color_map)
            hierarchy_data.loc[hierarchy_data['parent'] == '', 'color'] = '#E5ECF6'

            hierarchy_data['display_text'] = hierarchy_data['label']
            if label_threshold > 0:
                hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''

            update_log(f"Top N discrete colors dtype: {hierarchy_data['color'].dtype}, sample: {hierarchy_data['color'].head().tolist()}")

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

            colors = hierarchy_data['count'].astype(float).copy()
            update_log(f"Top N continuous colors dtype: {colors.dtype}, sample: {colors.head().tolist()}")

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
                text=f"Top {top_n} Materials Sunburst Chart",
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

def create_expanded_sunburst(df, top_ns, colormap_choice, discrete_mode, show_labels, label_fontsize, 
                            excluded_labels, year_range=None, chart_height=800, branchvalues='total',
                            label_threshold=1.0, show_values=True, show_percentages=True, 
                            qualitative_scale='D3'):
    """Create a sunburst chart with expandable layers, showing top N materials and expanding 'Other' categories."""
    try:
        if df is None or df.empty:
            st.error("No data available for visualization")
            update_log("No data available for expanded sunburst chart")
            return None, None
        
        # Validate colormap_choice for continuous mode
        if colormap_choice.lower() not in COLOR_SCALES:
            update_log(f"Invalid colormap '{colormap_choice}' selected. Falling back to 'viridis'.")
            colormap_choice = 'viridis'

        # Validate qualitative_scale for discrete mode
        if qualitative_scale not in QUALITATIVE_COLOR_SCALES:
            update_log(f"Invalid qualitative color scale '{qualitative_scale}'. Falling back to 'D3'.")
            qualitative_scale = 'D3'

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
            update_log(f"Missing required columns for expanded sunburst: {', '.join(missing)}")
            return None, None
        
        # Validate count column for non-numeric or NaN values
        if 'count' in df.columns:
            df['count'] = pd.to_numeric(df['count'], errors='coerce')
            if df['count'].isna().any():
                st.warning("NaN values found in 'count' column. Filling with 0.")
                update_log("NaN values in 'count' column. Filling with 0.")
                df['count'] = df['count'].fillna(0)
        
        # Check if enough data for expansion
        unique_formulas = df.groupby(['material_type', 'year'] if 'year' in df.columns else ['material_type'])['formula'].nunique()
        max_formulas = unique_formulas.min()
        if max_formulas < max(top_ns):
            st.warning(f"Insufficient unique formulas ({max_formulas}) for requested top {max(top_ns)} in some categories. Adjusting layers.")
            update_log(f"Insufficient unique formulas ({max_formulas}) for top {max(top_ns)}")
            top_ns = [min(n, max_formulas) for n in top_ns]
        
        # Initialize hierarchy data
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
                        type_data['parent'] = type_id if layer_idx == 0 else f"{year}_{mtype}_other_{layer_idx-1}"
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
            update_log(f"Expanded count column dtype: {hierarchy_data['count'].dtype}")
            if hierarchy_data['count'].isna().any():
                update_log("Warning: NaN values in hierarchy_data['count'] after conversion")
            
            if discrete_mode:
                unique_labels = hierarchy_data[hierarchy_data['parent'].str.contains('_') | (hierarchy_data['parent'] == '')]['label'].unique()
                color_map = {}
                colors = getattr(px.colors.qualitative, qualitative_scale)
                update_log(f"Using qualitative color scale: {qualitative_scale}, colors: {colors[:3]}... (length: {len(colors)})")
                for i, t in enumerate(unique_labels):
                    if t.startswith('Other') or t.startswith('Sub-Other'):
                        mtype_parts = t.split(' ')[1].split('L')[0] if 'L' in t else t.split(' ')[1]
                        mtype = mtype_parts if mtype_parts in ['p-type', 'n-type'] else 'p-type'
                        other_colors = validate_color_scale(OTHER_COLOR_SCALES.get(mtype, 'Greys'))
                        color_map[t] = other_colors[min(i % len(other_colors), len(other_colors)-1)]
                    else:
                        color_map[t] = colors[i % len(colors)]
                
                hierarchy_data['color'] = hierarchy_data['label'].map(color_map)
                hierarchy_data.loc[hierarchy_data['parent'] == '', 'color'] = '#E5ECF6'
                
                hierarchy_data['display_text'] = hierarchy_data['label']
                if label_threshold > 0:
                    hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
                
                update_log(f"Expanded discrete colors dtype: {hierarchy_data['color'].dtype}, sample: {hierarchy_data['color'].head().tolist()}")
                
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
                
                colors = hierarchy_data['count'].astype(float).copy()
                update_log(f"Expanded continuous colors dtype: {colors.dtype}, sample: {colors.head().tolist()}")
                
                other_mask = hierarchy_data['label'].str.contains('Other|Sub-Other', na=False)
                if other_mask.any():
                    for mtype in ['p-type', 'n-type']:
                        mask = hierarchy_data['label'].str.contains(f'Other {mtype}|Sub-Other {mtype}', na=False)
                        if mask.any():
                            other_count = hierarchy_data.loc[mask, 'count'].astype(float).iloc[0]
                            other_colors = validate_color_scale(OTHER_COLOR_SCALES.get(mtype, 'Greys'))
                            update_log(f"Expanded other colors for {mtype}: {other_colors[:3]}... (length: {len(other_colors)})")
                            color_idx = min(int((other_count / hierarchy_data['count'].max()) * (len(other_colors) - 1)), len(other_colors) - 1)
                            colors[mask] = other_colors[color_idx]
                            update_log(f"Expanded assigned color {other_colors[color_idx]} to Other {mtype} (count: {other_count}, idx: {color_idx})")
                
                colors = pd.to_numeric(colors, errors='coerce').fillna(0)
                update_log(f"Expanded final colors dtype: {colors.dtype}, sample: {colors.head().tolist()}")
                
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
            sunburst_data = current_df.groupby(['material_type', 'formula']).size().reset_index(name='count')
            sunburst_data['count'] = pd.to_numeric(sunburst_data['count'], errors='coerce').fillna(0)
            
            for layer_idx, top_n in enumerate(top_ns):
                grouped_data = []
                parent_level = 'type_id' if layer_idx == 0 else f'other_{layer_idx-1}_id'
                
                for mtype in sunburst_data['material_type'].unique():
                    type_data = sunburst_data[sunburst_data['material_type'] == mtype]
                    if type_data.empty:
                        continue
                    
                    top_formulas = type_data.nlargest(top_n, 'count')
                    other_data = type_data[~type_data['formula'].isin(top_formulas['formula'])]
                    
                    if layer_idx == 0:
                        type_id = mtype
                    else:
                        type_id = f"{mtype}_other_{layer_idx-1}"
                    
                    type_data = top_formulas.copy()
                    type_data['parent'] = type_id if layer_idx == 0 else f"{mtype}_other_{layer_idx-1}"
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
                    update_log(f"No data for layer {layer_idx + 1}")
                    break
                
                if layer_idx < len(top_ns) - 1:
                    other_data = sunburst_data[sunburst_data['formula'].str.startswith('Other')]
                    if other_data.empty:
                        update_log(f"No 'Other' categories to expand in layer {layer_idx + 1}")
                        break
                    other_formulas = df.groupby(['material_type', 'formula']).size().reset_index(name='count')
                    for idx, row in other_data.iterrows():
                        mtype = row['material_type']
                        prev_formulas = sunburst_data[sunburst_data['id'].str.startswith(mtype) & 
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
                        update_log(f"No remaining formulas for layer {layer_idx + 2}")
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
            update_log(f"Expanded count column dtype: {hierarchy_data['count'].dtype}")
            if hierarchy_data['count'].isna().any():
                update_log("Warning: NaN values in hierarchy_data['count'] after conversion")
            
            if discrete_mode:
                unique_labels = hierarchy_data[hierarchy_data['parent'] != '']['label'].unique()
                color_map = {}
                colors = getattr(px.colors.qualitative, qualitative_scale)
                update_log(f"Using qualitative color scale: {qualitative_scale}, colors: {colors[:3]}... (length: {len(colors)})")
                for i, t in enumerate(unique_labels):
                    if t.startswith('Other') or t.startswith('Sub-Other'):
                        mtype_parts = t.split(' ')[1].split('L')[0] if 'L' in t else t.split(' ')[1]
                        mtype = mtype_parts if mtype_parts in ['p-type', 'n-type'] else 'p-type'
                        other_colors = validate_color_scale(OTHER_COLOR_SCALES.get(mtype, 'Greys'))
                        color_map[t] = other_colors[min(i % len(other_colors), len(other_colors)-1)]
                    else:
                        color_map[t] = colors[i % len(colors)]
                
                hierarchy_data['color'] = hierarchy_data['label'].map(color_map)
                hierarchy_data.loc[hierarchy_data['parent'] == '', 'color'] = '#E5ECF6'
                
                hierarchy_data['display_text'] = hierarchy_data['label']
                if label_threshold > 0:
                    hierarchy_data.loc[hierarchy_data['percentage'] < label_threshold, 'display_text'] = ''
                
                update_log(f"Expanded discrete colors dtype: {hierarchy_data['color'].dtype}, sample: {hierarchy_data['color'].head().tolist()}")
                
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
                
                colors = hierarchy_data['count'].astype(float).copy()
                update_log(f"Expanded continuous colors dtype: {colors.dtype}, sample: {colors.head().tolist()}")
                
                other_mask = hierarchy_data['label'].str.contains('Other|Sub-Other', na=False)
                if other_mask.any():
                    for mtype in ['p-type', 'n-type']:
                        mask = hierarchy_data['label'].str.contains(f'Other {mtype}|Sub-Other {mtype}', na=False)
                        if mask.any():
                            other_count = hierarchy_data.loc[mask, 'count'].astype(float).iloc[0]
                            other_colors = validate_color_scale(OTHER_COLOR_SCALES.get(mtype, 'Greys'))
                            update_log(f"Expanded other colors for {mtype}: {other_colors[:3]}... (length: {len(other_colors)})")
                            color_idx = min(int((other_count / hierarchy_data['count'].max()) * (len(other_colors) - 1)), len(other_colors) - 1)
                            colors[mask] = other_colors[color_idx]
                            update_log(f"Expanded assigned color {other_colors[color_idx]} to Other {mtype} (count: {other_count}, idx: {color_idx})")
                
                colors = pd.to_numeric(colors, errors='coerce').fillna(0)
                update_log(f"Expanded final colors dtype: {colors.dtype}, sample: {colors.head().tolist()}")
                
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
st.set_page_config(page_title="Thermoelectric Materials Sunburst Analysis", layout="wide")
st.title("Thermoelectric Materials Sunburst Analysis")

# Initialize session state
if 'data_df' not in st.session_state:
    st.session_state.data_df = None
if 'top_n' not in st.session_state:
    st.session_state.top_n = 5
if 'expanded_layers' not in st.session_state:
    st.session_state.expanded_layers = [5, 5, 5]
if 'year_range' not in st.session_state:
    st.session_state.year_range = None

# Sidebar for database input and customization
st.sidebar.header("Database Connection")
db_path = st.sidebar.text_input("Database Path", "thermoelectric.db")
table_name = st.sidebar.text_input("Table Name", "materials")
if st.sidebar.button("Load Data"):
    with st.spinner("Loading data..."):
        st.session_state.data_df = load_data_from_db(db_path, table_name)
        if st.session_state.data_df is not None and 'year' in st.session_state.data_df.columns:
            years = st.session_state.data_df['year'].dropna().astype(int).sort_values().unique()
            if len(years) > 1:
                st.session_state.year_range = [int(years[0]), int(years[-1])]
            else:
                st.session_state.year_range = None

st.sidebar.header("Chart Customization")
discrete_mode = st.sidebar.radio("Color Mode", ["Discrete (by type)", "Continuous (by count)"]) == "Discrete (by type)"
if discrete_mode:
    qualitative_scale = st.sidebar.selectbox(
        "Choose Qualitative Color Scale (Discrete mode)",
        QUALITATIVE_COLOR_SCALES,
        index=1  # Default to D3
    )
    st.sidebar.info(f"In Discrete mode, colors are assigned by material type using Plotly's {qualitative_scale} colors. Color map selection is used only in Continuous mode.")
else:
    qualitative_scale = 'D3'  # Default for discrete mode calls in function

colormap_choice = st.sidebar.selectbox(
    "Choose Color Map (Continuous mode)",
    COLOR_SCALES,
    index=0
)

show_labels = st.sidebar.checkbox("Show Labels", value=True)
label_fontsize = st.sidebar.slider("Label Font Size", 8, 24, 12)
label_threshold = st.sidebar.slider("Label Threshold (% of total)", 0.0, 5.0, 1.0, 0.1)
show_values = st.sidebar.checkbox("Show Values", value=True)
show_percentages = st.sidebar.checkbox("Show Percentages", value=True)
chart_height = st.sidebar.slider("Chart Height", 400, 1200, 800, 50)
branchvalues = st.sidebar.radio("Branch Values", ["total", "remainder"], index=0)

excluded_labels = st.sidebar.multiselect(
    "Exclude Material Types or Formulas",
    options=[] if st.session_state.data_df is None else (
        list(st.session_state.data_df.get('material_type', []).unique()) +
        list(st.session_state.data_df.get('classification', []).unique()) +
        list(st.session_state.data_df.get('material', []).unique()) +
        list(st.session_state.data_df.get('formula', []).unique())
    )
)

if st.session_state.data_df is not None and 'year' in st.session_state.data_df.columns:
    years = st.session_state.data_df['year'].dropna().astype(int).sort_values().unique()
    if len(years) > 1:
        year_range = st.sidebar.slider(
            "Year Range",
            min_value=int(years[0]),
            max_value=int(years[-1]),
            value=(int(years[0]), int(years[-1]))
        )
        if year_range != st.session_state.year_range:
            st.session_state.year_range = year_range
            st.session_state.data_df = load_data_from_db(db_path, table_name, year_range)
else:
    year_range = None

st.sidebar.header("Chart Parameters")
st.session_state.top_n = st.sidebar.number_input("Top N Materials", min_value=1, max_value=50, value=st.session_state.top_n)
num_layers = st.sidebar.number_input("Number of Expanded Layers", min_value=1, max_value=5, value=len(st.session_state.expanded_layers))
st.session_state.expanded_layers = []
for i in range(num_layers):
    layer_n = st.sidebar.number_input(f"Top N for Layer {i+1}", min_value=1, max_value=50, value=st.session_state.expanded_layers[i] if i < len(st.session_state.expanded_layers) else 5)
    st.session_state.expanded_layers.append(layer_n)

# Generate charts
if st.session_state.data_df is not None:
    st.subheader("Three-Tier Sunburst Chart")
    if st.sidebar.button("Generate Three-Tier Sunburst Chart"):
        with st.spinner("Generating three-tier sunburst chart..."):
            fig_three_tier, three_tier_hierarchy_data = create_three_tier_sunburst(
                st.session_state.data_df,
                colormap_choice,
                discrete_mode,
                show_labels,
                label_fontsize,
                excluded_labels,
                st.session_state.year_range,
                chart_height,
                branchvalues,
                label_threshold,
                show_values,
                show_percentages
            )
            
            if fig_three_tier:
                st.plotly_chart(fig_three_tier, use_container_width=True)
                if three_tier_hierarchy_data is not None:
                    st.download_button(
                        label="Download Three-Tier Hierarchy Data as CSV",
                        data=three_tier_hierarchy_data.to_csv(index=False),
                        file_name=f"three_tier_hierarchy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        label="Download Three-Tier Hierarchy Data as JSON",
                        data=three_tier_hierarchy_data.to_json(orient="records", lines=True),
                        file_name=f"three_tier_hierarchy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.warning("Unable to generate three-tier sunburst chart. Check data format or logs for details.")

    st.subheader(f"Top {st.session_state.top_n} Materials Sunburst Chart")
    if st.sidebar.button("Generate Top N Sunburst Chart"):
        with st.spinner("Generating top N sunburst chart..."):
            fig_top_n, top_n_hierarchy_data = create_top_n_sunburst(
                st.session_state.data_df,
                st.session_state.top_n,
                colormap_choice,
                discrete_mode,
                show_labels,
                label_fontsize,
                excluded_labels,
                st.session_state.year_range,
                chart_height,
                branchvalues,
                label_threshold,
                show_values,
                show_percentages
            )
            
            if fig_top_n:
                st.plotly_chart(fig_top_n, use_container_width=True)
                if top_n_hierarchy_data is not None:
                    st.download_button(
                        label="Download Top N Hierarchy Data as CSV",
                        data=top_n_hierarchy_data.to_csv(index=False),
                        file_name=f"top_n_hierarchy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        label="Download Top N Hierarchy Data as JSON",
                        data=top_n_hierarchy_data.to_json(orient="records", lines=True),
                        file_name=f"top_n_hierarchy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.warning("Unable to generate top N sunburst chart. Check data format or logs for details.")

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
                st.session_state.year_range,
                chart_height,
                branchvalues,
                label_threshold,
                show_values,
                show_percentages,
                qualitative_scale=qualitative_scale
            )
            
            if fig_expanded:
                st.plotly_chart(fig_expanded, use_container_width=True)
                if expanded_hierarchy_data is not None:
                    st.download_button(
                        label="Download Expanded Hierarchy Data as CSV",
                        data=expanded_hierarchy_data.to_csv(index=False),
                        file_name=f"expanded_hierarchy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        label="Download Expanded Hierarchy Data as JSON",
                        data=expanded_hierarchy_data.to_json(orient="records", lines=True),
                        file_name=f"expanded_hierarchy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.warning("Unable to generate expanded sunburst chart. Check data format or logs for details.")
else:
    st.warning("Please load data to generate charts.")
