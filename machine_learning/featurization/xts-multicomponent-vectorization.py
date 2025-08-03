import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core.composition import Composition
import re
import io
import json
import matplotlib
import os

# Set matplotlib to non-interactive backend for Streamlit
matplotlib.use('Agg')

# Electronegativity and thermoelectric weights
electronegativity = {
    'O': 3.44, 'Cl': 3.16, 'N': 3.04, 'Br': 2.96, 'I': 2.66, 'S': 2.58, 'Se': 2.55, 'Te': 2.1, 'P': 2.19, 'As': 2.18,
    'Sb': 2.05, 'Bi': 2.02, 'Si': 1.90, 'Ge': 2.01, 'Sn': 1.96, 'Pb': 2.33, 'B': 2.04, 'Al': 1.61, 'Ga': 1.81,
    'In': 1.78, 'Tl': 2.04, 'Mg': 1.31, 'Ca': 1.00, 'Sr': 0.95, 'Ba': 0.89, 'Li': 0.98, 'Na': 0.93, 'K': 0.82,
    'Rb': 0.82, 'Cs': 0.79, 'Sc': 1.36, 'Y': 1.22, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Sm': 1.17,
    'Eu': 1.2, 'Gd': 1.2, 'Tb': 1.1, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.1, 'Lu': 1.27,
    'Ti': 1.54, 'Zr': 1.33, 'Hf': 1.3, 'V': 1.63, 'Nb': 1.6, 'Ta': 1.5, 'Cr': 1.66, 'Mo': 2.16, 'Mn': 1.55,
    'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Cd': 1.69, 'Ag': 1.93, 'Au': 2.54, 'Pd': 2.20, 'Ru': 2.2
}

thermoelectric_weights = {
    'Bi': 2.0, 'Te': 2.0, 'Sb': 1.8, 'Pb': 1.8, 'Se': 1.5, 'Sn': 1.5, 'Ge': 1.3, 'Si': 1.3, 'Mg': 1.2
}

# List of all 85 elements in the specified order
all_elements = [
    'In', 'Tl', 'La', 'Sr', 'Mn', 'Ni', 'Ru', 'Pd', 'Hf', 'Cs', 'Sc', 'Co', 'Si', 'Fe', 'Li', 'Cl', 'Yb', 'Te',
    'N', 'Ti', 'Cd', 'Zr', 'Y', 'Ga', 'Cr', 'Pr', 'Tm', 'Br', 'Ca', 'Mg', 'Rb', 'Au', 'Nd', 'Ce', 'Ho', 'I', 'Ba',
    'Se', 'Pb', 'Ge', 'Gd', 'Tb', 'Dy', 'Cu', 'Na', 'Sb', 'Bi', 'P', 'As', 'Sm', 'Zn', 'Al', 'Sn', 'Ag', 'Nb', 'Mo',
    'V', 'S', 'K', 'Lu', 'O', 'Eu', 'Ta', 'B', 'Er', 'H', 'He', 'Be', 'C', 'F', 'Ne', 'Ar', 'Kr',
    'Xe', 'Tc', 'Rh', 'Pm', 'Re', 'Os', 'Ir', 'Pt', 'Hg', 'W'
]

# Define color map for elements
default_color_list = (
    px.colors.qualitative.Plotly +
    px.colors.qualitative.Pastel1 +
    px.colors.qualitative.D3 +
    px.colors.qualitative.G10 +
    px.colors.qualitative.T10
)
default_element_color_map = dict(zip(all_elements, default_color_list[:len(all_elements)]))

def parse_formula(formula):
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)?'
    elements = re.findall(pattern, formula)
    return list(set([element[0] for element in elements]))

def extract_multiplier_and_replace(input_formula):
    pattern = r'\)(\d*\.?\d*)?'
    match = re.search(pattern, input_formula)
    if match:
        multiplier = float(match.group(1)) if match.group(1) else 1.0
        parts = re.split(pattern, input_formula)
        formula_without_multiplier = parts[0]
        content_within_parentheses = formula_without_multiplier.split('(')[-1]
        elements_within_parentheses = re.findall(r'([A-Za-z]+)(\d*\.?\d*)', content_within_parentheses)
        modified_elements = [(element, str(float(stoichiometry) * multiplier) if stoichiometry else '0.0') for element, stoichiometry in elements_within_parentheses]
        modified_formula = formula_without_multiplier.split('(')[0]
        modified_formula += ''.join(element + stoichiometry for element, stoichiometry in modified_elements)
        return modified_formula
    return input_formula

def count_elements(df):
    elements = set()
    for formula in df['Formula']:
        try:
            elements.update(parse_formula(formula))
        except Exception as e:
            st.warning(f"Error parsing formula {formula}: {e}")
            continue
    return sorted(list(elements))

def featurize_materials(df, available_elements, csv_columns):
    features = []
    for _, row in df.iterrows():
        try:
            modified_formula = extract_multiplier_and_replace(row['Formula'])
            composition = Composition(modified_formula)
            composition_dict = composition.fractional_composition.as_dict()
            feature_vector = {col: 0.0 for col in csv_columns if col not in ['Formula', 'modformula', 'temperature(K)', 'seebeck_coefficient(μV/K)']}
            for element in available_elements:
                feature_vector[element] = composition_dict.get(element, 0.0)
            features.append(feature_vector)
        except Exception as e:
            st.warning(f"Error processing formula {row['Formula']}: {e}")
            continue
    return features

def plot_periodic_table(all_elements, present_elements, element_color_map, fontsize=12):
    periodic_table_positions = {
        'H': (1, 1), 'He': (1, 18),
        'Li': (2, 1), 'Be': (2, 2), 'B': (2, 13), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18),
        'Na': (3, 1), 'Mg': (3, 2), 'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
        'K': (4, 1), 'Ca': (4, 2), 'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6), 'Mn': (4, 7), 'Fe': (4, 8),
        'Co': (4, 9), 'Ni': (4, 10), 'Cu': (4, 11), 'Zn': (4, 12), 'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15),
        'Se': (4, 16), 'Br': (4, 17), 'Kr': (4, 18),
        'Rb': (5, 1), 'Sr': (5, 2), 'Y': (5, 3), 'Zr': (5, 4), 'Nb': (5, 5), 'Mo': (5, 6), 'Tc': (5, 7), 'Ru': (5, 8),
        'Rh': (5, 9), 'Pd': (5, 10), 'Ag': (5, 11), 'Cd': (5, 12), 'In': (5, 13), 'Sn': (5, 14), 'Sb': (5, 15),
        'Te': (5, 16), 'I': (5, 17), 'Xe': (5, 18),
        'Cs': (6, 1), 'Ba': (6, 2), 'La': (6, 3), 'Ce': (7, 3), 'Pr': (7, 4), 'Nd': (7, 5), 'Pm': (7, 6), 'Sm': (7, 7),
        'Eu': (7, 8), 'Gd': (7, 9), 'Tb': (7, 10), 'Dy': (7, 11), 'Ho': (7, 12), 'Er': (7, 13), 'Tm': (7, 14),
        'Yb': (7, 15), 'Lu': (7, 16), 'Hf': (6, 4), 'Ta': (6, 5), 'W': (6, 6), 'Re': (6, 7), 'Os': (6, 8),
        'Ir': (6, 9), 'Pt': (6, 10), 'Au': (6, 11), 'Hg': (6, 12), 'Tl': (6, 13), 'Pb': (6, 14), 'Bi': (6, 15)
    }
    fig = go.Figure()
    for element in all_elements:
        if element in periodic_table_positions:
            row, col = periodic_table_positions[element]
            en = electronegativity.get(element, 1.0)
            tw = thermoelectric_weights.get(element, 1.0)
            color = element_color_map.get(element, '#D3D3D3') if element in present_elements else '#D3D3D3'
            fig.add_trace(go.Scatter(
                x=[col], y=[-row],
                mode='markers+text',
                text=[element],
                textposition='middle center',
                marker=dict(size=40, color=color, line=dict(width=2, color='black')),
                hoverinfo='text',
                hovertext=[f"Element: {element}<br>Electronegativity: {en:.2f}<br>Thermoelectric Weight: {tw:.2f}"],
                customdata=[element],
                name=element,
                showlegend=False
            ))
    fig.update_layout(
        title=dict(text='Interactive Periodic Table', x=0.5, xanchor='center', font=dict(size=fontsize + 4, family='Arial')),
        xaxis=dict(range=[0, 19], showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(range=[-8, 0], showgrid=False, zeroline=False, showticklabels=False, title=''),
        plot_bgcolor='white', paper_bgcolor='white',
        width=900, height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Streamlit UI
st.title("Thermoelectric Material Featurization")
st.markdown("""
Enter the number of components (n) for the system (e.g., 3 for ternary, 4 for quaternary). Upload a pre-existing featurized CSV to define the column structure. Then, upload one or more JSON files with the naming convention `AaBbCc...Xn.json` (e.g., `Bi1Sb1Te2.json` for ternary), where `a`, `b`, `c`, ..., `n` are stoichiometric coefficients. Each JSON file contains a list of dictionaries with `x` (temperature in K) and `y` (Seebeck coefficient in μV/K). The script appends new data to the CSV, setting absent features to 0, and allows downloading the updated CSV.
**Date and Time**: 06:31 AM CEST, Sunday, August 03, 2025
""")

# User input for number of components
n_components = st.number_input("Enter the number of components (n, e.g., 3 for ternary, 4 for quaternary)", min_value=2, max_value=10, value=3, step=1)

# File uploader for pre-existing featurized CSV
csv_file = st.file_uploader("Upload pre-existing featurized CSV (e.g., featurized_thermoelectric_data.csv)", type=["csv"])

# Load CSV to get column structure
if csv_file:
    try:
        df_csv = pd.read_csv(csv_file)
        csv_columns = df_csv.columns.tolist()
        st.write("Loaded CSV column structure:", csv_columns)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()
else:
    # Default columns if no CSV is uploaded
    csv_columns = ['Formula', 'modformula'] + all_elements + [
        'temperature(K)', 'seebeck_coefficient(μV/K)', 'electrical_conductivity(S/m)',
        'thermal_conductivity(W/mK)', 'power_factor(W/mK2)', 'ZT', 'reference', 'sum_elements'
    ]
    st.warning("No CSV uploaded. Using default column structure with 95 columns.")

# File uploader for JSON files
uploaded_files = st.file_uploader(f"Upload JSON files for {n_components}-component system (e.g., {'Bi1Sb1Te2.json' if n_components == 3 else 'Bi1Sb1Te2Se1.json'})", type=["json"], accept_multiple_files=True)

if not uploaded_files:
    st.error("Please upload at least one JSON file.")
    st.stop()

# Process uploaded JSON files
data = []
for uploaded_file in uploaded_files:
    try:
        # Extract formula from filename
        filename = uploaded_file.name
        formula = os.path.splitext(filename)[0]
        
        try:
            # Validate formula with pymatgen
            comp = Composition(formula)
            if len(comp.elements) != n_components:
                st.warning(f"Formula {formula} does not have exactly {n_components} elements (found {len(comp.elements)}). Skipping file.")
                continue
            comp_dict = comp.as_dict()
            total = sum(comp_dict.values())
            if total <= 0:
                st.warning(f"Formula {formula} has invalid stoichiometric sum {total}. Skipping file.")
                continue
        except Exception as e:
            st.warning(f"Invalid chemical formula in filename {filename}: {e}. Skipping file.")
            continue

        # Read JSON content
        json_content = json.load(uploaded_file)
        if not isinstance(json_content, list):
            st.warning(f"File {filename} does not contain a list of dictionaries. Skipping file.")
            continue

        for entry in json_content:
            if not isinstance(entry, dict) or 'x' not in entry or 'y' not in entry:
                st.warning(f"Invalid data format in {filename}. Expected dictionaries with 'x' and 'y' keys. Skipping entry.")
                continue
            try:
                temperature = float(entry['x'])
                seebeck = float(entry['y'])
                data.append({
                    'Formula': formula,
                    'temperature(K)': temperature,
                    'seebeck_coefficient(μV/K)': seebeck
                })
            except (ValueError, TypeError) as e:
                st.warning(f"Error parsing entry {entry} in {filename}: {e}. Skipping entry.")
                continue
    except Exception as e:
        st.warning(f"Error reading JSON file {uploaded_file.name}: {e}. Skipping file.")
        continue

if not data:
    st.error("No valid data extracted from uploaded JSON files. Please check the file format and content.")
    st.stop()

# Create initial DataFrame
df = pd.DataFrame(data)

# Validate required columns
required_columns = ['Formula', 'temperature(K)', 'seebeck_coefficient(μV/K)']
if not all(col in df.columns for col in required_columns):
    st.error("Generated DataFrame is missing required columns. Please check the JSON files.")
    st.stop()

# Display initial DataFrame
st.subheader("Initial Thermoelectric Data")
st.write("Data extracted from uploaded JSON files:")
st.dataframe(df)

# Allow download of initial CSV
st.download_button(
    label="Download Initial CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='thermoelectric_data.csv',
    mime='text/csv',
    key='download_initial'
)

# Count present elements
present_elements = count_elements(df)
st.write("Number of elements present in the new data:", len(present_elements))
st.write("Present elements:", present_elements)

# Featurize materials
features = featurize_materials(df, all_elements, csv_columns)
if not features:
    st.error("No valid formulas found in the data. Please check the JSON files.")
    st.stop()

# Create DataFrame for feature vectors
df_features = pd.DataFrame(features)

# Insert modified formulas
modified_formulas = df['Formula'].apply(extract_multiplier_and_replace)
df['modformula'] = modified_formulas

# Combine feature vectors with original DataFrame
df_combined = pd.concat([df[['Formula', 'modformula', 'temperature(K)', 'seebeck_coefficient(μV/K)']], df_features], axis=1)

# Calculate sum of elemental columns
df_combined['sum_elements'] = df_combined[all_elements].sum(axis=1)

# Check for duplicate columns
if df_combined.columns.duplicated().any():
    duplicate_columns = df_combined.columns[df_combined.columns.duplicated()].tolist()
    st.error(f"Duplicate columns found in DataFrame: {duplicate_columns}")
    st.stop()

# Ensure all CSV columns are present
for col in csv_columns:
    if col not in df_combined.columns:
        df_combined[col] = 0.0

# Reorder columns to match CSV structure
df_combined = df_combined[csv_columns]

# Append to existing CSV data if loaded
if csv_file:
    df_final = pd.concat([df_csv, df_combined], ignore_index=True)
else:
    df_final = df_combined

# Display interactive periodic table
st.subheader("Interactive Periodic Table")
st.write("Elements present in the new data are colored; absent elements are gray. Hover to see electronegativity and thermoelectric weight.")
fig_periodic = plot_periodic_table(all_elements, present_elements, default_element_color_map, fontsize=12)
st.plotly_chart(fig_periodic, use_container_width=True)

# Display and allow download of updated featurized DataFrame
st.subheader("Updated Featurized Data")
st.write("The DataFrame includes the pre-existing CSV data (if uploaded) and new data with feature vectors for all 85 elements, setting absent features to 0.")
st.dataframe(df_final)

st.download_button(
    label="Download Updated Featurized CSV",
    data=df_final.to_csv(index=False).encode('utf-8'),
    file_name='featurized_thermoelectric_data_updated.csv',
    mime='text/csv',
    key='download_featurized'
)
