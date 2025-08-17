import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core.composition import Composition
import sqlite3
import os
import joblib
import colorsys
from scipy.optimize import minimize
from itertools import combinations

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

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

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim=66, latent_dim=8):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128, momentum=0.05), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64, momentum=0.05), nn.Dropout(0.4),
        )
        self.z_mean = nn.Linear(64, latent_dim)
        self.z_log_var = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.BatchNorm1d(64, momentum=0.05), nn.Dropout(0.4),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128, momentum=0.05), nn.Dropout(0.4),
            nn.Linear(128, input_dim), nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.z_mean(h)
        log_var = self.z_log_var(h)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

# Regressor Model
class Regressor(nn.Module):
    def __init__(self, latent_dim=8):
        super(Regressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(), nn.BatchNorm1d(16, momentum=0.05), nn.Dropout(0.4),
            nn.Linear(16, 8), nn.ReLU(), nn.BatchNorm1d(8, momentum=0.05), nn.Dropout(0.4),
            nn.Linear(8, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Preprocessing for prediction
def featurize_composition(composition_dict, available_elements, temperature):
    feature_vector = {element: composition_dict.get(element, 0) for element in available_elements}
    feature_vector['temperature(K)'] = temperature
    return pd.DataFrame([feature_vector])

def preprocess_new_data(df, available_elements, scaler):
    features_df = df
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(features_df)
    X_scaled = scaler.transform(X_imputed)
    return X_scaled

def predict_seebeck(composition_dict, temperature, available_elements, scaler, vae, regressor, y_scaler):
    try:
        df = featurize_composition(composition_dict, available_elements, temperature)
        X_scaled = preprocess_new_data(df, available_elements, scaler)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        vae.eval()
        regressor.eval()
        with torch.no_grad():
            _, z_mean, _ = vae(X_tensor)
            y_scaled_pred = regressor(z_mean)
            y_pred = y_scaler.inverse_transform(y_scaled_pred.cpu().numpy().reshape(-1, 1)).ravel()
        return y_pred[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# Load models and scalers
script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    vae = VAE().to(device)
    regressor = Regressor().to(device)
    vae.load_state_dict(torch.load(os.path.join(script_dir, 'vae_model.pt'), map_location=device))
    regressor.load_state_dict(torch.load(os.path.join(script_dir, 'regressor_model.pt'), map_location=device))
    scaler = joblib.load(os.path.join(script_dir, 'scaler.pkl'))
    y_scaler = joblib.load(os.path.join(script_dir, 'y_scaler.pkl'))
except FileNotFoundError as e:
    st.error(f"Required files not found: {e}")
    st.stop()
except RuntimeError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Available elements
available_elements = [
    'Mg', 'Cs', 'Co', 'Zr', 'Se', 'Dy', 'Pb', 'Ga', 'O', 'Sn', 'Yb', 'B', 'La', 'Si', 'V', 'Fe', 'S', 'Sc', 'Tl', 'Zn',
    'Cl', 'Ce', 'Er', 'Nd', 'Pd', 'Y', 'P', 'Ta', 'In', 'Te', 'Ru', 'Rb', 'Tm', 'Tb', 'Sb', 'Al', 'Lu', 'Bi', 'Pr', 'Eu',
    'Sm', 'Ba', 'Cr', 'Sr', 'Ni', 'Ca', 'As', 'Mn', 'Mo', 'Cd', 'Ti', 'Nb', 'Hf', 'Gd', 'Ag', 'Ge', 'Li', 'Br', 'Au', 'I',
    'N', 'Na', 'Cu', 'Ho', 'K'
]

# All elements for full periodic table
all_elements = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'
]

# Enhanced color map for all elements
base_color_list = (
    px.colors.qualitative.Plotly +
    px.colors.qualitative.Pastel1 +
    px.colors.qualitative.D3 +
    px.colors.qualitative.G10 +
    px.colors.qualitative.T10 +
    px.colors.qualitative.Set1 +
    px.colors.qualitative.Set2 +
    px.colors.qualitative.Set3 +
    px.colors.qualitative.Pastel2 +
    px.colors.qualitative.Dark2
)
num_additional_colors = len(all_elements) - len(base_color_list)
additional_colors = []
for i in range(num_additional_colors):
    hue = i / num_additional_colors
    rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    additional_colors.append(hex_color)
default_color_list = base_color_list + additional_colors
default_element_color_map = dict(zip(all_elements, default_color_list[:len(all_elements)]))

# Streamlit UI
st.title("Ternary Seebeck Coefficient Predictor")
st.markdown("""
This application predicts the Seebeck coefficient for a ternary composition of selected elements at a specified temperature, visualized in a ternary diagram. Select up to three elements from the dropdown below, input their stoichiometric coefficients, and view the absolute Seebeck coefficient across compositions. The app also identifies compositions with minimum and maximum absolute Seebeck coefficients and plots their variation with temperature.
**Date and Time**: 06:32 PM CEST, Sunday, August 17, 2025
""")

# Initialize session state with fallback
try:
    if 'selected_elements' not in st.session_state:
        st.session_state.selected_elements = []
    if 'compositions' not in st.session_state:
        st.session_state.compositions = {}
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 300
except Exception as e:
    st.warning(f"Session state initialization failed: {e}. Resetting to defaults.")
    st.session_state.selected_elements = []
    st.session_state.compositions = {}
    st.session_state.temperature = 300

# Periodic Table for Reference
st.header("Periodic Table Reference")
st.write("Use the dropdown below to select up to three elements. The periodic table below shows available elements (colored), unavailable elements (gray), and selected elements (bold outline).")
def plot_periodic_table(available_elements, selected_elements, element_color_map, fontsize=14):
    periodic_table_positions = {
        'Li': (3, 1), 'Na': (4, 1), 'K': (5, 1), 'Rb': (6, 1), 'Cs': (7, 1),
        'Be': (3, 2), 'Mg': (4, 2), 'Ca': (5, 2), 'Sr': (6, 2), 'Ba': (7, 2),
        'Sc': (5, 3), 'Y': (6, 3), 'La': (8, 3), 'Ce': (8, 4), 'Pr': (8, 5), 'Nd': (8, 6),
        'Sm': (8, 7), 'Eu': (8, 8), 'Gd': (8, 9), 'Tb': (8, 10), 'Dy': (8, 11), 'Ho': (8, 12),
        'Er': (8, 13), 'Tm': (8, 14), 'Yb': (8, 15), 'Lu': (8, 16),
        'Ti': (5, 4), 'Zr': (6, 4), 'Hf': (7, 4), 'V': (5, 5), 'Nb': (6, 5), 'Ta': (7, 5),
        'Cr': (5, 6), 'Mo': (6, 6), 'Mn': (5, 7), 'Fe': (5, 8), 'Co': (5, 9), 'Ni': (5, 10),
        'Cu': (5, 11), 'Zn': (5, 12), 'B': (3, 13), 'Al': (4, 13), 'Ga': (5, 13), 'In': (6, 13),
        'Tl': (7, 13), 'Si': (4, 14), 'Ge': (5, 14), 'Sn': (6, 14), 'Pb': (7, 14),
        'P': (4, 15), 'As': (5, 15), 'Sb': (6, 15), 'Bi': (7, 15), 'S': (4, 16), 'Se': (5, 16),
        'Te': (6, 16), 'Cl': (4, 17), 'Br': (5, 17), 'I': (6, 17), 'Au': (7, 11), 'Ag': (6, 11),
        'Cd': (6, 12), 'Pd': (6, 10), 'Ru': (6, 8), 'N': (3, 15), 'Na': (3, 1), 'K': (4, 1)
    }
    fig = go.Figure()
    for element in all_elements:
        if element in periodic_table_positions:
            row, col = periodic_table_positions[element]
            color = element_color_map.get(element, '#D3D3D3') if element in available_elements else '#D3D3D3'
            opacity = 1.0 if element in selected_elements else (0.7 if element in available_elements else 0.3)
            line_width = 4 if element in selected_elements else 2  # Bold outline for selected elements
            fig.add_trace(go.Scatter(
                x=[col], y=[-row],
                mode='markers+text',
                text=[element],
                textposition='middle center',
                textfont=dict(size=fontsize, family='Arial'),
                marker=dict(size=40, color=color, opacity=opacity, line=dict(width=line_width, color='black')),
                hoverinfo='text',
                hovertext=[f"Element: {element}<br>Electronegativity: {electronegativity.get(element, 1.0):.2f}<br>Thermoelectric Weight: {thermoelectric_weights.get(element, 1.0):.2f}"],
                name=element,
                showlegend=False
            ))
    fig.update_layout(
        title=dict(text='Periodic Table Reference (Available Elements in Color, Selected Elements with Bold Outline)', x=0.5, xanchor='center', font=dict(size=fontsize + 4, family='Arial')),
        xaxis=dict(range=[0, 19], showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(range=[-9, -2], showgrid=False, zeroline=False, showticklabels=False, title=''),
        plot_bgcolor='white', paper_bgcolor='white',
        width=900, height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Plot periodic table for reference
fig_periodic = plot_periodic_table(available_elements, st.session_state.selected_elements, default_element_color_map)
st.plotly_chart(fig_periodic, use_container_width=True)

# Element selection via dropdown
st.header("Select Elements")
st.session_state.selected_elements = st.multiselect(
    "Select up to three elements",
    options=available_elements,
    default=st.session_state.selected_elements,
    max_selections=3,
    key='element_selector'
)

# Update compositions based on selected elements
for element in st.session_state.selected_elements:
    if element not in st.session_state.compositions:
        st.session_state.compositions[element] = 0.0
st.session_state.compositions = {k: v for k, v in st.session_state.compositions.items() if k in st.session_state.selected_elements}

# Display selected elements and allow composition input
st.header("Input Stoichiometric Coefficients")
if st.session_state.selected_elements:
    st.write(f"Selected Elements: {', '.join(st.session_state.selected_elements)}")
    cols = st.columns(len(st.session_state.selected_elements))
    for idx, element in enumerate(st.session_state.selected_elements):
        with cols[idx]:
            st.session_state.compositions[element] = st.number_input(
                f"Composition for {element} (0 to 1)", min_value=0.0, max_value=1.0,
                value=st.session_state.compositions.get(element, 0.0), step=0.1, key=f"comp_{element}"
            )
    # Normalize compositions
    if st.button("Normalize Compositions"):
        total = sum(st.session_state.compositions.values())
        if total > 0:
            for element in st.session_state.compositions:
                st.session_state.compositions[element] /= total
            st.rerun()
else:
    st.write("Please select up to three elements from the dropdown.")

# Temperature input
st.session_state.temperature = st.number_input("Enter Temperature (K):", min_value=0, max_value=5000, value=st.session_state.temperature, step=10)

# Complete to three elements if fewer are selected
def complete_to_three_elements(selected_elements, compositions, available_elements):
    while len(selected_elements) < 3:
        remaining_elements = [e for e in available_elements if e not in selected_elements]
        if remaining_elements:
            random_element = np.random.choice(remaining_elements)
            selected_elements.append(random_element)
            compositions[random_element] = 0.0
        else:
            st.error("Not enough available elements to complete the ternary composition.")
            return selected_elements, compositions
    return selected_elements, compositions

# Generate ternary diagram
def generate_ternary_data(elements, temperature, available_elements, scaler, vae, regressor, y_scaler, steps=10):
    compositions = []
    seebeck_values = []
    for a in np.linspace(0, 1, steps):
        for b in np.linspace(0, 1 - a, steps):
            c = 1 - a - b
            if c >= 0:
                comp_dict = {elements[0]: a, elements[1]: b, elements[2]: c}
                seebeck = predict_seebeck(comp_dict, temperature, available_elements, scaler, vae, regressor, y_scaler)
                if seebeck is not None:
                    compositions.append([a, b, c])
                    seebeck_values.append(abs(seebeck))  # Use absolute value
    return np.array(compositions), np.array(seebeck_values)

# Optimize for maximum and minimum absolute Seebeck coefficient
def optimize_seebeck(elements, temperature, available_elements, scaler, vae, regressor, y_scaler, maximize=True):
    def objective(x):
        comp_dict = {elements[i]: x[i] for i in range(3)}
        seebeck = predict_seebeck(comp_dict, temperature, available_elements, scaler, vae, regressor, y_scaler)
        if seebeck is None:
            return float('inf') if maximize else float('-inf')
        return -abs(seebeck) if maximize else abs(seebeck)  # Optimize absolute value
    initial_guess = [1/3, 1/3, 1/3]
    constraints = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bounds = [(0, 1)] * 3
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_comp = result.x
    optimal_seebeck = predict_seebeck({elements[i]: optimal_comp[i] for i in range(3)}, temperature, available_elements, scaler, vae, regressor, y_scaler)
    return optimal_comp, abs(optimal_seebeck) if optimal_seebeck is not None else (float('-inf') if maximize else float('inf'))

def plot_ternary_diagram(compositions, seebeck_values, elements, user_composition, user_seebeck, min_comp, min_seebeck, max_comp, max_seebeck, color_scale='viridis'):
    fig = go.Figure()
    # Ternary scatter plot
    fig.add_trace(go.Scatterternary(
        a=compositions[:, 0], b=compositions[:, 1], c=compositions[:, 2],
        mode='markers',
        marker=dict(size=10, color=seebeck_values, colorscale=color_scale, showscale=True, colorbar=dict(title='|Seebeck| (μV/K)')),
        text=[f"|Seebeck|: {s:.2f}" for s in seebeck_values],
        hoverinfo='text',
        name='Compositions'
    ))
    # User composition
    if user_seebeck is not None:
        fig.add_trace(go.Scatterternary(
            a=[user_composition[0]], b=[user_composition[1]], c=[user_composition[2]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            text=[f"User Composition<br>|Seebeck|: {abs(user_seebeck):.2f}"],
            hoverinfo='text',
            name='User Composition'
        ))
    # Minimum Seebeck
    if min_seebeck != float('inf'):
        fig.add_trace(go.Scatterternary(
            a=[min_comp[0]], b=[min_comp[1]], c=[min_comp[2]],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='diamond'),
            text=[f"Min |Seebeck|: {min_seebeck:.2f}"],
            hoverinfo='text',
            name='Min |Seebeck|'
        ))
    # Maximum Seebeck
    if max_seebeck != float('-inf'):
        fig.add_trace(go.Scatterternary(
            a=[max_comp[0]], b=[max_comp[1]], c=[max_comp[2]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='square'),
            text=[f"Max |Seebeck|: {max_seebeck:.2f}"],
            hoverinfo='text',
            name='Max |Seebeck|'
        ))
    fig.update_layout(
        title=dict(text=f"Ternary Diagram: |Seebeck Coefficient| at {st.session_state.temperature} K", x=0.5, xanchor='center', font=dict(size=16, family='Arial')),
        ternary=dict(
            sum=1,
            aaxis=dict(title=elements[0], tickformat='.2f'),
            baxis=dict(title=elements[1], tickformat='.2f'),
            caxis=dict(title=elements[2], tickformat='.2f')
        ),
        showlegend=True,
        legend=dict(x=1.05, y=1, font=dict(size=12)),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def plot_temperature_variance(elements, user_composition, min_comp, max_comp, temp_range, available_elements, scaler, vae, regressor, y_scaler):
    temps = np.linspace(temp_range[0], temp_range[1], 20)
    user_seebeck = []
    min_seebeck = []
    max_seebeck = []
    for temp in temps:
        user_val = predict_seebeck({elements[i]: user_composition[i] for i in range(3)}, temp, available_elements, scaler, vae, regressor, y_scaler)
        min_val = predict_seebeck({elements[i]: min_comp[i] for i in range(3)}, temp, available_elements, scaler, vae, regressor, y_scaler)
        max_val = predict_seebeck({elements[i]: max_comp[i] for i in range(3)}, temp, available_elements, scaler, vae, regressor, y_scaler)
        user_seebeck.append(abs(user_val) if user_val is not None else np.nan)
        min_seebeck.append(abs(min_val) if min_val is not None else np.nan)
        max_seebeck.append(abs(max_val) if max_val is not None else np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temps, y=user_seebeck, mode='lines+markers', name='User Composition', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=temps, y=min_seebeck, mode='lines+markers', name='Min |Seebeck|', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=temps, y=max_seebeck, mode='lines+markers', name='Max |Seebeck|', line=dict(color='green')))
    fig.update_layout(
        title=dict(text='|Seebeck Coefficient| vs Temperature', x=0.5, xanchor='center', font=dict(size=16, family='Arial')),
        xaxis_title='Temperature (K)', yaxis_title='|Seebeck Coefficient| (μV/K)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=2, linecolor='black', tickfont=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=2, linecolor='black', tickfont=dict(size=12)),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=1.05, y=1, font=dict(size=12)),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# Generate ternary diagram and temperature variance plot
if st.button("Generate Ternary Diagram"):
    if len(st.session_state.selected_elements) > 0:
        elements, compositions = complete_to_three_elements(
            st.session_state.selected_elements.copy(),
            st.session_state.compositions.copy(),
            available_elements
        )
        total = sum(compositions.values())
        if total == 0:
            st.error("Please provide non-zero compositions for at least one element.")
        else:
            # Normalize user composition
            user_composition = [compositions.get(elements[i], 0) / total if total > 0 else 0 for i in range(3)]
            # Predict Seebeck for user composition
            user_seebeck = predict_seebeck({elements[i]: user_composition[i] for i in range(3)}, st.session_state.temperature, available_elements, scaler, vae, regressor, y_scaler)
            if user_seebeck is None:
                st.error("Failed to predict Seebeck coefficient for user composition.")
            else:
                # Generate ternary data
                compositions_array, seebeck_values = generate_ternary_data(elements, st.session_state.temperature, available_elements, scaler, vae, regressor, y_scaler)
                if len(compositions_array) == 0:
                    st.error("Failed to generate ternary data due to prediction errors.")
                else:
                    # Optimize for min and max absolute Seebeck
                    min_comp, min_seebeck = optimize_seebeck(elements, st.session_state.temperature, available_elements, scaler, vae, regressor, y_scaler, maximize=False)
                    max_comp, max_seebeck = optimize_seebeck(elements, st.session_state.temperature, available_elements, scaler, vae, regressor, y_scaler, maximize=True)
                    # Display composition and Seebeck
                    st.write("### Composition and Seebeck Coefficient")
                    st.write(f"**User Composition**: {elements[0]}: {user_composition[0]:.2f}, {elements[1]}: {user_composition[1]:.2f}, {elements[2]}: {user_composition[2]:.2f}")
                    st.write(f"**User |Seebeck Coefficient|**: {abs(user_seebeck):.2f} μV/K")
                    st.write(f"**Minimum |Seebeck| Composition**: {elements[0]}: {min_comp[0]:.2f}, {elements[1]}: {min_comp[1]:.2f}, {elements[2]}: {min_comp[2]:.2f}")
                    st.write(f"**Minimum |Seebeck Coefficient|**: {min_seebeck:.2f} μV/K")
                    st.write(f"**Maximum |Seebeck| Composition**: {elements[0]}: {max_comp[0]:.2f}, {elements[1]}: {max_comp[1]:.2f}, {elements[2]}: {max_comp[2]:.2f}")
                    st.write(f"**Maximum |Seebeck Coefficient|**: {max_seebeck:.2f} μV/K")
                    # Plot ternary diagram
                    st.write("### Ternary Diagram")
                    fig_ternary = plot_ternary_diagram(compositions_array, seebeck_values, elements, user_composition, user_seebeck, min_comp, min_seebeck, max_comp, max_seebeck)
                    st.plotly_chart(fig_ternary, use_container_width=True)
                    try:
                        fig_ternary.write_html(os.path.join(script_dir, 'ternary_diagram.html'))
                    except Exception as e:
                        st.warning(f"Failed to save ternary diagram: {e}")
                    # Plot temperature variance
                    st.write("### |Seebeck Coefficient| vs Temperature")
                    fig_temp = plot_temperature_variance(elements, user_composition, min_comp, max_comp, [100, 1000], available_elements, scaler, vae, regressor, y_scaler)
                    st.plotly_chart(fig_temp, use_container_width=True)
                    try:
                        fig_temp.write_html(os.path.join(script_dir, 'temperature_variance.html'))
                    except Exception as e:
                        st.warning(f"Failed to save temperature variance plot: {e}")
    else:
        st.error("Please select at least one element.")
