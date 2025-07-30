import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import joblib
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core.composition import Composition
import sqlite3
import re
import os
from matplotlib.lines import Line2D

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Set matplotlib font for publication quality
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

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

# Preprocessing function for visualization
def preprocess_data(df, scaler, y_scaler):
    input_features = df[['Mg', 'Cs', 'Co', 'Zr', 'Se', 'Dy', 'Pb', 'Ga', 'O', 'Sn',
                         'Yb', 'B', 'La', 'Si', 'V', 'Fe', 'S', 'Sc', 'Tl', 'Zn',
                         'Cl', 'Ce', 'Er', 'Nd', 'Pd', 'Y', 'P', 'Ta', 'In', 'Te',
                         'Ru', 'Rb', 'Tm', 'Tb', 'Sb', 'Al', 'Lu', 'Bi', 'Pr', 'Eu',
                         'Sm', 'Ba', 'Cr', 'Sr', 'Ni', 'Ca', 'As', 'Mn', 'Mo', 'Cd',
                         'Ti', 'Nb', 'Hf', 'Gd', 'Ag', 'Ge', 'Li', 'Br', 'Au', 'I',
                         'N', 'Na', 'Cu', 'Ho', 'K', 'temperature(K)']]
    output_feature = df['seebeck_coefficient(μV/K)']
    imputer_input = SimpleImputer(strategy='mean')
    input_features_imputed = imputer_input.fit_transform(input_features)
    imputer_output = SimpleImputer(strategy='mean')
    output_feature_imputed = imputer_output.fit_transform(output_feature.values.reshape(-1, 1)).ravel()
    iso_forest = IsolationForest(contamination=0.1)
    outliers = iso_forest.fit_predict(input_features_imputed) == -1
    input_features_cleaned = input_features_imputed[~outliers]
    output_feature_cleaned = output_feature_imputed[~outliers]
    valid_indices = np.where(~outliers)[0]
    mask = (output_feature_cleaned >= -1174.0) & (output_feature_cleaned <= 1052.0)
    input_features_cleaned = input_features_cleaned[mask]
    output_feature_cleaned = output_feature_cleaned[mask]
    valid_indices = valid_indices[mask]
    X_scaled = scaler.transform(input_features_cleaned)
    y_scaled = y_scaler.transform(output_feature_cleaned.reshape(-1, 1)).ravel()
    return X_scaled, y_scaled, output_feature_cleaned, valid_indices

# Preprocessing for prediction
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
        modified_formula = formula_without_multiplier.split('(')[0] + ''.join(element + stoichiometry for element, stoichiometry in modified_elements)
        return modified_formula
    return input_formula

def featurize_materials(df, available_elements):
    features = []
    for _, row in df.iterrows():
        modified_formula = extract_multiplier_and_replace(row['Formula'])
        composition = Composition(modified_formula)
        composition_dict = composition.fractional_composition.as_dict()
        feature_vector = {element: composition_dict.get(element, 0) for element in available_elements}
        feature_vector['temperature(K)'] = row['temperature(K)']
        features.append(feature_vector)
    return pd.DataFrame(features)

def preprocess_new_data(df, available_elements, scaler):
    features_df = featurize_materials(df, available_elements)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(features_df)
    X_scaled = scaler.transform(X_imputed)
    return X_scaled

def plot_radar(data, labels, title, max_samples=10, alpha=0.3, linewidth=2, fontsize=16, legend_pos='upper right', axis_linewidth=1.5):
    num_vars = data.shape[1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, min(max_samples, len(data))))
    for i in range(min(max_samples, len(data))):
        values = data[i].tolist()
        values += values[:1]  # Close the radar shape
        ax.fill(angles, values, color=colors[i], alpha=alpha, label=labels[i])
        ax.plot(angles, values, color=colors[i], linewidth=linewidth)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), [f'Latent Dim {i+1}' for i in range(num_vars)],
                      fontsize=fontsize, weight='bold')
    ax.set_title(title, fontsize=fontsize + 2, pad=20, weight='bold')
    ax.legend(loc=legend_pos, bbox_to_anchor=(1.2, 1.1), fontsize=fontsize - 2, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_linewidth(axis_linewidth)
    plt.tight_layout()
    return fig

# Enhanced Training History Plot (Matplotlib)
def plot_training_history_matplotlib(history_df, title, filename, linewidth=2.5, fontsize=12, train_color='#1f77b4', val_color='#ff7f0e', tick_fontsize=10, axis_linewidth=1.5):
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    ax1.plot(history_df['loss'], label='Training Loss', linewidth=linewidth, color=train_color)
    ax1.plot(history_df['val_loss'], label='Validation Loss', linewidth=linewidth, color=val_color)
    ax1.set_xlabel('Epoch', fontsize=fontsize, weight='bold')
    ax1.set_ylabel('Loss', fontsize=fontsize, weight='bold')
    ax1.set_title(f'{title} Loss', fontsize=fontsize + 2, weight='bold')
    ax1.legend(fontsize=fontsize - 2, frameon=True, edgecolor='black')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(axis_linewidth)
    ax1.spines['bottom'].set_linewidth(axis_linewidth)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=axis_linewidth, length=6)
    ax2.plot(history_df['mse'], label='Training MSE', linewidth=linewidth, color=train_color)
    ax2.plot(history_df['val_mse'], label='Validation MSE', linewidth=linewidth, color=val_color)
    ax2.set_xlabel('Epoch', fontsize=fontsize, weight='bold')
    ax2.set_ylabel('MSE', fontsize=fontsize, weight='bold')
    ax2.set_title(f'{title} MSE', fontsize=fontsize + 2, weight='bold')
    ax2.legend(fontsize=fontsize - 2, frameon=True, edgecolor='black')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(axis_linewidth)
    ax2.spines['bottom'].set_linewidth(axis_linewidth)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=axis_linewidth, length=6)
    plt.tight_layout()
    return fig

# Enhanced Training History Plot (Plotly)
def plot_training_history_plotly(history_df, title, train_color='#1f77b4', val_color='#ff7f0e', linewidth=3, label_fontsize=12, tick_fontsize=10, axis_linewidth=2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['loss'], name='Training Loss', line=dict(width=linewidth, color=train_color), mode='lines+markers', marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['val_loss'], name='Validation Loss', line=dict(width=linewidth, color=val_color), mode='lines+markers', marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['mse'], name='Training MSE', line=dict(width=linewidth, dash='dash', color=train_color), mode='lines+markers', marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=history_df.index, y=history_df['val_mse'], name='Validation MSE', line=dict(width=linewidth, dash='dash', color=val_color), mode='lines+markers', marker=dict(size=6)))
    fig.update_layout(
        title=dict(text=f'{title} Training Metrics', x=0.5, xanchor='center', font=dict(size=label_fontsize + 4, family='Arial')),
        xaxis_title='Epoch', yaxis_title='Value',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=axis_linewidth, linecolor='black', tickfont=dict(size=tick_fontsize)),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=axis_linewidth, linecolor='black', tickfont=dict(size=tick_fontsize)),
        plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial', size=label_fontsize),
        legend=dict(x=1.05, y=1, font=dict(size=label_fontsize - 2), bordercolor='black', borderwidth=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# Plotly Box Plot for Latent Dimensions
def plot_latent_box(z_train, box_linewidth=1, label_fontsize=12, axis_linewidth=2):
    fig = go.Figure()
    for i in range(z_train.shape[1]):
        fig.add_trace(go.Box(y=z_train[:, i], name=f'Latent Dim {i+1}', boxmean=True, line=dict(width=box_linewidth), marker=dict(size=6)))
    fig.update_layout(
        title=dict(text='Distribution of Latent Dimensions', x=0.5, xanchor='center', font=dict(size=label_fontsize + 4, family='Arial')),
        xaxis_title='Latent Dimensions', yaxis_title='Value',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=axis_linewidth, linecolor='black', tickfont=dict(size=label_fontsize)),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=axis_linewidth, linecolor='black', tickfont=dict(size=label_fontsize)),
        plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial', size=label_fontsize),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# Periodic Table Plot (Present Elements)
def plot_periodic_table(available_elements, element_color_map, fontsize=12):
    periodic_table_positions = {
        'Li': (3, 1), 'Na': (4, 1), 'K': (5, 1), 'Rb': (6, 1), 'Cs': (7, 1),
        'Be': (3, 2), 'Mg': (4, 2), 'Ca': (5, 2), 'Sr': (6, 2), 'Ba': (7, 2),
        'Sc': (5, 3), 'Y': (6, 3),
        'Ti': (5, 4), 'Zr': (6, 4), 'Hf': (7, 4),
        'V': (5, 5), 'Nb': (6, 5), 'Ta': (7, 5),
        'Cr': (5, 6), 'Mo': (6, 6),
        'Mn': (5, 7),
        'Fe': (5, 8), 'Co': (5, 9), 'Ni': (5, 10), 'Cu': (5, 11), 'Zn': (5, 12),
        'B': (3, 13), 'Al': (4, 13), 'Ga': (5, 13), 'In': (6, 13), 'Tl': (7, 13),
        'C': (3, 14), 'Si': (4, 14), 'Ge': (5, 14), 'Sn': (6, 14), 'Pb': (7, 14),
        'N': (3, 15), 'P': (4, 15), 'As': (5, 15), 'Sb': (6, 15), 'Bi': (7, 15),
        'O': (3, 16), 'S': (4, 16), 'Se': (5, 16), 'Te': (6, 16),
        'F': (3, 17), 'Cl': (4, 17), 'Br': (5, 17), 'I': (6, 17),
        'Au': (7, 11), 'Ag': (6, 11), 'Cd': (6, 12), 'Pd': (6, 10), 'Ru': (6, 8),
        'La': (8, 3), 'Ce': (8, 4), 'Pr': (8, 5), 'Nd': (8, 6), 'Sm': (8, 7), 'Eu': (8, 8),
        'Gd': (8, 9), 'Tb': (8, 10), 'Dy': (8, 11), 'Ho': (8, 12), 'Er': (8, 13), 'Tm': (8, 14), 'Yb': (8, 15), 'Lu': (8, 16)
    }
    fig = go.Figure()
    for element in available_elements:
        if element in periodic_table_positions:
            row, col = periodic_table_positions[element]
            en = electronegativity.get(element, 1.0)
            tw = thermoelectric_weights.get(element, 1.0)
            fig.add_trace(go.Scatter(
                x=[col], y=[-row],
                mode='markers+text',
                text=[element],
                textposition='middle center',
                marker=dict(size=40, color=element_color_map.get(element, '#FFFFFF'), line=dict(width=2, color='black')),
                hoverinfo='text',
                hovertext=[f"Element: {element}<br>Electronegativity: {en:.2f}<br>Thermoelectric Weight: {tw:.2f}"],
                customdata=[element],
                name=element,
                showlegend=False
            ))
    fig.update_layout(
        title=dict(text='Periodic Table Legend (Present Elements)', x=0.5, xanchor='center', font=dict(size=fontsize + 4, family='Arial')),
        xaxis=dict(range=[0, 19], showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(range=[-9, -2], showgrid=False, zeroline=False, showticklabels=False, title=''),
        plot_bgcolor='white', paper_bgcolor='white',
        width=900, height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Full Periodic Table Plot (All Elements)
def plot_full_periodic_table(all_elements, available_elements, element_color_map, fontsize=12):
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
            color = element_color_map.get(element, '#D3D3D3') if element in available_elements else '#D3D3D3'
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
        title=dict(text='Periodic Table Legend (All Elements)', x=0.5, xanchor='center', font=dict(size=fontsize + 4, family='Arial')),
        xaxis=dict(range=[0, 19], showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(range=[-8, 0], showgrid=False, zeroline=False, showticklabels=False, title=''),
        plot_bgcolor='white', paper_bgcolor='white',
        width=900, height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

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

# Color map options (lowercase for Matplotlib compatibility)
color_map_options = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'jet', 'rainbow',
    'bluered', 'electric', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
    'greys', 'greens', 'blues', 'reds', 'purples', 'oranges',
    'ylorrd', 'ylorbr', 'ylgnbu', 'ylgn', 'rdpu', 'purd', 'pubugn', 'pubu',
    'orrd', 'gnbu', 'bupu', 'bugn', 'pinkyl', 'coolwarm', 'spectral',
    'rdylbu', 'rdylgn', 'rdbu', 'piyg', 'prgn', 'brbg', 'puor', 'rdgy',
    'viridis_r', 'plasma_r', 'inferno_r', 'magma_r', 'cividis_r', 'turbo_r',
    'jet_r', 'rainbow_r', 'greys_r', 'blues_r', 'reds_r'
]

# Define default color map for scatter plots
default_color_list = (
    px.colors.qualitative.Plotly +
    px.colors.qualitative.Pastel1 +
    list(plt.cm.tab20(np.linspace(0, 1, 20))) +
    list(plt.cm.tab20b(np.linspace(0, 1, 20))) +
    list(plt.cm.tab20c(np.linspace(0, 1, 20)))
)
default_color_list = [matplotlib.colors.to_hex(c) if isinstance(c, tuple) else c for c in default_color_list]
default_element_color_map = dict(zip(available_elements, default_color_list[:len(available_elements)]))

# Streamlit UI
st.title("Thermoelectric Material Analysis and Seebeck Coefficient Prediction")

# Database connection
db_path = os.path.join(script_dir, 'thermoelectric_data.db')
try:
    conn = sqlite3.connect(db_path)
except sqlite3.Error as e:
    st.error(f"Error connecting to database: {e}")
    st.stop()

# Tabs for different functionalities
tab1, tab2 = st.tabs(["Visualizations", "Seebeck Prediction"])

with tab1:
    st.header("Data Visualizations")
    
    # Load data from database
    try:
        df = pd.read_sql("SELECT * FROM thermoelectric_materials;", conn)
        vae_history_df = pd.read_sql("SELECT * FROM vae_training_history;", conn)
        regressor_history_df = pd.read_sql("SELECT * FROM regressor_training_history;", conn)
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        conn.close()
        st.stop()

    # Sidebar for visualization settings
    st.sidebar.header("Visualization Settings")
    marker_size = st.sidebar.slider("Scatter Marker Size", 5, 100, 50, 5)
    marker_alpha = st.sidebar.slider("Scatter Marker Transparency", 0.1, 1.0, 0.6, 0.1)
    color_scale = st.sidebar.selectbox(
        "Color Scale for Seebeck Plot",
        color_map_options,
        index=0
    )
    scatter_label_fontsize = st.sidebar.slider("Scatter Label Font Size", 8, 16, 12, 1)
    scatter_axis_linewidth = st.sidebar.slider("Scatter Axis Line Width", 0.5, 5.0, 2.0, 0.5)
    fig_width = st.sidebar.slider("Matplotlib Figure Width", 6, 12, 8, 1)
    fig_height = st.sidebar.slider("Matplotlib Figure Height", 4, 10, 6, 1)
    history_linewidth = st.sidebar.slider("Training History Line Width", 1.0, 5.0, 2.5, 0.5)
    history_label_fontsize = st.sidebar.slider("Training History Label Font Size", 8, 16, 12, 1)
    history_tick_fontsize = st.sidebar.slider("Training History Tick Font Size", 6, 14, 10, 1)
    history_axis_linewidth = st.sidebar.slider("Training History Axis Line Width", 0.5, 5.0, 1.5, 0.5)
    color_options = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
        '#7f7f7f', '#bcbd22', '#17becf', '#000000', '#FFD700', '#00FF00', '#FF1493',
        '#00CED1', '#FF4500', '#6A5ACD'
    ]
    train_color = st.sidebar.selectbox("Training Line Color", color_options, index=0)
    val_color = st.sidebar.selectbox("Validation Line Color", color_options, index=1)
    max_samples = st.sidebar.slider("Number of Samples in Radar Plot", 1, 10, 5, 1)
    radar_alpha = st.sidebar.slider("Radar Fill Transparency", 0.1, 0.5, 0.3, 0.05)
    radar_linewidth = st.sidebar.slider("Radar Line Width", 1.0, 5.0, 2.0, 0.5)
    radar_fontsize = st.sidebar.slider("Radar Font Size", 8, 16, 12, 1)
    periodic_table_fontsize = st.sidebar.slider("Periodic Table Font Size", 8, 16, 12, 1)
    radar_legend_pos = st.sidebar.selectbox("Radar Legend Position", ['upper right', 'upper left', 'lower right', 'lower left'], index=0)
    radar_axis_linewidth = st.sidebar.slider("Radar Axis Line Width", 0.5, 5.0, 1.5, 0.5)
    box_linewidth = st.sidebar.slider("Box Plot Line Width", 0.5, 5.0, 1.0, 0.5)
    box_label_fontsize = st.sidebar.slider("Box Plot Label Font Size", 8, 16, 12, 1)
    box_axis_linewidth = st.sidebar.slider("Box Plot Axis Line Width", 0.5, 5.0, 2.0, 0.5)
    parallel_color_scale = st.sidebar.selectbox(
        "Color Scale for Parallel Coordinates",
        color_map_options,
        index=0
    )
    parallel_label_fontsize = st.sidebar.slider("Parallel Coordinates Label Font Size", 8, 16, 12, 1)

    # Latent Space Visualizations
    if not df.empty:
        st.subheader("Latent Space Visualizations")
        X_scaled, y_scaled, output_feature_cleaned, valid_indices = preprocess_data(df, scaler, y_scaler)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        vae.eval()
        with torch.no_grad():
            _, z_mean, _ = vae(X_tensor)
        z_train = z_mean.cpu().numpy()
        z_scaler = MinMaxScaler()
        z_normalized = z_scaler.fit_transform(z_train)
        pca = PCA(n_components=2)
        z_2d_pca = pca.fit_transform(z_train)
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
        z_2d_tsne = tsne.fit_transform(z_train)
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        z_2d_umap = umap_reducer.fit_transform(z_train)

        def get_dominant_element(formula):
            try:
                comp = Composition(formula)
                if not comp.valid:
                    return 'Unknown'
                comp_dict = comp.get_el_amt_dict()
                scores = {
                    el: comp_dict[el] * electronegativity.get(el, 1.0) * thermoelectric_weights.get(el, 1.0)
                    for el in comp_dict
                }
                return max(scores, key=scores.get)
            except:
                return 'Unknown'

        dominant_elements = df['Formula'].apply(get_dominant_element)
        dominant_elements_filtered = dominant_elements.iloc[valid_indices].values
        formulas_filtered = df['Formula'].iloc[valid_indices].values

        st.sidebar.header("Filter by Dominant Element")
        unique_elements = np.unique(dominant_elements_filtered)
        selected_element = st.sidebar.selectbox("Select Dominant Element", ['All'] + list(unique_elements))
        if selected_element != 'All':
            mask = dominant_elements_filtered == selected_element
            z_2d_pca_filtered = z_2d_pca[mask]
            z_2d_tsne_filtered = z_2d_tsne[mask]
            z_2d_umap_filtered = z_2d_umap[mask]
            output_feature_cleaned_filtered = output_feature_cleaned[mask]
            dominant_elements_filtered_filtered = dominant_elements_filtered[mask]
            formulas_filtered_filtered = formulas_filtered[mask]
        else:
            z_2d_pca_filtered = z_2d_pca
            z_2d_tsne_filtered = z_2d_tsne
            z_2d_umap_filtered = z_2d_umap
            output_feature_cleaned_filtered = output_feature_cleaned
            dominant_elements_filtered_filtered = dominant_elements_filtered
            formulas_filtered_filtered = formulas_filtered

        st.write("#### Periodic Table Legend (Present Elements)")
        st.write("Click an element in the periodic table or use the dropdown to filter scatter plots. Colors match the Dominant Element scatter plots.")
        fig_periodic = plot_periodic_table(available_elements, default_element_color_map, fontsize=periodic_table_fontsize)
        st.plotly_chart(fig_periodic, use_container_width=True)
        fig_periodic.write_html(os.path.join(script_dir, 'periodic_table_present.html'))

        st.write("#### Periodic Table Legend (All Elements)")
        st.write("Elements not in the database are shown in gray. Colors match the Dominant Element scatter plots.")
        fig_full_periodic = plot_full_periodic_table(all_elements, available_elements, default_element_color_map, fontsize=periodic_table_fontsize)
        st.plotly_chart(fig_full_periodic, use_container_width=True)
        fig_full_periodic.write_html(os.path.join(script_dir, 'periodic_table_all.html'))

        st.write("#### Dominant Element Distribution")
        element_counts = pd.Series(dominant_elements_filtered).value_counts()
        fig_bar = px.bar(x=element_counts.index, y=element_counts.values, labels={'x': 'Dominant Element', 'y': 'Count'},
                         title='Distribution of Dominant Elements')
        fig_bar.update_traces(marker_color=[default_element_color_map.get(elem, '#FFFFFF') for elem in element_counts.index])
        fig_bar.update_layout(
            title=dict(text='Distribution of Dominant Elements', x=0.5, xanchor='center', font=dict(size=scatter_label_fontsize + 4, family='Arial')),
            xaxis=dict(tickfont=dict(size=scatter_label_fontsize), showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black'),
            yaxis=dict(tickfont=dict(size=scatter_label_fontsize), showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black'),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial', size=scatter_label_fontsize)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        fig_bar.write_html(os.path.join(script_dir, 'dominant_element_distribution.html'))

        st.write("#### Latent Dimensions Box Plot")
        fig_box = plot_latent_box(z_train, box_linewidth=box_linewidth, label_fontsize=box_label_fontsize, axis_linewidth=box_axis_linewidth)
        st.plotly_chart(fig_box, use_container_width=True)
        fig_box.write_html(os.path.join(script_dir, 'latent_box_plotly.html'))

        st.write("#### PCA Latent Space: Seebeck Coefficient")
        fig_pca_seebeck = px.scatter(
            x=z_2d_pca_filtered[:, 0], y=z_2d_pca_filtered[:, 1], color=output_feature_cleaned_filtered, color_continuous_scale=color_scale,
            labels={'x': 'PC1', 'y': 'PC2', 'color': 'Seebeck Coefficient (μV/K)'},
            title=f'PCA Latent Space: Seebeck Coefficient ({selected_element})',
            hover_data={'Formula': formulas_filtered_filtered, 'Dominant Element': dominant_elements_filtered_filtered, 'Seebeck (μV/K)': output_feature_cleaned_filtered}
        )
        fig_pca_seebeck.update_traces(marker=dict(size=marker_size, opacity=marker_alpha))
        fig_pca_seebeck.update_layout(
            title=dict(text=f'PCA Latent Space: Seebeck Coefficient ({selected_element})', x=0.5, xanchor='center', font=dict(size=scatter_label_fontsize + 4, family='Arial')),
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial', size=scatter_label_fontsize),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        st.plotly_chart(fig_pca_seebeck, use_container_width=True)
        fig_pca_seebeck.write_html(os.path.join(script_dir, 'latent_pca_seebeck_plotly.html'))

        st.write("#### PCA Latent Space: Dominant Element")
        fig_pca_elements = px.scatter(
            x=z_2d_pca_filtered[:, 0], y=z_2d_pca_filtered[:, 1], color=dominant_elements_filtered_filtered,
            color_discrete_map=default_element_color_map,
            labels={'x': 'PC1', 'y': 'PC2', 'color': 'Dominant Element'},
            title=f'PCA Latent Space: Dominant Element ({selected_element})',
            hover_data={'Formula': formulas_filtered_filtered, 'Seebeck (μV/K)': output_feature_cleaned_filtered}
        )
        fig_pca_elements.update_traces(marker=dict(size=marker_size, opacity=marker_alpha))
        fig_pca_elements.update_layout(
            title=dict(text=f'PCA Latent Space: Dominant Element ({selected_element})', x=0.5, xanchor='center', font=dict(size=scatter_label_fontsize + 4, family='Arial')),
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial', size=scatter_label_fontsize),
            legend=dict(x=1.05, y=1, font=dict(size=scatter_label_fontsize - 2), bordercolor='black', borderwidth=1),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        st.plotly_chart(fig_pca_elements, use_container_width=True)
        fig_pca_elements.write_html(os.path.join(script_dir, 'latent_pca_elements_plotly.html'))

        st.write("#### t-SNE Latent Space: Seebeck Coefficient")
        fig_tsne_seebeck = px.scatter(
            x=z_2d_tsne_filtered[:, 0], y=z_2d_tsne_filtered[:, 1], color=output_feature_cleaned_filtered, color_continuous_scale=color_scale,
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Seebeck Coefficient (μV/K)'},
            title=f't-SNE Latent Space: Seebeck Coefficient ({selected_element})',
            hover_data={'Formula': formulas_filtered_filtered, 'Dominant Element': dominant_elements_filtered_filtered, 'Seebeck (μV/K)': output_feature_cleaned_filtered}
        )
        fig_tsne_seebeck.update_traces(marker=dict(size=marker_size, opacity=marker_alpha))
        fig_tsne_seebeck.update_layout(
            title=dict(text=f't-SNE Latent Space: Seebeck Coefficient ({selected_element})', x=0.5, xanchor='center', font=dict(size=scatter_label_fontsize + 4, family='Arial')),
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial', size=scatter_label_fontsize),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        st.plotly_chart(fig_tsne_seebeck, use_container_width=True)
        fig_tsne_seebeck.write_html(os.path.join(script_dir, 'latent_tsne_seebeck_plotly.html'))

        st.write("#### t-SNE Latent Space: Dominant Element")
        fig_tsne_elements = px.scatter(
            x=z_2d_tsne_filtered[:, 0], y=z_2d_tsne_filtered[:, 1], color=dominant_elements_filtered_filtered,
            color_discrete_map=default_element_color_map,
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Dominant Element'},
            title=f't-SNE Latent Space: Dominant Element ({selected_element})',
            hover_data={'Formula': formulas_filtered_filtered, 'Seebeck (μV/K)': output_feature_cleaned_filtered}
        )
        fig_tsne_elements.update_traces(marker=dict(size=marker_size, opacity=marker_alpha))
        fig_tsne_elements.update_layout(
            title=dict(text=f't-SNE Latent Space: Dominant Element ({selected_element})', x=0.5, xanchor='center', font=dict(size=scatter_label_fontsize + 4, family='Arial')),
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial', size=scatter_label_fontsize),
            legend=dict(x=1.05, y=1, font=dict(size=scatter_label_fontsize - 2), bordercolor='black', borderwidth=1),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        st.plotly_chart(fig_tsne_elements, use_container_width=True)
        fig_tsne_elements.write_html(os.path.join(script_dir, 'latent_tsne_elements_plotly.html'))

        st.write("#### UMAP Latent Space: Seebeck Coefficient")
        fig_umap_seebeck = px.scatter(
            x=z_2d_umap_filtered[:, 0], y=z_2d_umap_filtered[:, 1], color=output_feature_cleaned_filtered, color_continuous_scale=color_scale,
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'color': 'Seebeck Coefficient (μV/K)'},
            title=f'UMAP Latent Space: Seebeck Coefficient ({selected_element})',
            hover_data={'Formula': formulas_filtered_filtered, 'Dominant Element': dominant_elements_filtered_filtered, 'Seebeck (μV/K)': output_feature_cleaned_filtered}
        )
        fig_umap_seebeck.update_traces(marker=dict(size=marker_size, opacity=marker_alpha))
        fig_umap_seebeck.update_layout(
            title=dict(text=f'UMAP Latent Space: Seebeck Coefficient ({selected_element})', x=0.5, xanchor='center', font=dict(size=scatter_label_fontsize + 4, family='Arial')),
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial', size=scatter_label_fontsize),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        st.plotly_chart(fig_umap_seebeck, use_container_width=True)
        fig_umap_seebeck.write_html(os.path.join(script_dir, 'latent_umap_seebeck_plotly.html'))

        st.write("#### UMAP Latent Space: Dominant Element")
        fig_umap_elements = px.scatter(
            x=z_2d_umap_filtered[:, 0], y=z_2d_umap_filtered[:, 1], color=dominant_elements_filtered_filtered,
            color_discrete_map=default_element_color_map,
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'color': 'Dominant Element'},
            title=f'UMAP Latent Space: Dominant Element ({selected_element})',
            hover_data={'Formula': formulas_filtered_filtered, 'Seebeck (μV/K)': output_feature_cleaned_filtered}
        )
        fig_umap_elements.update_traces(marker=dict(size=marker_size, opacity=marker_alpha))
        fig_umap_elements.update_layout(
            title=dict(text=f'UMAP Latent Space: Dominant Element ({selected_element})', x=0.5, xanchor='center', font=dict(size=scatter_label_fontsize + 4, family='Arial')),
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, showline=True, linewidth=scatter_axis_linewidth, linecolor='black', tickfont=dict(size=scatter_label_fontsize)),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial', size=scatter_label_fontsize),
            legend=dict(x=1.05, y=1, font=dict(size=scatter_label_fontsize - 2), bordercolor='black', borderwidth=1),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        st.plotly_chart(fig_umap_elements, use_container_width=True)
        fig_umap_elements.write_html(os.path.join(script_dir, 'latent_umap_elements_plotly.html'))

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        scatter1 = ax1.scatter(z_2d_pca_filtered[:, 0], z_2d_pca_filtered[:, 1], c=output_feature_cleaned_filtered, cmap='viridis', s=marker_size, alpha=marker_alpha)
        ax1.set_xlabel('PC1', fontsize=scatter_label_fontsize, weight='bold')
        ax1.set_ylabel('PC2', fontsize=scatter_label_fontsize, weight='bold')
        ax1.set_title(f'PCA Latent Space: Seebeck Coefficient ({selected_element})', fontsize=scatter_label_fontsize + 2, weight='bold')
        cbar1 = plt.colorbar(scatter1, ax=ax1, label='Seebeck Coefficient (μV/K)', pad=0.02)
        cbar1.ax.tick_params(labelsize=scatter_label_fontsize - 2, width=scatter_axis_linewidth, length=6)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(scatter_axis_linewidth)
        ax1.spines['bottom'].set_linewidth(scatter_axis_linewidth)
        ax1.tick_params(axis='both', which='major', labelsize=scatter_label_fontsize - 2, width=scatter_axis_linewidth, length=6)
        
        # FIXED: Use single color per element group
        unique_elements = np.unique(dominant_elements_filtered_filtered)
        for element in unique_elements:
            idx = dominant_elements_filtered_filtered == element
            if np.sum(idx) > 0:  # Only plot if there are points
                color = default_element_color_map.get(element, '#FFFFFF')
                ax2.scatter(z_2d_pca_filtered[idx, 0], z_2d_pca_filtered[idx, 1], 
                            c=color, label=element, s=marker_size, alpha=marker_alpha)
        
        ax2.set_xlabel('PC1', fontsize=scatter_label_fontsize, weight='bold')
        ax2.set_ylabel('PC2', fontsize=scatter_label_fontsize, weight='bold')
        ax2.set_title(f'PCA Latent Space: Dominant Element ({selected_element})', fontsize=scatter_label_fontsize + 2, weight='bold')
        ax2.legend(title='Dominant Element', fontsize=scatter_label_fontsize - 2, loc='upper right', bbox_to_anchor=(1.3, 1), frameon=True, edgecolor='black')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(scatter_axis_linewidth)
        ax2.spines['bottom'].set_linewidth(scatter_axis_linewidth)
        ax2.tick_params(axis='both', which='major', labelsize=scatter_label_fontsize - 2, width=scatter_axis_linewidth, length=6)
        plt.tight_layout()
        st.pyplot(fig)
        try:
            plt.savefig(os.path.join(script_dir, 'latent_2d_matplotlib.pdf'), dpi=300, bbox_inches='tight', format='pdf')
        except Exception as e:
            st.warning(f"Error saving Matplotlib figure: {e}")
        plt.close(fig)

        st.write("#### 8D Latent Space Radar Plot")
        sample_indices = np.random.choice(len(z_normalized), size=max_samples, replace=False)
        radar_data = z_normalized[sample_indices]
        radar_labels = [f"Sample {i+1} (Seebeck: {output_feature_cleaned[i]:.1f})" for i in sample_indices]
        fig_radar = plot_radar(radar_data, radar_labels, "8D Latent Space Radar Plot", max_samples=max_samples,
                               alpha=radar_alpha, linewidth=radar_linewidth, fontsize=radar_fontsize,
                               legend_pos=radar_legend_pos, axis_linewidth=radar_axis_linewidth)
        st.pyplot(fig_radar)
        try:
            plt.savefig(os.path.join(script_dir, 'latent_radar.pdf'), dpi=300, bbox_inches='tight', format='pdf')
        except Exception as e:
            st.warning(f"Error saving radar plot: {e}")
        plt.close(fig_radar)

        st.write("#### 8D Latent Space Parallel Coordinates")
        parallel_df = pd.DataFrame(z_normalized, columns=[f'Latent Dim {i+1}' for i in range(8)])
        parallel_df['Seebeck'] = output_feature_cleaned
        fig_parallel = px.parallel_coordinates(
            parallel_df, color='Seebeck', color_continuous_scale=parallel_color_scale,
            labels={f'Latent Dim {i+1}': f'Latent Dim {i+1}' for i in range(8)},
            title='8D Latent Space Parallel Coordinates'
        )
        fig_parallel.update_layout(
            title=dict(text='8D Latent Space Parallel Coordinates', x=0.5, xanchor='center', font=dict(size=parallel_label_fontsize + 4, family='Arial')),
            font=dict(family='Arial', size=parallel_label_fontsize),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        st.plotly_chart(fig_parallel, use_container_width=True)
        fig_parallel.write_html(os.path.join(script_dir, 'latent_parallel_plotly.html'))

    # Training History Visualizations
    st.subheader("Training History Visualizations")
    if not vae_history_df.empty:
        st.write("#### VAE Training History")
        fig_vae = plot_training_history_matplotlib(vae_history_df, "VAE", os.path.join(script_dir, 'vae_history_matplotlib.pdf'),
                                                  linewidth=history_linewidth, fontsize=history_label_fontsize,
                                                  train_color=train_color, val_color=val_color,
                                                  tick_fontsize=history_tick_fontsize, axis_linewidth=history_axis_linewidth)
        st.pyplot(fig_vae)
        try:
            plt.savefig(os.path.join(script_dir, 'vae_history_matplotlib.pdf'), dpi=300, bbox_inches='tight', format='pdf')
        except Exception as e:
            st.warning(f"Error saving VAE history plot: {e}")
        plt.close(fig_vae)
        fig_vae_plotly = plot_training_history_plotly(vae_history_df, "VAE", train_color=train_color, val_color=val_color,
                                                     linewidth=history_linewidth, label_fontsize=history_label_fontsize,
                                                     tick_fontsize=history_tick_fontsize, axis_linewidth=history_axis_linewidth)
        st.plotly_chart(fig_vae_plotly, use_container_width=True)
        fig_vae_plotly.write_html(os.path.join(script_dir, 'vae_history_plotly.html'))

    if not regressor_history_df.empty:
        st.write("#### Regressor Training History")
        fig_regressor = plot_training_history_matplotlib(regressor_history_df, "Regressor", os.path.join(script_dir, 'regressor_history_matplotlib.pdf'),
                                                        linewidth=history_linewidth, fontsize=history_label_fontsize,
                                                        train_color=train_color, val_color=val_color,
                                                        tick_fontsize=history_tick_fontsize, axis_linewidth=history_axis_linewidth)
        st.pyplot(fig_regressor)
        try:
            plt.savefig(os.path.join(script_dir, 'regressor_history_matplotlib.pdf'), dpi=300, bbox_inches='tight', format='pdf')
        except Exception as e:
            st.warning(f"Error saving regressor history plot: {e}")
        plt.close(fig_regressor)
        fig_regressor_plotly = plot_training_history_plotly(regressor_history_df, "Regressor", train_color=train_color, val_color=val_color,
                                                           linewidth=history_linewidth, label_fontsize=history_label_fontsize,
                                                           tick_fontsize=history_tick_fontsize, axis_linewidth=history_axis_linewidth)
        st.plotly_chart(fig_regressor_plotly, use_container_width=True)
        fig_regressor_plotly.write_html(os.path.join(script_dir, 'regressor_history_plotly.html'))

    if not df.empty or not vae_history_df.empty or not regressor_history_df.empty:
        st.success("Visualizations generated successfully!")
    else:
        st.warning("No data available in the database.")

with tab2:
    st.header("Seebeck Coefficient Prediction")
    formula_input = st.text_input("Enter the chemical formula (e.g., Mg2Si):")
    temperature_input = st.number_input("Enter the temperature (K):", min_value=0, max_value=5000, value=300)
    
    if st.button("Predict Seebeck Coefficient"):
        if formula_input:
            try:
                new_data = pd.DataFrame({'Formula': [formula_input], 'temperature(K)': [temperature_input]})
                X_scaled = preprocess_new_data(new_data, available_elements, scaler)
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                vae.eval()
                regressor.eval()
                with torch.no_grad():
                    _, z_mean, _ = vae(X_tensor)
                    y_scaled_pred = regressor(z_mean)
                    y_pred = y_scaler.inverse_transform(y_scaled_pred.cpu().numpy().reshape(-1, 1)).ravel()
                st.write(f"Predicted Seebeck Coefficient (Scaled): {y_scaled_pred[0].item():.2f}")
                st.write(f"Predicted Seebeck Coefficient (Absolute): {y_pred[0]:.2f} μV/K")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Please enter a valid chemical formula.")

# Close database connection
conn.close()
