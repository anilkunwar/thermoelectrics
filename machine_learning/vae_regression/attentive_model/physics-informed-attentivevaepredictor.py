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
import os
import joblib
import colorsys
from itertools import combinations
import logging
import sqlite3
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Attention-Based Classifier for Material Type
class AttentionClassifier(nn.Module):
    def __init__(self, input_dim=8, num_heads=1):
        super(AttentionClassifier, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = input_dim // num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 3), nn.Softmax(dim=-1)  # Outputs probabilities for p-type, n-type, neutral
        )

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.attention_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.attention_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.attention_dim).transpose(1, 2)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.attention_dim)
        attention_probs = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_probs, V).transpose(1, 2).contiguous().view(batch_size, -1)
        output = self.out(attention_output) + x  # Residual connection
        probs = self.classifier(output)
        return probs

# Preprocessing functions
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

def featurize_materials(df, available_elements):
    features = []
    for _, row in df.iterrows():
        try:
            composition = Composition(row['Formula'])
            composition_dict = composition.fractional_composition.as_dict()
            feature_vector = {element: composition_dict.get(element, 0) for element in available_elements}
            feature_vector['temperature(K)'] = row['temperature(K)']
            features.append(feature_vector)
        except Exception as e:
            logger.warning(f"Failed to parse formula {row['Formula']}: {e}")
            continue
    return pd.DataFrame(features)

# Compute z_mean statistics and bias vector
def compute_z_mean_stats_and_bias(elements, temperature, available_elements, scaler, vae, steps=30):
    z_means = []
    try:
        if len(elements) != 3:
            raise ValueError("Exactly 3 elements required")
        if not all(e in available_elements for e in elements):
            raise ValueError("All elements must be in available_elements")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("Temperature must be a non-negative number")
        vae.eval()
        with torch.no_grad():
            for a in np.linspace(0, 1, steps):
                for b in np.linspace(0, 1 - a, steps):
                    c = 1 - a - b
                    if c >= 0:
                        comp_dict = {elements[0]: a, elements[1]: b, elements[2]: c}
                        df = featurize_composition(comp_dict, available_elements, temperature)
                        X_scaled = preprocess_new_data(df, available_elements, scaler)
                        X_tensor = torch.FloatTensor(X_scaled).to(device)
                        _, z_mean, _ = vae(X_tensor)
                        z_means.append(z_mean.cpu().numpy())
        if not z_means:
            raise ValueError("No valid compositions generated")
        z_means = np.vstack(z_means)
        z_mean_avg = np.mean(z_means, axis=0)
        z_mean_std = np.std(z_means, axis=0)
        p_type_comp = {elements[0]: 0.0, elements[1]: 0.4, elements[2]: 0.6}
        n_type_comp = {elements[0]: 0.33, elements[1]: 0.33, elements[2]: 0.34}
        df_p = featurize_composition(p_type_comp, available_elements, temperature)
        df_n = featurize_composition(n_type_comp, available_elements, temperature)
        X_scaled_p = preprocess_new_data(df_p, available_elements, scaler)
        X_scaled_n = preprocess_new_data(df_n, available_elements, scaler)
        X_tensor_p = torch.FloatTensor(X_scaled_p).to(device)
        X_tensor_n = torch.FloatTensor(X_scaled_n).to(device)
        _, z_mean_p, _ = vae(X_tensor_p)
        _, z_mean_n, _ = vae(X_tensor_n)
        bias_vector = (z_mean_p - z_mean_n).cpu().numpy()
        bias_norm = np.linalg.norm(bias_vector)
        if bias_norm > 0:
            bias_vector = bias_vector / bias_norm
        else:
            bias_vector = np.ones(vae.latent_dim) / np.sqrt(vae.latent_dim)
        bias_magnitude = 0.5 * np.mean(z_mean_std)
        return z_mean_avg, z_mean_std, bias_vector, bias_magnitude
    except Exception as e:
        logger.warning(f"Using fallback statistics due to error: {e}")
        fallback_z_mean_avg = np.array([-0.0003, -0.0000, 0.0004, 0.0003, 0.0003, -0.0006, 0.0009, -0.0001])
        fallback_z_mean_std = np.array([0.0003, 0.0007, 0.0003, 0.0005, 0.0005, 0.0010, 0.0011, 0.0003])
        fallback_bias_vector = np.ones(8) / np.sqrt(8)
        fallback_bias_magnitude = 0.5 * np.mean(fallback_z_mean_std)
        return fallback_z_mean_avg, fallback_z_mean_std, fallback_bias_vector, fallback_bias_magnitude

# Predict material type probabilities
@st.cache_resource
def predict_material_type_probs(_vae, _classifier, _scaler, elements, temperature, available_elements, steps=30):
    p_type_counts = 0
    n_type_counts = 0
    probs_list = []
    try:
        _vae.eval()
        _classifier.eval()
        with torch.no_grad():
            for a in np.linspace(0, 1, steps):
                for b in np.linspace(0, 1 - a, steps):
                    c = 1 - a - b
                    if c >= 0:
                        comp_dict = {elements[0]: a, elements[1]: b, elements[2]: c}
                        df = featurize_composition(comp_dict, available_elements, temperature)
                        X_scaled = preprocess_new_data(df, available_elements, _scaler)
                        X_tensor = torch.FloatTensor(X_scaled).to(device)
                        _, z_mean, _ = _vae(X_tensor)
                        probs = _classifier(z_mean)
                        probs = probs.cpu().numpy()[0]  # [p-type, n-type, neutral]
                        probs_list.append(probs)
                        seebeck, _ = predict_seebeck(comp_dict, temperature, available_elements, _scaler, _vae, regressor, y_scaler)
                        if seebeck is not None:
                            if seebeck > 10:
                                p_type_counts += 1
                            elif seebeck < -10:
                                n_type_counts += 1
        probs_array = np.array(probs_list)
        avg_probs = np.mean(probs_array, axis=0) if probs_array.size > 0 else np.array([0.33, 0.33, 0.34])
        return p_type_counts, n_type_counts, avg_probs
    except Exception as e:
        logger.warning(f"Material type prediction failed: {e}")
        return 0, 0, np.array([0.33, 0.33, 0.34])

# Predict Seebeck coefficient
def predict_seebeck(composition_dict, temperature, available_elements, scaler, vae, regressor, y_scaler, sign_bias=None, bias_vector=None, bias_magnitude=0.0003):
    try:
        df = featurize_composition(composition_dict, available_elements, temperature)
        X_scaled = preprocess_new_data(df, available_elements, scaler)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        vae.eval()
        regressor.eval()
        with torch.no_grad():
            _, z_mean, _ = vae(X_tensor)
            z_mean_original = z_mean.clone()
            y_scaled_pred_unbiased = regressor(z_mean_original)
            y_pred_unbiased = y_scaler.inverse_transform(y_scaled_pred_unbiased.cpu().numpy().reshape(-1, 1)).ravel()
            y_pred_unbiased = np.clip(y_pred_unbiased, -300, 300)
            if sign_bias is not None and bias_vector is not None:
                bias_vector = torch.FloatTensor(bias_vector).to(device) * bias_magnitude
                if sign_bias == 'p-type':
                    z_mean = z_mean + bias_vector
                elif sign_bias == 'n-type':
                    z_mean = z_mean - bias_vector
                y_scaled_pred = regressor(z_mean)
                y_pred = y_scaler.inverse_transform(y_scaled_pred.cpu().numpy().reshape(-1, 1)).ravel()
                y_pred = np.clip(y_pred, -300, 300)
                if sign_bias == 'n-type' and y_pred[0] > 0:
                    y_pred = -y_pred
                if abs(y_pred[0]) > 0:
                    y_pred = y_pred * (abs(y_pred_unbiased[0]) / abs(y_pred[0]))
            else:
                y_pred = y_pred_unbiased
            return y_pred[0], y_pred_unbiased[0], z_mean.cpu().numpy()
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None, None, None

# Plot periodic table
def plot_periodic_table(available_elements, selected_elements, element_color_map, fontsize=14):
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
            color = element_color_map.get(element, '#D3D3D3') if element in available_elements else '#D3D3D3'
            opacity = 1.0 if element in selected_elements else (0.7 if element in available_elements else 0.3)
            line_width = 4 if element in selected_elements else 2
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
        title=dict(text="Periodic Table: Full (Unavailable in Gray, Selected with Bold Outline)", x=0.5, xanchor='center', font=dict(size=fontsize + 4, family='Arial')),
        xaxis=dict(range=[0, 19], showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(range=[-8, 0], showgrid=False, zeroline=False, showticklabels=False, title=''),
        plot_bgcolor='white', paper_bgcolor='white',
        width=900, height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Plot material type histogram
def plot_material_type_histogram(p_type_counts, n_type_counts, font_size):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['p-type', 'n-type'],
        y=[p_type_counts, n_type_counts],
        marker=dict(color=['#1f77b4', '#ff7f0e']),
        text=[p_type_counts, n_type_counts],
        textposition='auto'
    ))
    fig.update_layout(
        title=dict(text='Material Type Distribution', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Material Type',
        yaxis_title='Count',
        xaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# Plot probability distribution
def plot_probability_distribution(probs, font_size):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['p-type', 'n-type', 'neutral'],
        y=probs,
        marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c']),
        text=[f'{p:.2%}' for p in probs],
        textposition='auto'
    ))
    fig.update_layout(
        title=dict(text='Attention-Based Material Type Probabilities', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Material Type',
        yaxis_title='Probability',
        xaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size)), range=[0, 1]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# Plot element correlation network
def plot_element_correlation_network(elements, available_elements, scaler, vae, font_size):
    G = nx.Graph()
    for element in elements:
        G.add_node(element)
    correlation_matrix = np.zeros((len(elements), len(elements)))
    try:
        compositions = []
        for a in np.linspace(0, 1, 10):
            for b in np.linspace(0, 1 - a, 10):
                c = 1 - a - b
                if c >= 0:
                    comp_dict = {elements[0]: a, elements[1]: b, elements[2]: c}
                    compositions.append(comp_dict)
        feature_vectors = []
        for comp_dict in compositions:
            df = featurize_composition(comp_dict, available_elements, 300)  # Fixed temperature for consistency
            X_scaled = preprocess_new_data(df, available_elements, scaler)
            feature_vectors.append(X_scaled[0, :-1])  # Exclude temperature
        feature_matrix = np.array(feature_vectors)
        for i, j in combinations(range(len(elements)), 2):
            corr = np.corrcoef(feature_matrix[:, elements.index(elements[i])], feature_matrix[:, elements.index(elements[j])])[0, 1]
            correlation_matrix[i, j] = correlation_matrix[j, i] = corr
            if abs(corr) > 0.5:  # Threshold for visualization
                G.add_edge(elements[i], elements[j], weight=abs(corr))
        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            textposition='top center',
            textfont=dict(size=font_size, family='Arial'),
            marker=dict(size=20, color=[default_element_color_map[node] for node in G.nodes()], line=dict(width=2, color='black')),
            hoverinfo='text',
            hovertext=[f"Element: {node}<br>Degree: {G.degree[node]}" for node in G.nodes()]
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=dict(text='Element Correlation Network', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        return fig, correlation_matrix
    except Exception as e:
        logger.warning(f"Failed to compute correlation network: {e}")
        return None, correlation_matrix

# Load models and scalers
script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    vae = VAE().to(device)
    regressor = Regressor().to(device)
    classifier = AttentionClassifier().to(device)  # New attention-based classifier
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

# Note: Classifier is initialized with random weights
st.warning("The attention-based classifier is initialized with random weights. For accurate material type predictions, train the classifier with labeled data.")

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

# Enhanced color map
base_color_list = (
    px.colors.qualitative.Plotly + px.colors.qualitative.Pastel1 + px.colors.qualitative.D3 +
    px.colors.qualitative.G10 + px.colors.qualitative.T10 + px.colors.qualitative.Set1 +
    px.colors.qualitative.Set2 + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel2 +
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
st.title("Thermoelectric Material Type and Seebeck Coefficient Predictor")
st.markdown("""
This application predicts the Seebeck coefficient and material type (p-type, n-type, neutral) for a ternary composition using a Variational Autoencoder (VAE) and an attention-based classifier. Select up to three elements, input their proportions, and view statistical probabilities for material types based on an attention model. Confirm or adjust the material type before predicting the Seebeck coefficient.

**Date and Time**: 06:47 AM CEST, Tuesday, August 19, 2025
""")

# Sidebar for periodic table customization
st.sidebar.header("Periodic Table Customization")
font_size = st.sidebar.slider("Font Size (Periodic Table)", 8, 20, 14)

# Periodic Table Visualization
st.header("Periodic Table Reference")
st.write("The periodic table shows all elements, with unavailable elements in gray and selected elements with bold outlines.")
fig_periodic = plot_periodic_table(available_elements, st.session_state.get('selected_elements', []), default_element_color_map, fontsize=font_size)
st.plotly_chart(fig_periodic, use_container_width=True)
try:
    fig_periodic.write_html(os.path.join(script_dir, 'periodic_table.html'))
except Exception as e:
    st.warning(f"Failed to save periodic table: {e}")

# Database connection
db_path = os.path.join(script_dir, 'thermoelectric_data.db')
try:
    conn = sqlite3.connect(db_path)
except sqlite3.Error as e:
    st.error(f"Error connecting to database: {e}")
    st.stop()

# Load data for histogram
try:
    df = pd.read_sql("SELECT * FROM thermoelectric_materials;", conn)
except Exception as e:
    st.error(f"Error loading data from database: {e}")
    conn.close()
    st.stop()

# Element selection
st.header("Select Elements")
st.session_state.selected_elements = st.multiselect(
    "Select up to three elements",
    options=available_elements,
    default=st.session_state.get('selected_elements', []),
    max_selections=3,
    key='element_selector'
)

# Update proportions and compositions
st.session_state.proportions = st.session_state.get('proportions', {})
st.session_state.compositions = st.session_state.get('compositions', {})
for element in st.session_state.selected_elements:
    if element not in st.session_state.proportions:
        st.session_state.proportions[element] = 0.0
    if element not in st.session_state.compositions:
        st.session_state.compositions[element] = 0.0
st.session_state.proportions = {k: v for k, v in st.session_state.proportions.items() if k in st.session_state.selected_elements}
st.session_state.compositions = {k: v for k, v in st.session_state.compositions.items() if k in st.session_state.selected_elements}

# Proportion and composition input
st.header("Input Proportions and View Normalized Compositions")
if st.session_state.selected_elements:
    st.write(f"Selected Elements: {', '.join(st.session_state.selected_elements)}")
    st.subheader("Proportions")
    cols = st.columns(len(st.session_state.selected_elements))
    for idx, element in enumerate(st.session_state.selected_elements):
        with cols[idx]:
            st.session_state.proportions[element] = st.number_input(
                f"Proportion for {element}", min_value=0.0, value=st.session_state.proportions.get(element, 0.0), step=0.1, key=f"prop_{element}"
            )
    
    if st.button("Normalize Proportions"):
        total = sum(st.session_state.proportions.values())
        if total > 0:
            for element in st.session_state.proportions:
                st.session_state.compositions[element] = st.session_state.proportions[element] / total
            st.rerun()
        else:
            st.error("Please provide non-zero proportions for at least one element.")
    
    st.subheader("Normalized Compositions")
    cols = st.columns(len(st.session_state.selected_elements))
    for idx, element in enumerate(st.session_state.selected_elements):
        with cols[idx]:
            st.number_input(
                f"Composition for {element}", min_value=0.0, max_value=1.0,
                value=st.session_state.compositions.get(element, 0.0), step=0.1, key=f"comp_{element}", disabled=True
            )
else:
    st.write("Please select up to three elements from the dropdown.")

# Temperature input
st.session_state.temperature = st.number_input("Enter Temperature (K):", min_value=0, max_value=5000, value=st.session_state.get('temperature', 300), step=10)

# Material type and Seebeck prediction
st.header("Material Type and Seebeck Coefficient Prediction")
if st.button("Analyze Material Type"):
    if len(st.session_state.selected_elements) > 0:
        elements, proportions, compositions = st.session_state.selected_elements.copy(), st.session_state.proportions.copy(), st.session_state.compositions.copy()
        total = sum(proportions.values())
        if total == 0:
            st.error("Please provide non-zero proportions for at least one element.")
        else:
            # Complete to three elements
            while len(elements) < 3:
                remaining_elements = [e for e in available_elements if e in ['Ag', 'Bi', 'Te'] and e not in elements]
                if not remaining_elements:
                    remaining_elements = [e for e in available_elements if e not in elements]
                if remaining_elements:
                    random_element = np.random.choice(remaining_elements)
                    elements.append(random_element)
                    proportions[random_element] = 0.0
                    compositions[random_element] = 0.0
                else:
                    st.error("Not enough available elements to complete the ternary composition.")
                    st.stop()
            
            # Compute z_mean statistics and bias vector
            z_mean_avg, z_mean_std, bias_vector, bias_magnitude = compute_z_mean_stats_and_bias(elements, st.session_state.temperature, available_elements, scaler, vae)
            
            # Predict material type probabilities
            p_type_counts, n_type_counts, avg_probs = predict_material_type_probs(vae, classifier, scaler, elements, st.session_state.temperature, available_elements)
            
            # Plot material type histogram
            st.subheader("Material Type Distribution")
            fig_histogram = plot_material_type_histogram(p_type_counts, n_type_counts, font_size)
            st.plotly_chart(fig_histogram, use_container_width=True)
            try:
                fig_histogram.write_html(os.path.join(script_dir, 'material_type_histogram.html'))
            except Exception as e:
                st.warning(f"Failed to save histogram: {e}")
            
            # Plot probability distribution
            st.subheader("Attention-Based Material Type Probabilities")
            st.write(f"p-type: {avg_probs[0]:.2%}, n-type: {avg_probs[1]:.2%}, neutral: {avg_probs[2]:.2%}")
            fig_probs = plot_probability_distribution(avg_probs, font_size)
            st.plotly_chart(fig_probs, use_container_width=True)
            try:
                fig_probs.write_html(os.path.join(script_dir, 'material_type_probs.html'))
            except Exception as e:
                st.warning(f"Failed to save probability plot: {e}")
            
            # Plot element correlation network
            st.subheader("Element Correlation Network")
            fig_network, correlation_matrix = plot_element_correlation_network(elements, available_elements, scaler, vae, font_size)
            if fig_network:
                st.plotly_chart(fig_network, use_container_width=True)
                try:
                    fig_network.write_html(os.path.join(script_dir, 'element_correlation_network.html'))
                except Exception as e:
                    st.warning(f"Failed to save network plot: {e}")
            st.write("Correlation Matrix:")
            st.write(pd.DataFrame(correlation_matrix, index=elements, columns=elements).round(2))
            
            # User composition
            user_composition = [compositions.get(elements[i], 0) for i in range(3)]
            user_composition_dict = {elements[i]: user_composition[i] for i in range(3)}
            
            # Predict material type for user composition
            df = featurize_composition(user_composition_dict, available_elements, st.session_state.temperature)
            X_scaled = preprocess_new_data(df, available_elements, scaler)
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            vae.eval()
            classifier.eval()
            with torch.no_grad():
                _, z_mean, _ = vae(X_tensor)
                user_probs = classifier(z_mean).cpu().numpy()[0]
            st.subheader("User Composition Material Type Probabilities")
            st.write(f"p-type: {user_probs[0]:.2%}, n-type: {user_probs[1]:.2%}, neutral: {user_probs[2]:.2%}")
            suggested_type = ['p-type', 'n-type', 'neutral'][np.argmax(user_probs)]
            st.write(f"Suggested Material Type: {suggested_type}")
            
            # Material type selection
            st.session_state.sign_bias = st.selectbox(
                "Confirm or Select Material Type",
                options=['p-type', 'n-type', 'neutral'],
                index=['p-type', 'n-type', 'neutral'].index(suggested_type),
                key='sign_bias_selector'
            )
            
            # Predict Seebeck coefficient
            if st.button("Predict Seebeck Coefficient"):
                sign_bias = st.session_state.sign_bias if st.session_state.sign_bias != 'neutral' else None
                seebeck, seebeck_unbiased, z_mean = predict_seebeck(
                    user_composition_dict, st.session_state.temperature, available_elements, scaler, vae, regressor, y_scaler,
                    sign_bias=sign_bias, bias_vector=bias_vector, bias_magnitude=bias_magnitude
                )
                if seebeck is None:
                    st.error("Failed to predict Seebeck coefficient. Please check inputs or model files.")
                else:
                    st.subheader("Seebeck Coefficient Prediction")
                    st.write(f"**User Composition**: {elements[0]}: {user_composition[0]:.2f}, {elements[1]}: {user_composition[1]:.2f}, {elements[2]}: {user_composition[2]:.2f}")
                    st.write(f"**Temperature**: {st.session_state.temperature} K")
                    st.write(f"**Signed Seebeck Coefficient (Biased)**: {seebeck:.2f} μV/K ({'p-type' if seebeck > 0 else 'n-type' if seebeck < 0 else 'neutral'})")
                    st.write(f"**Signed Seebeck Coefficient (Unbiased)**: {seebeck_unbiased:.2f} μV/K ({'p-type' if seebeck_unbiased > 0 else 'n-type' if seebeck_unbiased < 0 else 'neutral'})")
    else:
        st.error("Please select at least one element.")

# Close database connection
conn.close()
