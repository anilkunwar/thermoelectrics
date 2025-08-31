import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition, Element
import os
import joblib
import colorsys
from itertools import combinations
import logging
import sqlite3
from io import BytesIO
from collections import Counter
from physics_attention import extract_material_type, plot_material_type_histogram, plot_material_probabilities, plot_relevance_box_plot, plot_pmi_network

# Set page config
st.set_page_config(page_title="Ternary Seebeck Coefficient Predictor", layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Electronegativity and thermoelectric weights
electronegativity = {
    'H': 2.20, 'He': 0.0, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': 0.0,
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': 0.0, 'K': 0.82, 'Ca': 1.00,
    'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 0.0, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
    'Nb': 1.6, 'Mo': 2.16, 'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96,
    'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 0.0, 'Cs': 0.79, 'Ba': 0.89, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
    'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.2, 'Gd': 1.2, 'Tb': 1.1, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.1,
    'Lu': 1.27, 'Hf': 1.3, 'Ta': 1.5, 'W': 2.36, 'Re': 1.9, 'Os': 2.2, 'Ir': 2.2, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00,
    'Tl': 2.04, 'Pb': 2.33, 'Bi': 2.02, 'Po': 2.0, 'At': 2.2, 'Rn': 0.0, 'Fr': 0.7, 'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3,
    'Pa': 1.5, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.3, 'Cm': 1.3, 'Bk': 1.3, 'Cf': 1.3, 'Es': 1.3, 'Fm': 1.3,
    'Md': 1.3, 'No': 1.3, 'Lr': 1.3, 'Rf': 1.3, 'Db': 1.2, 'Sg': 1.1, 'Bh': 1.1, 'Hs': 1.1, 'Mt': 1.1, 'Ds': 1.1,
    'Rg': 1.1, 'Cn': 1.1, 'Nh': 1.1, 'Fl': 1.1, 'Mc': 1.1, 'Lv': 1.1, 'Ts': 1.1, 'Og': 1.1
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

# ANN Classifier for Material Type
class MaterialClassifier(nn.Module):
    def __init__(self, input_dim=5):
        super(MaterialClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 2), nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)

# Load database
def load_thermoelectric_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT formula, seebeck_coefficient, material_type 
        FROM thermoelectric_materials 
        WHERE material_type IN ('p-type', 'n-type')
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Failed to load thermoelectric_universe.db: {e}")
        return pd.DataFrame()

# Load material classification CSV
def load_material_classification_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logger.error(f"Failed to load material classification CSV: {e}")
        return pd.DataFrame()

# Feature extraction for MaterialClassifier
def featurize_formula_for_classifier(composition_dict, temperature=None):
    """Generate 5-dimensional feature vector for MaterialClassifier."""
    element_properties = {
        el.symbol: [
            float(el.Z or 0),  # Atomic number
            float(el.X or 0),  # Electronegativity
            float(el.group or 0),  # Group
            float(el.row or 0),  # Row
            float(el.atomic_mass or 0)  # Atomic mass
        ] for el in Element
    }
    
    try:
        total = sum(composition_dict.values())
        if total == 0:
            raise ValueError("Composition dictionary has zero total proportion")
        
        feature_vector = np.zeros(5)
        for el, amt in composition_dict.items():
            weight = amt / total
            props = element_properties.get(el, [0.0] * 5)
            feature_vector += np.array(props) * weight
        
        if np.any(np.isnan(feature_vector)):
            raise ValueError("NaN features in feature vector")
        
        return feature_vector
    except Exception as e:
        logger.error(f"Failed to featurize composition: {str(e)}")
        return None

# Preprocessing for VAE and Regressor
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

# Compute z_mean statistics and bias vector
def compute_z_mean_stats_and_bias(elements, temperature, available_elements, _scaler, _vae, steps=30):
    z_means = []
    try:
        if len(elements) != 3:
            raise ValueError("Exactly 3 elements required")
        if not all(e in available_elements for e in elements):
            raise ValueError("All elements must be in available_elements")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("Temperature must be a non-negative number")
        expected_features = len(available_elements) + 1
        if _scaler.n_features_in_ != expected_features:
            raise ValueError("Scaler feature mismatch")
        if _vae.input_dim != expected_features:
            raise ValueError("VAE input dimension mismatch")
        
        _vae.eval()
        with torch.no_grad():
            for a in np.linspace(0, 1, steps):
                for b in np.linspace(0, 1 - a, steps):
                    c = 1 - a - b
                    if c >= 0:
                        comp_dict = {elements[0]: a, elements[1]: b, elements[2]: c}
                        df = featurize_composition(comp_dict, available_elements, temperature)
                        X_scaled = preprocess_new_data(df, available_elements, _scaler)
                        if X_scaled.shape[1] != _vae.input_dim:
                            raise ValueError("Input shape mismatch")
                        X_tensor = torch.FloatTensor(X_scaled).to(device)
                        _, z_mean, _ = _vae(X_tensor)
                        if z_mean.shape[1] != _vae.latent_dim:
                            raise ValueError("z_mean shape mismatch")
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
        X_scaled_p = preprocess_new_data(df_p, available_elements, _scaler)
        X_scaled_n = preprocess_new_data(df_n, available_elements, _scaler)
        X_tensor_p = torch.FloatTensor(X_scaled_p).to(device)
        X_tensor_n = torch.FloatTensor(X_scaled_n).to(device)
        _, z_mean_p, _ = _vae(X_tensor_p)
        _, z_mean_n, _ = _vae(X_tensor_n)
        bias_vector = (z_mean_p - z_mean_n).cpu().numpy()
        bias_norm = np.linalg.norm(bias_vector)
        if bias_norm > 0:
            bias_vector = bias_vector / bias_norm
        else:
            bias_vector = np.ones(_vae.latent_dim) / np.sqrt(_vae.latent_dim)
        bias_magnitude = 0.5 * np.mean(z_mean_std)
        return z_mean_avg, z_mean_std, bias_vector, bias_magnitude
    except Exception as e:
        logger.error(f"Failed to compute z_mean statistics and bias: {e}")
        fallback_z_mean_avg = np.array([-0.0003, -0.0000, 0.0004, 0.0003, 0.0003, -0.0006, 0.0009, -0.0001])
        fallback_z_mean_std = np.array([0.0003, 0.0007, 0.0003, 0.0005, 0.0005, 0.0010, 0.0011, 0.0003])
        fallback_bias_vector = np.ones(8) / np.sqrt(8)
        fallback_bias_magnitude = 0.5 * np.mean(fallback_z_mean_std)
        return fallback_z_mean_avg, fallback_z_mean_std, fallback_bias_vector, fallback_bias_magnitude

# Plot z_mean bar chart (Plotly)
def plot_z_mean_bar_chart(z_mean_avg, z_mean_std, font_size):
    dimensions = [f"Dim {i+1}" for i in range(len(z_mean_avg))]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dimensions,
        y=z_mean_avg,
        error_y=dict(type='data', array=z_mean_std, visible=True),
        marker=dict(color='blue', opacity=0.7),
        name='z_mean'
    ))
    fig.update_layout(
        title=dict(text='Latent Space z_mean: Mean with SD Error Bars', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Latent Dimension',
        yaxis_title='z_mean Value',
        xaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        template='seaborn'
    )
    return fig

# Plot z_mean bar chart (Matplotlib)
def plot_z_mean_bar_chart_matplotlib(z_mean_avg, z_mean_std, output_path='z_mean_bar_chart.pdf'):
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    dimensions = [f'Dim {i+1}' for i in range(len(z_mean_avg))]
    x = np.arange(len(z_mean_avg))
    ax.bar(x, z_mean_avg, yerr=z_mean_std, capsize=5, color='#1f77b4', edgecolor='black', alpha=0.7, label='z_mean')
    ax.set_xlabel('Latent Dimension', fontsize=14, fontfamily='Arial')
    ax.set_ylabel('z_mean Value', fontsize=14, fontfamily='Arial')
    ax.set_title('Latent Space z_mean: Mean with SD Error Bars', fontsize=16, fontfamily='Arial', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=12, fontfamily='Arial')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    try:
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        logger.info(f"Saved Matplotlib figure to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Matplotlib figure: {e}")
    return fig

# Plot formula histogram
def plot_formula_histogram(verbatim_matches, font_size):
    formulas = [match['matched_formula'] for match in verbatim_matches if 'matched_formula' in match]
    formula_counts = Counter(formulas)
    if not formula_counts:
        return None
    df = pd.DataFrame.from_dict(formula_counts, orient='index').reset_index()
    df.columns = ['Formula', 'Count']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Formula'],
        y=df['Count'],
        marker=dict(color='teal', opacity=0.7),
        name='Formula Frequency'
    ))
    fig.update_layout(
        title=dict(text='Histogram of Matched Chemical Formulas', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Chemical Formula',
        yaxis_title='Frequency',
        xaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size)), tickangle=45),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        template='seaborn'
    )
    return fig

# Plot literature Seebeck values
def plot_literature_seebeck(summary_dict, font_size):
    if 'seebeck_values' not in summary_dict or not summary_dict['seebeck_values']:
        return None
    seebeck_values = summary_dict['seebeck_values']
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=seebeck_values,
        nbinsx=30,
        marker=dict(color='#1f77b4', opacity=0.7),
        name='Literature Seebeck'
    ))
    fig.update_layout(
        title=dict(text='Histogram of Literature Seebeck Coefficients (μV/K)', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Seebeck Coefficient (μV/K)',
        yaxis_title='Frequency',
        xaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        yaxis=dict(tickfont=dict(size=font_size), title=dict(font=dict(size=font_size))),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        template='seaborn'
    )
    return fig

# Predict material type using ANN
def predict_material_type(composition_dict, temperature, available_elements, _scaler_classifier, _classifier):
    try:
        features = featurize_formula_for_classifier(composition_dict)
        if features is None:
            raise ValueError("Failed to generate features for classifier")
        X_scaled = _scaler_classifier.transform([features])
        if X_scaled.shape[1] != _classifier.model[0].in_features:
            raise ValueError("Input shape mismatch for ANN classifier")
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        _classifier.eval()
        with torch.no_grad():
            probs = _classifier(X_tensor)
            p_type_prob, n_type_prob = probs[0].cpu().numpy()
            material_type = 'p-type' if p_type_prob > n_type_prob else 'n-type'
            return material_type, {'p_type_prob': p_type_prob, 'n_type_prob': n_type_prob}
    except Exception as e:
        logger.error(f"ANN material type prediction failed: {e}")
        return 'Neutral', {'p_type_prob': 0.5, 'n_type_prob': 0.5}

# Predict Seebeck coefficient
def predict_seebeck(composition_dict, temperature, available_elements, _scaler, _vae, _regressor, _y_scaler, sign_bias=None, bias_vector=None, bias_magnitude=0.0003, summary_dict=None):
    try:
        df = featurize_composition(composition_dict, available_elements, temperature)
        X_scaled = preprocess_new_data(df, available_elements, _scaler)
        if X_scaled.shape[1] != _vae.input_dim:
            raise ValueError("Input shape mismatch")
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        _vae.eval()
        _regressor.eval()
        with torch.no_grad():
            _, z_mean, _ = _vae(X_tensor)
            z_mean_original = z_mean.clone()
            y_scaled_pred_unbiased = _regressor(z_mean_original)
            y_pred_unbiased = _y_scaler.inverse_transform(y_scaled_pred_unbiased.cpu().numpy().reshape(-1, 1)).ravel()
            y_pred_unbiased = np.clip(y_pred_unbiased, -300, 300)
            if sign_bias is not None and bias_vector is not None:
                if summary_dict and 'seebeck_values' in summary_dict and summary_dict['seebeck_values']:
                    avg_seebeck = np.mean([abs(v) for v in summary_dict['seebeck_values']])
                    bias_magnitude = bias_magnitude * (avg_seebeck / 100.0)
                bias_vector = torch.FloatTensor(bias_vector).to(device) * bias_magnitude
                if sign_bias == 'p-type':
                    z_mean = z_mean + bias_vector
                elif sign_bias == 'n-type':
                    z_mean = z_mean - bias_vector
                y_scaled_pred = _regressor(z_mean)
                y_pred = _y_scaler.inverse_transform(y_scaled_pred.cpu().numpy().reshape(-1, 1)).ravel()
                y_pred = np.clip(y_pred, -300, 300)
                if sign_bias == 'n-type' and y_pred[0] > 0:
                    y_pred = -y_pred
            else:
                y_pred = y_pred_unbiased
        return y_pred[0], y_pred_unbiased[0]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if sign_bias is not None:
            return predict_seebeck(composition_dict, temperature, available_elements, _scaler, _vae, _regressor, _y_scaler, sign_bias=None, bias_vector=None, bias_magnitude=bias_magnitude, summary_dict=summary_dict)
        return None, None

# Load models and scalers
script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    vae = VAE(input_dim=66).to(device)
    regressor = Regressor().to(device)
    classifier = MaterialClassifier(input_dim=5).to(device)
    vae.load_state_dict(torch.load(os.path.join(script_dir, 'vae_model.pt'), map_location=device))
    regressor.load_state_dict(torch.load(os.path.join(script_dir, 'regressor_model.pt'), map_location=device))
    classifier.load_state_dict(torch.load(os.path.join(script_dir, 'material_classifier.pt'), map_location=device))
    #scaler_vae = joblib.load(os.path.join(script_dir, 'scaler_vae.pkl'))
    scaler_vae = joblib.load(os.path.join(script_dir, 'vrnumeric_scaler.pkl'))
    scaler_classifier = joblib.load(os.path.join(script_dir, 'scaler_classifier.pkl'))
    #y_scaler = joblib.load(os.path.join(script_dir, 'y_scaler.pkl'))
    y_scaler = joblib.load(os.path.join(script_dir, 'y_vrnumeric_scaler.pkl'))
    thermoelectric_db = load_thermoelectric_db(os.path.join(script_dir, 'thermoelectric_universe.db'))
    material_classification_df = load_material_classification_csv(os.path.join(script_dir, 'material_classification.csv'))
except FileNotFoundError as e:
    st.error(f"Required files not found: {e}")
    st.stop()
except RuntimeError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Complete periodic table elements
all_elements = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

# Available elements
available_elements = [
    'Mg', 'Cs', 'Co', 'Zr', 'Se', 'Dy', 'Pb', 'Ga', 'O', 'Sn', 'Yb', 'B', 'La', 'Si', 'V', 'Fe', 'S', 'Sc', 'Tl', 'Zn',
    'Cl', 'Ce', 'Er', 'Nd', 'Pd', 'Y', 'P', 'Ta', 'In', 'Te', 'Ru', 'Rb', 'Tm', 'Tb', 'Sb', 'Al', 'Lu', 'Bi', 'Pr', 'Eu',
    'Sm', 'Ba', 'Cr', 'Sr', 'Ni', 'Ca', 'As', 'Mn', 'Mo', 'Cd', 'Ti', 'Nb', 'Hf', 'Gd', 'Ag', 'Ge', 'Li', 'Br', 'Au', 'I',
    'N', 'Na', 'Cu', 'Ho', 'K'
]

# Enhanced color map
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
This application predicts the Seebeck coefficient for a ternary composition at a specified temperature, visualized in a ternary diagram. Select up to three elements, choose optimization mode, and input proportions or use optimized compositions based on SciBERT, database, or ANN analysis. Material type (p-type, n-type, or Neutral) is determined manually or using a combination of SciBERT, database, and ANN-based classification from thermoelectric_universe.db and material_classification.csv.
**Date and Time**: 08:27 PM CEST, Friday, August 22, 2025
""")

# Sidebar for figure customization
st.sidebar.header("Figure Customization")
color_scales = [
    'aggrnyl', 'agsunset', 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brwnyl',
    'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'darkmint', 'electric', 'emrld', 'gnbu',
    'greens', 'greys', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet', 'magma', 'mint',
    'orrd', 'oranges', 'oryel', 'peach', 'pinkyl', 'plasma', 'plotly3', 'pubu', 'pubugn',
    'purp', 'purples', 'purpor', 'rainbow', 'rdpu', 'reds', 'sunset', 'sunsetdark', 'teal',
    'tealgrn', 'turbo', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
]
color_scale = st.sidebar.selectbox("Ternary Color Scale", color_scales, index=color_scales.index('viridis'))
legend_font_size = st.sidebar.slider("Legend Font Size", 8, 20, 12)
axes_line_width = st.sidebar.slider("Axes Line Width", 1, 5, 2)
font_size = st.sidebar.slider("Font Size (Axes/Title)", 8, 20, 16)
grid_width = st.sidebar.slider("Grid Width", 0.5, 3.0, 1.0, step=0.5)
user_point_color = st.sidebar.color_picker("User Composition Point Color", '#FF0000')
max_point_color = st.sidebar.color_picker("Max |Seebeck| Point Color", '#00FF00')
literature_point_color = st.sidebar.color_picker("Literature Formula Point Color", '#FFFF00')
ternary_grid_color = st.sidebar.color_picker("Ternary Grid Color", '#000000')
ternary_axes_color = st.sidebar.color_picker("Ternary Axes Color", '#000000')
point_size = st.sidebar.slider("Point Size (Ternary/Temperature)", 5, 20, 10)
axes_box_thickness = st.sidebar.slider("Axes Box Thickness", 1, 5, 2)
legend_spacing = st.sidebar.slider("Legend Spacing", 0.0, 0.5, 0.3, step=0.05)
periodic_table_marker_size = st.sidebar.slider("Periodic Table Marker Size", 20, 60, 40)
periodic_table_font_size = st.sidebar.slider("Periodic Table Font Size", 8, 20, 14)

# Physics attention options
st.sidebar.header("Physics Attention Options")
custom_keywords = st.sidebar.text_input("Custom Keywords (comma-separated)", "band gap, thermal conductivity, electrical conductivity").split(", ")
custom_keywords = [kw.strip() for kw in custom_keywords if kw.strip()]
year_range = st.sidebar.slider("Publication Year Range", 2000, 2025, (2020, 2025))
pmi_threshold = st.sidebar.slider("PMI Threshold", 0.0, 5.0, 1.0, step=0.1)

# Initialize session state
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []
if 'proportions' not in st.session_state:
    st.session_state.proportions = {}
if 'compositions' not in st.session_state:
    st.session_state.compositions = {}
if 'temperature' not in st.session_state:
    st.session_state.temperature = 800
if 'sign_bias' not in st.session_state:
    st.session_state.sign_bias = 'Neutral'
if 'use_manual_material_type' not in st.session_state:
    st.session_state.use_manual_material_type = False
if 'optimization_mode' not in st.session_state:
    st.session_state.optimization_mode = 'Informatics-Attention Optimize'
if 'summary_dict' not in st.session_state:
    st.session_state.summary_dict = {}
if 'verbatim_matches' not in st.session_state:
    st.session_state.verbatim_matches = []

# Periodic Table for Reference
st.header("Periodic Table Reference")
def plot_periodic_table(available_elements, selected_elements, element_color_map, show_all_elements=False, fontsize=14, marker_size=40):
    periodic_table_positions = {
        'H': (1, 1), 'He': (1, 18), 'Li': (2, 1), 'Be': (2, 2), 'B': (2, 13), 'C': (2, 14), 'N': (2, 15), 'O': (2, 16), 'F': (2, 17), 'Ne': (2, 18),
        'Na': (3, 1), 'Mg': (3, 2), 'Al': (3, 13), 'Si': (3, 14), 'P': (3, 15), 'S': (3, 16), 'Cl': (3, 17), 'Ar': (3, 18),
        'K': (4, 1), 'Ca': (4, 2), 'Sc': (4, 3), 'Ti': (4, 4), 'V': (4, 5), 'Cr': (4, 6), 'Mn': (4, 7), 'Fe': (4, 8), 'Co': (4, 9), 'Ni': (4, 10),
        'Cu': (4, 11), 'Zn': (4, 12), 'Ga': (4, 13), 'Ge': (4, 14), 'As': (4, 15), 'Se': (4, 16), 'Br': (4, 17), 'Kr': (4, 18),
        'Rb': (5, 1), 'Sr': (5, 2), 'Y': (5, 3), 'Zr': (5, 4), 'Nb': (5, 5), 'Mo': (5, 6), 'Tc': (5, 7), 'Ru': (5, 8), 'Rh': (5, 9), 'Pd': (5, 10),
        'Ag': (5, 11), 'Cd': (5, 12), 'In': (5, 13), 'Sn': (5, 14), 'Sb': (5, 15), 'Te': (5, 16), 'I': (5, 17), 'Xe': (5, 18),
        'Cs': (6, 1), 'Ba': (6, 2), 'La': (6, 3), 'Ce': (6, 4), 'Pr': (6, 5), 'Nd': (6, 6), 'Pm': (6, 7), 'Sm': (6, 8), 'Eu': (6, 9), 'Gd': (6, 10),
        'Tb': (6, 11), 'Dy': (6, 12), 'Ho': (6, 13), 'Er': (6, 14), 'Tm': (6, 15), 'Yb': (6, 16), 'Lu': (6, 17),
        'Hf': (7, 4), 'Ta': (7, 5), 'W': (7, 6), 'Re': (7, 7), 'Os': (7, 8), 'Ir': (7, 9), 'Pt': (7, 10), 'Au': (7, 11), 'Hg': (7, 12),
        'Tl': (7, 13), 'Pb': (7, 14), 'Bi': (7, 15), 'Po': (7, 16), 'At': (7, 17), 'Rn': (7, 18),
        'Fr': (8, 1), 'Ra': (8, 2), 'Ac': (8, 3), 'Th': (8, 4), 'Pa': (8, 5), 'U': (8, 6), 'Np': (8, 7), 'Pu': (8, 8), 'Am': (8, 9),
        'Cm': (8, 10), 'Bk': (8, 11), 'Cf': (8, 12), 'Es': (8, 13), 'Fm': (8, 14), 'Md': (8, 15), 'No': (8, 16), 'Lr': (8, 17),
        'Rf': (9, 4), 'Db': (9, 5), 'Sg': (9, 6), 'Bh': (9, 7), 'Hs': (9, 8), 'Mt': (9, 9), 'Ds': (9, 10), 'Rg': (9, 11), 'Cn': (9, 12),
        'Nh': (9, 13), 'Fl': (9, 14), 'Mc': (9, 15), 'Lv': (9, 16), 'Ts': (9, 17), 'Og': (9, 18)
    }
    elements_to_plot = all_elements if show_all_elements else [e for e in all_elements if e in available_elements or e in selected_elements]
    fig = go.Figure()
    for element in elements_to_plot:
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
                marker=dict(size=marker_size, color=color, opacity=opacity, line=dict(width=line_width, color='black')),
                hoverinfo='text',
                hovertext=[f"Element: {element}<br>Electronegativity: {electronegativity.get(element, 1.0):.2f}<br>Thermoelectric Weight: {thermoelectric_weights.get(element, 1.0):.2f}"],
                name=element,
                showlegend=False
            ))
    title_text = 'Periodic Table: Available Elements' if not show_all_elements else 'Periodic Table: Full (Unavailable in Gray)'
    fig.update_layout(
        title=dict(text=f"{title_text} (Selected Elements with Bold Outline)", x=0.5, xanchor='center', font=dict(size=fontsize + 4, family='Arial')),
        xaxis=dict(range=[0, 20], showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(range=[-11, 0], showgrid=False, zeroline=False, showticklabels=False, title=''),
        plot_bgcolor='white', paper_bgcolor='white',
        autosize=True,
        margin=dict(l=40, r=40, t=60, b=40),
        template='seaborn'
    )
    return fig

# Plot periodic tables
st.subheader("Full Periodic Table")
fig_full = plot_periodic_table(available_elements, st.session_state.selected_elements, default_element_color_map, show_all_elements=True, fontsize=periodic_table_font_size, marker_size=periodic_table_marker_size)
st.plotly_chart(fig_full, use_container_width=True)

st.subheader("Available Elements Only")
fig_present = plot_periodic_table(available_elements, st.session_state.selected_elements, default_element_color_map, show_all_elements=False, fontsize=periodic_table_font_size, marker_size=periodic_table_marker_size)
st.plotly_chart(fig_present, use_container_width=True)

# Optimization mode selection
st.header("Optimization Mode")
optimization_mode = st.selectbox("Select Optimization Mode", ["Informatics-Attention Optimize", "Manual"], index=0, key='optimization_mode')
if optimization_mode != st.session_state.optimization_mode:
    st.session_state.optimization_mode = optimization_mode

# Element selection
st.header("Select Elements")
st.session_state.selected_elements = st.multiselect(
    "Select up to three elements",
    options=available_elements,
    default=st.session_state.selected_elements,
    max_selections=3,
    key='element_selector'
)

# Update proportions and compositions
for element in st.session_state.selected_elements:
    if element not in st.session_state.proportions:
        st.session_state.proportions[element] = 0.0
    if element not in st.session_state.compositions:
        st.session_state.compositions[element] = 0.0
st.session_state.proportions = {k: v for k, v in st.session_state.proportions.items() if k in st.session_state.selected_elements}
st.session_state.compositions = {k: v for k, v in st.session_state.compositions.items() if k in st.session_state.selected_elements}

# Composition input
st.header("Input Proportions and View Normalized Compositions")
if st.session_state.selected_elements:
    st.write(f"Selected Elements: {', '.join(st.session_state.selected_elements)}")
    
    if st.session_state.optimization_mode == "Manual":
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
    else:
        try:
            # Check database for matching formula
            matched_formula = None
            for _, row in thermoelectric_db.iterrows():
                try:
                    comp = Composition(row['formula'])
                    comp_elements = set(comp.elements)
                    selected_elements_set = set([Composition(el) for el in st.session_state.selected_elements])
                    if comp_elements.issubset(selected_elements_set):
                        matched_formula = row['formula']
                        break
                except:
                    continue
            if matched_formula:
                comp = Composition(matched_formula)
                st.session_state.compositions = {el: comp[el] / comp.num_atoms for el in comp if el in st.session_state.selected_elements}
                st.write(f"Optimized composition from thermoelectric_universe.db: {matched_formula}")
            else:
                # Fall back to material classification CSV
                for _, row in material_classification_df.iterrows():
                    try:
                        comp = Composition(row['formula'])
                        comp_elements = set(comp.elements)
                        selected_elements_set = set([Composition(el) for el in st.session_state.selected_elements])
                        if comp_elements.issubset(selected_elements_set):
                            matched_formula = row['formula']
                            break
                    except:
                        continue
                if matched_formula:
                    comp = Composition(matched_formula)
                    st.session_state.compositions = {el: comp[el] / comp.num_atoms for el in comp if el in st.session_state.selected_elements}
                    st.write(f"Optimized composition from material_classification.csv: {matched_formula}")
                else:
                    # Use SciBERT-derived formula
                    material_type, summary_dict, verbatim_matches = extract_material_type(
                        st.session_state.selected_elements, 
                        {el: 1.0 / len(st.session_state.selected_elements) for el in st.session_state.selected_elements}, 
                        custom_keywords, year_range, pmi_threshold
                    )
                    matched_formulas = list(set([match['matched_formula'] for match in verbatim_matches if 'matched_formula' in match]))
                    if matched_formulas:
                        formula_seebeck = {}
                        for formula in matched_formulas:
                            formula_matches = [m for m in verbatim_matches if m.get('matched_formula') == formula]
                            seebeck_values = summary_dict.get('seebeck_values', [])
                            if seebeck_values:
                                formula_seebeck[formula] = np.mean([abs(v) for v in seebeck_values])
                        if formula_seebeck:
                            best_formula = max(formula_seebeck, key=formula_seebeck.get)
                            comp = Composition(best_formula)
                            st.session_state.compositions = {el: comp[el] / comp.num_atoms for el in comp if el in st.session_state.selected_elements}
                            st.write(f"Optimized composition based on SciBERT formula: {best_formula}")
                        else:
                            comp = Composition({el: 1.0 for el in st.session_state.selected_elements})
                            st.session_state.compositions = {el: comp[el] / comp.num_atoms for el in comp}
                            st.write(f"Default composition (equal proportions): {comp.reduced_formula}")
                    else:
                        comp = Composition({el: 1.0 for el in st.session_state.selected_elements})
                        st.session_state.compositions = {el: comp[el] / comp.num_atoms for el in comp}
                        st.write(f"No SciBERT formulas found, using default composition: {comp.reduced_formula}")
        except Exception as e:
            st.error(f"Invalid composition: {str(e)}")
            st.session_state.compositions = {el: 0.0 for el in st.session_state.selected_elements}
    
    # Display normalized compositions
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
st.session_state.temperature = st.number_input("Enter Temperature (K):", min_value=0, max_value=5000, value=st.session_state.temperature, step=10)

# Material type selection
st.header("Material Type Selection")
st.session_state.use_manual_material_type = st.checkbox("Manually Select Material Type", value=st.session_state.use_manual_material_type)
if st.session_state.use_manual_material_type:
    st.session_state.sign_bias = st.selectbox(
        "Select Material Type (Sign Bias)",
        options=['Neutral', 'p-type', 'n-type'],
        index=['Neutral', 'p-type', 'n-type'].index(st.session_state.sign_bias),
        key='sign_bias_selector'
    )
    st.session_state.summary_dict = {}
    st.session_state.verbatim_matches = []
else:
    st.write("Using combined database, CSV, and ANN-based material type suggestion.")
    if st.session_state.selected_elements and sum(st.session_state.compositions.values()) > 0:
        try:
            # Check database for material type
            material_type = None
            for _, row in thermoelectric_db.iterrows():
                try:
                    comp = Composition(row['formula'])
                    comp_elements = set(comp.elements)
                    selected_elements_set = set([Composition(el) for el in st.session_state.selected_elements])
                    if comp_elements.issubset(selected_elements_set):
                        material_type = row['material_type']
                        break
                except:
                    continue
            if material_type:
                st.session_state.sign_bias = material_type
                st.session_state.summary_dict = {}
                st.session_state.verbatim_matches = []
                st.write(f"**Material Type from Database**: {material_type}")
            else:
                # Check CSV for material type
                for _, row in material_classification_df.iterrows():
                    try:
                        comp = Composition(row['formula'])
                        comp_elements = set(comp.elements)
                        selected_elements_set = set([Composition(el) for el in st.session_state.selected_elements])
                        if comp_elements.issubset(selected_elements_set):
                            material_type = row.get('material_type', 'Neutral')
                            break
                    except:
                        continue
                if material_type:
                    st.session_state.sign_bias = material_type
                    st.session_state.summary_dict = {}
                    st.session_state.verbatim_matches = []
                    st.write(f"**Material Type from CSV**: {material_type}")
                else:
                    # Use ANN classifier
                    material_type, ann_probs = predict_material_type(
                        st.session_state.compositions, 
                        st.session_state.temperature, 
                        available_elements, 
                        scaler_classifier, 
                        classifier
                    )
                    if material_type != 'Neutral':
                        st.session_state.sign_bias = material_type
                        st.session_state.summary_dict = ann_probs
                        st.session_state.verbatim_matches = []
                        st.write(f"**Material Type from ANN**: {material_type}")
                        st.write(f"**p-type Probability**: {ann_probs['p_type_prob']:.3f}")
                        st.write(f"**n-type Probability**: {ann_probs['n_type_prob']:.3f}")
                    else:
                        # Fall back to SciBERT
                        material_type, summary_dict, verbatim_matches = extract_material_type(
                            st.session_state.selected_elements, 
                            st.session_state.compositions, 
                            custom_keywords, year_range, pmi_threshold
                        )
                        st.session_state.sign_bias = material_type
                        st.session_state.summary_dict = summary_dict
                        st.session_state.verbatim_matches = verbatim_matches
                        st.write(f"**Suggested Material Type (SciBERT)**: {material_type}")
                        st.write(f"**p-type Probability**: {summary_dict['p_type_prob']:.3f}")
                        st.write(f"**n-type Probability**: {summary_dict['n_type_prob']:.3f}")
                        st.write(f"**Total Abstracts Analyzed**: {summary_dict['total_abstracts']}")
                        st.write(f"**Abstracts with Formula Matches**: {summary_dict['formula_matches']}")
                        st.write(f"**Matched Terms**: {', '.join(summary_dict['matched_terms'])}")
                        if summary_dict.get('seebeck_values'):
                            st.write(f"**Literature Seebeck Coefficients (μV/K)**: Mean = {np.mean(summary_dict['seebeck_values']):.2f}, Std = {np.std(summary_dict['seebeck_values']):.2f}")
                        st.write(f"**Matched Formulas**: {', '.join(set([m.get('matched_formula', 'N/A') for m in verbatim_matches]))}")
                        st.subheader("PMI Scores")
                        for term_pair, pmi in summary_dict['pmi_scores'].items():
                            st.write(f"{term_pair}: {pmi:.2f}")
                        st.subheader("Relevant Abstract Snippets")
                        for match in verbatim_matches:
                            st.write(f"**Title**: {match['title']}")
                            st.write(f"**arXiv ID**: {match['arxiv_id']}")
                            st.write(f"**Snippet**: {match['snippet']}")
                            st.write(f"**Label**: {match['label']}")
                            st.write(f"**Score**: {match['score']:.3f}")
                            st.write(f"**Matched Formula**: {match.get('matched_formula', 'N/A')}")
                            st.markdown("---")
                        if verbatim_matches:
                            abstracts_df = pd.DataFrame(verbatim_matches)
                            csv = abstracts_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Abstracts as CSV",
                                data=csv,
                                file_name="abstracts.csv",
                                mime="text/csv"
                            )
                            output = BytesIO()
                            with sqlite3.connect(':memory:') as conn:
                                abstracts_df.to_sql('abstracts', conn, index=False, if_exists='replace')
                                conn.commit()
                                with output:
                                    output.write(conn.backup(conn).read())
                                    sqlite_data = output.getvalue()
                            st.download_button(
                                label="Download Abstracts as SQLite DB",
                                data=sqlite_data,
                                file_name="abstracts.db",
                                mime="application/x-sqlite3"
                            )
                        st.subheader("Matched Formula Histogram")
                        fig_formula = plot_formula_histogram(verbatim_matches, font_size)
                        if fig_formula:
                            st.plotly_chart(fig_formula, use_container_width=True)
                        else:
                            st.write("No matched formulas available for histogram.")
                        st.subheader("Literature Seebeck Coefficient Histogram")
                        fig_seebeck = plot_literature_seebeck(summary_dict, font_size)
                        if fig_seebeck:
                            st.plotly_chart(fig_seebeck, use_container_width=True)
                        else:
                            st.write("No Seebeck coefficients extracted from literature.")
        except Exception as e:
            st.error(f"Failed to determine material type: {e}. Defaulting to Neutral.")
            st.session_state.sign_bias = 'Neutral'
            st.session_state.summary_dict = {}
            st.session_state.verbatim_matches = []
    else:
        st.warning("Please select elements and normalize proportions to determine material type.")
        st.session_state.sign_bias = 'Neutral'
        st.session_state.summary_dict = {}
        st.session_state.verbatim_matches = []

# Complete to three elements
def complete_to_three_elements(selected_elements, proportions, compositions, available_elements):
    while len(selected_elements) < 3:
        remaining_elements = [e for e in available_elements if e in ['Ag', 'Bi', 'Te'] and e not in selected_elements]
        if not remaining_elements:
            remaining_elements = [e for e in available_elements if e not in selected_elements]
        if remaining_elements:
            random_element = np.random.choice(remaining_elements)
            selected_elements.append(random_element)
            proportions[random_element] = 0.0
            compositions[random_element] = 0.0
        else:
            st.error("Not enough available elements to complete the ternary composition.")
            return selected_elements, proportions, compositions
    return selected_elements, proportions, compositions

# Generate ternary data
@st.cache_resource
def generate_ternary_data(_vae, _regressor, _scaler, _y_scaler, elements, temperature, available_elements, sign_bias, bias_vector, bias_magnitude, summary_dict, steps=30):
    compositions = []
    seebeck_values = []
    for a in np.linspace(0, 1, steps):
        for b in np.linspace(0, 1 - a, steps):
            c = 1 - a - b
            if c >= 0:
                comp_dict = {elements[0]: a, elements[1]: b, elements[2]: c}
                seebeck, _ = predict_seebeck(comp_dict, temperature, available_elements, _scaler, _vae, _regressor, _y_scaler, sign_bias=sign_bias, bias_vector=bias_vector, bias_magnitude=bias_magnitude, summary_dict=summary_dict)
                if seebeck is not None:
                    compositions.append([a, b, c])
                    seebeck_values.append(abs(seebeck))
    return np.array(compositions), np.array(seebeck_values)

# Plot ternary diagram
def plot_ternary_diagram(compositions, seebeck_values, elements, user_composition, user_seebeck, max_comp, max_seebeck, literature_compositions, literature_seebeck, color_scale, font_size, axes_line_width, point_size, axes_box_thickness, legend_spacing, user_point_color, max_point_color, literature_point_color, ternary_grid_color, ternary_axes_color):
    if len(compositions) == 0:
        st.warning("No valid ternary data to plot.")
        return None
    fig = go.Figure()
    hover_texts = [f"{elements[0]}: {comp[0]:.2f}<br>{elements[1]}: {comp[1]:.2f}<br>{elements[2]}: {comp[2]:.2f}<br>|Seebeck|: {s:.2f} μV/K" for comp, s in zip(compositions, seebeck_values)]
    fig.add_trace(go.Scatterternary(
        a=compositions[:, 0], b=compositions[:, 1], c=compositions[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=seebeck_values,
            colorscale=color_scale,
            showscale=True,
            colorbar=dict(
                title='|Seebeck| (μV/K)',
                tickfont=dict(size=font_size),
                x=1.0 + legend_spacing / 2,
                y=0.5,
                len=0.75
            )
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Compositions'
    ))
    if user_seebeck is not None:
        user_hover_text = f"User Composition<br>{elements[0]}: {user_composition[0]:.2f}<br>{elements[1]}: {user_composition[1]:.2f}<br>{elements[2]}: {user_composition[2]:.2f}<br>|Seebeck|: {abs(user_seebeck):.2f} μV/K"
        fig.add_trace(go.Scatterternary(
            a=[user_composition[0]], b=[user_composition[1]], c=[user_composition[2]],
            mode='markers',
            marker=dict(size=point_size + 5, color=user_point_color, symbol='star'),
            text=[user_hover_text],
            hoverinfo='text',
            name='User Composition'
        ))
    if max_seebeck != float('-inf'):
        max_hover_text = f"Max |Seebeck|<br>{elements[0]}: {max_comp[0]:.2f}<br>{elements[1]}: {max_comp[1]:.2f}<br>{elements[2]}: {max_comp[2]:.2f}<br>|Seebeck|: {max_seebeck:.2f} μV/K"
        fig.add_trace(go.Scatterternary(
            a=[max_comp[0]], b=[max_comp[1]], c=[max_comp[2]],
            mode='markers',
            marker=dict(size=point_size + 5, color=max_point_color, symbol='square'),
            text=[max_hover_text],
            hoverinfo='text',
            name='Max |Seebeck|'
        ))
    if literature_compositions:
        for comp, seebeck in zip(literature_compositions, literature_seebeck):
            lit_hover_text = f"Literature Composition<br>{elements[0]}: {comp[0]:.2f}<br>{elements[1]}: {comp[1]:.2f}<br>{elements[2]}: {comp[2]:.2f}<br>|Seebeck|: {seebeck:.2f} μV/K"
            fig.add_trace(go.Scatterternary(
                a=[comp[0]], b=[comp[1]], c=[comp[2]],
                mode='markers',
                marker=dict(size=point_size + 5, color=literature_point_color, symbol='diamond'),
                text=[lit_hover_text],
                hoverinfo='text',
                name='Literature Composition'
            ))
    fig.update_layout(
        title=dict(text=f"Ternary Diagram: |Seebeck Coefficient| at {st.session_state.temperature} K", x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        ternary=dict(
            sum=1,
            aaxis=dict(
                title=dict(text=elements[0], font=dict(size=font_size)),
                tickfont=dict(size=font_size),
                gridcolor=ternary_grid_color,
                linecolor=ternary_axes_color,
                linewidth=axes_line_width
            ),
            baxis=dict(
                title=dict(text=elements[1], font=dict(size=font_size)),
                tickfont=dict(size=font_size),
                gridcolor=ternary_grid_color,
                linecolor=ternary_axes_color,
                linewidth=axes_line_width
            ),
            caxis=dict(
                title=dict(text=elements[2], font=dict(size=font_size)),
                tickfont=dict(size=font_size),
                gridcolor=ternary_grid_color,
                linecolor=ternary_axes_color,
                linewidth=axes_line_width
            )
        ),
        showlegend=True,
        legend=dict(x=-(0.15 + legend_spacing), y=1, xanchor='right', font=dict(size=legend_font_size)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        template='seaborn'
    )
    return fig

# Plot temperature variance
def plot_temperature_variance(elements, user_composition, max_comp, literature_compositions, literature_seebeck, temp_range, available_elements, _scaler, _vae, _regressor, _y_scaler, sign_bias, bias_vector, bias_magnitude, summary_dict, font_size, axes_line_width, grid_width, user_point_color, max_point_color, literature_point_color, point_size, axes_box_thickness):
    temps = np.linspace(temp_range[0], temp_range[1], 20)
    user_seebeck = []
    max_seebeck = []
    literature_seebeck_vals = []
    for temp in temps:
        user_val, _ = predict_seebeck({elements[i]: user_composition[i] for i in range(3)}, temp, available_elements, _scaler, _vae, _regressor, _y_scaler, sign_bias=sign_bias, bias_vector=bias_vector, bias_magnitude=bias_magnitude, summary_dict=summary_dict)
        max_val, _ = predict_seebeck({elements[i]: max_comp[i] for i in range(3)}, temp, available_elements, _scaler, _vae, _regressor, _y_scaler, sign_bias=sign_bias, bias_vector=bias_vector, bias_magnitude=bias_magnitude, summary_dict=summary_dict)
        user_seebeck.append(abs(user_val) if user_val is not None else np.nan)
        max_seebeck.append(abs(max_val) if max_val is not None else np.nan)
        lit_vals = []
        for comp in literature_compositions:
            comp_dict = {elements[i]: comp[i] for i in range(3)}
            lit_val, _ = predict_seebeck(comp_dict, temp, available_elements, _scaler, _vae, _regressor, _y_scaler, sign_bias=sign_bias, bias_vector=bias_vector, bias_magnitude=bias_magnitude, summary_dict=summary_dict)
            lit_vals.append(abs(lit_val) if lit_val is not None else np.nan)
        literature_seebeck_vals.append(lit_vals)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temps, y=user_seebeck, mode='lines+markers', name='User Composition', line=dict(color=user_point_color, width=axes_line_width), marker=dict(size=point_size)))
    fig.add_trace(go.Scatter(x=temps, y=max_seebeck, mode='lines+markers', name='Max |Seebeck|', line=dict(color=max_point_color, width=axes_line_width), marker=dict(size=point_size)))
    for idx, lit_vals in enumerate(np.array(literature_seebeck_vals).T):
        fig.add_trace(go.Scatter(x=temps, y=lit_vals, mode='lines+markers', name=f'Literature Comp {idx+1}', line=dict(color=literature_point_color, width=axes_line_width, dash='dash'), marker=dict(size=point_size)))
    fig.update_layout(
        title=dict(text='|Seebeck Coefficient| vs Temperature', x=0.5, xanchor='center', font=dict(size=font_size + 4, family='Arial')),
        xaxis_title='Temperature (K)', yaxis_title='|Seebeck Coefficient| (μV/K)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=grid_width,
            zeroline=False,
            showline=True,
            linewidth=axes_box_thickness,
            linecolor='black',
            title=dict(text='Temperature (K)', font=dict(size=font_size)),
            tickfont=dict(size=font_size)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=grid_width,
            zeroline=False,
            showline=True,
            linewidth=axes_box_thickness,
            linecolor='black',
            title=dict(text='|Seebeck Coefficient| (μV/K)', font=dict(size=font_size)),
            tickfont=dict(size=font_size)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(x=1.05, y=1, font=dict(size=legend_font_size)),
        margin=dict(l=50, r=50, t=80, b=50),
        template='seaborn'
    )
    return fig, temps, user_seebeck, max_seebeck

# ... (previous code remains unchanged until the section below)

# Generate ternary diagram and temperature variance plot
if st.button("Generate Ternary Diagram"):
    if len(st.session_state.selected_elements) > 0:
        elements, proportions, compositions = complete_to_three_elements(
            st.session_state.selected_elements.copy(),
            st.session_state.proportions.copy(),
            st.session_state.compositions.copy(),
            available_elements
        )
        total = sum(proportions.values())
        if total == 0 and st.session_state.optimization_mode == "Manual":
            st.error("Please provide non-zero proportions for at least one element.")
        else:
            z_mean_avg, z_mean_std, bias_vector, bias_magnitude = compute_z_mean_stats_and_bias(elements, st.session_state.temperature, available_elements, scaler_vae, vae)
            st.write("### Latent Space Statistics (z_mean)")
            st.write(f"**Mean per dimension**: {[f'{x:.4f}' for x in z_mean_avg]}")
            st.write(f"**Std per dimension**: {[f'{x:.4f}' for x in z_mean_std]}")
            st.write(f"**Bias vector (p-type to n-type)**: {[f'{x:.4f}' for x in bias_vector]}")
            st.write(f"**Applied bias magnitude**: {bias_magnitude:.4f}")
            fig_z_mean = plot_z_mean_bar_chart(z_mean_avg, z_mean_std, font_size)
            st.plotly_chart(fig_z_mean, use_container_width=True)
            fig_z_mean_matplotlib = plot_z_mean_bar_chart_matplotlib(z_mean_avg, z_mean_std, os.path.join(script_dir, 'z_mean_bar_chart.pdf'))
            st.pyplot(fig_z_mean_matplotlib)
            st.download_button(
                label="Download z_mean Bar Chart as PDF",
                data=open(os.path.join(script_dir, 'z_mean_bar_chart.pdf'), 'rb').read(),
                file_name="z_mean_bar_chart.pdf",
                mime="application/pdf"
            )
            
            user_composition = [compositions.get(elements[i], 0) for i in range(3)]
            user_composition_dict = {elements[i]: user_composition[i] for i in range(3)}
            sign_bias = st.session_state.sign_bias if st.session_state.sign_bias != 'Neutral' else None
            user_seebeck, user_seebeck_unbiased = predict_seebeck(
                user_composition_dict,
                st.session_state.temperature,
                available_elements,
                scaler_vae,
                vae,
                regressor,
                y_scaler,
                sign_bias=sign_bias,
                bias_vector=bias_vector,
                bias_magnitude=bias_magnitude,
                summary_dict=st.session_state.summary_dict
            )
            if user_seebeck is None:
                user_seebeck, user_seebeck_unbiased = predict_seebeck(
                    user_composition_dict,
                    st.session_state.temperature,
                    available_elements,
                    scaler_vae,
                    vae,
                    regressor,
                    y_scaler,
                    sign_bias=None,
                    bias_vector=None,
                    bias_magnitude=bias_magnitude,
                    summary_dict=st.session_state.summary_dict
                )
                if user_seebeck is None:
                    user_seebeck = 0.0
                    user_seebeck_unbiased = 0.0
            try:
                compositions_array, seebeck_values = generate_ternary_data(
                    vae, regressor, scaler_vae, y_scaler, elements, st.session_state.temperature, available_elements, 
                    sign_bias=sign_bias, bias_vector=bias_vector, bias_magnitude=bias_magnitude, summary_dict=st.session_state.summary_dict
                )
            except Exception as e:
                st.error(f"Failed to generate ternary data: {e}")
                compositions_array, seebeck_values = [], []
            literature_compositions = []
            literature_seebeck = []
            for _, row in thermoelectric_db.iterrows():
                try:
                    comp = Composition(row['formula'])
                    comp_dict = {el: comp[el] / comp.num_atoms for el in comp if el in elements}
                    if len(comp_dict) == len(elements):
                        lit_comp = [comp_dict.get(elements[i], 0) for i in range(3)]
                        total = sum(lit_comp)
                        if total > 0:
                            lit_comp = [x / total for x in lit_comp]
                            lit_seebeck = row['seebeck_coefficient']
                            literature_compositions.append(lit_comp)
                            literature_seebeck.append(abs(lit_seebeck))
                except:
                    continue
            if len(compositions_array) == 0:
                max_comp, max_seebeck_abs, max_seebeck_signed = user_composition, abs(user_seebeck), user_seebeck
            else:
                ternary_df = pd.DataFrame(compositions_array, columns=[elements[0], elements[1], elements[2]])
                ternary_df['|Seebeck| (μV/K)'] = seebeck_values
                max_row = ternary_df.loc[ternary_df['|Seebeck| (μV/K)'].idxmax()]
                max_comp = [max_row[elements[0]], max_row[elements[1]], max_row[elements[2]]]
                max_seebeck_abs = max_row['|Seebeck| (μV/K)']
                max_seebeck_signed, _ = predict_seebeck(
                    {elements[i]: max_comp[i] for i in range(3)},
                    st.session_state.temperature,
                    available_elements,
                    scaler_vae,
                    vae,
                    regressor,
                    y_scaler,
                    sign_bias=sign_bias,
                    bias_vector=bias_vector,
                    bias_magnitude=bias_magnitude,
                    summary_dict=st.session_state.summary_dict
                )
                if max_seebeck_signed is None:
                    max_seebeck_signed, _ = predict_seebeck(
                        {elements[i]: max_comp[i] for i in range(3)},
                        st.session_state.temperature,
                        available_elements,
                        scaler_vae,
                        vae,
                        regressor,
                        y_scaler,
                        sign_bias=None,
                        bias_vector=None,
                        bias_magnitude=bias_magnitude,
                        summary_dict=st.session_state.summary_dict
                    )
                    if max_seebeck_signed is None:
                        max_seebeck_signed = user_seebeck
                        max_comp = user_composition
                        max_seebeck_abs = abs(user_seebeck)
            st.write("### Composition and Seebeck Coefficient")
            st.write(f"**User Composition**: {elements[0]}: {user_composition[0]:.2f}, {elements[1]}: {user_composition[1]:.2f}, {elements[2]}: {user_composition[2]:.2f}")
            st.write(f"**Material Type Used**: {st.session_state.sign_bias} ({'Manual' if st.session_state.use_manual_material_type else 'Database/CSV/ANN/SciBERT'})")
            st.write(f"**User |Seebeck Coefficient| (Biased)**: {abs(user_seebeck):.2f} μV/K")
            st.write(f"**User Signed Seebeck Coefficient (Biased)**: {user_seebeck:.2f} μV/K ({'p-type' if user_seebeck > 0 else 'n-type' if user_seebeck < 0 else 'neutral'})")
            st.write(f"**User |Seebeck Coefficient| (Unbiased)**: {abs(user_seebeck_unbiased):.2f} μV/K")
            st.write(f"**User Signed Seebeck Coefficient (Unbiased)**: {user_seebeck_unbiased:.2f} μV/K ({'p-type' if user_seebeck_unbiased > 0 else 'n-type' if user_seebeck_unbiased < 0 else 'neutral'})")
            st.write(f"**Max |Seebeck Coefficient| Composition**: {elements[0]}: {max_comp[0]:.2f}, {elements[1]}: {max_comp[1]:.2f}, {elements[2]}: {max_comp[2]:.2f}")
            st.write(f"**Max |Seebeck Coefficient| (Biased)**: {max_seebeck_abs:.2f} μV/K")
            st.write(f"**Max Signed Seebeck Coefficient (Biased)**: {max_seebeck_signed:.2f} μV/K ({'p-type' if max_seebeck_signed > 0 else 'n-type' if max_seebeck_signed < 0 else 'neutral'})")

            # Generate ternary diagram
            st.subheader("Ternary Diagram")
            fig_ternary = plot_ternary_diagram(
                compositions_array, seebeck_values, elements, user_composition, user_seebeck, 
                max_comp, max_seebeck_abs, literature_compositions, literature_seebeck, 
                color_scale, font_size, axes_line_width, point_size, axes_box_thickness, 
                legend_spacing, user_point_color, max_point_color, literature_point_color, 
                ternary_grid_color, ternary_axes_color
            )
            if fig_ternary:
                st.plotly_chart(fig_ternary, use_container_width=True)
            else:
                st.warning("Unable to generate ternary diagram due to insufficient data.")

            # Generate temperature variance plot
            st.subheader("Seebeck Coefficient vs Temperature")
            temp_range = (300, 1000)  # Default temperature range
            fig_temp, temps, user_seebeck_vals, max_seebeck_vals = plot_temperature_variance(
                elements, user_composition, max_comp, literature_compositions, literature_seebeck, 
                temp_range, available_elements, scaler_vae, vae, regressor, y_scaler, 
                sign_bias, bias_vector, bias_magnitude, st.session_state.summary_dict, 
                font_size, axes_line_width, grid_width, user_point_color, max_point_color, 
                literature_point_color, point_size, axes_box_thickness
            )
            if fig_temp:
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Provide downloadable data for temperature variance
                temp_df = pd.DataFrame({
                    'Temperature (K)': temps,
                    'User |Seebeck| (μV/K)': user_seebeck_vals,
                    'Max |Seebeck| (μV/K)': max_seebeck_vals
                })
                for idx, lit_vals in enumerate(np.array(literature_seebeck_vals).T):
                    temp_df[f'Literature Comp {idx+1} |Seebeck| (μV/K)'] = lit_vals
                csv = temp_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Temperature Variance Data as CSV",
                    data=csv,
                    file_name="temperature_variance.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Unable to generate temperature variance plot due to insufficient data.")

            # Physics attention visualizations
            if st.session_state.summary_dict and not st.session_state.use_manual_material_type:
                st.subheader("Physics Attention Visualizations")
                
                # Material type histogram
                fig_material_hist = plot_material_type_histogram(st.session_state.summary_dict, font_size)
                if fig_material_hist:
                    st.plotly_chart(fig_material_hist, use_container_width=True)
                
                # Material probabilities
                fig_probs = plot_material_probabilities(st.session_state.summary_dict, font_size)
                if fig_probs:
                    st.plotly_chart(fig_probs, use_container_width=True)
                
                # Relevance box plot
                fig_box = plot_relevance_box_plot(st.session_state.verbatim_matches, font_size)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # PMI network
                fig_network = plot_pmi_network(st.session_state.summary_dict['pmi_scores'], pmi_threshold, font_size)
                if fig_network:
                    st.plotly_chart(fig_network, use_container_width=True)

            # Download ternary data as CSV
            if len(compositions_array) > 0:
                ternary_df['Signed Seebeck (μV/K)'] = [
                    predict_seebeck(
                        {elements[i]: comp[i] for i in range(3)},
                        st.session_state.temperature,
                        available_elements,
                        scaler_vae,
                        vae,
                        regressor,
                        y_scaler,
                        sign_bias=sign_bias,
                        bias_vector=bias_vector,
                        bias_magnitude=bias_magnitude,
                        summary_dict=st.session_state.summary_dict
                    )[0] or 0.0 for comp in compositions_array
                ]
                csv = ternary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Ternary Data as CSV",
                    data=csv,
                    file_name="ternary_data.csv",
                    mime="text/csv"
                )

                # Save ternary data to SQLite
                output = BytesIO()
                with sqlite3.connect(':memory:') as conn:
                    ternary_df.to_sql('ternary_data', conn, index=False, if_exists='replace')
                    conn.commit()
                    with output:
                        output.write(conn.backup(conn).read())
                        sqlite_data = output.getvalue()
                st.download_button(
                    label="Download Ternary Data as SQLite DB",
                    data=sqlite_data,
                    file_name="ternary_data.db",
                    mime="application/x-sqlite3"
                )
    else:
        st.error("Please select at least one element to generate the ternary diagram.")
