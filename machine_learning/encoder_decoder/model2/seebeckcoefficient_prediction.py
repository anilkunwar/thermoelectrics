import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
from pymatgen.core.composition import Composition
import re
import os

class VAE(nn.Module):
    def __init__(self, input_dim=66, latent_dim=8):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(2 * 3 * 128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.z_mean = nn.Linear(64, latent_dim)
        self.z_log_var = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2 * 3 * 64),
            nn.ReLU(),
            nn.BatchNorm1d(2 * 3 * 64),
            nn.Unflatten(1, (64, 2, 3)),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 11)),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(6 * 11, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, 1, 6, 11)
        h = self.encoder(x)
        mu = self.z_mean(h)
        log_var = self.z_log_var(h)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

class Regressor(nn.Module):
    def __init__(self, latent_dim=8):
        super(Regressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Dropout(0.3),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.BatchNorm1d(2),
            nn.Dropout(0.3),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

script_dir = os.path.dirname(os.path.abspath(__file__))
vae = VAE().to(device)
regressor = Regressor().to(device)
vae.load_state_dict(torch.load(os.path.join(script_dir, 'vae_model.pt'), map_location=device))
regressor.load_state_dict(torch.load(os.path.join(script_dir, 'regressor_model.pt'), map_location=device))
scaler = joblib.load(os.path.join(script_dir, 'scaler.pkl'))
y_scaler = joblib.load(os.path.join(script_dir, 'y_scaler.pkl'))

available_elements = [
    'Mg', 'Cs', 'Co', 'Zr', 'Se', 'Dy', 'Pb', 'Ga', 'O', 'Sn', 
    'Yb', 'B', 'La', 'Si', 'V', 'Fe', 'S', 'Sc', 'Tl', 'Zn', 
    'Cl', 'Ce', 'Er', 'Nd', 'Pd', 'Y', 'P', 'Ta', 'In', 'Te', 
    'Ru', 'Rb', 'Tm', 'Tb', 'Sb', 'Al', 'Lu', 'Bi', 'Pr', 'Eu', 
    'Sm', 'Ba', 'Cr', 'Sr', 'Ni', 'Ca', 'As', 'Mn', 'Mo', 'Cd', 
    'Ti', 'Nb', 'Hf', 'Gd', 'Ag', 'Ge', 'Li', 'Br', 'Au', 'I', 
    'N', 'Na', 'Cu', 'Ho', 'K'
]

def parse_formula(formula):
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)?'
    elements = re.findall(pattern, formula)
    return list(set([element[0] for element in elements]))

def extract_multiplier_and_replace(input_formula):
    pattern = r'\)(\d*\.?\d*)'
    match = re.search(pattern, input_formula)
    if match:
        multiplier = match.group(1)
        multiplier = float(multiplier) if multiplier else 1.0
        parts = re.split(pattern, input_formula)
        formula_without_multiplier = parts[0]
        content_within_parentheses = formula_without_multiplier.split('(')[-1]
        elements_within_parentheses = re.findall(r'([A-Za-z]+)(\d*\.?\d*)', content_within_parentheses)
        modified_elements = [(element, str(float(stoichiometry) * multiplier) if stoichiometry else '0.0') for element, stoichiometry in elements_within_parentheses]
        modified_formula = formula_without_multiplier.split('(')[0] + ''.join(element + stoichiometry for element, stoichiometry in modified_elements)
        return modified_formula
    else:
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

def preprocess_new_data(df, available_elements):
    features_df = featurize_materials(df, available_elements)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(features_df)
    X_scaled = scaler.transform(X_imputed)
    return X_scaled

st.title("Thermoelectric Material Seebeck Coefficient Prediction")
formula_input = st.text_input("Enter the chemical formula:")
temperature_input = st.number_input("Enter the temperature (K):", min_value=0, max_value=5000, value=300)

if st.button("Predict Seebeck Coefficient"):
    if formula_input:
        new_data = pd.DataFrame({
            'Formula': [formula_input],
            'temperature(K)': [temperature_input]
        })
        X_scaled = preprocess_new_data(new_data, available_elements)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        vae.eval()
        regressor.eval()
        with torch.no_grad():
            _, z_mean, _ = vae(X_tensor)
            y_scaled_pred = regressor(z_mean)
            y_pred = y_scaler.inverse_transform(y_scaled_pred.cpu().numpy().reshape(-1, 1)).ravel()
        
        st.write(f"Predicted Seebeck Coefficient (Scaled): {y_scaled_pred[0].item():.2f}")
        st.write(f"Predicted Seebeck Coefficient (Absolute): {y_pred[0]:.2f} μV/K")
    else:
        st.error("Please enter a chemical formula.")