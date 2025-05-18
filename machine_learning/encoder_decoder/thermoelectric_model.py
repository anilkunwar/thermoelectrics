import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
from pymatgen.core.composition import Composition
import re
import os

st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load models with explicit paths
encoder = load_model(os.path.join(current_dir, 'encoder_model.h5'))
decoder = load_model(os.path.join(current_dir, 'decoder_model.h5'))  # Fix typo if needed
regressor = load_model(os.path.join(current_dir, 'regressor_model.h5'))
scaler = joblib.load(os.path.join(current_dir, 'scaler.pkl'))
y_scaler = joblib.load(os.path.join(current_dir, 'y_scaler.pkl'))

# Define the base path for model files (relative to thermoelectric_model.py)
#BASE_DIR = os.path.dirname(__file__)

# Load the trained models and scalers
#encoder = load_model(os.path.join(BASE_DIR, 'encoder_model.h5'))
#decoder = load_model(os.path.join(BASE_DIR, 'decoder_model.h5'))
#regressor = load_model(os.path.join(BASE_DIR, 'regressor_model.h5'))
#scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
#y_scaler = joblib.load(os.path.join(BASE_DIR, 'y_scaler.pkl'))

# Load the trained models and scalers
#encoder = tf.keras.models.load_model('encoder_model')
#decoder = tf.keras.models.load_model('decoder_model')
#regressor = joblib.load('regressor.pkl')
#encoder = joblib.load('encoder_model.pkl')
#decoder = joblib.load('decoder_model.pkl')
#regressor = joblib.load('regressor_model.pkl')
# Load models
#encoder = load_model('encoder_model.h5')
#decoder = load_model('decoder_model.h5')
#regressor = load_model('regressor_model.h5')
#scaler = joblib.load('scaler.pkl')
#y_scaler = joblib.load('y_scaler.pkl')

# Define the available elements
available_elements = [
    'Mg', 'Cs', 'Co', 'Zr', 'Se', 'Dy', 'Pb', 'Ga', 'O', 'Sn', 
    'Yb', 'B', 'La', 'Si', 'V', 'Fe', 'S', 'Sc', 'Tl', 'Zn', 
    'Cl', 'Ce', 'Er', 'Nd', 'Pd', 'Y', 'P', 'Ta', 'In', 'Te', 
    'Ru', 'Rb', 'Tm', 'Tb', 'Sb', 'Al', 'Lu', 'Bi', 'Pr', 'Eu', 
    'Sm', 'Ba', 'Cr', 'Sr', 'Ni', 'Ca', 'As', 'Mn', 'Mo', 'Cd', 
    'Ti', 'Nb', 'Hf', 'Gd', 'Ag', 'Ge', 'Li', 'Br', 'Au', 'I', 
    'N', 'Na', 'Cu', 'Ho', 'K'
]

# Define the function to parse formulas
def parse_formula(formula):
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)?'
    elements = re.findall(pattern, formula)
    return list(set([element[0] for element in elements]))

# Define the function to extract and multiply stoichiometry
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

# Function to featurize materials based on available elements
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

# Function to preprocess new data
def preprocess_new_data(df, available_elements):
    features_df = featurize_materials(df, available_elements)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(features_df)
    X_scaled = scaler.transform(X_imputed)
    return X_scaled

# Streamlit UI
st.title("Thermoelectric Material Seebeck Coefficient Prediction")
formula_input = st.text_input("Enter the chemical formula:")
temperature_input = st.number_input("Enter the temperature (K):", min_value=0, max_value=5000, value=300)

if st.button("Predict Seebeck Coefficient"):
    if formula_input:
        # Create a DataFrame for the new input
        new_data = pd.DataFrame({
            'Formula': [formula_input],
            'temperature(K)': [temperature_input]
        })

        # Preprocess new data
        X_scaled = preprocess_new_data(new_data, available_elements)

        # Predict using the encoder and decoder
        z_mean, z_log_var, z = encoder.predict(X_scaled)
        X_reconstructed = decoder.predict(z)

        # Predict thermal conductivity
        y_scaled_pred = regressor.predict(z)
        y_pred = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()

        st.write(f"Predicted Seebeck Coefficient (Scaled): {y_scaled_pred[0][0]:.2f} ")
        st.write(f"Predicted Seebeck Coefficient (Absolute): {y_pred[0]:.2f} μV/K")
    else:
        st.error("Please enter a chemical formula.")

