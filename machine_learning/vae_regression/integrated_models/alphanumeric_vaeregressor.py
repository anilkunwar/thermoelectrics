import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
import os
import sqlite3
import joblib
import re
import colorsys
from itertools import combinations
import logging
from difflib import SequenceMatcher
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    st.error("PyTorch Geometric is required for GNN classification. Install with: `pip install torch-geometric`")
    st.stop()

# Set up logging
script_dir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(script_dir, 'thermoelectric_app.log'),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define 65 thermoelectric elements
THERMOELECTRIC_ELEMENTS = [
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K',
    'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
    'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Pb', 'Bi'
]  # 65 elements

# Initialize session state
if 'log_buffer' not in st.session_state:
    st.session_state.log_buffer = []
if 'error_summary' not in st.session_state:
    st.session_state.error_summary = []
if 'selected_elements' not in st.session_state:
    st.session_state.selected_elements = []
if 'compositions' not in st.session_state:
    st.session_state.compositions = {}
if 'material_classifications' not in st.session_state:
    st.session_state.material_classifications = None
if 'ann_model' not in st.session_state:
    st.session_state.ann_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_files' not in st.session_state:
    st.session_state.model_files = {
        'gnn_model.pt': os.path.join(script_dir, 'gnn_model.pt'),
        'classifier_scaler.pkl': os.path.join(script_dir, 'classifier_scaler.pkl'),
        'classifier_scaler.pt': os.path.join(script_dir, 'classifier_scaler.pt'),
        'gnn_models.h5': os.path.join(script_dir, 'gnn_models.h5')
    }

def update_log(message):
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    st.session_state.log_buffer.append(log_message)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logger.info(log_message)

# Define valid chemical elements
VALID_ELEMENTS = set(Element.__members__.keys())

# GNN Classifier
class GNNClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return F.softmax(x, dim=-1)

# Standardize material formula
def standardize_material_formula(formula, preserve_stoichiometry=False, canonical_order=True):
    if not formula or not isinstance(formula, str):
        update_log(f"Invalid input formula: {formula}")
        st.session_state.error_summary.append(f"Invalid formula: {formula}")
        return None
    
    formula = re.sub(r'\s+', '', formula)
    formula = re.sub(r'[\[\]\{\}]', '', formula)
    
    doping_pattern = r'(.+?)(?::|doped\s+)([A-Za-z0-9,\.]+)'
    doping_match = re.match(doping_pattern, formula, re.IGNORECASE)
    dopants = None
    if doping_match:
        base_formula, dopants = doping_match.groups()
        formula = base_formula.strip()
        dopants = dopants.split(',')
        update_log(f"Detected doped material: base='{formula}', dopants='{','.join(dopants)}'")
    
    try:
        comp = Composition(formula)
        if not comp.valid:
            update_log(f"Invalid chemical formula '{formula}': not a valid composition")
            st.session_state.error_summary.append(f"Invalid formula '{formula}': not a valid composition")
            return None
        
        elements = comp.elements
        if not all(isinstance(el, Element) for el in elements):
            update_log(f"Invalid elements in formula '{formula}'")
            st.session_state.error_summary.append(f"Invalid elements in formula '{formula}'")
            return None
        
        if preserve_stoichiometry:
            el_amt_dict = comp.get_el_amt_dict()
            standardized_formula = ''.join(
                f"{el}{amt:.2f}" if amt != int(amt) else f"{el}{int(amt)}"
                for el, amt in (sorted(el_amt_dict.items()) if canonical_order else el_amt_dict.items())
            )
        else:
            standardized_formula = comp.reduced_formula
        
        if dopants:
            valid_dopants = []
            for dopant in dopants:
                try:
                    dopant_comp = Composition(dopant.strip())
                    valid_dopants.append(dopant_comp.reduced_formula)
                except Exception as e:
                    update_log(f"Failed to parse dopant '{dopant}' in '{formula}': {e}")
                    st.session_state.error_summary.append(f"Failed to parse dopant '{dopant}' in '{formula}'")
            if valid_dopants:
                standardized_formula = f"{standardized_formula}:{','.join(valid_dopants)}"
        
        update_log(f"Standardized formula '{formula}' to '{standardized_formula}' using pymatgen")
        return standardized_formula
    except Exception as e:
        update_log(f"pymatgen could not parse formula '{formula}': {str(e)}")
        st.session_state.error_summary.append(f"pymatgen failed for '{formula}': {str(e)}")
        return None

# Validate formula
def validate_formula(formula):
    if not formula or not isinstance(formula, str):
        return False
    
    base_formula = re.sub(r':.+', '', formula)
    
    non_chemical_terms = {
        'DFT', 'TOC', 'PDOS', 'UTS', 'TEs', 'PFU', 'CNO', 'DOS', 'III', 
        'S10', 'K35', 'Ca5', 'Sb6', 'Te3', 'Te4', 'Bi2'
    }
    if base_formula.upper() in non_chemical_terms:
        return False
    
    if len(base_formula) <= 2 or re.match(r'^[A-Z](?:-[A-Z]|\.\d+|)$', base_formula):
        return False
    
    try:
        comp = Composition(base_formula)
        if not comp.valid:
            return False
        elements = [el.symbol for el in comp.elements]
        total_atoms = sum(comp.get_el_amt_dict().values())
        if total_atoms < 2:
            return False
        return all(el in VALID_ELEMENTS for el in elements)
    except Exception:
        return False

# Featurize formulas for GNN
def featurize_formulas(formulas, labels=None):
    data_list = []
    valid_formulas = []
    valid_labels = [] if labels is not None else None

    element_properties = {
        el.symbol: [
            float(el.Z or 0),
            float(el.X or 0),
            float(el.group or 0),
            float(el.row or 0),
            float(el.atomic_mass or 0)
        ] for el in Element
    }

    for i, formula in enumerate(formulas):
        if not validate_formula(formula):
            update_log(f"Skipped featurization for invalid formula '{formula}'")
            st.session_state.error_summary.append(f"Invalid formula '{formula}' for featurization")
            continue

        try:
            comp = Composition(formula)
            if not comp.valid:
                update_log(f"Invalid composition for formula '{formula}'")
                continue

            el_amt_dict = comp.get_el_amt_dict()
            el_amt_dict = {k: max(1, round(v)) for k, v in el_amt_dict.items()}
            total_atoms = sum(el_amt_dict.values())
            if total_atoms < 2:
                update_log(f"Formula '{formula}' has fewer than 2 atoms: {el_amt_dict}")
                continue

            species = []
            frac_coords = []
            pos = 0
            for el, amt in el_amt_dict.items():
                for _ in range(int(amt)):
                    species.append(el)
                    frac_coords.append([pos * 0.1, 0, 0])
                    pos += 1

            if len(species) < 2:
                update_log(f"No valid species for formula '{formula}'")
                continue

            lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
            structure = Structure(lattice, species, frac_coords, coords_are_cartesian=False)

            strategy = MinimumDistanceNN(cutoff=10.0)
            sg = StructureGraph.with_local_env_strategy(structure, strategy)

            node_features = []
            for site in structure:
                el = site.specie.symbol
                props = element_properties.get(el, [0.0] * 5)
                node_features.append(props)
            node_features = torch.tensor(node_features, dtype=torch.float32)

            edge_index = []
            edge_weights = []
            adjacency = list(sg.graph.adjacency())
            if not adjacency or len(structure) < 2:
                update_log(f"No edges found for '{formula}'; using fully connected graph")
                for i in range(len(structure)):
                    for j in range(i + 1, len(structure)):
                        edge_index.append([i, j])
                        edge_index.append([j, i])
                        edge_weights.append(1.0)
            else:
                for i, neighbor_dict in enumerate(adjacency):
                    for neighbor_idx, data in neighbor_dict[1].items():
                        edge_index.append([i, neighbor_idx])
                        edge_weights.append(data.get('weight', 1.0))

            if not edge_index:
                update_log(f"No valid edges for formula '{formula}' after fallback")
                continue

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_weights.unsqueeze(-1),
                y=torch.tensor([1 if labels[i] == "p-type" else 0], dtype=torch.long) if labels is not None else None
            )

            data_list.append(data)
            valid_formulas.append(formula)
            if labels is not None:
                valid_labels.append(labels[i])

        except Exception as e:
            update_log(f"Failed to featurize formula '{formula}': {str(e)}")
            st.session_state.error_summary.append(f"Featurization failed for '{formula}': {str(e)}")
            continue

    if not data_list:
        update_log("No valid graph data generated for GNN")
        return [], [], [] if labels is not None else None

    update_log(f"Generated {len(data_list)} valid graph data objects")
    return data_list, valid_formulas, valid_labels if labels is not None else None

# Classify formula using GNN
def classify_formula(formula, material_df, fuzzy_match=False):
    try:
        if not formula.strip():
            update_log("Empty formula input provided")
            return None, "Please enter a valid chemical formula.", None
        
        normalized_formula = standardize_material_formula(formula, preserve_stoichiometry=False)
        if not normalized_formula:
            update_log(f"Invalid chemical formula: {formula}")
            return None, f"Invalid formula: {formula}", None
        
        update_log(f"Normalized formula '{formula}' to '{normalized_formula}'")
        
        if material_df is None or material_df.empty:
            update_log("No material classifications available for formula lookup")
            return None, "No material classification data available.", None
        
        formula_matches = material_df[material_df["material"].str.lower() == normalized_formula.lower()]
        similar_formula = None
        
        if formula_matches.empty and fuzzy_match:
            materials = material_df["material"].unique()
            similarities = [(m, SequenceMatcher(None, normalized_formula.lower(), m.lower()).ratio()) for m in materials]
            best_match, similarity = max(similarities, key=lambda x: x[1]) if similarities else (None, 0)
            if similarity > 0.8:
                formula_matches = material_df[material_df["material"].str.lower() == best_match.lower()]
                similar_formula = best_match
                update_log(f"Fuzzy matched '{normalized_formula}' to '{best_match}' (similarity: {similarity:.2%})")
        
        if not formula_matches.empty:
            classifications = formula_matches["classification"].value_counts()
            total_matches = len(formula_matches)
            paper_ids = formula_matches["paper_id"].unique() if "paper_id" in formula_matches.columns else []
            contexts = formula_matches["context"].tolist() if "context" in formula_matches.columns else []
            
            confidence = {cls: count / total_matches for cls, count in classifications.items()}
            primary_classification = classifications.idxmax()
            confidence_score = confidence.get(primary_classification, 0.0)
            
            update_log(f"Formula '{normalized_formula}' classified as {primary_classification} (confidence: {confidence_score:.2%})")
            return {
                "formula": normalized_formula,
                "classification": primary_classification,
                "confidence": confidence_score,
                "paper_ids": paper_ids.tolist(),
                "count": total_matches,
                "contexts": contexts,
                "all_classifications": confidence
            }, None, similar_formula
        else:
            if st.session_state.ann_model is None:
                update_log("No GNN model available for prediction")
                return None, "No GNN model available for prediction.", None
            
            data_list, valid_formulas, _ = featurize_formulas([normalized_formula])
            if not data_list:
                update_log(f"Failed to featurize formula '{normalized_formula}' for GNN")
                return None, f"Could not featurize formula '{normalized_formula}' for prediction.", None
            
            data = data_list[0]
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            with torch.no_grad():
                prob = st.session_state.ann_model(data).numpy()[0]
            prediction = "p-type" if prob[1] > prob[0] else "n-type"
            confidence = max(prob)

            update_log(f"GNN predicted '{normalized_formula}' as {prediction} (confidence: {confidence:.2%})")
            return {
                "formula": normalized_formula,
                "classification": prediction,
                "confidence": float(confidence),
                "paper_ids": [],
                "count": 0,
                "contexts": [],
                "all_classifications": {"p-type": float(prob[1]), "n-type": float(prob[0])}
            }, None, None
    
    except Exception as e:
        update_log(f"Error classifying formula '{formula}': {str(e)}")
        return None, f"Error classifying formula: {str(e)}", None

# Featurize composition for VAE
def featurize_composition(composition_dict, available_elements, temperature):
    feature_vector = {element: composition_dict.get(element, 0) for element in available_elements}
    feature_vector['temperature(K)'] = temperature
    update_log(f"Feature vector created with {len(feature_vector)} features: {list(feature_vector.keys())}")
    return pd.DataFrame([feature_vector])

# Preprocess data for VAE
def preprocess_new_data(df, scaler):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df)
    X_scaled = scaler.transform(X_imputed)
    update_log(f"Preprocessed data shape: {X_scaled.shape}")
    return X_scaled

# Load material classifications and models
def load_material_data():
    db_path = os.path.join(script_dir, 'thermoelectric_universe.db')
    if not os.path.exists(db_path):
        st.error(f"Database file {db_path} not found. Please ensure it exists.")
        st.session_state.material_classifications = pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
        st.stop()
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='standardized_formulas'")
        if cursor.fetchone():
            material_df = pd.read_sql_query("SELECT * FROM standardized_formulas", conn)
            update_log(f"Loaded {len(material_df)} material classifications from database")
        else:
            material_df = pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
            update_log("No standardized_formulas table found in database")
        conn.close()
        st.session_state.material_classifications = material_df
    except Exception as e:
        st.error(f"Failed to load material classifications: {str(e)}")
        st.session_state.material_classifications = pd.DataFrame(columns=["paper_id", "title", "material", "classification", "context"])
        st.stop()

    if "gnn_model.pt" in st.session_state.model_files and os.path.exists(st.session_state.model_files["gnn_model.pt"]):
        try:
            model = GNNClassifier(input_dim=5, hidden_dim=64, output_dim=2)
            model.load_state_dict(torch.load(st.session_state.model_files["gnn_model.pt"], map_location='cpu'))
            model.eval()
            st.session_state.ann_model = model
            update_log("Loaded GNN model from gnn_model.pt")
        except Exception as e:
            update_log(f"Failed to load GNN model: {str(e)}")
            st.session_state.error_summary.append(f"GNN model load error: {str(e)}")
            st.session_state.ann_model = None
    
    if "classifier_scaler.pkl" in st.session_state.model_files and os.path.exists(st.session_state.model_files["classifier_scaler.pkl"]):
        try:
            st.session_state.scaler = joblib.load(st.session_state.model_files["classifier_scaler.pkl"])
            update_log("Loaded scaler from classifier_scaler.pkl")
        except Exception as e:
            update_log(f"Failed to load scaler: {str(e)}")
            st.session_state.error_summary.append(f"Scaler load error: {str(e)}")
            st.session_state.scaler = StandardScaler()

# Streamlit app
st.set_page_config(page_title="Thermoelectric Material Analysis", layout="wide")
st.title("Thermoelectric Material Analysis Tool")
st.markdown("""
This tool predicts thermoelectric properties using a VAE and regression model, with material classification based on p-type/n-type.
**Date and Time**: 09:52 AM CEST, Sunday, August 31, 2025
**Dependencies**: `pip install streamlit pandas numpy torch torch-geometric sklearn plotly matplotlib pymatgen joblib`
""")

# Load material data and models
load_material_data()

# Sidebar for element selection
with st.sidebar:
    st.header("Element Selection")
    elements = sorted(THERMOELECTRIC_ELEMENTS)
    selected_elements = st.multiselect("Select Elements", elements, default=["Bi", "Te", "Ag"])
    if selected_elements != st.session_state.selected_elements:
        st.session_state.selected_elements = selected_elements
        st.session_state.compositions = {e: st.session_state.compositions.get(e, 1.0) for e in selected_elements}
    
    st.header("Composition Proportions")
    for element in st.session_state.selected_elements:
        st.session_state.compositions[element] = st.slider(
            f"Proportion of {element}", 0.0, 5.0, st.session_state.compositions.get(element, 1.0), 0.1, key=f"comp_{element}"
        )
    
    st.header("Material Type")
    use_manual_material_type = st.checkbox("Manually Specify Material Type", value=False, key="use_manual_material_type")
    if use_manual_material_type:
        material_type = st.selectbox("Select Material Type", ["p-type", "n-type", "Neutral"], key="material_type")
        st.session_state.sign_bias = material_type
    else:
        st.session_state.sign_bias = None
        if st.session_state.selected_elements and sum(st.session_state.compositions.values()) > 0:
            try:
                comp_dict = {e: st.session_state.compositions[e] for e in st.session_state.selected_elements}
                formula = "".join(f"{e}{comp_dict[e]:.2f}" if comp_dict[e] > 0 else "" for e in comp_dict)
                if not formula:
                    st.warning("No valid formula generated from compositions.")
                    st.session_state.sign_bias = 'Neutral'
                else:
                    result, error, similar_formula = classify_formula(formula, st.session_state.material_classifications, fuzzy_match=True)
                    if error:
                        st.error(f"Classification error: {error}")
                        st.session_state.sign_bias = 'Neutral'
                    else:
                        st.session_state.sign_bias = result['classification']
                        st.write(f"Auto-detected Material Type: {result['classification']} (Confidence: {result['confidence']:.2%})")
            except Exception as e:
                st.error(f"Error generating formula: {str(e)}")
                st.session_state.sign_bias = 'Neutral'

# Load VAE and regressor models
vae_model_path = os.path.join(script_dir, 'vae_model.pt')
regressor_model_path = os.path.join(script_dir, 'regressor_model.pt')
scaler_path = os.path.join(script_dir, 'scaler.pkl')
y_scaler_path = os.path.join(script_dir, 'y_scaler.pkl')

try:
    vae_model = torch.load(vae_model_path, map_location='cpu')
    regressor_model = torch.load(regressor_model_path, map_location='cpu')
    scaler = joblib.load(scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    update_log(f"Scaler expects {scaler.n_features_in_} features")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    update_log(f"Error loading models: {str(e)}")
    st.stop()

# Main content
st.header("Input Parameters")
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature (K)", 300, 1000, 300, 10)
with col2:
    carrier_concentration = st.slider("Carrier Concentration (cm⁻³)", 1e17, 1e21, 1e19, 1e17, format="%.2e")

if st.session_state.sign_bias:
    st.write(f"Material Type: {st.session_state.sign_bias}")
else:
    st.warning("No material type selected or detected.")

# Prediction logic
if st.button("Predict Thermoelectric Properties"):
    if not st.session_state.selected_elements:
        st.error("Please select at least one element.")
    elif sum(st.session_state.compositions.values()) == 0:
        st.error("Please set non-zero composition proportions.")
    else:
        try:
            comp_dict = {e: st.session_state.compositions[e] for e in st.session_state.selected_elements}
            formula = "".join(f"{e}{comp_dict[e]:.2f}" if comp_dict[e] > 0 else "" for e in comp_dict)
            comp = Composition(formula)
            update_log(f"Processing formula: {formula}")
            
            # Featurize composition
            df = featurize_composition(comp_dict, THERMOELECTRIC_ELEMENTS, temperature)
            if df.shape[1] != 66:
                update_log(f"Feature vector has {df.shape[1]} features, expected 66")
                st.error(f"Feature vector has {df.shape[1]} features, expected 66")
                st.stop()
            
            X_scaled = preprocess_new_data(df, scaler)
            if X_scaled.shape[1] != 66:
                update_log(f"Scaled data has {X_scaled.shape[1]} features, expected 66")
                st.error(f"Scaled data has {X_scaled.shape[1]} features, expected 66")
                st.stop()
            
            with torch.no_grad():
                vae_model.eval()
                regressor_model.eval()
                latent, _, _ = vae_model(torch.tensor(X_scaled, dtype=torch.float32))
                predictions = regressor_model(latent).numpy()
                predictions = y_scaler.inverse_transform(predictions)
            
            st.subheader("Prediction Results")
            st.write(f"Formula: {formula}")
            st.write(f"Seebeck Coefficient: {predictions[0][0]:.2f} µV/K")
            st.write(f"Electrical Conductivity: {predictions[0][1]:.2f} S/m")
            st.write(f"Thermal Conductivity: {predictions[0][2]:.2f} W/m·K")
            st.write(f"Power Factor: {predictions[0][3]:.2f} µW/m·K²")
            st.write(f"Figure of Merit (ZT): {predictions[0][4]:.2f}")
            
            fig = go.Figure(data=[
                go.Bar(name='Seebeck Coefficient', x=['Value'], y=[predictions[0][0]], yaxis='y', offsetgroup=1),
                go.Bar(name='Electrical Conductivity', x=['Value'], y=[predictions[0][1]], yaxis='y2', offsetgroup=2),
                go.Bar(name='Thermal Conductivity', x=['Value'], y=[predictions[0][2]], yaxis='y3', offsetgroup=3),
                go.Bar(name='Power Factor', x=['Value'], y=[predictions[0][3]], yaxis='y4', offsetgroup=4),
                go.Bar(name='ZT', x=['Value'], y=[predictions[0][4]], yaxis='y5', offsetgroup=5)
            ])
            fig.update_layout(
                yaxis=dict(title='Seebeck (µV/K)', side='left'),
                yaxis2=dict(title='Elec. Cond. (S/m)', overlaying='y', side='right', position=0.85),
                yaxis3=dict(title='Therm. Cond. (W/m·K)', overlaying='y', side='right', position=0.9),
                yaxis4=dict(title='Power Factor (µW/m·K²)', overlaying='y', side='right', position=0.95),
                yaxis5=dict(title='ZT', overlaying='y', side='right', position=1.0),
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            update_log(f"Prediction error: {str(e)}")
            st.session_state.error_summary.append(f"Prediction error: {str(e)}")

st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="logs")
