import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core.composition import Composition
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Streamlit UI
st.title("Thermoelectric Material Latent Space and Training History Visualization")
st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir())
st.write("TensorFlow version:", tf.__version__)
st.write("Streamlit version:", st.__version__)
st.write("Numpy version:", np.__version__)

# Preprocessing function
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

    st.write("Imputed Seebeck Coefficient Values (Absolute):")
    st.write(output_feature_imputed)

    iso_forest = IsolationForest(contamination=0.1)
    outliers = iso_forest.fit_predict(input_features_imputed) == -1
    input_features_cleaned = input_features_imputed[~outliers]
    output_feature_cleaned = output_feature_imputed[~outliers]
    valid_indices = np.where(~outliers)[0]

    st.write(f"Initial Minimum Seebeck Coefficient (μV/K): {output_feature_cleaned.min()}")
    st.write(f"Initial Maximum Seebeck Coefficient (μV/K): {output_feature_cleaned.max()}")

    mask = (output_feature_cleaned >= -1174.0) & (output_feature_cleaned <= 1052.0)
    input_features_cleaned = input_features_cleaned[mask]
    output_feature_cleaned = output_feature_cleaned[mask]
    valid_indices = valid_indices[mask]

    X_scaled = scaler.transform(input_features_cleaned)
    y_scaled = y_scaler.transform(output_feature_cleaned.reshape(-1, 1)).ravel()

    st.write("Imputed Seebeck Coefficient Values (Scaled):")
    st.write(f"Scaled Minimum Seebeck Coefficient (μV/K): {y_scaled.min()}")
    st.write(f"Scaled Maximum Seebeck Coefficient (μV/K): {y_scaled.max()}")

    if X_scaled.shape[1] != 66:
        raise ValueError(f"Expected 66 features, got {X_scaled.shape[1]}")

    return X_scaled, y_scaled, output_feature_cleaned, valid_indices

# Radar Plot Function
def plot_radar(data, labels, title, max_samples=10):
    num_vars = data.shape[1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i in range(min(max_samples, len(data))):
        values = data[i].tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.2, label=labels[i])
        ax.plot(angles, values, linewidth=2)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), [f'Latent Dim {i+1}' for i in range(num_vars)], fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    return fig

# Training History Plot (Matplotlib)
def plot_training_history_matplotlib(history_df, title, filename):
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('ggplot')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history_df['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history_df['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{title} Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(history_df['mean_squared_error'], label='Training MSE', linewidth=2)
    ax2.plot(history_df['val_mean_squared_error'], label='Validation MSE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title(f'{title} MSE', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    return fig

# Training History Plot (Plotly)
def plot_training_history_plotly(history_df, title):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df['loss'],
        name='Training Loss',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df['val_loss'],
        name='Validation Loss',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df['mean_squared_error'],
        name='Training MSE',
        line=dict(width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df['val_mean_squared_error'],
        name='Validation MSE',
        line=dict(width=2, dash='dash')
    ))

    fig.update_layout(
        title=f'{title} Training Metrics',
        xaxis_title='Epoch',
        yaxis_title='Value',
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black'),
        plot_bgcolor='white',
        font=dict(size=12),
        legend=dict(font=dict(size=10))
    )
    
    return fig

# Load model and scalers
script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    encoder = tf.keras.models.load_model(os.path.join(script_dir, 'encoder_model.h5'))
    scaler = joblib.load(os.path.join(script_dir, 'scaler.pkl'))
    y_scaler = joblib.load(os.path.join(script_dir, 'y_scaler.pkl'))
except FileNotFoundError:
    st.error("Required files (encoder_model.h5, scaler.pkl, y_scaler.pkl) not found in the script directory.")
    st.stop()

# Upload dataset
dataset_file = st.file_uploader("Upload Dataset CSV", type=["csv"])
if dataset_file is not None:
    df = pd.read_csv(dataset_file)
    
    # Preprocess data
    X_scaled, y_scaled, output_feature_cleaned, valid_indices = preprocess_data(df, scaler, y_scaler)
    
    # Get latent representations
    z_train = encoder.predict(X_scaled)[2]  # z output from encoder
    
    # Normalize latent representations for radar plot
    z_scaler = MinMaxScaler()
    z_normalized = z_scaler.fit_transform(z_train)
    
    # 2D PCA projection
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_train)
    
    # Filter dominant elements
    dominant_elements = df['Formula'].apply(
        lambda x: Composition(x).get_el_amt_dict().get(
            max(Composition(x).get_el_amt_dict(), key=Composition(x).get_el_amt_dict().get), 'Unknown'
        ) if Composition(x).valid else 'Unknown'
    )
    dominant_elements_filtered = dominant_elements.iloc[valid_indices].values
    
    # Latent space visualizations
    st.write("### Latent Space Visualizations")
    
    # Plotly: 2D Scatter (Seebeck Coefficient)
    fig = px.scatter(
        x=z_2d[:, 0],
        y=z_2d[:, 1],
        color=output_feature_cleaned,
        color_continuous_scale='Plasma',
        labels={'x': 'Latent Dim 1', 'y': 'Latent Dim 2', 'color': 'Seebeck Coefficient (μV/K)'},
        title='Latent Space (Seebeck Coefficient)'
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black'),
        plot_bgcolor='white',
        font=dict(size=12)
    )
    st.plotly_chart(fig)
    fig.write_image(os.path.join(script_dir, 'latent_seebeck_plotly.png'), format='png')
    
    # Plotly: 2D Scatter (Dominant Element)
    fig_elements = px.scatter(
        x=z_2d[:, 0],
        y=z_2d[:, 1],
        color=dominant_elements_filtered,
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={'x': 'Latent Dim 1', 'y': 'Latent Dim 2', 'color': 'Dominant Element'},
        title='Latent Space (Dominant Element)'
    )
    fig_elements.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black'),
        plot_bgcolor='white',
        font=dict(size=12)
    )
    st.plotly_chart(fig_elements)
    fig_elements.write_image(os.path.join(script_dir, 'latent_elements_plotly.png'), format='png')
    
    # Matplotlib: 2D Scatter
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    scatter1 = ax1.scatter(z_2d[:, 0], z_2d[:, 1], c=output_feature_cleaned, cmap='plasma', s=50, alpha=0.6)
    ax1.set_xlabel('Latent Dim 1', fontsize=12)
    ax1.set_ylabel('Latent Dim 2', fontsize=12)
    ax1.set_title('Latent Space (Seebeck Coefficient)', fontsize=14)
    plt.colorbar(scatter1, ax=ax1, label='Seebeck Coefficient (μV/K)', pad=0.02)
    scatter2 = ax2.scatter(z_2d[:, 0], z_2d[:, 1], c=pd.Categorical(dominant_elements_filtered).codes, cmap='tab20', s=50, alpha=0.6)
    ax2.set_xlabel('Latent Dim 1', fontsize=12)
    ax2.set_ylabel('Latent Dim 2', fontsize=12)
    ax2.set_title('Latent Space (Dominant Element)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'latent_2d_matplotlib.png'), dpi=300, bbox_inches='tight')
    st.pyplot(fig)
    
    # Radar Plot (8D)
    st.write("#### 8D Latent Space Radar Plot")
    sample_indices = np.random.choice(len(z_normalized), size=10, replace=False)
    radar_data = z_normalized[sample_indices]
    radar_labels = [f"Sample {i+1} (Seebeck: {output_feature_cleaned[i]:.1f})" for i in sample_indices]
    fig_radar = plot_radar(radar_data, radar_labels, "8D Latent Space Radar Plot")
    plt.savefig(os.path.join(script_dir, 'latent_radar.png'), dpi=300, bbox_inches='tight')
    st.pyplot(fig_radar)
    
    # Parallel Coordinates (8D)
    st.write("#### 8D Latent Space Parallel Coordinates")
    parallel_df = pd.DataFrame(z_normalized, columns=[f'Latent Dim {i+1}' for i in range(8)])
    parallel_df['Seebeck'] = output_feature_cleaned
    fig_parallel = px.parallel_coordinates(
        parallel_df,
        color='Seebeck',
        color_continuous_scale='Plasma',
        labels={f'Latent Dim {i+1}': f'Latent Dim {i+1}' for i in range(8)},
        title='8D Latent Space Parallel Coordinates'
    )
    fig_parallel.update_layout(font=dict(size=12))
    st.plotly_chart(fig_parallel)
    fig_parallel.write_image(os.path.join(script_dir, 'latent_parallel_plotly.png'), format='png')

# Upload training history CSVs
st.write("### Training History Visualizations")
vae_history_file = st.file_uploader("Upload VAE Training History CSV", type=["csv"], key="vae_history")
regressor_history_file = st.file_uploader("Upload Regressor Training History CSV", type=["csv"], key="regressor_history")

if vae_history_file is not None:
    vae_history_df = pd.read_csv(vae_history_file)
    
    # Matplotlib: VAE History
    fig_vae = plot_training_history_matplotlib(vae_history_df, "VAE", os.path.join(script_dir, 'vae_history_matplotlib.png'))
    st.pyplot(fig_vae)
    
    # Plotly: VAE History
    fig_vae_plotly = plot_training_history_plotly(vae_history_df, "VAE")
    st.plotly_chart(fig_vae_plotly)
    fig_vae_plotly.write_image(os.path.join(script_dir, 'vae_history_plotly.png'), format='png')

if regressor_history_file is not None:
    regressor_history_df = pd.read_csv(regressor_history_file)
    
    # Matplotlib: Regressor History
    fig_regressor = plot_training_history_matplotlib(regressor_history_df, "Regressor", os.path.join(script_dir, 'regressor_history_matplotlib.png'))
    st.pyplot(fig_regressor)
    
    # Plotly: Regressor History
    fig_regressor_plotly = plot_training_history_plotly(regressor_history_df, "Regressor")
    st.plotly_chart(fig_regressor_plotly)
    fig_regressor_plotly.write_image(os.path.join(script_dir, 'regressor_history_plotly.png'), format='png')

if dataset_file is not None or vae_history_file is not None or regressor_history_file is not None:
    st.success("Visualizations generated successfully!")
else:
    st.warning("Please upload at least one CSV file (dataset or training history).")
