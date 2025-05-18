import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
import re
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")
st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir())
st.write("PyTorch version:", torch.__version__)
st.write("Streamlit version:", st.__version__)
st.write("Numpy version:", np.__version__)

# Preprocessing function
def preprocess_data(df):
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

    st.write(f"Initial Minimum seebeck coefficient(μV/K): {output_feature_cleaned.min()}")
    st.write(f"Initial Maximum seebeck coefficient(μV/K): {output_feature_cleaned.max()}")

    mask = (output_feature_cleaned >= -1174.0) & (output_feature_cleaned <= 1052.0)
    input_features_cleaned = input_features_cleaned[mask]
    output_feature_cleaned = output_feature_cleaned[mask]
    valid_indices = valid_indices[mask]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(input_features_cleaned)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(output_feature_cleaned.reshape(-1, 1)).ravel()

    st.write("Imputed Seebeck Coefficient Values (Scaled):")
    st.write(f"Scaled Minimum seebeck coefficient(μV/K): {y_scaled.min()}")
    st.write(f"Scaled Maximum seebeck coefficient(μV/K): {y_scaled.max()}")

    if X_scaled.shape[1] != 66:
        raise ValueError(f"Expected 66 features, got {X_scaled.shape[1]}")

    return X_scaled, y_scaled, scaler, y_scaler, input_features_cleaned, output_feature_cleaned, valid_indices

# VAE Model
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

# Regressor Model
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

# VAE Loss Function
def vae_loss(recon_x, x, mu, log_var):
    mse_loss = nn.MSELoss(reduction='mean')(recon_x, x)
    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return mse_loss + kl_div

# Custom callback equivalent for PyTorch
class TrainingHistory:
    def __init__(self, name):
        self.name = name
        self.history = {'loss': [], 'mse': [], 'val_loss': [], 'val_mse': []}

    def append(self, loss, mse, val_loss, val_mse):
        self.history['loss'].append(loss)
        self.history['mse'].append(mse)
        self.history['val_loss'].append(val_loss)
        self.history['val_mse'].append(val_mse)

    def save(self):
        df = pd.DataFrame(self.history)
        df.to_csv(f'{self.name}_training_history.csv', index=False)

# Training Loop for VAE
def train_vae(model, train_loader, optimizer, scheduler, epochs=10):
    model.train()
    history = TrainingHistory('vae')
    placeholder = st.empty()
    
    for epoch in range(epochs):
        train_loss = 0
        train_mse = 0
        val_loss = 0
        val_mse = 0
        train_batches = 0
        val_batches = 0

        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(data)
            loss = vae_loss(recon, data.view(-1, model.input_dim), mu, log_var)
            mse = nn.MSELoss(reduction='mean')(recon, data.view(-1, model.input_dim))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mse += mse.item()
            train_batches += 1

        model.eval()
        with torch.no_grad():
            for data, _ in train_loader:
                data = data.to(device)
                recon, mu, log_var = model(data)
                loss = vae_loss(recon, data.view(-1, model.input_dim), mu, log_var)
                mse = nn.MSELoss(reduction='mean')(recon, data.view(-1, model.input_dim))
                val_loss += loss.item()
                val_mse += mse.item()
                val_batches += 1
        model.train()

        scheduler.step()
        train_loss /= train_batches
        train_mse /= train_batches
        val_loss /= val_batches
        val_mse /= val_batches

        history.append(train_loss, train_mse, val_loss, val_mse)
        placeholder.text(f"Epoch {epoch+1}, VAE Loss: {train_loss:.4f}, VAE MSE: {train_mse:.4f}, Val VAE Loss: {val_loss:.4f}, Val VAE MSE: {val_mse:.4f}")

    history.save()
    return history.history

# Training Loop for Regressor
def train_regressor(model, train_loader, optimizer, scheduler, epochs=100):
    model.train()
    history = TrainingHistory('regressor')
    placeholder = st.empty()
    
    for epoch in range(epochs):
        train_loss = 0
        train_mse = 0
        val_loss = 0
        val_mse = 0
        train_batches = 0
        val_batches = 0

        for z, y in train_loader:
            z, y = z.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(z)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mse += loss.item()
            train_batches += 1

        model.eval()
        with torch.no_grad():
            for z, y in train_loader:
                z, y = z.to(device), y.to(device)
                pred = model(z)
                loss = nn.MSELoss()(pred, y)
                val_loss += loss.item()
                val_mse += loss.item()
                val_batches += 1
        model.train()

        scheduler.step()
        train_loss /= train_batches
        train_mse /= train_batches
        val_loss /= val_batches
        val_mse /= val_batches

        history.append(train_loss, train_mse, val_loss, val_mse)
        placeholder.text(f"Epoch {epoch+1}, Regressor Loss: {train_loss:.4f}, Regressor MSE: {train_mse:.4f}, Val Regressor Loss: {val_loss:.4f}, Val Regressor MSE: {val_mse:.4f}")

    history.save()
    return history.history

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
    ax.set_thetagrids(np.degrees(angles[:-1]), [f'Latent Dim {i+1}' for i in range(num_vars)])
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    return fig

# Load and preprocess data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    X_scaled, y_scaled, scaler, y_scaler, input_features_cleaned, output_feature_cleaned, valid_indices = preprocess_data(df)

    st.write(f"Final Minimum temperature(K): {input_features_cleaned[:, -1].min()}")
    st.write(f"Final Maximum temperature(K): {input_features_cleaned[:, -1].max()}")
    st.write(f"Final Minimum seebeck coefficient(μV/K): {output_feature_cleaned.min()}")
    st.write(f"Final Maximum seebeck coefficient(μV/K): {output_feature_cleaned.max()}")
    st.write(f"Original number of rows: {len(df)}")
    st.write(f"Number of rows after exclusion: {len(input_features_cleaned)}")

    X_tensor = torch.FloatTensor(X_scaled).to(device)
    y_tensor = torch.FloatTensor(y_scaled).reshape(-1, 1).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    vae = VAE(input_dim=66, latent_dim=8).to(device)
    regressor = Regressor(latent_dim=8).to(device)

    vae_optimizer = optim.Adam(vae.parameters(), lr=4e-3)
    regressor_optimizer = optim.Adam(regressor.parameters(), lr=4e-3)

    def lr_schedule(epoch):
        initial_lr = 4.0e-3
        if epoch < 10:
            return initial_lr
        elif epoch < 50:
            return initial_lr * 1.5
        else:
            return initial_lr * 1.5

    vae_scheduler = optim.lr_scheduler.LambdaLR(vae_optimizer, lr_lambda=lr_schedule)
    regressor_scheduler = optim.lr_scheduler.LambdaLR(regressor_optimizer, lr_lambda=lr_schedule)

    st.write("Training VAE...")
    vae_history = train_vae(vae, train_loader, vae_optimizer, vae_scheduler, epochs=10)

    vae.eval()
    with torch.no_grad():
        _, z_mean, _ = vae(X_tensor)
    z_train = z_mean.cpu().numpy()

    st.write("Training Regressor...")
    z_tensor = torch.FloatTensor(z_train).to(device)
    regressor_dataset = TensorDataset(z_tensor, y_tensor)
    regressor_loader = DataLoader(regressor_dataset, batch_size=32, shuffle=True)
    regressor_history = train_regressor(regressor, regressor_loader, regressor_optimizer, regressor_scheduler, epochs=100)

    torch.save(vae.state_dict(), 'vae_model.pt')
    torch.save(regressor.state_dict(), 'regressor_model.pt')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(y_scaler, 'y_scaler.pkl')

    # Latent space visualization
    st.write("Visualizing Latent Space...")

    z_scaler = MinMaxScaler()
    z_normalized = z_scaler.fit_transform(z_train)

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_train)

    dominant_elements = df['Formula'].apply(lambda x: Composition(x).get_el_amt_dict().get(max(Composition(x).get_el_amt_dict(), key=Composition(x).get_el_amt_dict().get), 'Unknown') if Composition(x).valid else 'Unknown')
    dominant_elements_filtered = dominant_elements.iloc[valid_indices].values

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
        plot_bgcolor='white'
    )
    st.plotly_chart(fig)

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
        plot_bgcolor='white'
    )
    st.plotly_chart(fig_elements)

    # Matplotlib: 2D Scatter
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    scatter1 = ax1.scatter(z_2d[:, 0], z_2d[:, 1], c=output_feature_cleaned, cmap='plasma', s=50, alpha=0.6)
    ax1.set_xlabel('Latent Dim 1')
    ax1.set_ylabel('Latent Dim 2')
    ax1.set_title('Latent Space (Seebeck Coefficient)')
    plt.colorbar(scatter1, ax=ax1, label='Seebeck Coefficient (μV/K)')
    scatter2 = ax2.scatter(z_2d[:, 0], z_2d[:, 1], c=pd.Categorical(dominant_elements_filtered).codes, cmap='tab20', s=50, alpha=0.6)
    ax2.set_xlabel('Latent Dim 1')
    ax2.set_ylabel('Latent Dim 2')
    ax2.set_title('Latent Space (Dominant Element)')
    plt.tight_layout()
    st.pyplot(fig)

    # Radar Plot (8D)
    st.write("Radar Plot for 8D Latent Space...")
    sample_indices = np.random.choice(len(z_normalized), size=10, replace=False)
    radar_data = z_normalized[sample_indices]
    radar_labels = [f"Sample {i+1} (Seebeck: {output_feature_cleaned[i]:.1f})" for i in sample_indices]
    fig_radar = plot_radar(radar_data, radar_labels, "8D Latent Space Radar Plot")
    st.pyplot(fig_radar)

    # Parallel Coordinates (8D)
    st.write("Parallel Coordinates for 8D Latent Space...")
    parallel_df = pd.DataFrame(z_normalized, columns=[f'Latent Dim {i+1}' for i in range(8)])
    parallel_df['Seebeck'] = output_feature_cleaned
    fig_parallel = px.parallel_coordinates(
        parallel_df,
        color='Seebeck',
        color_continuous_scale='Plasma',
        labels={f'Latent Dim {i+1}': f'Latent Dim {i+1}' for i in range(8)},
        title='8D Latent Space Parallel Coordinates'
    )
    st.plotly_chart(fig_parallel)

    # Training history plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(vae_history['loss'], label='Training Loss')
    ax[0].plot(vae_history['val_loss'], label='Validation Loss')
    ax[0].set_title('VAE Training and Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(vae_history['mse'], label='Training MSE')
    ax[1].plot(vae_history['val_mse'], label='Validation MSE')
    ax[1].set_title('VAE Training and Validation MSE')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('MSE')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('vae_training_history.png')
    st.pyplot(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(regressor_history['loss'], label='Training Loss')
    ax[0].plot(regressor_history['val_loss'], label='Validation Loss')
    ax[0].set_title('Regressor Training and Validation Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(regressor_history['mse'], label='Training MSE')
    ax[1].plot(regressor_history['val_mse'], label='Validation MSE')
    ax[1].set_title('Regressor Training and Validation MSE')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('MSE')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('regressor_training_history.png')
    st.pyplot(fig)

    st.success("Training completed successfully!")
else:
    st.warning("Please upload a CSV file.")