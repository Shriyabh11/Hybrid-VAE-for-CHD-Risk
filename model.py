import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import QuantileTransformer
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BoostedVAE(nn.Module):
    """
    The Variational Autoencoder model class.
    """
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[64, 128]):
        super(BoostedVAE, self).__init__()
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2)])
            prev_dim = hidden_dim
        self.encoder_base = nn.Sequential(*encoder_layers)
        self.fc_mu, self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim), nn.Linear(hidden_dims[-1], latent_dim)
        
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2)])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder_base(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def train_boosted_vae(model, dataloader, epochs=1000, lr=0.0005):
    """
    Training loop for the BoostedVAE model.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for data_batch in dataloader:
            data, = data_batch
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            recon_loss = F.mse_loss(recon_batch, data, reduction='mean')
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + (0.01 + epoch / epochs) * kl_div
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
    return model

def tri_brid_generate_data(vae, real_minority_df, well_behaved_cont, problematic_cont, cat_features, n_samples, latent_dim):
    """
    Generates synthetic data using the three-part hybrid strategy.
    """
    # 1. VAE for well-behaved continuous data
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=max(min(len(real_minority_df)//2, 50), 10))
    real_wb_cont_transformed = qt.fit_transform(real_minority_df[well_behaved_cont])
    real_wb_cont_tensor = torch.FloatTensor(real_wb_cont_transformed).to(device)
    
    vae.eval()
    with torch.no_grad():
        mu_real, _ = vae.encode(real_wb_cont_tensor)
        z = torch.randn(n_samples, latent_dim).to(device) + mu_real.mean(dim=0)
        synthetic_wb_cont_transformed = vae.decode(z).cpu().numpy()
    synthetic_wb_cont_df = pd.DataFrame(qt.inverse_transform(synthetic_wb_cont_transformed), columns=well_behaved_cont)

    # 2. KDE for problematic continuous data
    synthetic_prob_df = pd.DataFrame()
    for col in problematic_cont:
        kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(real_minority_df[col].values[:, np.newaxis])
        samples = kde.sample(n_samples)[:, 0]
        samples = np.round(samples)
        samples[samples < 0] = 0
        synthetic_prob_df[col] = samples

    # 3. Proportional sampling for categorical data
    synthetic_cat_df = pd.DataFrame()
    for col in cat_features:
        props = real_minority_df[col].value_counts(normalize=True)
        synthetic_cat_df[col] = np.random.choice(props.index, size=n_samples, p=props.values)
        
    return pd.concat([synthetic_wb_cont_df, synthetic_prob_df, synthetic_cat_df], axis=1)
