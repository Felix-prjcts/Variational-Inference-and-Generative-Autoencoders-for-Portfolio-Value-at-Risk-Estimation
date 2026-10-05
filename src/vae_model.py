"""VAE on multivariate (normalised) returns, MLP encoder/decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # no activation on the output layer: returns can be negative
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    def generate_samples(self, n_samples, device='cpu'):
        """Decode n_samples draws of z ~ N(0, I)."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self.decode(z)
        return samples


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """MSE reconstruction + beta * KL(q(z|x) || N(0, I)).

    Returns (total, reconstruction, kl).
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


class VAETrainer:

    def __init__(self, model, device='cpu', learning_rate=1e-3, beta=1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.beta = beta
        self.history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, recon_loss, kl_loss = 0, 0, 0
        n_batches = 0

        for x_batch in train_loader:
            x_batch = x_batch[0].to(self.device)

            recon_x, mu, logvar, z = self.model(x_batch)
            loss, r_loss, kl = vae_loss(recon_x, x_batch, mu, logvar, self.beta)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            recon_loss += r_loss.item()
            kl_loss += kl.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = recon_loss / n_batches
        avg_kl = kl_loss / n_batches

        return avg_loss, avg_recon, avg_kl

    def validate(self, val_loader):
        self.model.eval()
        total_loss, recon_loss, kl_loss = 0, 0, 0
        n_batches = 0

        with torch.no_grad():
            for x_batch in val_loader:
                x_batch = x_batch[0].to(self.device)
                recon_x, mu, logvar, z = self.model(x_batch)
                loss, r_loss, kl = vae_loss(recon_x, x_batch, mu, logvar, self.beta)

                total_loss += loss.item()
                recon_loss += r_loss.item()
                kl_loss += kl.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = recon_loss / n_batches
        avg_kl = kl_loss / n_batches

        return avg_loss, avg_recon, avg_kl

    def train(self, X_train, X_val, n_epochs, batch_size):
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(1, n_epochs + 1):
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)
            val_loss, val_recon, val_kl = self.validate(val_loader)

            self.history['total_loss'].append(train_loss)
            self.history['recon_loss'].append(train_recon)
            self.history['kl_loss'].append(train_kl)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{n_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")
