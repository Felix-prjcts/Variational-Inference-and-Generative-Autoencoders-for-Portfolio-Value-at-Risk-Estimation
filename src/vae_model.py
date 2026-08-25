"""
Variational Autoencoder (VAE) for portfolio returns
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm


class VAE(nn.Module):
    """Variational Autoencoder for multivariate returns"""
    
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        """
        Initialize VAE
        
        Args:
            input_dim: dimension of input (number of indices)
            latent_dim: dimension of latent space
            hidden_dim: dimension of hidden layers
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        """Encode input to latent space"""
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
    def generate_samples(self, n_samples, device='cpu'):
        """
        Generate new samples from the latent space
        
        Args:
            n_samples: number of samples to generate
            device: 'cpu' or 'cuda'
            
        Returns:
            generated samples
        """
        z = torch.randn(n_samples, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self.decode(z)
        return samples


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        recon_x: reconstructed data
        x: original data
        mu: mean of latent distribution
        logvar: log variance of latent distribution
        beta: weight for KL divergence (beta-VAE)
        
    Returns:
        total loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


class VAETrainer:
    """Trainer for VAE model"""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3, beta=1.0):
        """
        Initialize trainer
        
        Args:
            model: VAE model
            device: 'cpu' or 'cuda'
            learning_rate: learning rate for optimizer
            beta: weight for KL divergence
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.beta = beta
        self.history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss, recon_loss, kl_loss = 0, 0, 0
        n_batches = 0
        
        for x_batch in train_loader:
            x_batch = x_batch[0].to(self.device)
            
            # Forward pass
            recon_x, mu, logvar, z = self.model(x_batch)
            
            # Compute loss
            loss, r_loss, kl = vae_loss(recon_x, x_batch, mu, logvar, self.beta)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            recon_loss += r_loss.item()
            kl_loss += kl.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_recon = recon_loss / n_batches
        avg_kl = kl_loss / n_batches
        
        return avg_loss, avg_recon, avg_kl
    
    def validate(self, val_loader):
        """Validate model"""
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
        """
        Train VAE
        
        Args:
            X_train: training data
            X_val: validation data
            n_epochs: number of epochs
            batch_size: batch size
        """
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(torch.FloatTensor(X_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        print(f"Training VAE for {n_epochs} epochs...")
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
        
        print("Training completed!")
