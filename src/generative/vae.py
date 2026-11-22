import torch
from torch import nn
class TinyVAE(nn.Module):
    """Small fully-connected VAE for 28x28 grayscale demonstration data."""
    def __init__(self,latent_dim=8):
        super().__init__(); self.enc=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU())
        self.mu=nn.Linear(256,latent_dim); self.logvar=nn.Linear(256,latent_dim)
        self.dec=nn.Sequential(nn.Linear(latent_dim,256),nn.ReLU(),nn.Linear(256,784),nn.Sigmoid())
    def forward(self,x):
        h=self.enc(x); mu=self.mu(h); lv=self.logvar(h); z=mu+torch.randn_like(mu)*torch.exp(.5*lv)
        return self.dec(z).view(-1,1,28,28),mu,lv
def vae_loss(recon,x,mu,logvar):
    return nn.functional.binary_cross_entropy(recon,x,reduction="sum")-.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
