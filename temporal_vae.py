import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Input Data
df = pd.read_csv("vae_input_data.csv")
input = df.values.astype(np.float32)

# Normalize the input
mean = input.mean(axis = 0)
std = input.std(axis = 0)
input_norm = (input - mean)/ std

# Convert to Tensor
tensor_data = torch.tensor(input_norm)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size = 64, shuffle = True)

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim = 12, latent_dim = 3):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    


# Loss Function
def loss_function(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction = 'sum')
    kld = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

n_epochs = 50
losses = []

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x_batch = batch[0].to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x_batch)
        loss = loss_function(recon, x_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader.dataset)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")


# Plot Training Loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss of VAE")
plt.grid()
plt.show()

# Save Model
torch.save(model.state_dict(), "temporal_vae_model.pt")

