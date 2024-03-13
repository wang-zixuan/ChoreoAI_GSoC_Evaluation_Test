import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, n_vtx, latent_dim, n_units):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(n_vtx * 3, n_units)

        self.mean = nn.Linear(in_features=n_units, out_features=latent_dim)
        self.log_var = nn.Linear(in_features=n_units, out_features=latent_dim)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)

        x = F.leaky_relu(self.linear(x))

        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, seq_len, n_vtx, latent_dim=32, n_units=32):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.n_vtx = n_vtx
        self.linear = nn.Linear(latent_dim, n_units)

        self.output_linear = nn.Linear(n_units, self.n_vtx * 3)
    
    def forward(self, x):
        x = self.output_linear(F.relu(self.linear(x)))
        x = x.reshape(x.shape[0], 1, -1, 3)
        return x


class VAELSTM_Static(nn.Module):
    def __init__(self, seq_len, latent_dim, n_units, reduced_joints=False, device='cuda', is_train=True):
        super(VAELSTM_Static, self).__init__()
        self.num_joints = 18 if reduced_joints else 53
        self.device = device

        self.encoder = Encoder(n_vtx=self.num_joints, latent_dim=latent_dim, n_units=n_units)
        self.decoder = Decoder(seq_len=seq_len, n_vtx=self.num_joints, latent_dim=latent_dim, n_units=n_units)

    def sample_z(self, mean, log_var):
        batch, dim = mean.shape
        epsilon = torch.randn(batch, dim).to(self.device)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def forward(self, x):
        # input size: [batch_size, seq_len, joints, 6]

        mean, log_var = self.encoder(x)
        z = self.sample_z(mean, log_var)
        x = self.decoder(z)
        return x, mean, log_var

