import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RotationLayer(nn.Module):
    def __init__(self, theta, is_train=True, device='cpu'):
        super(RotationLayer, self).__init__()
        self.is_train = is_train
        self.R = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) * torch.cos(theta) + torch.Tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) * torch.sin(theta) + torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.R = self.R.to(device)

    def forward(self, x):
        if self.is_train:
            return torch.matmul(x, self.R)
        return x

class Encoder(nn.Module):
    def __init__(self, n_vtx, latent_dim, n_units, theta, is_train=True, device='cpu'):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.rotation = RotationLayer(theta=theta, is_train=is_train, device=device)

        self.lstm = nn.LSTM(input_size=n_vtx * 3, hidden_size=n_units, batch_first=True, num_layers=2)
        self.mean = nn.Linear(in_features=n_units * 2, out_features=latent_dim)
        self.log_var = nn.Linear(in_features=n_units * 2, out_features=latent_dim)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x_locations, x_velocity = x[:, :3, :, :], x[:, 3:, :, :]

        x_locations_new = x_locations + self.conv1(x_velocity)
        # [batch_size, seq_len, joints * 3]
        x = x_locations_new.permute(0, 2, 3, 1)
        # rotation layer
        x = self.rotation(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        _, (h_n, _) = self.lstm(x)
        h_n = h_n.reshape(h_n.shape[1], -1)
        z_mean = self.mean(h_n)
        z_log_var = self.log_var(h_n)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, seq_len, target_seq_len, n_vtx, theta, latent_dim=32, n_units=32, is_train=True, device='cpu'):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.target_seq_len = target_seq_len
        self.n_vtx = n_vtx
        self.rotation = RotationLayer(theta=theta, is_train=is_train, device=device)
        self.linear = nn.Linear(latent_dim, n_units)
        self.lstm = nn.LSTM(input_size=n_units, hidden_size=self.n_vtx * 3, batch_first=True, num_layers=2)
    
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.reshape(-1, lstm_out.shape[1], self.n_vtx, 3)
        lstm_out = self.rotation(lstm_out)
        return lstm_out


class VAELSTM(nn.Module):
    def __init__(self, seq_len, target_seq_len, latent_dim, n_units, reduced_joints=False, device='cuda', is_train=True):
        super(VAELSTM, self).__init__()
        self.num_joints = 18 if reduced_joints else 53
        self.target_seq_len = target_seq_len
        self.device = device

        theta = 2 * np.pi * torch.rand(1)

        self.encoder = Encoder(n_vtx=self.num_joints, latent_dim=latent_dim, n_units=n_units, theta=theta, is_train=is_train, device=device)
        self.decoder = Decoder(seq_len=seq_len, target_seq_len=target_seq_len, n_vtx=self.num_joints, theta=-theta, is_train=is_train, device=device, latent_dim=latent_dim, n_units=n_units)

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


class GNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
