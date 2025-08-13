
import torch
import torch.nn as nn

class SudokuEBM(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, hidden_dim, 3, padding=1),  # [B,9,9,9] → [B,hd,9,9]
            nn.Softplus(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Softplus(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.Softplus(),
            nn.Flatten(),
            nn.Linear(hidden_dim*9*9, 1)           # energy scalar
        )
    def forward(self, x):
        
        ### input batch [B, 9, 9, 9]: we treat the one-hot dimension as channels 
        ### ouput [B]
        return self.net(x).squeeze(-1)  # [B]
