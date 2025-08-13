# In your model/ebm.py file

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A modern residual block with GroupNorm and SiLU activation.
    """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        
        # The main convolutional path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(), # SiLU (Swish) is a smooth, non-monotonic activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels)
        )
        
        # A 1x1 convolution for the skip connection if channel sizes differ
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.final_activation = nn.SiLU()

    def forward(self, x):
        # The skip connection ensures the gradient can flow directly
        # and adds the initial information to the transformed information.
        return self.final_activation(self.main_path(x) + self.skip_connection(x))



# In your model/ebm.py file

class SudokuEBM(nn.Module): # You can rename this to SudokuEBM_ResNet if you like
    def __init__(self, in_channels=9, hidden_dim=128, num_blocks=4):
        super().__init__()
        
        # Initial convolution to project input to the hidden dimension
        self.initial_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # A sequence of residual blocks to learn complex features
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
        
        # Final layers to produce the scalar energy value
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 9 * 9, 1) # Reshape and project to a single energy value
        )

    def forward(self, x):
        # The input x is expected to be [B, 9, 9, 9] where the one-hot
        # dimension is treated as channels. For your data loader which
        # likely produces [B, 9, 81], you'll need to reshape it first.
        
        if x.shape[-1] == 81:
            x = x.view(-1, 9, 9, 9) # Reshape [B, 9, 81] -> [B, 9, 9, 9]

        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.output_head(x)
        
        return x.squeeze(-1) # Return shape [B]