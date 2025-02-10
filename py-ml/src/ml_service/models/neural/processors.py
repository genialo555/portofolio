import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import numpy as np

class DeepProcessor(nn.Module):
    """Deep neural processor for complex feature extraction."""
    
    def __init__(self, 
                input_dim: int,
                hidden_dims: List[int],
                dropout: float = 0.3):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        # Construction des couches profondes
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # GELU pour une meilleure performance que ReLU
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
            
        self.deep_layers = nn.Sequential(*layers)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=4,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass avec attention.
        Returns: (processed_features, attention_weights)
        """
        features = self.deep_layers(x)
        attn_output, attn_weights = self.attention(
            features, features, features
        )
        return attn_output, attn_weights


class FastProcessor(nn.Module):
    """Fast neural processor for efficient processing."""
    
    def __init__(self, 
                input_dim: int,
                hidden_dim: int,
                num_groups: int = 8):
        super().__init__()
        
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            groups=num_groups  # Convolution groupée pour la vitesse
        )
        
        self.gn = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=hidden_dim
        )
        
        self.fast_attention = FastAttention(
            dim=hidden_dim,
            heads=4,
            dim_head=hidden_dim // 4
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution rapide
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.gn(x)
        x = F.gelu(x)
        
        # Attention rapide
        return self.fast_attention(x)


class FastAttention(nn.Module):
    """Attention rapide avec approximation linéaire."""
    
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, -1).transpose(1, 2), qkv)
        
        # Attention linéaire approximée
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out) 