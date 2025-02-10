import torch
import torch.nn as nn
from typing import Tuple, Dict

class TeacherLSTM(nn.Module):
    """Teacher LSTM model with complex architecture."""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=4
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2)
        ) 