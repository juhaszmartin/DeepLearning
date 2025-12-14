"""Transformer model for time series classification."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [Batch, Seq_Len, d_model]
            
        Returns:
            Input with positional encoding added
        """
        # x shape: [Batch, Seq_Len, d_model]
        # Add PE to the input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series classification."""
    
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.1):
        """
        Initialize transformer model.
        
        Args:
            input_size: Input feature size (typically 1 for univariate time series)
            d_model: Model dimension (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(TimeSeriesTransformer, self).__init__()
        
        # 1. Input Projection
        # Projects your 1 feature (Close price) up to d_model size (e.g. 64)
        self.input_net = nn.Linear(input_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # 3. Transformer Encoder
        # batch_first=True ensures input is [Batch, Seq, Feature]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Classifier
        self.fc = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model

    def forward(self, x, lengths):
        """
        Forward pass.
        
        Args:
            x: Input tensor [Batch, Seq_Len, input_size]
            lengths: Actual sequence lengths [Batch]
            
        Returns:
            Logits [Batch, num_classes]
        """
        # x shape: [Batch, Seq_Len, 1]
        
        # --- SAFETY FIX: Ensure lengths is on correct device ---
        lengths = lengths.to(x.device)

        # --- A. Create Padding Mask ---
        batch_size, max_len, _ = x.shape
        range_tensor = torch.arange(max_len, device=x.device).expand(batch_size, max_len)
        
        # Mask is True where index >= length (padding areas)
        mask = range_tensor >= lengths.unsqueeze(1)
        
        # --- B. Project & Encode ---
        x = self.input_net(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        # src_key_padding_mask expects [Batch, Seq_Len]
        trans_out = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # --- C. Pooling ---
        valid_mask = ~mask.unsqueeze(-1) # Invert mask: True = Valid
        masked_out = trans_out * valid_mask.float()
        
        sum_out = masked_out.sum(dim=1)
        
        # Avoid division by zero
        lengths_safe = lengths.unsqueeze(1).float()
        lengths_safe[lengths_safe == 0] = 1.0 
        mean_pool = sum_out / lengths_safe
        
        # --- D. Classify ---
        logits = self.fc(mean_pool)
        return logits


def print_model_summary(model):
    """Print a summary of model parameters."""
    print("-" * 60)
    print(f"{'Layer':<25} {'Shape':<20} {'Params':<10}")
    print("-" * 60)
    
    total_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue
        
        # Get shape as a list
        shape_list = list(param.shape)
        param_count = param.numel()
        total_params += param_count
        
        print(f"{name:<25} {str(shape_list):<20} {param_count:<10,}")
        
    print("-" * 60)
    print(f"Total Trainable Params: {total_params:,}")
    print("-" * 60)


