"""Dataset classes for time series classification."""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder


class TimeSeriesDataset(Dataset):
    """Dataset for variable-length time series sequences."""
    
    def __init__(self, df, label_encoder=None):
        """
        Initialize dataset from DataFrame.
        
        Args:
            df: DataFrame with columns: segment_id, close, label
            label_encoder: Optional pre-fitted LabelEncoder for consistency
        """
        self.segments = []
        self.labels = []
        
        # Group by segment_id to reconstruct the sequences
        # We assume df is already sorted by timestamp per segment
        grouped = df.groupby('segment_id')
        
        for _, group in grouped:
            # Feature: Close price
            prices = group['close'].values.astype(np.float32)
            
            # Normalize per sequence (Z-score) to make patterns scale-invariant
            # This is critical so the model sees the *shape*, not the price level ($10 vs $1000)
            if len(prices) > 1 and prices.std() > 0:
                prices = (prices - prices.mean()) / prices.std()
            else:
                prices = (prices - prices.mean()) # Fallback if flat
                
            self.segments.append(torch.tensor(prices).unsqueeze(1)) # [Seq_Len, 1]
            
            # Label
            label_str = group['label'].iloc[0]
            self.labels.append(label_str)
            
        # Encode labels to integers
        if label_encoder is None:
            self.le = LabelEncoder()
            self.encoded_labels = self.le.fit_transform(self.labels)
        else:
            self.le = label_encoder
            self.encoded_labels = self.le.transform(self.labels)
            
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return self.segments[idx], self.encoded_labels[idx]

    def get_num_classes(self):
        return len(self.le.classes_)
    
    def get_label_encoder(self):
        return self.le


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of (sequence, label) tuples
        
    Returns:
        padded_seqs: Padded sequences [batch, max_seq_len, features]
        labels: Tensor of labels [batch]
        lengths: Tensor of actual sequence lengths [batch]
    """
    sequences, labels = zip(*batch)
    
    # Get lengths for packing (needed for efficiency)
    lengths = torch.tensor([len(s) for s in sequences])
    
    # Pad sequences to the max length in this specific batch
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    labels = torch.tensor(labels).long()
    
    return padded_seqs, labels, lengths

