"""Configuration file containing all hyperparameters."""

import torch


class Config:
    """Central configuration for model hyperparameters."""
    
    # Data paths
    ROOT_DIR = "./flags/bullflagdetector"
    
    # Data preprocessing
    MAX_SEGMENT_LEN = 100  # Maximum segment length for outlier removal
    TEST_SIZE = 0.30  # Proportion of data for test set
    RANDOM_STATE = 42  # Random seed for reproducibility
    
    # Training hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 2000
    LEARNING_RATE = 0.0005
    DROPOUT = 0.2
    
    # Model architecture
    INPUT_SIZE = 1  # Number of input features (close price)
    D_MODEL = 32  # Internal dimension (must be divisible by NHEAD)
    NHEAD = 16  # Number of attention heads
    NUM_LAYERS = 2  # Number of transformer encoder layers
    MAX_LEN_POS_ENCODING = 5000  # Maximum sequence length for positional encoding
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def to_dict(cls):
        """Return configuration as a dictionary."""
        return {
            'ROOT_DIR': cls.ROOT_DIR,
            'MAX_SEGMENT_LEN': cls.MAX_SEGMENT_LEN,
            'TEST_SIZE': cls.TEST_SIZE,
            'RANDOM_STATE': cls.RANDOM_STATE,
            'BATCH_SIZE': cls.BATCH_SIZE,
            'EPOCHS': cls.EPOCHS,
            'LEARNING_RATE': cls.LEARNING_RATE,
            'DROPOUT': cls.DROPOUT,
            'INPUT_SIZE': cls.INPUT_SIZE,
            'D_MODEL': cls.D_MODEL,
            'NHEAD': cls.NHEAD,
            'NUM_LAYERS': cls.NUM_LAYERS,
            'MAX_LEN_POS_ENCODING': cls.MAX_LEN_POS_ENCODING,
            'DEVICE': str(cls.DEVICE),
        }

