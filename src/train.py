"""Training and evaluation functions."""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from utils import get_logger


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch, lengths in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch, lengths)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch, lengths in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch, lengths)
            loss = criterion(output, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, epochs, learning_rate, device, logger=None):
    """
    Train model for multiple epochs.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to run on
        logger: Optional logger instance
        
    Returns:
        Dictionary with training history
    """
    if logger is None:
        logger = get_logger('train')
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    logger.info("Starting Full Training...")
    
    for epoch in range(epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation/test phase
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1:4d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%} | "
                    f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.1%}")
        
        # Also print to console for real-time monitoring
        # print(f"Epoch {epoch+1:4d}/{epochs} | "
        #       f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%} | "
        #       f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.1%}")
    
    return history


def split_data(df, test_size=0.30, random_state=42):
    """
    Split data into train and test sets while preserving segment integrity.
    
    Args:
        df: DataFrame with segment_id and label columns
        test_size: Proportion of data for test set
        random_state: Random seed
        
    Returns:
        train_df, test_df
    """
    # Get unique segments and their labels for splitting
    segment_metadata = df.drop_duplicates(subset=['segment_id'])[['segment_id', 'label']]
    
    # Perform Stratified Split on the IDs
    train_ids, test_ids = train_test_split(
        segment_metadata['segment_id'], 
        test_size=test_size, 
        random_state=random_state, 
        stratify=segment_metadata['label']
    )
    
    # Create the actual DataFrames
    train_df = df[df['segment_id'].isin(train_ids)].copy()
    test_df = df[df['segment_id'].isin(test_ids)].copy()
    
    # Verification
    logger = get_logger('train')
    
    logger.info(f"Total Segments: {len(segment_metadata)}")
    logger.info(f"Train Segments: {len(train_ids)} ({len(train_ids)/len(segment_metadata):.1%})")
    logger.info(f"Test Segments:  {len(test_ids)} ({len(test_ids)/len(segment_metadata):.1%})")
    
    logger.info("\n--- Train Distribution (Number of Flags) ---")
    logger.info(str(train_df.drop_duplicates('segment_id')['label'].value_counts()))
    
    logger.info("\n--- Test Distribution (Number of Flags) ---")
    logger.info(str(test_df.drop_duplicates('segment_id')['label'].value_counts()))
    
    # Also print to console
    # print(f"Total Segments: {len(segment_metadata)}")
    # print(f"Train Segments: {len(train_ids)} ({len(train_ids)/len(segment_metadata):.1%})")
    # print(f"Test Segments:  {len(test_ids)} ({len(test_ids)/len(segment_metadata):.1%})")
    
    # print("\n--- Train Distribution (Number of Flags) ---")
    # print(train_df.drop_duplicates('segment_id')['label'].value_counts())
    
    # print("\n--- Test Distribution (Number of Flags) ---")
    # print(test_df.drop_duplicates('segment_id')['label'].value_counts())
    
    return train_df, test_df

