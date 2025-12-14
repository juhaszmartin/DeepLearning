"""Evaluation and visualization functions."""

import os
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, test_loader, device, label_encoder):
    """
    Evaluate model and return predictions and labels.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run on
        label_encoder: LabelEncoder used for encoding
        
    Returns:
        all_preds: List of predicted labels (encoded)
        all_labels: List of true labels (encoded)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch, lengths in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            output = model(X_batch, lengths)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return all_preds, all_labels


def plot_confusion_matrix(all_labels, all_preds, class_names, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix and save to file.
    
    Args:
        all_labels: True labels
        all_preds: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot (if None, auto-generates from title)
    """
    # Ensure plots directory exists
    os.makedirs('output/plots', exist_ok=True)
    
    # Generate filename from title if not provided
    if save_path is None:
        filename = title.lower().replace(' ', '_').replace('-', '_') + '.png'
        save_path = os.path.join('output/plots', filename)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='jet', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_binary_confusion_matrix(all_labels, all_preds, label_encoder, save_path=None):
    """
    Plot binary confusion matrix (Bullish vs Bearish) and save to file.
    
    Args:
        all_labels: True labels (encoded)
        all_preds: Predicted labels (encoded)
        label_encoder: LabelEncoder to map indices back to class names
        save_path: Path to save the plot (if None, uses default name)
    
    Returns:
        binary_acc: Binary accuracy score
    """
    # Ensure plots directory exists
    os.makedirs('output/plots', exist_ok=True)
    
    # Generate filename if not provided
    if save_path is None:
        save_path = os.path.join('output/plots', 'binary_confusion_matrix_test.png')
    
    # Map specific indices back to high-level categories (Bullish/Bearish)
    idx_to_class = {i: name for i, name in enumerate(label_encoder.classes_)}
    
    def get_sentiment(idx):
        name = idx_to_class[idx]
        if "Bullish" in name:
            return "Bullish"
        elif "Bearish" in name:
            return "Bearish"
        else:
            return "Unknown"
    
    # Convert predictions/labels to binary categories
    binary_labels = [get_sentiment(y) for y in all_labels]
    binary_preds = [get_sentiment(p) for p in all_preds]
    
    # Create the 2x2 Confusion Matrix
    labels_order = ["Bullish", "Bearish"]
    cm_binary = confusion_matrix(binary_labels, binary_preds, labels=labels_order)
    
    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='RdYlGn', cbar=False,
                xticklabels=labels_order, yticklabels=labels_order, annot_kws={"size": 16})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Aggregated Performance: Bullish vs Bearish', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate binary accuracy
    correct = sum(1 for true, pred in zip(binary_labels, binary_preds) if true == pred)
    binary_acc = correct / len(binary_labels)
    
    return binary_acc


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy over epochs and save to file.
    
    Args:
        history: Dictionary with keys 'train_loss', 'train_acc', 'test_loss', 'test_acc'
        save_path: Path to save the plot (if None, uses default name)
    
    Returns:
        save_path: Path where plot was saved
    """
    # Ensure plots directory exists
    os.makedirs('output/plots', exist_ok=True)
    
    # Generate filename if not provided
    if save_path is None:
        save_path = os.path.join('output/plots', 'training_history.png')
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # --- PLOT LOSS ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['test_loss'], label='Test Loss', linestyle='--')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # --- PLOT ACCURACY ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', color='green')
    plt.plot(epochs_range, history['test_acc'], label='Test Acc', linestyle='--', color='orange')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def print_classification_report(all_labels, all_preds, class_names):
    """
    Print classification report.
    
    Args:
        all_labels: True labels
        all_preds: Predicted labels
        class_names: List of class names
    """
    # print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


def plot_train_test_confusion_matrices(model, train_loader, test_loader, device, label_encoder):
    """
    Plot confusion matrices for both training and test sets and save to files.
    
    Args:
        model: Trained model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to run on
        label_encoder: LabelEncoder used for encoding
    
    Returns:
        train_preds, train_labels, test_preds, test_labels: Predictions and labels
    """
    # Evaluate on train set
    train_preds, train_labels = evaluate_model(model, train_loader, device, label_encoder)
    
    # Evaluate on test set
    test_preds, test_labels = evaluate_model(model, test_loader, device, label_encoder)
    
    class_names = label_encoder.classes_
    
    # Plot training confusion matrix
    train_path = plot_confusion_matrix(
        train_labels,
        train_preds,
        class_names,
        title='Confusion Matrix - Training Set',
        save_path='output/plots/confusion_matrix_train.png'
    )
    
    # Plot test confusion matrix
    test_path = plot_confusion_matrix(
        test_labels,
        test_preds,
        class_names,
        title='Confusion Matrix - Test Set',
        save_path='output/plots/confusion_matrix_test.png'
    )
    
    return train_preds, train_labels, test_preds, test_labels

