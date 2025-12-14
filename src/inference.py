"""Inference script to evaluate trained model on test set."""

import os
import io
import sys
import pickle
import torch
from torch.utils.data import DataLoader

from src.utils import setup_logger, get_logger
from src.config import Config
from src.data_loader import load_all_labels, build_timeseries_df_local, remove_outliers
from src.dataset import TimeSeriesDataset, collate_fn
from src.model import TimeSeriesTransformer
from src.train import split_data
from src.evaluate import (
    evaluate_model, 
    plot_confusion_matrix, 
    plot_binary_confusion_matrix,
    print_classification_report
)


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    logger = get_logger('inference')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_config = checkpoint['model_config']
    num_classes = model_config['num_classes']
    
    # Create model with same architecture
    model = TimeSeriesTransformer(
        input_size=model_config['input_size'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        num_classes=num_classes,
        dropout=model_config['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Model loaded successfully.")
    return model, checkpoint['label_encoder_classes']


def main():
    """Run inference on test set."""
    # Use the same log file as training
    logger = setup_logger(name='inference', log_file='log/run.log')
    
    logger.info("=" * 60)
    logger.info("Running Inference on Test Set")
    logger.info("=" * 60)
    
    model_path = os.path.join('models', 'best_model.pth')
    split_info_path = os.path.join('models', 'split_info.pkl')
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train the model first using main.py")
        return
    
    if not os.path.exists(split_info_path):
        logger.error(f"Split info not found at {split_info_path}. Please train the model first using main.py")
        return
    
    # Load split information
    logger.info("Loading split information...")
    with open(split_info_path, 'rb') as f:
        split_info = pickle.load(f)
    
    train_segment_ids = set(split_info['train_segment_ids'])
    test_segment_ids = set(split_info['test_segment_ids'])
    root_dir = split_info['root_dir']
    
    # Reload and prepare data (same as training)
    logger.info("Loading and preparing data...")
    df_labels = load_all_labels(root_dir)
    
    if df_labels.empty:
        logger.error("No labels found. Exiting.")
        return
    
    df_timeseries = build_timeseries_df_local(df_labels, root_dir)
    
    if df_timeseries.empty:
        logger.error("No timeseries data extracted. Exiting.")
        return
    
    df_clean = remove_outliers(df_timeseries, max_len=Config.MAX_SEGMENT_LEN)
    
    # Split based on saved segment IDs
    train_df = df_clean[df_clean['segment_id'].isin(train_segment_ids)].copy()
    test_df = df_clean[df_clean['segment_id'].isin(test_segment_ids)].copy()
    
    logger.info(f"Train segments: {len(train_segment_ids)}")
    logger.info(f"Test segments: {len(test_segment_ids)}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TimeSeriesDataset(train_df)
    test_dataset = TimeSeriesDataset(test_df, label_encoder=train_dataset.get_label_encoder())
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    logger.info(f"Test batches: {len(test_loader)}")
    logger.info(f"Classes: {train_dataset.get_label_encoder().classes_}")
    
    # Load model
    model, label_classes = load_model(model_path, Config.DEVICE)
    
    # Run inference on test set
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE RESULTS - TEST SET")
    logger.info("=" * 60)
    
    test_preds, test_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=Config.DEVICE,
        label_encoder=train_dataset.get_label_encoder()
    )
    
    # Print classification report
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION REPORT - TEST SET")
    logger.info("=" * 60)
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    print_classification_report(
        test_labels,
        test_preds,
        train_dataset.get_label_encoder().classes_
    )
    report = buffer.getvalue()
    sys.stdout = old_stdout
    logger.info(report)
    
    # Calculate accuracies
    multi_class_acc = sum(p == l for p, l in zip(test_preds, test_labels)) / len(test_labels)
    logger.info(f"Multi-class accuracy: {multi_class_acc:.1%}")
    
    # Binary accuracy
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    binary_acc = plot_binary_confusion_matrix(
        test_labels,
        test_preds,
        train_dataset.get_label_encoder()
    )
    binary_report = buffer.getvalue()
    sys.stdout = old_stdout
    logger.info(binary_report)
    logger.info(f"Binary Accuracy (Bull vs Bear): {binary_acc:.1%}")
    
    logger.info("=" * 60)
    
    # Plot confusion matrices
    logger.info("\nGenerating confusion matrices...")
    conf_path = plot_confusion_matrix(
        test_labels,
        test_preds,
        train_dataset.get_label_encoder().classes_,
        title='Confusion Matrix - Test Set',
        save_path='plots/confusion_matrix_test_inference.png'
    )
    logger.info(f"Confusion matrix saved to {conf_path}")
    
    binary_path = plot_binary_confusion_matrix(
        test_labels,
        test_preds,
        train_dataset.get_label_encoder(),
        save_path='plots/binary_confusion_matrix_inference.png'
    )
    logger.info(f"Binary confusion matrix saved to plots/binary_confusion_matrix_inference.png")
    
    logger.info("\nInference completed successfully!")


if __name__ == "__main__":
    main()

