"""Main entry point for training the transformer model."""

import os
import io
import sys
import pickle
import torch
from torch.utils.data import DataLoader

from utils import setup_logger
from config import Config
from data_loader import load_all_labels, build_timeseries_df_local, remove_outliers
from dataset import TimeSeriesDataset, collate_fn
from model import TimeSeriesTransformer, print_model_summary
from train import train_model, split_data
from evaluate import (
    evaluate_model, 
    plot_confusion_matrix, 
    plot_binary_confusion_matrix, 
    plot_training_history,
    plot_train_test_confusion_matrices,
    print_classification_report
)


def main():
    """Main training pipeline."""
    # Set up logging (outputs to stdout, which will be captured to training_log.txt via Docker redirect)
    logger = setup_logger(name='main', log_file=None)
    
    logger.info("=" * 60)
    logger.info("Starting Bull Flag Detector Training Pipeline")
    logger.info("=" * 60)
    
    # Log hyperparameters
    logger.info("\n" + "=" * 60)
    logger.info("HYPERPARAMETERS")
    logger.info("=" * 60)
    config_dict = Config.to_dict()
    for key, value in config_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60 + "\n")
    
    logger.info(f"Using device: {Config.DEVICE}")
    
    # 1. Load labels
    logger.info("Step 1: Loading labels...")
    df_labels = load_all_labels(Config.ROOT_DIR)
    
    if df_labels.empty:
        logger.error("No labels found. Exiting.")
        return
    
    logger.info(f"Loaded {len(df_labels)} label segments.")
    logger.info(f"Label distribution:\n{df_labels['label'].value_counts()}")
    
    # 2. Build timeseries DataFrame
    logger.info("Step 2: Building timeseries DataFrame...")
    df_timeseries = build_timeseries_df_local(df_labels, Config.ROOT_DIR)
    
    if df_timeseries.empty:
        logger.error("No timeseries data extracted. Exiting.")
        return
    
    logger.info("Data preparation completed successfully.")
    
    # 3. Remove outliers
    logger.info("Step 3: Removing outliers...")
    df_clean = remove_outliers(df_timeseries, max_len=Config.MAX_SEGMENT_LEN)
    logger.info(f"Final dataset contains {len(df_clean)} rows across {df_clean['segment_id'].nunique()} segments.")
    
    # 4. Split data
    logger.info("Step 4: Splitting data...")
    train_df, test_df = split_data(
        df_clean, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE
    )
    
    # 5. Create datasets
    logger.info("Step 5: Creating datasets...")
    train_dataset = TimeSeriesDataset(train_df)
    test_dataset = TimeSeriesDataset(test_df, label_encoder=train_dataset.get_label_encoder())
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    logger.info(f"Classes: {train_dataset.get_label_encoder().classes_}")
    
    # 6. Create model
    logger.info("Step 6: Creating model...")
    logger.info("=" * 60)
    logger.info("MODEL ARCHITECTURE")
    logger.info("=" * 60)
    
    model = TimeSeriesTransformer(
        input_size=Config.INPUT_SIZE,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        num_classes=train_dataset.get_num_classes(),
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # Log model summary
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    print_model_summary(model)
    model_summary = buffer.getvalue()
    sys.stdout = old_stdout
    logger.info(model_summary)
    logger.info("=" * 60 + "\n")
    
    # 7. Train model
    logger.info("Step 7: Training model...")
    logger.info("=" * 60)
    logger.info("TRAINING METRICS")
    logger.info("=" * 60)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        device=Config.DEVICE,
        logger=logger
    )
    
    logger.info("=" * 60)
    
    # 8. Evaluate model
    logger.info("Step 8: Evaluating model...")
    logger.info("=" * 60)
    logger.info("VALIDATION METRICS")
    logger.info("=" * 60)
    
    all_preds, all_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=Config.DEVICE,
        label_encoder=train_dataset.get_label_encoder()
    )
    
    # Log final evaluation results
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION RESULTS - TEST SET")
    logger.info("=" * 60)
    
    # Calculate final accuracy
    multi_class_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    logger.info(f"Test Set Size: {len(all_labels)} samples")
    logger.info(f"Multi-class Accuracy: {multi_class_acc:.1%}")
    
    # Log classification report
    logger.info("\nClassification Report:")
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    print_classification_report(
        all_labels,
        all_preds,
        train_dataset.get_label_encoder().classes_
    )
    report = buffer.getvalue()
    sys.stdout = old_stdout
    logger.info(report)
    
    # Log confusion matrix info
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info("Confusion Matrix (Test Set):")
    logger.info(f"Matrix Shape: {cm.shape}")
    logger.info("Confusion Matrix Values:")
    for i, class_name in enumerate(train_dataset.get_label_encoder().classes_):
        logger.info(f"  {class_name}: {cm[i].tolist()}")
    
    # Log binary accuracy
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    binary_acc = plot_binary_confusion_matrix(
        all_labels,
        all_preds,
        train_dataset.get_label_encoder()
    )
    binary_report = buffer.getvalue()
    sys.stdout = old_stdout
    if binary_report.strip():
        logger.info(binary_report)
    logger.info(f"Binary Accuracy (Bull vs Bear): {binary_acc:.1%}")
    
    logger.info("=" * 60)
    
    # Save model and related data for inference
    logger.info("Saving model and metadata...")
    os.makedirs('output/models', exist_ok=True)
    model_path = os.path.join('output/models', 'best_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': Config.INPUT_SIZE,
            'd_model': Config.D_MODEL,
            'nhead': Config.NHEAD,
            'num_layers': Config.NUM_LAYERS,
            'num_classes': train_dataset.get_num_classes(),
            'dropout': Config.DROPOUT
        },
        'label_encoder_classes': train_dataset.get_label_encoder().classes_,
        'config': Config.to_dict()
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Also save train/test data paths info (we'll regenerate from same split)
    import pickle
    split_info_path = os.path.join('output/models', 'split_info.pkl')
    with open(split_info_path, 'wb') as f:
        pickle.dump({
            'train_segment_ids': train_df['segment_id'].unique().tolist(),
            'test_segment_ids': test_df['segment_id'].unique().tolist(),
            'root_dir': Config.ROOT_DIR,
            'random_state': Config.RANDOM_STATE,
            'test_size': Config.TEST_SIZE
        }, f)
    logger.info(f"Split information saved to {split_info_path}")
    
    # Plot results
    logger.info("Plotting results...")
    history_path = plot_training_history(history)
    logger.info(f"Training history plot saved to {history_path}")
    
    # Plot confusion matrices for both train and test sets
    logger.info("Generating confusion matrices for train and test sets...")
    train_preds, train_labels, test_preds, test_labels = plot_train_test_confusion_matrices(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=Config.DEVICE,
        label_encoder=train_dataset.get_label_encoder()
    )
    logger.info("Confusion matrices saved to output/plots/confusion_matrix_train.png and output/plots/confusion_matrix_test.png")
    
    # Also plot binary confusion matrix for test set
    binary_path = plot_binary_confusion_matrix(
        test_labels,
        test_preds,
        train_dataset.get_label_encoder()
    )
    logger.info(f"Binary confusion matrix saved to output/plots/binary_confusion_matrix_test.png")
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()

