### Data Preparation

The data preparation process involves several steps to handle the labeled flag pattern data collected from multiple annotators:

1. **Label Loading** (`src/data_loader.py:load_all_labels()`): 
   - Loads all JSON label files from student folders (excluding consensus/sample folders)
   - Handles two different JSON formats: Label Studio standard format and custom simple format
   - Extracts timestamps and converts them to Unix milliseconds (handles both Unix timestamps and ISO date strings)
   - Extracts flag labels from the `timeserieslabels` field

2. **Timeseries Extraction** (`src/data_loader.py:build_timeseries_df_local()`):
   - Matches label JSON files with corresponding CSV files in each student's folder
   - Uses fuzzy matching to handle UUID prefixes that Label Studio may add to filenames
   - Extracts time series segments based on start/end timestamps from labels
   - Standardizes column names and timestamps to ensure consistency

3. **Outlier Removal** (`src/data_loader.py:remove_outliers()`):
   - Removes segments with more than 100 datapoints (configurable via `MAX_SEGMENT_LEN`)
   - This filters out clear outliers that may include entire flag poles or incorrectly labeled regions

4. **Data Normalization** (`src/dataset.py:TimeSeriesDataset`):
   - Performs per-sequence Z-score normalization on close prices
   - This makes patterns scale-invariant so the model focuses on shape rather than price levels

5. **Train/Test Split** (`src/train.py:split_data()`):
   - Performs stratified split (70/30) based on segment IDs to preserve segment integrity
   - Ensures same label distribution in train and test sets

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Martin Áron Juhász
- **Aiming for +1 Mark**: No

### Solution Description

**Problem**: The task is to classify financial time series patterns into bull and bear flag categories. Flags are technical chart patterns that can appear in different variations (Normal, Pennant, Wedge) and directions (Bullish, Bearish), resulting in 6 distinct classes.

**Model Architecture**: The solution uses a Transformer Encoder architecture (`src/model.py:TimeSeriesTransformer`) that:
- Takes univariate time series (close prices) as input
- Projects the input to a higher dimensional space (d_model=32)
- Applies positional encoding to inject sequence order information
- Uses multi-head self-attention (16 heads) with 2 transformer encoder layers
- Performs mean pooling over valid (non-padded) positions
- Outputs logits for 6-way classification

**Training Methodology**:
- Data is split 70/30 (train/test) with stratified sampling to preserve label distribution
- Training uses Adam optimizer with learning rate 0.0005 for 2000 epochs
- Batch size of 32 with custom collation to handle variable-length sequences
- Dropout of 0.2 is applied for regularization
- Loss function is CrossEntropyLoss
- Training metrics (loss and accuracy) and validation metrics (test set performance) are computed every epoch

**Results**: 
- For baseline I used my older LSTM experiments, with around 30-35% 6-way classification accuracy
- For transformer:
- Multi-class accuracy: ~57% (6-way classification)
- Binary accuracy (Bullish vs Bearish): ~78%
- Training time: ~3 minutes on NVIDIA RTX 5070 (Blackwell) for 2000 epochs
- The model performs better at distinguishing bullish vs bearish patterns than at classifying specific flag sub-types


### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t my-dl-project-work-app:1.0 .
```

#### Run

To run the solution, you need to mount two directories:
- **Data directory**: Mount your data folder to `/data` inside the container
- **Output directory**: Mount an output folder to `/app/output` inside the container

**Linux/macOS:**
```bash
docker run --rm --gpus all \
 -v /teljes/eleresi/ut/az/adatokhoz:/data \
 -v /teljes/eleresi/ut/az/outputhoz:/app/output \
 my-dl-project-work-app:1.0 > training_log.txt 2>&1
```

**Windows (Command Prompt/PowerShell):**
```bash
docker run --rm --gpus all ^
 -v C:\eleresi\ut\az\adatokhoz:/data ^
 -v C:\eleresi\ut\az\outputhoz:/app/output ^
 my-dl-project-work-app:1.0 > training_log.txt 2>&1
```

**Important notes:**
- Replace `/teljes/eleresi/ut/az/adatokhoz` (Linux/macOS) or `C:\eleresi\ut\az\adatokhoz` (Windows) with the actual path to your data directory containing the labeled datasets
- Replace `/teljes/eleresi/ut/az/outputhoz` (Linux/macOS) or `C:\eleresi\ut\az\outputhoz` (Windows) with the actual path where you want the output files (models and plots)
- The `> training_log.txt 2>&1` redirects all output (stdout and stderr) to `training_log.txt` file
- After execution, all training logs will be in `training_log.txt`, and generated files (models, plots) will be in your output directory
- If you don't have GPU support, remove the `--gpus all` flag
- The `--rm` flag automatically removes the container after it finishes


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - **`config/`**: Configuration module
        - `config.py`: Centralized configuration class containing all hyperparameters (batch size, epochs, learning rate, model dimensions, etc.)
    - **`data_loader.py`**: Data loading and preprocessing functions
        - `load_all_labels()`: Loads label JSON files from student folders and extracts flag annotations
        - `build_timeseries_df_local()`: Matches labels with CSV files and extracts time series segments
        - `remove_outliers()`: Filters out segments exceeding maximum length threshold
    - **`dataset.py`**: PyTorch dataset implementation
        - `TimeSeriesDataset`: Custom dataset class for variable-length time series sequences with per-sequence normalization
        - `collate_fn()`: Custom collation function to pad variable-length sequences in batches
    - **`model.py`**: Model architecture definitions
        - `PositionalEncoding`: Positional encoding module for transformers
        - `TimeSeriesTransformer`: Transformer encoder model for time series classification
        - `print_model_summary()`: Utility to print model architecture and parameter counts
    - **`train.py`**: Training functions
        - `train_epoch()`: Trains model for one epoch
        - `evaluate()`: Evaluates model on a dataset
        - `train_model()`: Main training loop with logging
        - `split_data()`: Stratified train/test split preserving segment integrity
    - **`evaluate.py`**: Evaluation and visualization functions
        - `evaluate_model()`: Runs inference and returns predictions
        - `plot_confusion_matrix()`: Generates and saves confusion matrix plots
        - `plot_binary_confusion_matrix()`: Generates binary (Bullish/Bearish) confusion matrix
        - `plot_training_history()`: Plots loss and accuracy curves over epochs
        - `plot_train_test_confusion_matrices()`: Generates confusion matrices for both train and test sets
        - `print_classification_report()`: Prints detailed classification metrics
    - **`inference.py`**: Inference script
        - `load_model()`: Loads trained model from checkpoint
        - `main()`: Runs inference on test set using saved model and split information
    - **`utils.py`**: Utility functions
        - `setup_logger()`: Configures logging to stdout (captured by Docker redirect to training_log.txt)
        - `get_logger()`: Retrieves or creates logger instance
    - **`main.py`**: Main entry point that orchestrates the entire pipeline:
            1. Sets up logging
            2. Loads and preprocesses data
            3. Creates datasets and data loaders
            4. Initializes model
            5. Trains model
            6. Evaluates on test set
            7. Saves model and generates plots
    - **`run.sh`**: Shell script wrapper to execute the training pipeline

- **`notebook/`**: Contains Jupyter notebooks for development and experimentation
    - `transformer.ipynb`: Original development notebook (kept for reference)
    - `lstm.ipynb`: LSTM experimentation notebook
    - `data_gather/`: Data collection notebook

- **Root Directory**:
    - `Dockerfile`: Configuration for building Docker image with PyTorch and CUDA support
    - `.dockerignore`: I have stored the data folder locally, this is just so I don't copy that
    - `requirements.txt`: Python dependencies (pandas, numpy, torch, scikit-learn, matplotlib, seaborn, etc.)
    - `README.md`: Project documentation
    - `LICENSE`: License file


I used these locally, but will be mounted when executing:
- **`output/`**: Output directory for generated files (created during execution)
    - **`plots/`**: Generated plot files
        - `training_history.png`: Loss and accuracy curves over epochs
        - `confusion_matrix_train.png`: Training set confusion matrix
        - `confusion_matrix_test.png`: Test set confusion matrix
        - `binary_confusion_matrix_test.png`: Binary (Bullish vs Bearish) confusion matrix
    - **`models/`**: Saved model files
        - `best_model.pth`: Trained model checkpoint with architecture config and weights
        - `split_info.pkl`: Train/test split information for reproducible inference

- **`data/`**: Data directory containing labeled datasets from multiple annotators
    - Each student folder contains CSV files (time series data) and JSON files (label annotations)
    - This folder should be mounted to `/data` when running in Docker
    
- **`training_log.txt`**: Generated log file (created when using Docker redirect)
    - Contains all training information captured from stdout/stderr (hyperparameters, data processing, model architecture, training metrics, validation metrics, final evaluation results)
    - Created automatically when using `> training_log.txt 2>&1` in the Docker run command