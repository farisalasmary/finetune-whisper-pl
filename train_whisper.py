"""
Whisper Fine-tuning Training Script

This script orchestrates the fine-tuning of OpenAI's Whisper model using PyTorch Lightning.
It handles model loading, data preparation, training configuration, and monitoring.

Main components:
- Configuration management
- Model initialization with encoder freezing
- Data loading with caching support
- Training loop with checkpointing and logging
"""

import torch
from whisper_model_pl import WhisperModelModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from whisper_dataset import WhisperDataset, WhisperDataCollatorWhithPadding
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class Config:
    """
    Configuration class containing all hyperparameters and training settings.
    
    Attributes:
        learning_rate (float): Learning rate for the optimizer
        weight_decay (float): Weight decay for regularization
        adam_epsilon (float): Epsilon value for Adam optimizer numerical stability
        warmup_steps (int): Number of warmup steps for learning rate scheduler
        train_batch_size (int): Training batch size
        val_batch_size (int): Validation batch size
        train_num_workers (int): Number of workers for training data loader
        val_num_workers (int): Number of workers for validation data loader
        num_train_epochs (int): Total number of training epochs
        gradient_accumulation_steps (int): Steps to accumulate gradients before update
        sample_rate (int): Audio sample rate in Hz
        seed (int): Random seed for reproducibility
    """
    learning_rate = 1e-5             # Conservative learning rate for fine-tuning
    weight_decay = 0.01              # Weight decay for regularization to prevent overfitting
    adam_epsilon = 1e-8              # Small epsilon for numerical stability
    warmup_steps = 2000              # Gradual learning rate increase
    train_batch_size = 32            # Adjust based on GPU memory
    val_batch_size = 16              # Smaller batch for validation
    train_num_workers = 32           # CPU workers for data loading
    val_num_workers = 16             # Fewer workers for validation
    num_train_epochs = 200           # Maximum training epochs
    gradient_accumulation_steps = 1  # Accumulate gradients (effective batch size multiplier)
    sample_rate = 16000              # Whisper's expected sample rate
    seed = 1415                      # For reproducible results

# Initialize configuration
cfg = Config()

# Device selection: use GPU if available, otherwise CPU
device = "gpu" if torch.cuda.is_available() else "cpu"

# Set random seeds for reproducibility across workers
seed_everything(cfg.seed, workers=True)

# Directory for storing logs and checkpoints
logs_dir = "logs/"

# Training configuration
train_name = "whisper_turbo_v1"  # Experiment name for logging
model_name = "turbo"             # Whisper model variant (tiny, base, small, medium, large, turbo)
lang = "ar"                      # Target language code (Arabic in this case)

# TensorBoard logger for monitoring training progress
tflogger = TensorBoardLogger(
    save_dir=logs_dir,
    name=train_name,
)

# Model checkpoint callback to save best models
checkpoint_callback = ModelCheckpoint(
    dirpath=f"{logs_dir}/checkpoint",
    # Checkpoint filename includes key metrics for easy identification
    filename="whisper-turbo-v3-finetuned-{epoch:04d}-{val_loss:.5f}-{val_wer:.5f}-{val_cer:.5f}",
    save_top_k=5,        # Keep top 5 checkpoints based on validation WER
    monitor='val_wer',   # Monitor Word Error Rate for model selection
    mode='min'           # Lower WER is better
)

# List of callbacks for training
callback_list = [
    checkpoint_callback, 
    LearningRateMonitor(logging_interval="epoch")  # Log learning rate changes
]

# Initialize the Whisper model wrapped in PyTorch Lightning module
model = WhisperModelModule(cfg, model_name, lang)

# IMPORTANT: Fix for PyTorch DDP sparse tensor compatibility issue
# Whisper's alignment_heads buffer is sparse, which causes issues with DDP
# Solution: Convert sparse tensor to dense tensor
# See: https://discuss.pytorch.org/t/ddp-no-support-for-sparse-tensor/190375/1
alignment_heads_dense = model.model.get_buffer("alignment_heads").to_dense()
model.model.register_buffer("alignment_heads", alignment_heads_dense, persistent=False)

# Initialize PyTorch Lightning trainer
trainer = Trainer(
    precision=16,                                             # Use 16-bit precision for memory efficiency
    accelerator=device,                                       # Use GPU if available
    max_epochs=cfg.num_train_epochs,                          # Maximum training epochs
    accumulate_grad_batches=cfg.gradient_accumulation_steps,  # Gradient accumulation
    logger=tflogger,                                          # TensorBoard logging
    callbacks=callback_list                                   # Checkpointing and monitoring callbacks
)

# Get tokens to ignore during loss computation (special tokens like <|startoftranscript|>)
ignored_tokens = model.tokenizer.sot_sequence_including_notimestamps

# Training data configuration
train_data = 'YOUR_TRAIN_DATA.jsonl'  # Path to training data in JSONL format

# Optional: folder to cache computed mel spectrograms for faster training
# This significantly speeds up training after the first epoch
# Set to None to compute spectrograms on-the-fly (uses less disk space)
tmp_folder = None  # OR set to 'tmp/train_data' to enable caching
# tmp_folder = 'tmp/train_data'  # Uncomment to enable caching

# Initialize training dataset
train_dataset = WhisperDataset(
    train_data,
    model.tokenizer,
    n_mels=model.model.dims.n_mels, # Number of mel frequency bins
    min_duration=5,                 # Filter out audio shorter than 5 seconds
    max_duration=30,                # Filter out audio longer than 30 seconds
    tmp_folder=tmp_folder,          # Caching directory
    storage_threshold_gb=100.0      # Minimum disk space required for caching
)

# Training data loader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg.train_batch_size,
    num_workers=cfg.train_num_workers,
    shuffle=True,                                              # Shuffle training data
    collate_fn=WhisperDataCollatorWhithPadding(ignored_tokens) # Custom collation with padding
)

# Store dataset length for scheduler calculation
cfg.train_dataset_len = len(train_dataset)

# Validation data configuration
val_data = 'YOUR_VAL_DATA.jsonl'  # Path to validation data in JSONL format

# Optional: folder to cache validation spectrograms
tmp_folder = None  # OR set to 'tmp/val_data' to enable caching
# tmp_folder = 'tmp/val_data'  # Uncomment to enable caching

# Initialize validation dataset
val_dataset = WhisperDataset(
    val_data,
    model.tokenizer,
    n_mels=model.model.dims.n_mels, # Must match training dataset
    min_duration=5,                 # Same filtering as training
    max_duration=30,                # Same filtering as training
    tmp_folder=tmp_folder,          # Caching directory
    storage_threshold_gb=100.0      # Minimum disk space required for caching
)

# Validation data loader
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=cfg.val_batch_size,
    num_workers=cfg.val_num_workers,
    shuffle=False,                                             # Don't shuffle validation data
    collate_fn=WhisperDataCollatorWhithPadding(ignored_tokens) # Same collation as training
)

# Start training
# This will run the full training loop with validation after each epoch
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)