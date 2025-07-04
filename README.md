# Whisper Finetuning with PyTorch Lightning

This repository provides a complete pipeline for finetuning the official implementation of OpenAI's Whisper model for automatic speech recognition (ASR) using PyTorch Lightning.

## Features

- **Finetuning Support**: Finetune Whisper models (including Turbo) on custom datasets
- **Efficient Training**: Encoder freezing and decoder-only training for faster convergence
- **Memory Optimization**: Optional spectrogram caching to disk for faster data loading
- **Comprehensive Evaluation**: Word Error Rate (WER) and Character Error Rate (CER) metrics
- **Flexible Configuration**: Easy-to-modify training parameters
- **Multi-language Support**: Built-in support for different languages (Arabic example included)


## Project Structure

```
finetune-whisper-pl/
├── train_whisper.py          # Main training script
├── whisper_model_pl.py       # PyTorch Lightning model wrapper
├── whisper_dataset.py        # Dataset and data loading utilities
├── logs/                     # Training logs and checkpoints
├── tmp/                      # Optional spectrogram cache directory
└── data/
    ├── train_data.jsonl      # Training data
    └── val_data.jsonl        # Validation data
```

## Data Format

Your training and validation data should be in JSONL format with the following structure:

```json
{
  "utt": "unique_utterance_id",
  "audio_filepath": "/path/to/audio.wav",
  "text": "transcription text",
  "duration": 15.5,
  "offset": 0.0
}
```

### Required Fields:
- `utt`: Unique identifier for each utterance
- `audio_filepath`: Path to the audio file
- `text`: Ground truth transcription
- `duration`: Duration of the audio segment in seconds
- `offset`: Start time offset in the audio file (seconds)

## Quick Start

1. **Prepare your data** in the required JSONL format
2. **Update the data paths** in `train_whisper.py`:
   ```python
   train_data = 'path/to/your/train_data.jsonl'
   val_data = 'path/to/your/val_data.jsonl'
   ```
3. **Configure training parameters** in the `Config` class
4. **Run training**:
   ```bash
   python train_whisper.py
   ```

## Configuration Options

### Training Parameters

```python
class Config:
    learning_rate = 1e-5              # Learning rate
    weight_decay = 0.01               # Weight decay for regularization
    adam_epsilon = 1e-8               # Adam optimizer epsilon
    warmup_steps = 2000               # Learning rate warmup steps
    train_batch_size = 32             # Training batch size
    val_batch_size = 16               # Validation batch size
    train_num_workers = 32            # Training data loader workers
    val_num_workers = 16              # Validation data loader workers
    num_train_epochs = 200            # Number of training epochs
    gradient_accumulation_steps = 1    # Gradient accumulation steps
    sample_rate = 16000               # Audio sample rate
    seed = 1415                       # Random seed for reproducibility
```

### Model Configuration

- **Model**: Change `model_name` to use different Whisper variants (`tiny`, `base`, `small`, `medium`, `large`, `turbo`)
- **Language**: Set `lang` parameter for target language (e.g., `"ar"` for Arabic, `"en"` for English)
- **Audio Duration**: Adjust `min_duration` and `max_duration` in dataset configuration

## Training Features

### Encoder Freezing
The implementation freezes the Whisper encoder and only trains the decoder, which:
- Reduces training time and memory usage
- Maintains pre-trained audio feature extraction capabilities
- Focuses adaptation on text generation

### Spectrogram Caching
Optional caching of computed mel spectrograms to disk:
- Set `tmp_folder` to enable caching
- Automatically checks disk space before caching
- Significantly speeds up training after first epoch

### Evaluation Metrics
- **Word Error Rate (WER)**: Word-level transcription accuracy
- **Character Error Rate (CER)**: Character-level transcription accuracy
- **Validation Loss**: Cross-entropy loss on validation set

## Monitoring and Checkpointing

### TensorBoard Logging
View training progress:
```bash
tensorboard --logdir=logs/
```

### Model Checkpoints
- Automatically saves top 5 checkpoints based on validation WER
- Checkpoint naming includes epoch, validation loss, WER, and CER
- Stored in `logs/checkpoint/` directory

## Memory and Performance Tips

1. **Batch Size**: Adjust based on available GPU memory
2. **Num Workers**: Set based on CPU cores for optimal data loading
3. **Precision**: Uses 16-bit precision by default for memory efficiency
4. **Caching**: Enable spectrogram caching if you have sufficient disk space
5. **Gradient Accumulation**: Increase if you need larger effective batch sizes

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient accumulation
2. **Slow Data Loading**: Increase num_workers or enable spectrogram caching
3. **Poor Convergence**: Adjust learning rate or increase warmup steps
4. **Sparse Tensor Error**: The code includes a fix for PyTorch DDP sparse tensor issues

### Performance Optimization

- Use SSD storage for faster data loading
- Enable spectrogram caching for repeated training runs
- Monitor GPU utilization and adjust batch size accordingly
- Consider using multiple GPUs with PyTorch Lightning's DDP

## Model Inference

After training, load your finetuned model:

```python
import whisper
from whisper_model_pl import WhisperModelModule

# Load the trained model
model = WhisperModelModule.load_from_checkpoint('path/to/checkpoint.ckpt')
model.eval()

# Transcribe audio
result = model.model.transcribe('path/to/audio.wav')
print(result['text'])
```

## Acknowledgments

- The code in this project was inspired and partially copied from [this notebook](https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz?usp=sharing).
