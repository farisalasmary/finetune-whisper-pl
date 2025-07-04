"""
Whisper Dataset Module

This module provides dataset classes and utilities for loading and preprocessing
audio data for Whisper fine-tuning. It includes efficient audio loading,
spectrogram caching, and proper batching with padding.

Key features:
- Efficient audio loading with librosa
- Optional spectrogram caching to disk
- Automatic data filtering by duration
- Proper tokenization with special tokens
- Batch collation with padding
"""

import os
import torch
import pydub
import shutil
import librosa
import whisper
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def check_disk_space(threshold):
    """
    Check if there's enough free disk space for caching spectrograms.
    
    Args:
        threshold (float): Minimum required free space in gigabytes
        
    Returns:
        bool: True if sufficient space available, False otherwise
    """
    # Get disk usage statistics for the current directory
    total, used, free = shutil.disk_usage('.')
    
    # Convert free bytes to gigabytes (1 GB = 1024^3 bytes)
    free_gb = free / (1024 ** 3)
    
    return free_gb > threshold


def load_audio(audio_file_path, offset, duration, sample_rate=16000):
    """
    Load audio segment from file with specified time range.
    
    Args:
        audio_file_path (str): Path to the audio file
        offset (float): Start time in seconds
        duration (float): Duration in seconds
        sample_rate (int): Target sample rate (default: 16000 Hz for Whisper)
        
    Returns:
        np.ndarray: Audio signal as numpy array
    """
    
    # Load audio using librosa with specified parameters
    audio_signal, _ = librosa.load(
        audio_file_path,
        sr=sample_rate,          # Target sample rate
        mono=True,               # Convert to mono
        offset=offset,           # Start loading from this time
        duration=duration        # Load only this duration
    )
    
    return audio_signal


class WhisperDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing audio-text pairs for Whisper training.
    
    Features:
    - Loads audio segments based on JSONL metadata
    - Computes mel spectrograms using Whisper's preprocessing
    - Optional caching of spectrograms to disk for faster training
    - Automatic filtering by audio duration
    - Proper tokenization with Whisper's special tokens
    """

    def __init__(self, json_file, tokenizer, n_mels,
                       min_duration=5,
                       max_duration=30,
                       tmp_folder='tmp/',
                       storage_threshold_gb=20.0):
        """
        Initialize the WhisperDataset.
        
        Args:
            json_file (str): Path to JSONL file containing audio metadata
            tokenizer: Whisper tokenizer instance
            n_mels (int): Number of mel frequency bins for spectrograms
            min_duration (float): Minimum audio duration in seconds
            max_duration (float): Maximum audio duration in seconds
            tmp_folder (str): Directory for caching spectrograms (None to disable)
            storage_threshold_gb (float): Minimum disk space required for caching
        """
        # Load data from JSONL file
        self.data = pd.read_json(json_file, lines=True)
        
        # Print statistics before filtering
        print('Total samples BEFORE filtration:', len(self.data))
        print('Total duration BEFORE filtration (in hours):', self.data['duration'].sum() / 3600)
        
        # Filter data by duration to remove too short or too long segments
        self.data = self.data[
            (self.data['duration'] >= min_duration) & 
            (self.data['duration'] <= max_duration)
        ].reset_index(drop=True)
        
        # Print statistics after filtering
        print('Total samples AFTER filtration:', len(self.data))
        print('Total duration AFTER filtration (in hours):', self.data['duration'].sum() / 3600)
        
        # Store configuration
        self.tokenizer = tokenizer
        self.n_mels = n_mels
        self.tmp_folder = tmp_folder
        self.storage_threshold_gb = storage_threshold_gb
        
        # Create cache directory if caching is enabled
        if self.tmp_folder:
            os.makedirs(self.tmp_folder, exist_ok=True)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing:
                - mel_spects: Mel spectrogram tensor
                - labels: Target token IDs (shifted decoder input)
                - dec_input_ids: Decoder input token IDs
        """
        # Get sample metadata
        x = self.data.iloc[idx]
        segment_id = x['utt']          # Unique utterance ID
        text = x['text']               # Ground truth transcription
        
        # Check if cached spectrogram exists
        if self.tmp_folder:
            tmp_filepath = f'{self.tmp_folder}/{segment_id}.pt'
            
        # Load cached spectrogram if available
        if self.tmp_folder and os.path.exists(tmp_filepath):
            mel = torch.load(tmp_filepath)
        else:
            # Compute spectrogram from audio file
            audio_filepath = x['audio_filepath']
            offset = x['offset']
            duration = x['duration']
            
            # Load audio segment
            audio_signal = load_audio(audio_filepath, offset, duration, sample_rate=16000)
            
            # Apply Whisper's standard padding/trimming (ensures consistent length)
            audio_signal = whisper.pad_or_trim(audio_signal)
            
            # Compute log-mel spectrogram using Whisper's preprocessing
            mel = whisper.log_mel_spectrogram(audio_signal, n_mels=self.n_mels)
            
            # Cache spectrogram if conditions are met
            if self.tmp_folder and check_disk_space(self.storage_threshold_gb):
                torch.save(mel, tmp_filepath)
        
        # Tokenize text with special tokens
        # Add []"<|startoftranscript|>", "<|ar|>", "<|transcribe|>", "<|notimestamps|>"] + encoded text
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        
        # Create labels by shifting tokens to the right by one and adding "<|endoftext|>"
        labels = text[1:] + [self.tokenizer.eot]
        
        return {
            "mel_spects": mel,          # Mel spectrogram for encoder
            "labels": labels,           # Target tokens for loss computation
            "dec_input_ids": text       # Input tokens for decoder
        }


class WhisperDataCollatorWhithPadding:
    """
    Custom data collator for batching Whisper samples with proper padding.
    
    This collator handles:
    - Concatenating mel spectrograms
    - Padding token sequences to the same length
    - Masking special tokens in loss computation
    """

    def __init__(self, ignored_tokens):
        """
        Initialize the data collator.
        
        Args:
            ignored_tokens (list): List of token IDs to ignore in loss computation
        """
        # Convert to tensor for efficient masking
        self.ignored_tokens = torch.tensor(ignored_tokens)

    def __call__(self, features):
        """
        Collate a batch of samples into padded tensors.
        
        Args:
            features (list): List of sample dictionaries from dataset
            
        Returns:
            dict: Batched and padded data ready for model input
        """
        # Separate different components
        mel_spects, labels, dec_input_ids = [], [], []
        
        for f in features:
            mel_spects.append(f["mel_spects"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        # Concatenate mel spectrograms (add batch dimension)
        mel_spects = torch.concat([mel_spect[None, :] for mel_spect in mel_spects])

        # Calculate lengths for padding
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        
        # Find maximum length for padding
        max_label_len = max(label_lengths + dec_input_ids_length)

        # Pad labels with -100 (ignored in loss computation)
        labels = [
            np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)
        ]

        # Pad decoder input IDs with "<|endoftext|>" token (50257)
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ]

        # Create batch dictionary
        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        # Convert to tensors
        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        
        # Create mask for ignored tokens (special tokens that shouldn't contribute to loss)
        mask = torch.isin(batch['labels'], self.ignored_tokens)
        
        # Skip the first 3 or 4 tokens (depending on the input 'ignored_tokens')
        # to prevent computing loss on special tokens like <|startoftranscript|>
        # Tokens are usually like ("<|startoftranscript|>", "<|ar|>", "<|transcribe|>")
        # or ("<|startoftranscript|>", "<|ar|>", "<|transcribe|>", "<|notimestamps|>")
        # Masking these tokens keeps the original learned features unchanged for the language
        # identification and other important tokens during the original pretraining phase
        batch['labels'][mask] = -100
        
        # Add input spectrograms to batch
        batch["mel_spects"] = mel_spects
        
        return batch