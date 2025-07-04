"""
Whisper Model PyTorch Lightning Module

This module provides a PyTorch Lightning wrapper for OpenAI's Whisper model,
implementing efficient fine-tuning with encoder freezing.

Key features:
- Decoder-only fine-tuning (encoder frozen)
- WER and CER metric computation
- Configurable optimizer and scheduler
- Comprehensive logging and monitoring
"""

import re
import xer
import torch
import pydub
import whisper
import torchaudio
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
import torchaudio.transforms as at
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup


def remove_special_tokens(text):
    """
    Remove Whisper's special tokens and normalize whitespace.
    
    Whisper uses special tokens like <|startoftranscript|>, <|en|>, <|transcribe|>
    which need to be removed for proper evaluation.
    
    Args:
        text (str): Text containing special tokens
        
    Returns:
        str: Cleaned text without special tokens
    """
    # Remove Whisper's special tokens using regex pattern <|...|>
    text = re.sub(r'<\|[^|]*\|>', '', text)
    
    # Normalize whitespace: replace multiple spaces with single space and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class WhisperModelModule(LightningModule):
    """
    PyTorch Lightning module wrapping OpenAI's Whisper model for fine-tuning.
    
    This implementation freezes the encoder and only trains the decoder,
    which is more efficient and often produces better results for fine-tuning.
    
    Features:
    - Encoder freezing for efficient training
    - Comprehensive evaluation with WER/CER metrics
    - Configurable optimizer and learning rate scheduler
    - Automatic mixed precision support
    """
    
    def __init__(self, cfg, model_name="turbo", lang="ar") -> None:
        """
        Initialize the Whisper model module.
        
        Args:
            cfg: Configuration object containing training parameters
            model_name (str): Whisper model variant (tiny, base, small, medium, large, turbo)
            lang (str): Target language code (e.g., 'ar' for Arabic, 'en' for English)
        """
        super().__init__()
        
        # Configure decoding options for the model
        self.options = whisper.DecodingOptions(
            language=lang,           # Target language
            without_timestamps=True, # Don't predict timestamps
            task='transcribe'        # Task type (transcribe vs translate)
        )
        
        # Load the pre-trained Whisper model
        self.model = whisper.load_model(model_name)
        
        # Initialize tokenizer for the specific language and task
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            self.model.is_multilingual,             # Whether model supports multiple languages
            num_languages=self.model.num_languages, # Number of supported languages
            language=lang,                          # Target language
            task=self.options.task                  # Task type
        )
        
        # Set model to training mode
        self.model.train()
        
        # IMPORTANT: Freeze encoder parameters for efficient finetuning
        # This preserves the pretrained audio feature extraction capabilities
        # while allowing the decoder to adapt to the specific domain/language
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        # Initialize loss function
        # ignore_index=-100 means tokens with value -100 won't contribute to loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Store configuration
        self.cfg = cfg

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Model output
        """
        return self.model(x)

    def training_step(self, batch, batch_id):
        """
        Single training step.
        
        Args:
            batch (dict): Batch containing mel_spects, labels, and dec_input_ids
            batch_id (int): Batch index
            
        Returns:
            torch.Tensor: Training loss
        """
        # Extract batch components
        mel_spects = batch["mel_spects"]              # Mel spectrograms [B, n_mels, time]
        labels = batch["labels"].long()               # Target tokens [B, seq_len]
        dec_input_ids = batch["dec_input_ids"].long() # Decoder input tokens [B, seq_len]

        # Forward pass through frozen encoder
        audio_features = self.model.encoder(mel_spects)

        # Forward pass through decoder (only trainable part)
        out = self.model.decoder(dec_input_ids, audio_features)
        
        # Compute cross-entropy loss
        # Reshape to [batch_size * seq_len, vocab_size] for loss computation
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        
        # Log training loss
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_id):
        """
        Single validation step with comprehensive evaluation.
        
        Args:
            batch (dict): Batch containing mel_spects, labels, and dec_input_ids
            batch_id (int): Batch index
            
        Returns:
            dict: Dictionary containing CER, WER, and loss metrics
        """
        # Extract batch components
        mel_spects = batch["mel_spects"]              # Mel spectrograms [B, n_mels, time]
        labels = batch["labels"].long()               # Target tokens [B, seq_len]
        dec_input_ids = batch["dec_input_ids"].long() # Decoder input tokens [B, seq_len]

        # Forward pass through the model
        audio_features = self.model.encoder(mel_spects)
        out = self.model.decoder(dec_input_ids, audio_features)

        # Compute validation loss
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        # Replace -100 tokens with end-of-transcript token for decoding
        # This is necessary because -100 tokens can't be decoded
        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        # Initialize metrics accumulators
        total_val_wer_distance = 0      # Total WER edit distance
        total_val_wer_ref_length = 0    # Total WER reference length
        total_val_cer_distance = 0      # Total CER edit distance
        total_val_cer_ref_length = 0    # Total CER reference length
        
        # Compute metrics for each sample in the batch
        for o, l in zip(out, labels):
            # Get predicted tokens by taking argmax over vocabulary dimension
            o = torch.argmax(o, dim=1)
            
            # Decode tokens to text
            hyp_text = remove_special_tokens(self.tokenizer.decode(o))      # Hypothesis (predicted)
            ref_text = remove_special_tokens(self.tokenizer.decode(l))      # Reference (ground truth)
            
            # Compute CER (Character Error Rate) and WER (Word Error Rate)
            cer = xer.cer(ref_text, hyp_text)
            wer = xer.wer(ref_text, hyp_text)
            
            # Calculate error rates for this sample
            wer_err = wer['distance'] / wer['ref_length'] if wer['ref_length'] > 0 else 1.0
            cer_err = cer['distance'] / cer['ref_length'] if cer['ref_length'] > 0 else 1.0
            
            # Accumulate metrics for batch-level computation
            total_val_wer_distance += wer['distance']
            total_val_wer_ref_length += wer['ref_length']
            total_val_cer_distance += cer['distance']
            total_val_cer_ref_length += cer['ref_length']
            
            # Print individual sample results for debugging
            print('Hyp:', hyp_text)    # Hypothesis (model prediction)
            print('Ref:', ref_text)    # Reference (ground truth)
            print('WER:', wer_err)     # Word Error Rate for this sample
            print('CER:', cer_err)     # Character Error Rate for this sample
            print('-' * 89)
        
        # Compute total error rates across the entire batch
        total_wer_err = total_val_wer_distance / total_val_wer_ref_length if total_val_wer_ref_length > 0 else 1.0
        total_cer_err = total_val_cer_distance / total_val_cer_ref_length if total_val_cer_ref_length > 0 else 1.0
        
        # Print batch-level metrics
        print('Total WER:', total_wer_err)
        print('Total CER:', total_cer_err)
        print('-' * 89)

        # Log metrics to PyTorch Lightning
        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val_cer", total_cer_err, on_step=True, prog_bar=True, logger=True)
        self.log("val_wer", total_wer_err, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": total_cer_err,
            "wer": total_wer_err,
            "loss": loss
        }

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            tuple: (optimizers, schedulers) for PyTorch Lightning
        """
        model = self.model
        
        # Define parameters that should not have weight decay
        # Typically bias terms and layer normalization parameters
        no_decay = ["bias", "LayerNorm.weight"]
        
        # Group parameters based on whether they should have weight decay
        optimizer_grouped_parameters = [
            {
                # Parameters WITH weight decay (most model parameters)
                "params": [p for n, p in model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                # Parameters WITHOUT weight decay (bias and layer norm)
                "params": [p for n, p in model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Initialize AdamW optimizer
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.learning_rate,    # Learning rate
            eps=self.cfg.adam_epsilon     # Epsilon for numerical stability
        )
        self.optimizer = optimizer

        # Initialize learning rate scheduler with linear warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.cfg.warmup_steps,  # Warmup steps
            num_training_steps=self.t_total          # Total training steps
        )
        self.scheduler = scheduler

        # Return optimizer and scheduler configuration
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        """
        Setup method called by PyTorch Lightning before training.
        
        Computes total training steps needed for the learning rate scheduler.
        
        Args:
            stage (str): Training stage ('fit', 'validate', 'test', or None)
        """
        if stage == 'fit' or stage is None:
            self.t_total = (
                (self.cfg.train_dataset_len // (self.cfg.train_batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
