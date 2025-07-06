#!/usr/bin/env python3
"""
Whisper Model Converter: PyTorch Lightning to Official Whisper Format

This script converts a fine-tuned Whisper model from PyTorch Lightning checkpoint format
to the official Whisper model format that can be loaded with whisper.load_model().

Usage:
    python convert_whisper.py <whisper_model_name> <pytorch_lightning_checkpoint> <output_file>

Example:
    python convert_whisper.py turbo whisper_finetuned.ckpt whisper_converted.pt
"""

import sys
import torch
import whisper

# Check if the correct number of command line arguments are provided
if len(sys.argv) < 4:
    print('USAGE: python {} OFFICIAL_WHISPER_NAME WHISPER_PL.ckpt FINETUNED_OFFICIAL_WHISPER.pt'.format(sys.argv[0]))
    sys.exit(1)

# Parse command line arguments
whisper_name = sys.argv[1]              # Official whisper model name (e.g., 'base', 'small', 'medium', 'turbo', 'large')
whisper_pl_ckpt_path = sys.argv[2]      # Path to PyTorch Lightning checkpoint file
finetuned_whisper_pt_path = sys.argv[3] # Output path for converted whisper model

# Load the original whisper model to get the model dimensions/configuration
whisper_model = whisper.load_model(whisper_name)

# Load the PyTorch Lightning checkpoint and extract the state dictionary
# weights_only=False allows loading the full checkpoint with metadata
state_dict = torch.load(whisper_pl_ckpt_path, weights_only=False)['state_dict']

# Convert PyTorch Lightning state dict keys to standard Whisper format
# PyTorch Lightning adds 'model.' prefix to all parameter names, so we remove it
modified_state_dict = {}
for k, v in state_dict.items():
    k = k.replace('model.', '')  # Remove 'model.' prefix from parameter names
    modified_state_dict[k] = v

# Create the final whisper model dictionary with the required structure
whisper_pt_dict = {}
whisper_pt_dict['dims'] = whisper_model.dims.__dict__        # Model dimensions/configuration
whisper_pt_dict['model_state_dict'] = modified_state_dict    # Converted model weights

# Save the converted model in official Whisper format
torch.save(whisper_pt_dict, finetuned_whisper_pt_path)

print(f"Successfully converted {whisper_pl_ckpt_path} to {finetuned_whisper_pt_path}")
print(f"Model can now be loaded with: whisper.load_model('{finetuned_whisper_pt_path}')")