
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
    # Get disk usage statistics for the current directory
    total, used, free = shutil.disk_usage('.')
    # Convert free bytes to gigabytes (1 GB = 1024^3 bytes)
    free_gb = free / (1024 ** 3)
    return free_gb > threshold


def load_audio(audio_file_path, start_time, end_time, sample_rate=16000):
    duration = end_time - start_time
    audio_signal, _ = librosa.load(
        audio_file_path,
        sr=sample_rate,
        mono=True,
        offset=start_time,
        duration=duration
    )
    return audio_signal


class WhisperDataset(Dataset):

    def __init__(self, json_file, tokenizer, n_mels,
                       min_duration=5,
                       max_duration=30,
                       tmp_folder='tmp/',
                       storage_threshold_gb=20.0):

        self.data = pd.read_json(json_file, lines=True)
        print('Total samples BEFORE filtration:', len(self.data))
        print('Total duration BEFORE filtration (in hours):', self.data['duration'].sum() / 3600)
        self.data = self.data[(self.data['duration'] >= min_duration) & (self.data['duration'] <= max_duration)].reset_index(drop=True)
        print('Total samples AFTER filtration:', len(self.data))
        print('Total duration AFTER filtration (in hours):', self.data['duration'].sum() / 3600)
        self.tokenizer = tokenizer
        self.n_mels = n_mels
        self.tmp_folder = tmp_folder
        self.storage_threshold_gb = storage_threshold_gb
        if self.tmp_folder:
            os.makedirs(self.tmp_folder, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx]
        segment_id = x['utt']
        text = x['text']
        if self.tmp_folder:
            tmp_filepath = f'{self.tmp_folder}/{segment_id}.pt'
        if self.tmp_folder and os.path.exists(tmp_filepath):
            mel = torch.load(tmp_filepath)
        else:
            audio_filepath = x['audio_filepath']
            start_time = x['offset']
            duration = x['duration']
            end_time = start_time + duration
            audio_signal = load_audio(audio_filepath, start_time, end_time, sample_rate=16000)
            audio_signal = whisper.pad_or_trim(audio_signal)
            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio_signal, n_mels=self.n_mels)
            if self.tmp_folder and check_disk_space(self.storage_threshold_gb):
                torch.save(mel, tmp_filepath)
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }

class WhisperDataCollatorWhithPadding:

    def __init__(self, ignored_tokens):
        self.ignored_tokens = torch.tensor(ignored_tokens)

    def __call__(self, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [
                    np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100)
                      for lab, lab_len in zip(labels, label_lengths)
                 ]

        dec_input_ids = [
                          np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257)
                            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
                        ] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        mask = torch.isin(batch['labels'], self.ignored_tokens)
        
        # skip the first 3 or 4 tokens (depending on the input 'ignored_tokens') to prevent computing loss on them
        batch['labels'][mask] = -100
        batch["input_ids"] = input_ids
        return batch
