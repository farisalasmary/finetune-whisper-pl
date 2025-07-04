
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
import torchaudio.transforms as at
from pytorch_lightning import LightningModule

from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup
)

def remove_special_tokens(text):
    # Remove Whisper's special tokens and normalize whitespace in one go
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class WhisperModelModule(LightningModule):
    def __init__(self, cfg, model_name="turbo", lang="ar") -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True, task='transcribe')
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(self.model.is_multilingual,
                                                    num_languages=self.model.num_languages,
                                                    language=lang,
                                                    task=self.options.task)
        self.model.train()
        # Freeze the encoder and only train the decoder
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        total_val_wer_distance = 0
        total_val_wer_ref_length = 0
        total_val_cer_distance = 0
        total_val_cer_ref_length = 0
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            hyp_text = remove_special_tokens(self.tokenizer.decode(o))
            ref_text = remove_special_tokens(self.tokenizer.decode(l))
            cer = xer.cer(ref_text, hyp_text)
            wer = xer.wer(ref_text, hyp_text)
            wer_err = wer['distance'] / wer['ref_length']
            cer_err = cer['distance'] / cer['ref_length']
            total_val_wer_distance += wer['distance']
            total_val_wer_ref_length+= wer['ref_length']            
            total_val_cer_distance += cer['distance']
            total_val_cer_ref_length+= cer['ref_length']
            
            print('Hyp:', hyp_text)
            print('Ref:', ref_text)
            print('WER:', wer_err)
            print('CER:', cer_err)
            print('-'*89)
        
        total_wer_err = total_val_wer_distance / total_val_wer_ref_length
        total_cer_err = total_val_cer_distance / total_val_cer_ref_length
        print('Total WER:', total_wer_err)
        print('Total CER:', total_cer_err)
        print('-'*89)

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val_cer", total_cer_err, on_step=True, prog_bar=True, logger=True)
        self.log("val_wer", total_wer_err, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": total_cer_err,
            "wer": total_wer_err,
            "loss": loss
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.cfg.learning_rate,
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = (
                (self.cfg.train_dataset_len // (self.cfg.train_batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
