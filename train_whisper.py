
import torch
from whisper_model_pl import WhisperModelModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from whisper_dataset import WhisperDataset, WhisperDataCollatorWhithPadding
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class Config:
    learning_rate = 1e-5 #  0.0005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2000
    train_batch_size = 32
    val_batch_size = 16
    train_num_workers = 32
    val_num_workers = 16
    num_train_epochs = 200
    gradient_accumulation_steps = 1
    sample_rate = 16000
    seed = 1415

cfg = Config()

device = "gpu" if torch.cuda.is_available() else "cpu"
seed_everything(cfg.seed, workers=True)

logs_dir = "logs/"

train_name = "whisper_turbo_v1"
model_name = "turbo"
lang = "ar"

tflogger = TensorBoardLogger(
    save_dir=logs_dir,
    name=train_name,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{logs_dir}/checkpoint",
    filename="whisper-turbo-v3-finetuned-{epoch:04d}-{val_loss:.5f}-{val_wer:.5f}-{val_cer:.5f}",
    save_top_k=5, # save top 5 checkpoints,
    monitor='val_wer',
    mode='min'
)

callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
model = WhisperModelModule(cfg, model_name, lang)

# Solution to the error: RuntimeError: No support for sparse tensors
# See: https://discuss.pytorch.org/t/ddp-no-support-for-sparse-tensor/190375/1
alignment_heads_dense = model.model.get_buffer("alignment_heads").to_dense()
model.model.register_buffer("alignment_heads", alignment_heads_dense, persistent=False)

trainer = Trainer(
    precision=16,
    accelerator=device,
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list
)

ignored_tokens = model.tokenizer.sot_sequence_including_notimestamps

train_data = 'YOUR_TRAIN_DATA.jsonl'

# folder to cache computed mel spectrogram for faster use in the upcoming epochs
# tmp_folder = 'tmp/train_data' # specify the folder to cache spectrograms
tmp_folder = None # OR set it to None to compute spectrograms on-the-fly
train_dataset = WhisperDataset(train_data,
                               model.tokenizer,
                               n_mels=model.model.dims.n_mels,
                               min_duration=5,
                               max_duration=30,
                               tmp_folder=tmp_folder,
                               storage_threshold_gb=100.0
                               )
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.train_batch_size,
                                               num_workers=cfg.train_num_workers,
                                               shuffle=True,
                                               collate_fn=WhisperDataCollatorWhithPadding(ignored_tokens)
                                               )
cfg.train_dataset_len = len(train_dataset)

val_data = 'YOUR_VAL_DATA.jsonl'

# folder to cache computed mel spectrogram for faster use in the upcoming epochs
#tmp_folder = 'tmp/val_data' # specify the folder to cache spectrograms
tmp_folder = None # OR set it to None to compute spectrograms on-the-fly
val_dataset = WhisperDataset(val_data,
                             model.tokenizer,
                             n_mels=model.model.dims.n_mels,
                             min_duration=5,
                             max_duration=30,
                             tmp_folder=tmp_folder,
                             storage_threshold_gb=100.0
                             )
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.val_batch_size,
                                             num_workers=cfg.val_num_workers,
                                             shuffle=False,
                                             collate_fn=WhisperDataCollatorWhithPadding(ignored_tokens)
                                             )

trainer.fit(model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            )

