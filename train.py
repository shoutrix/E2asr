import os
import torch
from data_utils import prepare_datasets, collate_fn, SortedSampler
from E2asr_model import ASRconfig, E2ASR
from torch.utils.data import DataLoader
from trainer import Trainer, CosineScheduler


# HYPER PARAMETERS
data_path = "/speech/shoutrik/torch_exp/E2asr/data/LibriTTS"
expdir = "/speech/shoutrik/torch_exp/E2asr/exp/trial01"
train_set_name = "train"
valid_set_name = "dev_clean"
max_frames = 38400
batch_size = 64
max_epoch = 30
grad_norm_threshold = 1.0
save_last_step_freq = 100
save_global_step_freq = 50000
logging_freq = 100
grad_norm_threshold=1.0
seed=42
accum_grad=2
config = ASRconfig(
    sample_rate= 16000,
    n_fft=512,
    win_length=400,
    hop_length=160,
    n_mels=80,
    center=True,
    time_mask_param=30,
    freq_mask_param=15,
    norm_mean=True,
    norm_var=True,
    model_dim=512,
    feedforward_dim=208,
    dropout=0.1,
    num_heads=8,
    num_layers=18,
    max_len=4992,
)


train_dataset, valid_dataset, stoi, itos, sp = prepare_datasets(data_path, train_set_name, valid_set_name, expdir)
train_sampler = SortedSampler(train_dataset, max_frames, batch_size, seed=42, stft_center=config.center, win_length=config.win_length, hop_length=config.hop_length)
valid_sampler = SortedSampler(valid_dataset, max_frames, batch_size, seed=42, stft_center=config.center, win_length=config.win_length, hop_length=config.hop_length)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=1, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, collate_fn=collate_fn, num_workers=1, pin_memory=True)


vocab_size = len(stoi) + 1
model = E2ASR(config, vocab_size, training=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


trainer = Trainer(model=model,
                  train_loader=train_loader,
                  valid_loader=valid_loader,
                  device=device,
                  expdir=expdir,
                  accum_grad=accum_grad,
                  max_epoch=max_epoch,
                  save_last_step_freq=save_last_step_freq,
                  save_global_step_freq=save_global_step_freq,
                  resume=None,
                  logging_freq=logging_freq,
                  grad_norm_threshold=grad_norm_threshold,
                  seed=seed
                  )

trainer.train()
