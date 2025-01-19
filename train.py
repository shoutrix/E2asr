import os
import torch
from data_utils import prepare_datasets, collate_fn, SortedSampler
from E2asr_model import ASRconfig, E2ASR
from torch.utils.data import DataLoader
from trainer import Trainer, CosineScheduler
import wandb


# HYPER PARAMETERS
data_path = "/speech/shoutrik/torch_exp/E2asr/data/LibriTTS"
expdir = "/speech/shoutrik/torch_exp/E2asr/exp/test01"
train_set_name = "train"
valid_set_name = "dev_clean"
max_frames = 38400
batch_size = 64
max_epoch = 30
grad_norm_threshold = 1.0
save_last_step_freq = 2000
save_global_step_freq = 20000
logging_freq = 100
seed=42
accum_grad=2
learning_rate = 2e-4
warmup_steps = 10000
config = ASRconfig(
    sample_rate= 16000,
    n_fft=512,
    win_length=400,
    hop_length=160,
    n_mels=80,
    center=True,
    preemphasis=True,
    normalize_energy=True,
    time_mask_param=30,
    freq_mask_param=15,
    norm_mean=True,
    norm_var=True,
    model_dim=512,
    feedforward_dim=208,
    dropout=0.1,
    num_heads=8,
    num_layers=18,
    max_len=1600,
)


wandb.init(
    project="E2asr",
    name="libriTTS_trial01",
    id=wandb.util.generate_id(),
    resume="allow",
    config={
        "epochs": max_epoch,
        "learning_rate": learning_rate,
        "accum_grad": accum_grad,
        "warmup_steps": warmup_steps,
        "grad_norm_threshold": 1.0,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "model_dim": config.model_dim,
        "feedforward_dim": config.feedforward_dim,
    }
)

train_dataset, valid_dataset, stoi, itos, sp = prepare_datasets(data_path, train_set_name, valid_set_name, expdir)
train_sampler = SortedSampler(train_dataset, max_frames, batch_size, seed=42, stft_center=config.center, win_length=config.win_length, hop_length=config.hop_length)
valid_sampler = SortedSampler(valid_dataset, max_frames, batch_size, seed=42, stft_center=config.center, win_length=config.win_length, hop_length=config.hop_length)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True, shuffle=False)

vocab_size = len(stoi) + 1
model = E2ASR(config, vocab_size, training=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device : ", device)
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
                  resume_from_checkpoint=None,
                  logging_freq=logging_freq,
                  grad_norm_threshold=grad_norm_threshold,
                  seed=seed,
                  learning_rate=learning_rate,
                  warmup_steps=warmup_steps,
                  logger="wandb"
                  )

trainer.train()
wandb.finish()
