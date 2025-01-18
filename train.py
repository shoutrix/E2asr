import os
from data_utils import prepare_datasets, collate_fn, SortedSampler
from E2asr_model import ASRconfig, E2ASR
from torch.utils.data import DataLoader
import torch
import time
import random
import numpy as np
from tqdm import tqdm
import sys
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

set_seed(42)

torch.set_num_threads(8)

batch_size = 64
max_frames = 38400

train_set_name = "train"
valid_set_name= "dev_clean"

data_path = "/speech/shoutrik/torch_exp/E2asr/data/LibriTTS"
expdir = "/speech/shoutrik/torch_exp/E2asr/exp"
config = ASRconfig()

print("preparing datasets ...")
train_dataset, valid_dataset, stoi, itos, sp = prepare_datasets(data_path, train_set_name, valid_set_name, expdir)


train_sampler = SortedSampler(train_dataset, max_frames, batch_size, seed=42, stft_center=config.center, win_length=config.win_length, hop_length=config.hop_length)
valid_sampler = SortedSampler(valid_dataset, max_frames, batch_size, seed=42, stft_center=config.center, win_length=config.win_length, hop_length=config.hop_length)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=1, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, collate_fn=collate_fn, num_workers=1, pin_memory=True)


vocab_size = len(stoi) + 1
# vocab_size = 96
print("vocab_size : ", vocab_size)
print(stoi)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("using device : ", device)

model = E2ASR(config, vocab_size, training=True)

model = model.to(device)

print(model)


def compute_grad_norm(model):
    norm_type = 2.0
    total_grad_norm = 0.0
    high_grad_norm_modules = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(norm_type).item()
            grad_norm = grad_norm ** norm_type
            total_grad_norm += grad_norm
            if grad_norm > grad_norm_threshold:
                high_grad_norm_modules.append(f"{name} {grad_norm:.4f}")
    total_grad_norm = total_grad_norm ** (1 / norm_type)
    return total_grad_norm, high_grad_norm_modules
    


class WarmupScheduler:
    def __init__(self, optimizer, lr, warmup_steps):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = warmup_steps
    
    def step(self):
        lr_step = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_step
        
        

max_epoch = 30
grad_norm_threshold = 1.0
step = 0
lr = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = WarmupScheduler(optimizer, lr = lr, warmup_steps=10000)

for epoch in range(max_epoch):
    epoch_loss = 0
    for i, batch in enumerate(train_loader):

        batch_start_time = time.time()
        speech = batch["speech"]
        speech_lengths = batch["lengths"]
        y = batch["tokens"]
        speech = speech.to(device)
        
        speech, speech_lengths, y = speech.to(device), speech_lengths.to(device), y.to(device)
        logits, loss, acc = model(speech, speech_lengths, y)

        optimizer.zero_grad()
        loss.backward()
        
        grad_norm, high_grad_norm_modules = compute_grad_norm(model)
        print("grad norm : ", grad_norm)
        # print(f"modules with gradient higher than {grad_norm_threshold} ----->")
        # print(high_grad_norm_modules) 
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        lr_scheduler.step()
        optimizer.step()
        step += 1
        batch_end_time = time.time()
        n_frames = speech.shape[0]*speech.shape[1]
        print(f"batch : {i+1}/{len(train_loader)} | loss: {loss.item():.4f} | acc : {acc*100:.4f}% | throughput : {int(n_frames / (batch_end_time-batch_start_time))} frames/sec")
        epoch_loss += loss
    #     break
    # break
        
    epoch_loss = epoch_loss / (i+1)
    print(f"epoch : {epoch}/{max_epoch} | loss : {epoch_loss}")

    # validation
    print("validating ...")
    valid_loss = 0
    
    with torch.no_grad():
        model.eval()
        for batch in valid_loader:
            speech = batch["speech"]
            y = batch["tokens"]
            speech_lengths = batch["lengths"]
            speech, speech_lengths, y = speech.to(device), speech_lengths.to(device), y.to(device)
            logits, loss, acc = model(speech, speech_lengths, y)
            valid_loss += loss.item()
        
        valid_loss = valid_loss / len(valid_loader)
        print(f"epoch : {epoch}/{max_epoch} | valid loss : {valid_loss}")
