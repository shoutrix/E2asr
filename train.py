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
batch_size = 128
max_frames = 12800

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
print("vocab_size : ", vocab_size)
print(stoi)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("using device : ", device)



# torch.set_float32_matmul_precision("high")

def get_loss_fn(n_classes, padding_class):
    # class_weights = torch.ones(n_classes)
    # class_weights[padding_class] = 0.05
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn

loss_fn = get_loss_fn(vocab_size, 0)


model = E2ASR(config, vocab_size, loss_fn)

model = model.to(device)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# model = model.to('cuda')  # Move model to GPU
print(next(model.parameters()).dtype)  # Check dtype after moving to GPU

max_epoch = 30
# training
for epoch in range(max_epoch):
    epoch_loss = 0
    # print(f"EPOCH : {epoch+1}")
    # print("Training ...")
    for i, batch in enumerate(train_loader):
    #     print(batch)
        # print(batch)
        batch_start_time = time.time()
        speech = batch["speech"]
        speech_lengths = batch["lengths"]
        y = batch["tokens"]
        
        print(speech.dtype, speech_lengths.dtype, y.dtype)
        
        print(speech.shape)
        print(speech_lengths)
        print(y.shape)
        sys.exit()

    #     print(y)
    #     speech, speech_lengths, y = speech.to(device), speech_lengths.to(device), y.to(device)

    #     logits, loss, acc = model(speech, speech_lengths, y)
    #     sys.exit()

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     batch_end_time = time.time()
    #     n_frames = speech.shape[0]*speech.shape[1]
    #     # break
    #     print(f"batch : {i+1}/{len(train_loader)} | loss: {loss.item():.4f} | acc : {acc*100:.4f}% | throughput : {int(n_frames / (batch_end_time-batch_start_time))} frames/sec")
    #     epoch_loss += loss
    # epoch_loss = epoch_loss / (i+1)
    # print(f"epoch : {epoch}/{max_epoch} | loss : {epoch_loss}")
    # # break

    # # validation
    # print("validating ...")
    # valid_loss = 0
    
    # with torch.no_grad():
    #     model.eval()
    #     for batch in valid_loader:
    #         speech = batch["speech"]
    #         y = batch["tokens"]
    #         speech_lengths = batch["lengths"]
    #         speech, speech_lengths, y = speech.to(device), speech_lengths.to(device), y.to(device)
    #         logits, loss, acc = model(speech, speech_lengths, y)
    #         valid_loss += loss.item()
        
    #     valid_loss = valid_loss / len(valid_loader)
    #     print(f"epoch : {epoch}/{max_epoch} | valid loss : {valid_loss}")
