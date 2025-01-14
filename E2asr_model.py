import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
import sys

@dataclass
class ASRconfig:
    sample_rate: int = 16000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    center: bool = True
    time_mask_param: int = 30
    freq_mask_param: int = 15
    norm_mean: bool = True
    norm_var: bool = True
    model_dim: int = 256
    feedforward_dim: int = 1024
    dropout: float = 0.1
    num_heads: int = 4
    num_layers: int = 6
    encoder_normalize_first: bool = True
    max_len: int = 5000



class AudioFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.MelSpec = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            normalized=False,
            center=config.center
        )
        self.log = lambda x: torch.log1p(x)
    
    def forward(self, x, lengths):
        x = self.MelSpec(x) # shape : B, d, T

        if self.config.center:
            frame_lengths = 1 + lengths // self.config.hop_length
        else:
            frame_lengths = 1 + (lengths - self.config.win_length) // self.config.hop_length

        max_len = x.size(-1)
        range_ = torch.arange(max_len, device=x.device)
        mask = range_[None, :] <= frame_lengths[:, None]
        mask = mask.unsqueeze(1)
        x = torch.where(mask, x, torch.zeros_like(x))
        x = self.log(x)
        return x, frame_lengths, mask



class GlobalMVN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_mean = config.norm_mean
        self.norm_var = config.norm_var
    
    def forward(self, x, lengths, padding_mask):
        # shape of x : (B, T, d)
        padding_mask = padding_mask.transpose(1,2)
        x = x * padding_mask
        
        n_frames = torch.sum(lengths)
        mean = torch.sum(x, dim=(0,1), keepdims=True) / n_frames
        
        if self.norm_mean:
            x = x - mean
        
        if self.norm_var:
            var = ((x - mean)**2).sum(dim=(0,1), keepdims=True) / n_frames
            std = torch.clamp(torch.sqrt(var), min=1e-20)
            x = x / std  
    
        x = x * padding_mask
        return x
        

class SpecAugment(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time_mask = T.TimeMasking(config.time_mask_param)
        self.freq_mask = T.FrequencyMasking(config.freq_mask_param)
    
    def forward(self, x):
        x = self.time_mask(x)
        x = self.freq_mask(x)
        return x
    

class Conv2dSubsampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, config.model_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(config.model_dim, config.model_dim, 3, 2),
            nn.ReLU()
        )
        in_dim = config.n_mels
        flattened_shape = (((in_dim - 1) // 2) - 1) // 2
        self.linear = nn.Linear(flattened_shape * config.model_dim, config.model_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, T, d = x.shape
        x_feats = self.linear(x.transpose(1, 2).contiguous().view(B, T, C * d))
        return x_feats


class PositionWiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Linear(config.model_dim, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.model_dim)
        )
    
    def forward(self, x):
        return self.mod(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = nn.MultiheadAttention(config.model_dim, config.num_heads, dropout=config.dropout)
        self.feed_forward = PositionWiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layer = Conv2dSubsampling(config)
        self.positional_embedding = nn.Embedding(config.max_len, config.model_dim) # TODO use a better positional embedding
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, x):
        input_feats = self.input_layer(x)
        B, T, _ = input_feats.shape
        input_pos = self.positional_embedding(torch.arange(0, T, dtype=torch.long, device=x.device))
        input_pos = input_pos.unsqueeze(0).expand(B, -1, -1)
        
        x = input_feats + input_pos
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class E2ASR(nn.Module):
    def __init__(self, config, vocab_size, loss_fn):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.loss_fn = loss_fn
        self.feature_extractor = AudioFeatureExtractor(config)
        self.specaug = SpecAugment(config)
        self.normalization = GlobalMVN(config)
        self.encoder = TransformerEncoder(config)
        self.pred_head = nn.Linear(config.model_dim, self.vocab_size)

    def forward(self, speech, speech_lengths, y):
        y = y + 1  # Added 1 to make 0 the filler token
        
        print(y)
        
        
        feats, frame_lengths, padding_mask = self.feature_extractor(speech, speech_lengths)

        # feats = self.specaug(feats)
        # feats = feats.transpose(1,2) # B,d,T -> B,T,d
        # feats = self.normalization(feats, frame_lengths, padding_mask)           
        # out_ = self.encoder(feats)
        # logits = self.pred_head(out_)
        
        # y_mask = torch.ones_like(y, dtype=torch.bool, device=speech.device)
        # y_mask = F.pad(y_mask, (0, logits.shape[1] - y.shape[-1]), value=False)
        # y = F.pad(y, (0, logits.shape[1] - y.shape[-1]), value=0)
        
        # # print(logits.shape, y.shape)
        
        # # first_ = torch.argmax(logits[0], dim=-1)
        # # assert y[0].shape == first_.shape, f"shape of y[0] : {y[0].shape} | shape of first_ : {first_.shape}"
        # # print(y[0])
        # # print(first_)
        
        
        # assert y_mask.shape == y.shape, f"Shapes of y_mask {y_mask.shape} and y {y.shape} do not match."
        
        # loss = self.loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
        # acc = (torch.argmax(logits.view(-1, logits.shape[-1]), dim=-1)==y.view(-1)).sum().item() / len(y.view(-1))

        # return logits, loss, acc
