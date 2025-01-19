import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
import sys
import math
import wandb

@dataclass
class ASRconfig:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    center: bool = True
    preemphasis: bool = False
    normalize_energy: bool = False
    time_mask_param: int = 30
    freq_mask_param: int = 15
    norm_mean: bool = True
    norm_var: bool = True
    model_dim: int = 512
    feedforward_dim: int = 2048
    dropout: float = 0.1
    num_heads: int = 8
    num_layers: int = 18
    max_len: int = 4992
    
    def validate(self):
        assert self.model_dim % self.num_heads == 0, f"model dim should be divisible by num_heads"


def generate_padding_mask(lens, max_len=None):
    if max_len is None:
        max_len = torch.amax(lens)
    range_ = torch.arange(max_len).to(lens.device)
    mask = range_[None, :] < lens[:, None]
    return mask


def generate_attn_mask(lens, k):
    mask = generate_padding_mask(lens)
    mask = mask[:, None, :]
    if mask.dim() == 3:
        assert mask.shape[1] == 1
    elif mask.dim() == 2:
        mask.unsqueeze(1)
    else:
        raise ValueError(f"mask should be either 2 or 3 dimensional. If 3 dimensional then size of dimension 1 should be 1")
    B, _, T = mask.shape
    mask = mask.expand(-1, k, -1) # (B, 1, T) -> (B, K, T)
    mask = mask[:, :, :, None].expand(-1, -1, -1, T)
    return mask


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
        self.log = lambda x: torch.log(x + 1e-20)
        self.preemphasis = config.preemphasis
        self.normalize_energy = config.normalize_energy
    
    def pre_emphasis(self, signal, alpha=0.97):
        emp_signal = torch.cat((signal[:, :1], signal[:, 1:] - alpha * signal[:, :-1]), dim=1)
        return emp_signal
    
    def energy_normalization(self, mel_spec):
        energy = torch.sqrt(torch.sum(mel_spec ** 2, dim=1, keepdim=True))
        normalized_mel_spec = mel_spec / (energy + 1e-10)
        return normalized_mel_spec
    
    def forward(self, x, lengths):
        
        if self.preemphasis:
            x = self.pre_emphasis(x)
        x = self.MelSpec(x)
        if self.normalize_energy:
            x = self.energy_normalization(x)
        # print(x)
            
        if self.config.center:
            frame_lengths = 1 + lengths // self.config.hop_length
        else:
            frame_lengths = 1 + (lengths - self.config.win_length) // self.config.hop_length

        mask = generate_padding_mask(frame_lengths)[:, None, :]
        x = torch.where(mask, x, torch.zeros_like(x))
        x = self.log(x)
        # print(x)
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
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(config.model_dim, config.model_dim, 3, 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            # nn.AvgPool2d(kernel_size=2)
        )
        
        self.get_out_len = lambda x : (((x - 1) // 2) - 1) // 2
        in_dim = config.n_mels
        flattened_shape = self.get_out_len(in_dim)
        self.linear = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(flattened_shape * config.model_dim, config.model_dim)
        )

    def forward(self, x, frame_lengths):
        x = x.unsqueeze(1)
        x = self.conv(x)
        # wandb.log({"conv_std" : torch.std(x)})
        B, C, T, d = x.shape
        x_feats = self.linear(x.transpose(1, 2).contiguous().view(B, T, C * d))
        return x_feats, self.get_out_len(frame_lengths)

class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_dim = config.model_dim
        self.max_len = config.max_len
        self.positional_encodings = self._create_positional_encodings(self.max_len, self.model_dim)
    
    def _create_positional_encodings(self, max_len, model_dim):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_ = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))
        sinusoidal_pos_encodings = torch.zeros(max_len, model_dim)
        sinusoidal_pos_encodings[:, 0::2] = torch.sin(position * div_)
        sinusoidal_pos_encodings[:, 1::2] = torch.cos(position * div_)
        return sinusoidal_pos_encodings
    
    def forward(self, seq_len, device):
        assert seq_len <= self.max_len, f"sequence length exceeded max supported length of {self.max_len}"        
        return self.positional_encodings[:seq_len].to(device)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Linear(config.model_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.model_dim)
        )
    def forward(self, x):
        return self.mod(x)


class MultiheadAttention(nn.Module):
    def __init__(self, config, training=True):
        super(MultiheadAttention, self).__init__()
        self.config = config
        self.q = nn.Linear(config.model_dim, config.model_dim)
        self.k = nn.Linear(config.model_dim, config.model_dim)
        self.v = nn.Linear(config.model_dim, config.model_dim)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim)
        self.training = training
        if training:
            self.dropout_p = 0.2
        else:
            self.dropout_p = 0.0
        self.qk_scale = (self.config.model_dim // self.config.num_heads) ** -0.25

    def forward(self, x, attn_mask):
        B, T, _ = x.shape
        
        querry = self.q(x)
        key = self.k(x)
        value = self.v(x)

        querry = querry.view(B, T, self.config.num_heads, self.config.model_dim // self.config.num_heads).permute(0, 2, 1, 3)
        key = key.view(B, T, self.config.num_heads, self.config.model_dim // self.config.num_heads).permute(0, 2, 1, 3)
        value = value.view(B, T, self.config.num_heads, self.config.model_dim // self.config.num_heads).permute(0, 2, 1, 3)

        # print(querry.shape, key.shape, value.shape, attn_mask.shape)

        if hasattr(nn.functional, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(querry, key, value, dropout_p=self.dropout_p, attn_mask=attn_mask)

        else:
            scale = 1 / math.sqrt(self.config.model_dim // self.config.num_heads)
            score = torch.matmul(self.qk_scale * querry, self.qk_scale * key.transpose(2,3)) * scale
            score = score.float()
            norm_score = F.softmax(score, dim=-1).to(querry.dtype)
            norm_score = norm_score * attn_mask
            norm_score = F.dropout(norm_score, p = self.dropout_p, training=self.training)
            out = torch.matmul(norm_score, value)
        
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        return self.out_proj(out)
        
        
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.self_attn = MultiheadAttention(config)
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.feed_forward = PositionWiseFeedForward(config)
        self.norm_out = nn.LayerNorm(config.model_dim)
        
        # self.register_buffer("residual_scale", (self.config.num_layers*2)**-0.5)
        # self.register_buffer("residual_scale", torch.ones(1))
        self.residual_scale = 1 / math.sqrt(self.config.num_layers)

    def forward(self, x, attn_mask):  
        
        x = x + self.residual_scale * self.self_attn(self.norm1(x), attn_mask)
        x = x + self.residual_scale * self.feed_forward(self.norm2(x))
        return self.norm_out(x)


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_layer = Conv2dSubsampling(config)
        self.positional_encoding = SinusoidalPositionalEmbedding(config) # TODO use a better positional embedding
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, x, lengths):
        input_feats, feat_lengths = self.input_layer(x, lengths)
        B, T, _ = input_feats.shape
        std_list = []
        
        x = input_feats + self.positional_encoding(input_feats.shape[1], input_feats.device)
        attn_mask = generate_attn_mask(feat_lengths, self.config.num_heads)
        for i, layer in enumerate(self.layers):
            x = layer(x, attn_mask)
            std_list.append(torch.std(x))
        
        wandb.log({f"encoder_layer_{i}_std" : v for i, v in enumerate(std_list)})
        return x

class E2ASR(nn.Module):
    def __init__(self, config, vocab_size, training=True):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.training = training
        
        self.feature_extractor = AudioFeatureExtractor(config)
        self.specaug = SpecAugment(config)
        self.normalization = GlobalMVN(config)
        self.encoder = TransformerEncoder(config)
        self.pred_head = nn.Linear(config.model_dim, self.vocab_size)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.initialize_parameters()

    def forward(self, speech, speech_lengths, y):
        feats, frame_lengths, padding_mask = self.feature_extractor(speech, speech_lengths)
        if self.training:
            feats = self.specaug(feats)
        feats = feats.transpose(1, 2)  # B,d,T -> B,T,d
        feats = self.normalization(feats, frame_lengths, padding_mask)   
        # print("frame lengths : ", frame_lengths)
        # print("padding_mask shape : ", padding_mask.shape)
        # print(padding_mask) 
        # print(f"features scale | mean : {feats.mean()} | std : {feats.std()}")       
        out_ = self.encoder(feats, frame_lengths)
        logits = self.pred_head(out_)
        
        loss, acc = self.compute_masked_cross_entropy_loss_and_acc(logits, y)
        return logits, loss, acc
    
    def compute_masked_cross_entropy_loss_and_acc(self, logits, y):
        y = y + 1  # to make 0 the filler token
        # print(y)
        B, T, d = logits.shape
        y = F.pad(y, (0, T - y.shape[1]), value=0).flatten()
        y_mask = y != 0
            
        logits_flat = logits.view(-1, logits.shape[-1])[y_mask]
        y_flat = y[y_mask]
        
        loss = F.cross_entropy(logits_flat, y_flat)
        
        predicted = torch.argmax(logits_flat, dim=-1)
        # print(predicted)
        correct_predicted = (predicted == y_flat).sum().item()
        
        acc = correct_predicted / len(y_flat)
        return loss, acc

    def initialize_parameters(self):
        # encoder_PWFF_layer_linear_std = 0.5 * (self.config.num_layers * 2)**-0.5
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight.data.mul_(0.45)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                m.weight.data.mul_(0.45)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            
        nn.init.xavier_uniform_(self.pred_head.weight)
        nn.init.constant_(self.pred_head.bias, 0)