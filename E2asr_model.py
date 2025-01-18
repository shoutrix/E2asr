import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
import sys
import math

@dataclass
class ASRconfig:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    center: bool = True
    time_mask_param: int = 30
    freq_mask_param: int = 15
    norm_mean: bool = True
    norm_var: bool = True
    model_dim: int = 512
    feedforward_dim: int = 2048
    dropout: float = 0.1
    num_heads: int = 8
    num_layers: int = 18
    encoder_normalize_first: bool = True
    max_len: int = 4992
    
    def validate(self):
        assert self.model_dim % self.num_heads == 0, f"model dim should be divisible by num_heads"


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
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(config.model_dim, config.model_dim, 3, 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            # nn.AvgPool2d(kernel_size=2)
        )
        in_dim = config.n_mels
        flattened_shape = (((in_dim - 1) // 2) - 1) // 2
        self.linear = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(flattened_shape * config.model_dim, config.model_dim)
        )

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

    def forward(self, x):
        B, T, _ = x.shape
        
        querry = self.q(x)
        key = self.k(x)
        value = self.v(x)

        querry = querry.view(B, T, self.config.num_heads, self.config.model_dim // self.config.num_heads).permute(0, 2, 1, 3)
        key = key.view(B, T, self.config.num_heads, self.config.model_dim // self.config.num_heads).permute(0, 2, 1, 3)
        value = value.view(B, T, self.config.num_heads, self.config.model_dim // self.config.num_heads).permute(0, 2, 1, 3)

        if hasattr(nn.functional, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(querry, key, value, dropout_p=self.dropout_p)

        else:
            scale = 1 / math.sqrt(self.config.model_dim // self.config.num_heads)
            score = torch.matmul(querry, key.transpose(2,3)) * scale
            score = score.float()
            norm_score = F.softmax(score, dim=-1).to(querry.dtype)
            norm_score = F.dropout(norm_score, p = self.dropout_p, training=self.training)
            out = torch.matmul(norm_score, value)
        
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        return self.out_proj(out)
        
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = MultiheadAttention(config)
        self.feed_forward = PositionWiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # self.register_buffer("residual_scale", (self.config.num_layers*2)**-0.5)
        # self.register_buffer("residual_scale", torch.ones(1))
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):  
        
        x = x + self.residual_scale * self.self_attn(self.norm1(x))
        x = x + self.residual_scale * self.feed_forward(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layer = Conv2dSubsampling(config)
        self.positional_embedding = nn.Embedding(config.max_len, config.model_dim) # TODO use a better positional embedding
        import torch.backends.cuda as cuda
        print("using Flash sdpa : ", cuda.flash_sdp_enabled())
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, x):
        input_feats = self.input_layer(x)
        B, T, _ = input_feats.shape
        input_pos = self.positional_embedding(torch.arange(0, T, dtype=torch.long, device=x.device))
        input_pos = input_pos.unsqueeze(0).expand(B, -1, -1)
        
        x = input_feats + input_pos
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x_mean = x.mean()
            x_std = x.std()
            print(f"encoder layer {i} | mean : {x_mean} | std : {x_std}")
        return self.norm(x)

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
        print(f"features scale | mean : {feats.mean()} | std : {feats.std()}")       
        out_ = self.encoder(feats)
        logits = self.pred_head(out_)
        
        loss, acc = self.compute_masked_cross_entropy_loss_and_acc(logits, y)
        return logits, loss, acc
    
    def compute_masked_cross_entropy_loss_and_acc(self, logits, y):
        y = y + 1  # to make 0 the filler token
        B, T, d = logits.shape
        y = F.pad(y, (0, T - y.shape[1]), value=0)
        y_mask = y != 0
        y_mask = y_mask.view(-1)
            
        logits_flat = logits.view(-1, logits.shape[-1])[y_mask]
        y_flat = y.view(-1)[y_mask]
        
        loss = F.cross_entropy(logits_flat, y_flat)
        
        predicted = torch.argmax(logits_flat, dim=-1)
        correct_predicted = (predicted == y_flat).sum().item()
        
        acc = correct_predicted / len(y_flat)
        return loss, acc

    def initialize_parameters(self):
        # encoder_PWFF_layer_linear_std = 0.5 * (self.config.num_layers * 2)**-0.5
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight.data.mul_(0.1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.weight.data.mul_(0.1)
                # nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            
        nn.init.xavier_uniform_(self.pred_head.weight)
        nn.init.constant_(self.pred_head.bias, 0)

        
        
        
