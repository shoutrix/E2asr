import torch
import torch.nn as nn
import torchaudio.transforms as T
from dataclasses import dataclass
import torch.nn.functional as F
import math

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
    model_dim: int = 768
    feedforward_dim: int = 3072
    dropout: float = 0.1
    num_heads: int = 12
    num_layers: int = 24
    max_len: int = 4992
    stochastic_depth_p: float = 0.1
    unskipped_layers: list = None
    

    def validate(self):
        assert self.model_dim % self.num_heads == 0, "model_dim should be divisible by num_heads"
        assert self.sample_rate > 0, "sample_rate should be positive"
        assert self.n_fft > 0, "n_fft should be positive"
        assert self.n_mels > 0, "n_mels should be positive"


def generate_padding_mask(lens, max_len=None):
    if max_len is None:
        max_len = torch.amax(lens)
    range_ = torch.arange(max_len).to(lens.device)
    mask = range_[None, :] < lens[:, None]
    return mask


def generate_attention_mask(lens, k):
    max_len = torch.amax(lens)
    mask = generate_padding_mask(lens)
    B, T = mask.shape
    attn_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, k, max_len, -1)
    attn_mask = attn_mask.float()
    attn_mask = (1.0 - attn_mask) * -1e9
    return attn_mask
        


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
        return torch.cat((signal[:, :1], signal[:, 1:] - alpha * signal[:, :-1]), dim=1)
    
    def energy_normalization(self, mel_spec):
        energy = torch.sqrt(torch.sum(mel_spec ** 2, dim=1, keepdim=True))
        return mel_spec / (energy + 1e-10)
    
    def forward(self, x, lengths):
        if self.preemphasis:
            x = self.pre_emphasis(x)
        x = self.MelSpec(x)
        if self.normalize_energy:
            x = self.energy_normalization(x)
        frame_lengths = 1 + (lengths - self.config.win_length) // self.config.hop_length if not self.config.center else 1 + lengths // self.config.hop_length
        mask = generate_padding_mask(frame_lengths)[:, None, :]
        x = torch.where(mask, x, torch.zeros_like(x))
        x = self.log(x)
        return x, frame_lengths, mask


class GlobalMVN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_mean = config.norm_mean
        self.norm_var = config.norm_var
    
    def forward(self, x, lengths, padding_mask):
        padding_mask = padding_mask.transpose(1, 2)
        x = x * padding_mask
        n_frames = torch.sum(lengths)
        mean = torch.sum(x, dim=(0, 1), keepdims=True) / n_frames
        if self.norm_mean:
            x = x - mean
        if self.norm_var:
            var = ((x - mean)**2).sum(dim=(0, 1), keepdims=True) / n_frames
            std = torch.clamp(torch.sqrt(var), min=1e-20)
            x = x / std  
        return x * padding_mask


class SpecAugment(nn.Module):
    def __init__(self, config):
        super().__init__()
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
            nn.Dropout(p=config.dropout),
            nn.Conv2d(config.model_dim, config.model_dim, 3, 2),
            nn.GELU(),
        )
        
        self.get_out_len = lambda x: (((x - 1) // 2) - 1) // 2
        in_dim = config.n_mels
        flattened_shape = self.get_out_len(in_dim)
        self.linear = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(flattened_shape * config.model_dim, config.model_dim)
        )

    def forward(self, x, frame_lengths):
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, T, d = x.shape
        x_feats = self.linear(x.transpose(1, 2).contiguous().view(B, T, C * d))
        out_lens = self.get_out_len(frame_lengths)
        mask = generate_padding_mask(out_lens)
        out_feats = torch.where(mask[:, :, None], x_feats, torch.zeros_like(x_feats))
        return out_feats, out_lens

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_dim = config.model_dim
        self.max_len = config.max_len
        self.register_buffer('positional_encodings', self._create_positional_encodings(self.max_len, self.model_dim))
    
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


# class LearnablePositionalEncoding(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.model_dim = config.model_dim
#         self.max_len = config.max_len
#         self.positional_encodings = nn.Embedding(self.max_len, self.model_dim)
#         nn.init.normal_(self.positional_encodings.weight, mean=0, std=0.02)

#     def forward(self, seq_len, device):
#         positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
#         return self.positional_encodings(positions)



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
            self.dropout_p = config.dropout
        else:
            self.dropout_p = 0.0
        self.qk_scale = (self.config.model_dim // self.config.num_heads) ** -0.25

    def forward(self, x, lens):
        B, T, _ = x.shape
        
        querry = self.q(x)
        key = self.k(x)
        value = self.v(x)

        querry = querry.view(B, T, self.config.num_heads, self.config.model_dim // self.config.num_heads).permute(0, 2, 1, 3)
        key = key.view(B, T, self.config.num_heads, self.config.model_dim // self.config.num_heads).permute(0, 2, 1, 3)
        value = value.view(B, T, self.config.num_heads, self.config.model_dim // self.config.num_heads).permute(0, 2, 1, 3)

        # print(querry.shape, key.shape, value.shape, attn_mask.shape)
        
        attn_mask = generate_attention_mask(lens, k=self.config.num_heads)

        if hasattr(nn.functional, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(querry, key, value, dropout_p=self.dropout_p, attn_mask=attn_mask)
        else:
            out = self.scaled_dot_product_attention(querry, key, value, attn_mask)
        
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        return self.out_proj(out)
    
    def scaled_dot_product_attention(self, querry, key, value, attn_mask):
        scale = 1 / math.sqrt(self.config.model_dim // self.config.num_heads)
        score = torch.matmul(self.qk_scale * querry, self.qk_scale * key.transpose(2,3)) * scale
        score = score.float() + attn_mask
        norm_score = F.softmax(score, dim=-1).to(querry.dtype)
        norm_score = F.dropout(norm_score, p = self.dropout_p, training=self.training)
        out = torch.matmul(norm_score, value)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.self_attn = MultiheadAttention(config)
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.feed_forward = PositionWiseFeedForward(config)
        self.residual_scale = 1

    def forward(self, x, lens):
        x = x + self.residual_scale * self.self_attn(self.norm1(x), lens)
        x = x + self.residual_scale * self.feed_forward(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_layer = Conv2dSubsampling(config)
        self.positional_encoding = SinusoidalPositionalEmbedding(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.model_dim)
        self.stochastic_depth_p = self.config.stochastic_depth_p
        
    def forward(self, x, lengths, stochastic_depth):
        input_feats, feat_lengths = self.input_layer(x, lengths)
        # print("feat lengths after conv subsampling : ", feat_lengths)
        x = input_feats + self.positional_encoding(input_feats.shape[1], input_feats.device)
        for i, layer in enumerate(self.layers):
            if i not in self.config.unskipped_layers and stochastic_depth and torch.rand(1).item() < self.stochastic_depth_p:
                continue
            x = layer(x, feat_lengths)
        return self.norm(x)


class E2ASR(nn.Module):
    def __init__(self, config, vocab_size, training=True):
        super().__init__()
        self.config = config
        self.feature_extractor = AudioFeatureExtractor(config)
        self.specaug = SpecAugment(config)
        self.normalization = GlobalMVN(config)
        self.encoder = TransformerEncoder(config)
        self.pred_head = nn.Linear(config.model_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.initialize_parameters()


    def forward(self, speech, speech_lengths, y, stochastic_depth=False):
        feats, frame_lengths, padding_mask = self.feature_extractor(speech, speech_lengths)
        # print("frame lengths : ", frame_lengths)
        # print("text tokens : ", y)
        if self.training:
            feats = self.specaug(feats)
        feats = feats.transpose(1, 2)
        feats = self.normalization(feats, frame_lengths, padding_mask)
        out_ = self.encoder(feats, frame_lengths, stochastic_depth)
        logits = self.pred_head(out_)
        loss, acc = self.compute_masked_cross_entropy_loss_and_acc(logits, y)
        return logits, loss, acc

    def compute_masked_cross_entropy_loss_and_acc(self, logits, y):
        # print(logits.shape, y.shape)
        B, T, d = logits.shape
        y = F.pad(y, (0, T - y.shape[1]), "constant", 0).flatten()
        y_mask = y != 0
        logits_flat = logits.view(-1, logits.shape[-1])[y_mask]
        y_flat = y[y_mask]
        # print(logits_flat.shape, y_flat.shape)
        loss = self.loss_fn(logits_flat, y_flat)
        acc = (logits_flat.argmax(dim=-1) == y_flat).float().mean().item()
        # print("logits : ", logits_flat.argmax(dim=-1))
        # print("ground truth : ", y_flat)
        return loss, acc
    
    def initialize_parameters(self):
        print("\n\ninitializing parameters...")
        residual_scale = 1 / math.sqrt(2 * self.config.num_layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)

        for m in self.modules():
            if isinstance(m, MultiheadAttention):
                for name, param in m.named_parameters():
                    if "q.weight" in name or "k.weight" in name or "v.weight" in name or "out_proj.weight" in name:
                        param.data *= residual_scale
            elif isinstance(m, PositionWiseFeedForward):
                for sub_module in m.mod:
                    if isinstance(sub_module, nn.Linear):
                        sub_module.weight.data *= residual_scale
