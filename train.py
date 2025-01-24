import os
from data_utils import prepare_datasets, collate_fn, SortedSampler
from E2asr_model import ASRconfig, E2ASR
from trainer import Trainer
import wandb


# HYPER PARAMETERS
data_path = "/speech/shoutrik/torch_exp/E2asr/data/LibriTTS"
expdir = "/speech/shoutrik/torch_exp/E2asr/exp/LibriTTS_train_large_768_24_12_SLD"
train_set_name = "train"
valid_set_name = "dev_clean"
max_frames = 64000
batch_size = 96
max_epoch = 130
grad_norm_threshold = 1.0
save_last_step_freq = 10000
save_global_step_freq = 40000
logging_freq = 500
seed=42
accum_grad=2
learning_rate = 2e-4
warmup_steps = 40000
weight_decay=0.1

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
    model_dim=768,
    feedforward_dim=3072,
    dropout=0.1,
    num_heads=12,
    num_layers=24,
    max_len=1600,
    stochastic_depth_p=0.1,
    unskipped_layers=[0,1,2,3,4,5,21,22,23],
)

wandb.init(
    project="E2asr",
    name=os.path.basename(expdir),
    id="ay669h5r",
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
        "stochastic_depth_p": config.stochastic_depth_p
    }
)

train_dataset, valid_dataset, stoi, itos, sp = prepare_datasets(data_path, train_set_name, valid_set_name, expdir)
vocab_size = len(stoi) + 1
model = E2ASR(config, vocab_size, training=True)

trainer = Trainer(model=model,
                  train_dataset=train_dataset,
                  valid_dataset=valid_dataset,
                  max_frames=max_frames,
                  batch_size=batch_size,
                  config=config,
                  expdir=expdir,
                  accum_grad=accum_grad,
                  max_epoch=max_epoch,
                  save_last_step_freq=save_last_step_freq,
                  save_global_step_freq=save_global_step_freq,
                  resume_from_checkpoint=True,
                  logging_freq=logging_freq,
                  grad_norm_threshold=grad_norm_threshold,
                  seed=seed,
                  learning_rate=learning_rate,
                  warmup_steps=warmup_steps,
                  weight_decay=weight_decay,
                  step_to_start_layer_drop=50000,
                  logger="wandb"
                  )

trainer.train()
wandb.finish()
