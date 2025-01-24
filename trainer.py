import os
import time
import torch
import random
import numpy as np
from collections import defaultdict
import math
import torch.backends.cuda as cuda
import wandb
import inspect
from torch.utils.data import DataLoader

from data_utils import collate_fn, SortedSampler

class CheckpointManager:
    def __init__(self, expdir, metric):
        self.expdir = expdir
        self.ckpt_dir = os.path.join(expdir, "checkpoint")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.param = metric

    def save_checkpoint(self, model, optimizer, lr_scheduler, step, last_step=True):
        
        if last_step:
            # save_best = False
            # param_data = metrics[self.param]
            # best_ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_{self.param}_best.pt")

            # if len(param_data) > 1:
            #     if (self.param.endswith("acc") and param_data[-1] > param_data[-2]) or \
            #     (self.param.endswith("loss") and param_data[-1] < param_data[-2]):
            #         save_best = True
            # else:
            #     save_best = True

            # if save_best:
            #     if os.path.exists(best_ckpt_path):
            #         os.remove(best_ckpt_path)
            #     self.save_(model, optimizer, lr_scheduler, step, metrics, best_ckpt_path)


            last_ckpt_path = os.path.join(self.ckpt_dir, "checkpoint_last.pt")
            if os.path.exists(last_ckpt_path):
                os.remove(last_ckpt_path)
            ckpt_path = last_ckpt_path
            self.save_(model, optimizer, lr_scheduler, step, ckpt_path)
        
        else:
            ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_step_{step}.pt")
            self.save_(model, optimizer, lr_scheduler, step, ckpt_path)

    def save_(self, model, optimizer, lr_scheduler, step, ckpt_path):
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
        
    def load_(self, model, optimizer=None, lr_scheduler=None, device='cuda'):
        checkpoint_dir = os.path.join(self.expdir, "checkpoint")
        all_ckpts = [p for p in os.listdir(checkpoint_dir) if p.endswith(".pt")]

        if len(all_ckpts) < 1:
            print(f"No checkpoint found at {checkpoint_dir}\n")
            return None
        else:
            ckpt_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
            if not os.path.exists(ckpt_path):
                # If checkpoint_last.pt is not available, find the one with the highest step
                ckpt_step_map = {
                    int(p.split("_")[-1].split(".")[0]): os.path.join(checkpoint_dir, p)
                    for p in all_ckpts if "checkpoint_step_" in p
                }
                if len(ckpt_step_map) > 0:
                    max_step = max(ckpt_step_map.keys())
                    ckpt_path = ckpt_step_map[max_step]
                else:
                    print(f"No valid step checkpoint found in {checkpoint_dir}\n")
                    return None

            # Load the checkpoint
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)

            
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
            for state in optimizer.state.values():
                if isinstance(state, dict):
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)

        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        step = checkpoint['step']
        print(f"Checkpoint loaded from {ckpt_path}")
        return model, optimizer, lr_scheduler, step


class CosineScheduler:
    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * self.base_lr * (1 + math.cos(math.pi * progress))

    def step(self):
        self.current_step += 1
        return self.get_lr()

    def state_dict(self):
        return {
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict):
        self.base_lr = state_dict['base_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.current_step = state_dict['current_step']


class WarmupScheduler:
    def __init__(self, base_lr, warmup_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_lr(self):
        scale = self.base_lr * self.warmup_steps ** 0.5
        if self.current_step < self.warmup_steps:
            lr = scale * self.current_step * self.warmup_steps ** -1.5
        else:
            lr = scale * min(self.current_step ** -0.5, self.current_step * self.warmup_steps ** -1.5)
        return lr

    def step(self):
        self.current_step += 1
        return self.get_lr()

    def state_dict(self):
        return {
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict):
        self.base_lr = state_dict['base_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.current_step = state_dict['current_step']



class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, max_frames, batch_size, config, expdir, accum_grad, max_epoch, grad_norm_threshold, save_last_step_freq, save_global_step_freq, seed, learning_rate, warmup_steps, weight_decay, resume_from_checkpoint=False, logging_freq=100, step_to_start_layer_drop=10000, logger=None):
        
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.config = config
        self.expdir = expdir
        self.accum_grad = accum_grad
        self.max_epoch = max_epoch
        self.grad_norm_threshold = grad_norm_threshold
        self.save_last_step_freq = save_last_step_freq
        self.save_global_step_freq = save_global_step_freq
        self.seed = seed
        self.logging_freq = logging_freq
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.step_to_start_layer_drop = step_to_start_layer_drop
        self.logger = logger
    
        
        train_sampler = SortedSampler(train_dataset, max_frames, batch_size, seed=42, stft_center=config.center, win_length=config.win_length, hop_length=config.hop_length)
        valid_sampler = SortedSampler(valid_dataset, max_frames, batch_size, seed=42, stft_center=config.center, win_length=config.win_length, hop_length=config.hop_length)
        
        self.train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True, shuffle=False)
        self.valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True, shuffle=False)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("using device : ", self.device)
        
        self.ckpt_manager = CheckpointManager(expdir, "valid_acc")
        self.metrics = defaultdict(list)

        self.total_steps = ((len(self.train_loader) * max_epoch) / self.accum_grad)
        self.optimizer = self.configure_optimizer(model, weight_decay, learning_rate, self.device)
        self.lr_scheduler = CosineScheduler(base_lr=learning_rate, warmup_steps=warmup_steps, total_steps=self.total_steps)
        # self.lr_scheduler = WarmupScheduler(base_lr=learning_rate, warmup_steps=warmup_steps)

        if resume_from_checkpoint is True:
            loaded = self.ckpt_manager.load_(model, self.optimizer, self.lr_scheduler, device=self.device)
            if loaded:
                model, self.optimizer, self.lr_scheduler, self.step = loaded
            else:
                model = model.to(self.device)
                self.step = 0
            self.last_epoch = int(self.step * self.accum_grad) // len(self.train_loader)
        else:
            self.last_epoch = 0
            model = model.to(self.device)
            self.step = 0
        
        self.layer_drop = False
        
        # try:
        #     print("torch torch.compile() ...")
        #     self.model = torch.compile(model)
        # except:
        self.model = model
        print("setting all random seeds ...")
        self.set_seed()
        print("using Flash sdpa : ", cuda.flash_sdp_enabled())
        self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print("autocast dtype set to :", self.autocast_dtype)


    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def compute_grad_norm(self):
        norm_type = 2.0
        total_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(norm_type).item() ** norm_type
                total_grad_norm += grad_norm
        return total_grad_norm ** (1 / norm_type)
    
    
    def configure_optimizer(self, model, weight_decay, lr, device):
        params = {name:p for name, p in model.named_parameters() if p.requires_grad}
        decayed_params = [p for p in params.values() if p.dim()>=2]
        non_decayed_params = [p for p in params.values() if p.dim()<2]
        
        param_groups = [
            {"params":decayed_params, "weight_decay":weight_decay},
            {"params":non_decayed_params, "weight_decay":0}
        ]
        if "fused" in inspect.signature(torch.optim.AdamW).parameters and device==torch.device("cuda"):
            use_fused = True
        else:
            use_fused = False
        print("using Fused AdamW : ", use_fused)
        optimizer = torch.optim.AdamW(params=param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-8, fused=use_fused)
        return optimizer
           


    def train(self):
        self.model.to(self.device)
        print(f"Training started | epochs: {self.max_epoch} | batches per epoch: {len(self.train_loader)} | accum_grad: {self.accum_grad} | total updates: {self.total_steps}")
        for epoch in range(self.last_epoch, self.max_epoch):
            print(f"\nstarting epoch {epoch + 1}/{self.max_epoch}")
            self.train_epoch(epoch)
            self.validate_epoch(epoch)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss, train_acc = 0, 0
        for batch_idx, batch in enumerate(self.train_loader):
            speech, speech_lengths, y = (
                batch["speech"].to(self.device),
                batch["lengths"].to(self.device),
                batch["tokens"].to(self.device),
            )

            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                _, loss, acc = self.model(speech, speech_lengths, y, stochastic_depth=self.layer_drop)

            train_loss += loss.item()
            train_acc += acc

            scaled_loss = loss / self.accum_grad
            scaled_loss.backward()

            if (batch_idx + 1) % self.accum_grad == 0:
                grad_norm = self.compute_grad_norm()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_threshold)

                lr = self.lr_scheduler.step()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                self.optimizer.step()
                self.optimizer.zero_grad()

                wandb.log({
                    "step": self.step,
                    "train_loss": train_loss / self.accum_grad,
                    "train_acc": train_acc / self.accum_grad,
                    "learning_rate": lr,
                    "grad_norm": grad_norm,
                })

                train_loss, train_acc = 0, 0
                self.step += 1
                if self.step == self.step_to_start_layer_drop:
                    self.layer_drop = True
                    print(f"staring stochastic layer drop during training with p={self.model.config.stochastic_depth_p}")

                if self.step % self.logging_freq == 0:
                    print(f"epoch: {epoch + 1}/{self.max_epoch} | batch: {batch_idx + 1}/{len(self.train_loader)} | step: {self.step}/{int(self.total_steps)} | loss: {scaled_loss.item():.4f} | grad Norm: {grad_norm:.4f} | lr: {lr:.2e}")
                if self.step % self.save_global_step_freq == 0:
                    self.ckpt_manager.save_checkpoint(self.model, self.optimizer, self.lr_scheduler, self.step, last_step=False)
                if self.step % self.save_last_step_freq == 0:
                    self.ckpt_manager.save_checkpoint(self.model, self.optimizer, self.lr_scheduler, self.step, last_step=True)

    def validate_epoch(self, epoch):
        self.model.eval()
        total_valid_loss, total_valid_acc = 0, 0
        with torch.no_grad():
            for batch in self.valid_loader:
                speech, speech_lengths, y = (
                    batch["speech"].to(self.device),
                    batch["lengths"].to(self.device),
                    batch["tokens"].to(self.device),
                )
                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                    _, loss, acc = self.model(speech, speech_lengths, y, stochastic_depth=False)

                total_valid_loss += loss.item()
                total_valid_acc += acc

        avg_valid_loss = total_valid_loss / len(self.valid_loader)
        avg_valid_acc = total_valid_acc / len(self.valid_loader)

        print(f"validation :: epoch {epoch + 1} | loss: {avg_valid_loss:.4f} | accuracy: {avg_valid_acc:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "valid_loss": avg_valid_loss,
            "valid_acc": avg_valid_acc,
        })