import os
import time
import torch
import random
import numpy as np
from collections import defaultdict
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

class CheckpointManager:
    def __init__(self, expdir, metric):
        self.ckpt_dir = os.path.join(expdir, "checkpoint")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.param = metric

    def save_checkpoint(self, model, optimizer, lr_scheduler, step, metrics, last_step=True):
        save_best = False
        param_data = metrics[self.param]
        best_ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_{self.param}_best.pt")

        if len(param_data) > 1:
            if (self.param.endswith("acc") and param_data[-1] > param_data[-2]) or \
               (self.param.endswith("loss") and param_data[-1] < param_data[-2]):
                save_best = True
        else:
            save_best = True

        if save_best:
            if os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            self.save_(model, optimizer, lr_scheduler, step, metrics, best_ckpt_path)

        if last_step:
            last_ckpt_path = os.path.join(self.ckpt_dir, "checkpoint_last.pt")
            if os.path.exists(last_ckpt_path):
                os.remove(last_ckpt_path)
            ckpt_path = last_ckpt_path
        else:
            ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_step_{step}.pt")

        self.save_(model, optimizer, lr_scheduler, step, metrics, ckpt_path)

    def save_(self, model, optimizer, lr_scheduler, step, metrics, ckpt_path):
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")


    def load_checkpoint(self, ckpt_path, model, optimizer=None, lr_scheduler=None):
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint path {ckpt_path} does not exist.")
            return None
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        return checkpoint['step'], checkpoint['metrics']


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


class Trainer:
    def __init__(self, model, train_loader, valid_loader, device, expdir, accum_grad, max_epoch, grad_norm_threshold, save_last_step_freq, save_global_step_freq):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.expdir = expdir
        self.accum_grad = accum_grad
        self.max_epoch = max_epoch
        self.grad_norm_threshold = grad_norm_threshold
        self.save_last_step_freq = save_last_step_freq
        self.save_global_step_freq = save_global_step_freq
        self.ckpt_manager = CheckpointManager(expdir, "valid_acc")
        self.metrics = defaultdict(list)
        self.step = 0

        self.total_steps = (len(train_loader) * max_epoch) / self.accum_grad
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4)
        self.lr_scheduler = CosineScheduler(base_lr=2.5e-4, warmup_steps=10000, total_steps=self.total_steps)
        
        self.set_seed()
        
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.max_epoch):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            if epoch % self.save_last_step_freq == 0:
                self.ckpt_manager.save_checkpoint(self.model, self.optimizer, self.lr_scheduler, self.step, self.metrics, last_step=True)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_train_loss, epoch_train_acc = 0, 0
        for i, batch in enumerate(self.train_loader):
            speech, speech_lengths, y = batch["speech"].to(self.device), batch["lengths"].to(self.device), batch["tokens"].to(self.device)
            logits, loss, acc = self.model(speech, speech_lengths, y)
            epoch_train_loss += loss.item()
            epoch_train_acc += acc
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = self.compute_grad_norm()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_threshold)
            lr = self.lr_scheduler.step()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.optimizer.step()
            self.step += 1
            if self.step % self.save_global_step_freq == 0:
                self.ckpt_manager.save_checkpoint(self.model, self.optimizer, self.lr_scheduler, self.step, self.metrics, last_step=False)
            if self.step % self.save_last_step_freq == 0:
                self.ckpt_manager.save_checkpoint(self.model, self.optimizer, self.lr_scheduler, self.step, self.metrics, last_step=True)
            print(f"epoch {epoch}/{self.max_epoch} |  step {self.step}/{self.total_steps} |  loss {loss.item():.4f} | acc {acc:.4f} | grad Norm {grad_norm:.4f} | lR {lr}")

    def validate_epoch(self, epoch):
        self.model.eval()
        epoch_valid_loss, epoch_valid_acc = 0, 0
        with torch.no_grad():
            for batch in self.valid_loader:
                speech, speech_lengths, y = batch["speech"].to(self.device), batch["lengths"].to(self.device), batch["tokens"].to(self.device)
                logits, loss, acc = self.model(speech, speech_lengths, y)
                epoch_valid_loss += loss.item()
                epoch_valid_acc += acc
        self.metrics["valid_loss"].append(epoch_valid_loss / len(self.valid_loader))
        self.metrics["valid_acc"].append(epoch_valid_acc / len(self.valid_loader))
        print(f"Validation - epoch {epoch} | loss {epoch_valid_loss / len(self.valid_loader)} | acc {epoch_valid_acc / len(self.valid_loader)}")
