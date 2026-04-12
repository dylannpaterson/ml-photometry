import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import re
import numpy as np
from castor.models.dense_grid import compute_grid_loss, DenseGridModel
from castor.constants import GLOBAL_STRETCH_SCALE

def find_latest_checkpoint(checkpoint_dir="checkpoints", prefix="stage0"):
    if not os.path.exists(checkpoint_dir): return None, 0
    pattern = re.compile(rf"{prefix}_epoch_(\d+)\.pth")
    latest_epoch = 0
    latest_file = None
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            match = pattern.match(f)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch, latest_file = epoch, os.path.join(checkpoint_dir, f)
    return latest_file, latest_epoch

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, epochs=100, lr=0.0001, checkpoint_prefix="stage0"):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.config, self.device, self.checkpoint_prefix = config, device, checkpoint_prefix
        self.epochs, self.lr = epochs, lr
        
        # 1. Transition to AdamW for better weight decay handling
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # 2. Transition to OneCycleLR for faster convergence and local minima escape
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr * 3.0,
            steps_per_epoch=len(self.train_loader),
            epochs=self.epochs,
            pct_start=0.1, # 10% warmup
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        self.start_epoch = 0
        self.psf_library = None
        
        # FIX: Only use GradScaler on CUDA. CPU GradScaler is generally unnecessary and adds overhead.
        self.use_cuda = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_cuda else None
        
        # NEW: Memory-Aware Dynamic Batch Scaling
        self.target_batch_size = config.get("training_params", {}).get("BATCH_SIZE", 32)
        self.micro_batch_size = self.target_batch_size # Default
        self.accumulation_steps = 1
        
        if self.use_cuda:
            vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            # Conservative estimate: ~1.5GB per batch of 256x256 ResNet34
            if vram_gb < 12: # e.g. K80, 1080Ti
                self.micro_batch_size = 4
            elif vram_gb < 20: # e.g. T4, RTX 3080
                self.micro_batch_size = 8
            elif vram_gb < 30: # e.g. V100 16GB, A10
                self.micro_batch_size = 16
            else: # e.g. V100 32GB, A100
                self.micro_batch_size = 32
            
            # Ensure we don't exceed the configured batch size if it's smaller
            self.micro_batch_size = min(self.micro_batch_size, self.target_batch_size)
            self.accumulation_steps = max(1, self.target_batch_size // self.micro_batch_size)
            
            print(f"📟 GPU Memory Detected: {vram_gb:.1f}GB | Scaling Micro-batch: {self.micro_batch_size} | Accumulation: {self.accumulation_steps}")

        # FIX: Extract loss parameters from root of config instead of data_params
        self.loss_params = config.get("loss_params", {}).copy()
        self.loss_params["stretch_scale"] = config.get("data_params", {}).get("GLOBAL_STRETCH_SCALE", GLOBAL_STRETCH_SCALE)
        self.lambda_diffraction = self.loss_params.pop("lambda_diffraction_reg", 10.0)

    def resume(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path, self.start_epoch = find_latest_checkpoint(prefix=self.checkpoint_prefix)
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path} (Epoch {self.start_epoch})")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print("✅ Restored optimizer state.")
                if 'scheduler_state_dict' in ckpt:
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    print("✅ Restored scheduler state.")
                if 'scaler_state_dict' in ckpt and self.scaler:
                    self.scaler.load_state_dict(ckpt['scaler_state_dict'])
                    print("✅ Restored GradScaler state.")
                if 'psf_library' in ckpt:
                    self.psf_library = ckpt['psf_library']
                    print("✅ Restored PSF library from checkpoint.")
                self.start_epoch = ckpt.get('epoch', self.start_epoch - 1) + 1
            else:
                self.model.load_state_dict(ckpt)

    def train(self):
        print(f"Starting Training [{self.checkpoint_prefix}]: {self.epochs} epochs")
        
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train(); epoch_loss, start_time = 0, time.time()
            self.optimizer.zero_grad(set_to_none=True)
            
            for i, batch in enumerate(self.train_loader):
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device, non_blocking=True)
                    targets = batch["target"].to(self.device, non_blocking=True).float()
                    psf_library = batch["psf_library"].to(self.device, non_blocking=True)
                    
                    if self.psf_library is None:
                        raw_lib = psf_library.detach().cpu().squeeze()
                        if raw_lib.dim() == 2:
                            raw_lib = raw_lib.unsqueeze(0)
                        elif raw_lib.dim() > 3:
                            raw_lib = raw_lib.view(-1, raw_lib.shape[-2], raw_lib.shape[-1])
                        normalized_lib = raw_lib / (raw_lib.sum(dim=(-2, -1), keepdim=True) + 1e-9)
                        self.psf_library = normalized_lib
                else:
                    images, targets = batch
                    images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True).float()
                
                # --- LIVE NOISE INJECTION ---
                images_positive = torch.clamp(images, min=0.0) 
                images_noisy = torch.poisson(images_positive)
                images_noisy += torch.randn_like(images_noisy) * 5.0
                
                if isinstance(batch, dict) and "chunk_median" in batch:
                    batch_medians = batch["chunk_median"].to(self.device, non_blocking=True).float().view(-1, 1, 1, 1)
                else:
                    batch_medians = images_noisy.view(images_noisy.shape[0], -1).median(dim=1)[0].view(-1, 1, 1, 1)
                
                images_final = torch.asinh((images_noisy - batch_medians) / self.loss_params["stretch_scale"])
                
                # FIX: Disable autocast on CPU to avoid massive slowdown
                if self.use_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        preds = self.model(images_final)
                else:
                    preds = self.model(images_final)
                
                preds_fp32 = {k: v.float() for k, v in preds.items()}
                
                loss, p_loss, po_loss, f_loss, b_loss, e_loss = compute_grid_loss(
                    preds_fp32, targets, **self.loss_params,
                    log_task_vars=preds.get("log_task_vars")
                )
                
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
                
                diffraction_reg = self.model.diffraction_filter.get_regularization_loss()
                reg_loss_val = diffraction_reg.item()
                reg_loss = (self.lambda_diffraction * diffraction_reg) / self.accumulation_steps
                loss += reg_loss

                if torch.isnan(loss):
                    print(f"⚠️ NaN detected at step {i}"); continue
                    
                # 3. Backward Pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer Step (only every accumulation_steps)
                if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    
                    is_finite = True
                    for param in self.model.parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            is_finite = False; break
                    
                    if is_finite:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    
                    if self.scaler:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        if self.scaler.get_scale() >= scale_before:
                            self.scheduler.step()
                    else:
                        self.optimizer.step()
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                
                epoch_loss += loss.item() * self.accumulation_steps
                
                if i % 100 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch [{epoch+1}/{self.epochs}], Step [{i}/{len(self.train_loader)}], LR: {current_lr:.6f}, Loss: {loss.item()*self.accumulation_steps:.4f} (P:{p_loss.item():.4f}, Pos:{po_loss.item():.4f}, F:{f_loss.item():.4f}, B:{b_loss.item():.4f}, E:{e_loss.item():.4f}, DReg:{reg_loss_val:.6f})")

            avg_epoch_loss = epoch_loss/len(self.train_loader)
            print(f"==> Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f} | Time: {time.time()-start_time:.1f}s")
            val_loss = self.validate(); print(f"Validation Loss: {val_loss:.4f}")
            
            if np.isnan(avg_epoch_loss) or np.isnan(val_loss):
                print(f"❌ NaN detected in loss. Skipping checkpoint saving.")
                continue 

            os.makedirs("checkpoints", exist_ok=True)
            if self.psf_library is not None and not os.path.exists("master_psf_library.pt"):
                torch.save(self.psf_library, "master_psf_library.pt")

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'val_loss': val_loss,
                'psf_library': self.psf_library
            }
            torch.save(checkpoint, os.path.join("checkpoints", f"{self.checkpoint_prefix}_epoch_{epoch+1}.pth"))
            self._prune_checkpoints()
        
        final_ckpt = {'model_state_dict': self.model.state_dict(), 'psf_library': self.psf_library}
        torch.save(final_ckpt, os.path.join("checkpoints", f"{self.checkpoint_prefix}_final.pth"))

    def _prune_checkpoints(self, keep_last=10):
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir): return
        pattern = re.compile(rf"{self.checkpoint_prefix}_epoch_(\d+)\.pth")
        checkpoints = []
        for f in os.listdir(checkpoint_dir):
            match = pattern.match(f)
            if match: checkpoints.append((int(match.group(1)), f))
        checkpoints.sort()
        if len(checkpoints) > keep_last:
            for i in range(len(checkpoints) - keep_last):
                try: os.remove(os.path.join(checkpoint_dir, checkpoints[i][1]))
                except OSError: pass

    def validate(self):
        self.model.eval(); val_loss = 0
        num_batches = len(self.val_loader)
        if num_batches == 0: return 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device, non_blocking=True)
                    targets = batch["target"].to(self.device, non_blocking=True).float()
                else:
                    images, targets = batch
                    images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True).float()
                
                images_positive = torch.clamp(images, min=0.0) 
                images_noisy = torch.poisson(images_positive)
                images_noisy += torch.randn_like(images_noisy) * 5.0
                
                if isinstance(batch, dict) and "chunk_median" in batch:
                    batch_medians = batch["chunk_median"].to(self.device, non_blocking=True).float().view(-1, 1, 1, 1)
                else:
                    batch_medians = images_noisy.view(images_noisy.shape[0], -1).median(dim=1)[0].view(-1, 1, 1, 1)
                images_final = torch.asinh((images_noisy - batch_medians) / self.loss_params["stretch_scale"])
                
                if self.use_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        preds = self.model(images_final)
                else:
                    preds = self.model(images_final)
                
                preds_fp32 = {k: v.float() for k, v in preds.items()}
                loss, _, _, _, _, _ = compute_grid_loss(
                    preds_fp32, targets, **self.loss_params,
                    log_task_vars=preds.get("log_task_vars")
                )
                val_loss += loss.item()

        return val_loss / num_batches
