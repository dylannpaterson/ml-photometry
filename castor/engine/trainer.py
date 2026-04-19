import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
import re
import numpy as np
from castor.models.dense_grid import compute_grid_loss, DenseGridModel
from castor.constants import GLOBAL_STRETCH_SCALE, DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL

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

def render_confidence_prior(targets, img_size, cell_size, K, max_jitter=0.4, fwhm=3.5, sys_floor=0.01):
    """
    Renders a bilinear splat confidence prior map from target grid on-the-fly.
    Channel 0: P (bilinear sum = P)
    Fully vectorized. Handles overlapping splats via scatter_add_.
    """
    B, GH, GW, _ = targets.shape
    device = targets.device
    
    # 1. Extract targets
    num_params = (targets.shape[-1] - 1) // K
    star_targets = targets[..., :-1].view(B, GH, GW, K, num_params)
    p_targets = star_targets[..., 0]
    dx_targets = star_targets[..., 1]
    dy_targets = star_targets[..., 2]
    snr_targets = star_targets[..., 4]
    
    # Flatten across spatial and K dimensions: [B, GH*GW*K]
    p_flat = p_targets.reshape(B, -1)
    dx_flat = dx_targets.reshape(B, -1)
    dy_flat = dy_targets.reshape(B, -1)
    snr_flat = snr_targets.reshape(B, -1)
    
    # 2. Create Global Coordinate Grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(GH, device=device), 
        torch.arange(GW, device=device), 
        indexing='ij'
    )
    grid_y = grid_y.unsqueeze(-1).expand(-1, -1, K).reshape(-1)
    grid_x = grid_x.unsqueeze(-1).expand(-1, -1, K).reshape(-1)
    
    # Absolute center coordinates
    center_x = grid_x.unsqueeze(0) * cell_size + dx_flat
    center_y = grid_y.unsqueeze(0) * cell_size + dy_flat
    
    # Filter active stars (p > dropout threshold)
    mask = p_flat > 0.05
    batch_idx, star_idx = torch.where(mask)
    
    all_cx = center_x[batch_idx, star_idx]
    all_cy = center_y[batch_idx, star_idx]
    all_p  = p_flat[batch_idx, star_idx]
    all_snr = snr_flat[batch_idx, star_idx]
    
    # --- INJECT POISON (False Positives) ---
    num_poison = 5
    poison_batch_idx = torch.arange(B, device=device).view(-1, 1).expand(-1, num_poison).reshape(-1)
    poison_cx = torch.rand((B * num_poison,), device=device) * (img_size - 1)
    poison_cy = torch.rand((B * num_poison,), device=device) * (img_size - 1)
    poison_p = torch.empty((B * num_poison,), device=device).uniform_(0.5, 0.95)
    # Poison stars have low SNR for high uncertainty
    poison_snr = torch.empty((B * num_poison,), device=device).uniform_(1.0, 3.0)
    
    # Combine
    all_batch_idx = torch.cat([batch_idx, poison_batch_idx])
    all_cx = torch.cat([all_cx, poison_cx])
    all_cy = torch.cat([all_cy, poison_cy])
    all_p = torch.cat([all_p, poison_p])
    all_snr = torch.cat([all_snr, poison_snr])
    
    # --- DYNAMIC SNR-BASED JITTER (Cramér-Rao Lower Bound) ---
    snr_safe = torch.clamp(all_snr, min=1.0)
    theoretical_sigma = fwhm / (2.355 * snr_safe)
    dynamic_jitter = torch.sqrt(theoretical_sigma**2 + sys_floor**2)
    dynamic_jitter = torch.clamp(dynamic_jitter, max=max_jitter)

    # Apply random scatter based on dynamic jitter (with 10% dropout to teach the network perfect alignment)
    jitter_mask = torch.rand_like(all_cx) < 0.9
    jitter_x = (torch.rand_like(all_cx) * 2 - 1) * dynamic_jitter * jitter_mask
    jitter_y = (torch.rand_like(all_cy) * 2 - 1) * dynamic_jitter * jitter_mask
    all_cx = all_cx + jitter_x
    all_cy = all_cy + jitter_y

    # Anchors and sub-pixel remainders
    # We must ensure anchors are in [0, img_size-2] for 2x2 splat
    all_cx = torch.clamp(all_cx, 0, img_size - 1.001)
    all_cy = torch.clamp(all_cy, 0, img_size - 1.001)
    
    x0 = torch.floor(all_cx).long()
    y0 = torch.floor(all_cy).long()
    dx = all_cx - x0.float()
    dy = all_cy - y0.float()
    
    # Bilinear weights scaled by P
    w00 = (1 - dx) * (1 - dy) * all_p
    w10 = dx * (1 - dy) * all_p
    w01 = (1 - dx) * dy * all_p
    w11 = dx * dy * all_p
    
    # Flattened indices for 4 corners
    img_stride = img_size
    batch_stride = img_size * img_size
    
    idx00 = all_batch_idx * batch_stride + y0 * img_stride + x0
    idx10 = idx00 + 1
    idx01 = idx00 + img_stride
    idx11 = idx01 + 1
    
    prior_flat = torch.zeros(B * batch_stride, device=device)
    
    # Scatter add handles overlapping stars naturally
    prior_flat.scatter_add_(0, idx00, w00)
    prior_flat.scatter_add_(0, idx10, w10)
    prior_flat.scatter_add_(0, idx01, w01)
    prior_flat.scatter_add_(0, idx11, w11)
    
    return prior_flat.view(B, 1, img_size, img_size)

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, epochs=100, lr=0.0001, checkpoint_prefix="stage0"):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.config, self.device, self.checkpoint_prefix = config, device, checkpoint_prefix
        self.epochs, self.lr = epochs, lr
        
        self.use_cuda = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_cuda else None
        self.start_epoch = 0

        # 1. AdamW Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # NEW: Memory-Aware Dynamic Batch Scaling
        self.target_batch_size = config.get("training_params", {}).get("BATCH_SIZE", 32)
        self.micro_batch_size = self.target_batch_size # Default
        self.accumulation_steps = 1
        
        if self.use_cuda:
            vram_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            # Conservative estimate: ~1.5GB per batch of 256x256 ResNet34
            if vram_gb < 12: # e.g. K80, 1080Ti
                self.micro_batch_size = 8
            elif vram_gb < 16: # e.g. T4, RTX 3080
                self.micro_batch_size = 16
            elif vram_gb < 24: # e.g. V100 16GB, RTX 3090/4090
                self.micro_batch_size = 32
            else: # e.g. V100 32GB, A100
                self.micro_batch_size = 64
            
            # Ensure we don't exceed the configured batch size if it's smaller
            self.micro_batch_size = min(self.micro_batch_size, self.target_batch_size)
            self.accumulation_steps = max(1, self.target_batch_size // self.micro_batch_size)
            
            print(f"📟 GPU Memory Detected: {vram_gb:.1f}GB | Scaling Micro-batch: {self.micro_batch_size} | Accumulation: {self.accumulation_steps}")

        # 2. OneCycleLR Scheduler
        # FIX: Calculate actual steps the scheduler will see per epoch due to accumulation
        actual_steps_per_epoch = (len(self.train_loader) + self.accumulation_steps - 1) // self.accumulation_steps
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr * 2.0,
            steps_per_epoch=actual_steps_per_epoch,
            epochs=self.epochs,
            pct_start=0.1, # 10% warmup
            div_factor=25.0,
            final_div_factor=10000.0
        )

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
                self.start_epoch = ckpt.get('epoch', self.start_epoch - 1) + 1
            else:
                self.model.load_state_dict(ckpt)

    def train(self):
        print(f"Starting Training [{self.checkpoint_prefix}]: {self.epochs} epochs")
        
        # Get params for prior rendering
        img_size = self.config["data_params"]["image_size"]
        # Stage-specific cell size if available, else default
        cell_size = self.config["curriculum"].get(self.checkpoint_prefix, {}).get("cell_size", DEFAULT_CELL_SIZE)
        K = self.config["data_params"].get("max_capacity_per_cell", MAX_CAPACITY_PER_CELL)

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train(); epoch_loss, start_time = 0, time.time()
            self.optimizer.zero_grad(set_to_none=True)
            
            for i, batch in enumerate(self.train_loader):
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device, non_blocking=True)
                    targets = batch["target"].to(self.device, non_blocking=True).float()
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
                
                # --- LIVE CONFIDENCE PRIOR GENERATION ---
                with torch.no_grad():
                    B, GH, GW, _ = targets.shape
                    num_params = (targets.shape[-1] - 1) // K
                    st_view_orig = targets[..., :-1].view(B, GH, GW, K, num_params)
                    p_targets = st_view_orig[..., 0]
                    
                    survival_chance = p_targets * 0.9 + 0.05 
                    prior_mask = torch.rand_like(p_targets) < survival_chance
                    
                    partial_targets = targets.clone()
                    st_view = partial_targets[..., :-1].view(B, GH, GW, K, num_params)
                    st_view[~prior_mask] = 0.0
                    
                    # Render bilinear splat prior map (1 channel)
                    sigma_fixed = self.config.get("data_params", {}).get("physics_params", {}).get("sigma_fixed", 1.5)
                    calculated_fwhm = sigma_fixed * 2.355
                    prior_map = render_confidence_prior(
                        partial_targets, img_size, cell_size, K, 
                        max_jitter=0.4, fwhm=calculated_fwhm, sys_floor=0.01
                    )

                # 15% chance to drop the entire prior map
                if torch.rand(1).item() < 0.15:
                    prior_map = torch.zeros_like(prior_map)

                if self.use_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        preds = self.model(images_final, prior=prior_map)
                else:
                    preds = self.model(images_final, prior=prior_map)
                
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
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join("checkpoints", f"{self.checkpoint_prefix}_epoch_{epoch+1}.pth"))
            self._prune_checkpoints()
        
        final_ckpt = {'model_state_dict': self.model.state_dict()}
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
                
                # Use 1-channel zero prior
                prior_zeros = torch.zeros_like(images_final)

                if self.use_cuda:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        preds = self.model(images_final, prior=prior_zeros)
                else:
                    preds = self.model(images_final, prior=prior_zeros)
                
                preds_fp32 = {k: v.float() for k, v in preds.items()}
                loss, _, _, _, _, _ = compute_grid_loss(
                    preds_fp32, targets, **self.loss_params,
                    log_task_vars=preds.get("log_task_vars")
                )
                val_loss += loss.item()

        return val_loss / num_batches
