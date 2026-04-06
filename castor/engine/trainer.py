import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import re
from castor.models.dense_grid import compute_grid_loss
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
            max_lr=self.lr * 5,
            steps_per_epoch=len(self.train_loader),
            epochs=self.epochs,
            pct_start=0.1, # 10% warmup
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        self.start_epoch = 0
        
        # New: Track psf_library for checkpointing
        self.psf_library = None
        
        # Add the AMP GradScaler
        self.scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
        
        # FIX: Extract loss parameters from root of config instead of data_params
        self.loss_params = config.get("loss_params", {}).copy()
        self.loss_params["stretch_scale"] = GLOBAL_STRETCH_SCALE
        
        # Pop lambda_diffraction_reg so it doesn't collide with compute_grid_loss kwargs
        self.lambda_diffraction = self.loss_params.pop("lambda_diffraction_reg", 10.0)

        # FIX: Inject global standardization scale into the model buffer
        # This allows the model to un-standardize its own outputs during eval()
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'global_weights_std'):
            print(f"🛰️ Trainer: Injecting Global PCA StdDev into model buffer...")
            self.model.pca_std.data = torch.from_numpy(dataset.global_weights_std).float().to(self.device)

    def resume(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path, self.start_epoch = find_latest_checkpoint(prefix=self.checkpoint_prefix)
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path} (Epoch {self.start_epoch})")
            # Handle full checkpoint dict or state dict
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'])
                
                # Restore optimizer and scheduler states
                if 'optimizer_state_dict' in ckpt:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print("✅ Restored optimizer state.")
                
                if 'scheduler_state_dict' in ckpt:
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    print("✅ Restored scheduler state.")
                
                if 'scaler_state_dict' in ckpt:
                    self.scaler.load_state_dict(ckpt['scaler_state_dict'])
                    print("✅ Restored GradScaler state.")

                # Restore psf_library if it exists in the checkpoint
                if 'psf_library' in ckpt:
                    self.psf_library = ckpt['psf_library']
                    print("✅ Restored PSF library from checkpoint.")
                
                # Resume from next epoch
                self.start_epoch = ckpt.get('epoch', self.start_epoch - 1) + 1
            else:
                self.model.load_state_dict(ckpt)

    def train(self):
        print(f"Starting Training [{self.checkpoint_prefix}]: {self.epochs} epochs")
        
        # Warm up the Numba JIT compiler in the main thread to prevent LLVM crashes in forked workers
        print("🔥 Warming up Numba JIT compiler...")
        try:
            _ = self.train_loader.dataset[0]
        except Exception as e:
            print(f"⚠️ Numba warmup failed: {e}")

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train(); epoch_loss, start_time = 0, time.time()
            for i, batch in enumerate(self.train_loader):
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device, non_blocking=True)
                    # HDF5 transition: targets may be float16 on disk, must be float32 for loss
                    targets = batch["target"].to(self.device, non_blocking=True).float()
                    psf_library = batch["psf_library"].to(self.device, non_blocking=True)
                    
                    # Capture psf_library once for checkpointing
                    if self.psf_library is None:
                        self.psf_library = psf_library[0:1].detach().cpu()
                else:
                    images, targets = batch
                    images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True).float()
                    psf_library = None
                
                # --- GPU-ACCELERATED LIVE NOISE INJECTION ---
                # Ensure values are strictly positive before poisson
                images_positive = torch.clamp(images, min=0.0) 
                
                # 1. Poisson Noise (Photon Noise)
                images_noisy = torch.poisson(images_positive)
                
                # 2. Gaussian Read Noise (e.g., 5.0)
                images_noisy += torch.randn_like(images_noisy) * 5.0
                
                # Normalize via median (Done locally on GPU)
                batch_medians = images_noisy.view(images_noisy.shape[0], -1).median(dim=1)[0]
                batch_medians = batch_medians.view(-1, 1, 1, 1)
                
                # Apply your stretch scale
                images_final = torch.asinh((images_noisy - batch_medians) / GLOBAL_STRETCH_SCALE)
                # --------------------------------------------
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # 1. Forward pass in Mixed Precision
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    preds = self.model(images_final)
                
                # 2. Force FP32 before numerically sensitive loss calculation
                preds_fp32 = {k: v.float() for k, v in preds.items()}
                
                # FIX: Remove 'psf_library' argument which is no longer supported by compute_grid_loss
                loss, p_loss, po_loss, f_loss, s_loss, b_loss = compute_grid_loss(
                    preds_fp32, targets, pca_std=self.model.pca_std, **self.loss_params
                )
                
                # --- DIFFRACTION FILTER REGULARIZATION ---
                # This prevents the physics prior from drifting too far from initialization
                # and becoming a random convolutional layer.
                diffraction_reg = self.model.diffraction_filter.get_regularization_loss()
                reg_loss_val = diffraction_reg.item() # Raw L2 Distance
                reg_loss = self.lambda_diffraction * diffraction_reg
                loss += reg_loss
                # ------------------------------------------

                if torch.isnan(loss):
                    print(f"⚠️ NaN detected at step {i}")
                    continue
                    
                # 3. Scaled Backward Pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Only step scheduler if scaler didn't skip the optimizer step
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                epoch_loss += loss.item()
                
                if i % 100 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch [{epoch+1}/{self.epochs}], Step [{i}/{len(self.train_loader)}], LR: {current_lr:.6f}, Loss: {loss.item():.4f} (P:{p_loss.item():.4f}, Pos:{po_loss.item():.4f}, F:{f_loss.item():.4f}, S:{s_loss.item():.4f}, B:{b_loss.item():.4f}, DReg:{reg_loss_val:.6f})")

                # 4. FIX: Step scheduler if the optimizer was actually stepped
                if self.scaler.get_scale() >= scale_before:
                    self.scheduler.step()
            
            avg_epoch_loss = epoch_loss/len(self.train_loader)
            print(f"==> Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f} | Time: {time.time()-start_time:.1f}s")
            val_loss = self.validate(); print(f"Validation Loss: {val_loss:.4f}")
            
            os.makedirs("checkpoints", exist_ok=True)
            
            # Persist PSF Library to disk if it doesn't exist (Safety Layer)
            if self.psf_library is not None and not os.path.exists("master_psf_library.pt"):
                torch.save(self.psf_library, "master_psf_library.pt")
                print("💾 Persisted Master PSF Library from training batch to disk.")

            # Save full checkpoint dict for easier resuming
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'val_loss': val_loss,
                'psf_library': self.psf_library # Save PSF library
            }
            torch.save(checkpoint, os.path.join("checkpoints", f"{self.checkpoint_prefix}_epoch_{epoch+1}.pth"))
            self._prune_checkpoints()
        
        # Save final model state dict and include PSF library in a wrapper for inference compatibility
        final_ckpt = {
            'model_state_dict': self.model.state_dict(),
            'psf_library': self.psf_library
        }
        torch.save(final_ckpt, os.path.join("checkpoints", f"{self.checkpoint_prefix}_final.pth"))

    def _prune_checkpoints(self, keep_last=10):
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir): return
        
        pattern = re.compile(rf"{self.checkpoint_prefix}_epoch_(\d+)\.pth")
        checkpoints = []
        for f in os.listdir(checkpoint_dir):
            match = pattern.match(f)
            if match:
                epoch = int(match.group(1))
                checkpoints.append((epoch, f))
        
        # Sort by epoch
        checkpoints.sort()
        
        # Delete old ones
        if len(checkpoints) > keep_last:
            for i in range(len(checkpoints) - keep_last):
                file_to_delete = os.path.join(checkpoint_dir, checkpoints[i][1])
                try:
                    os.remove(file_to_delete)
                    print(f"🗑️ Pruned old checkpoint: {file_to_delete}")
                except OSError:
                    pass

    def validate(self):
        self.model.eval(); val_loss = 0
        num_batches = len(self.val_loader)
        if num_batches == 0:
            return 0.0 # Safety for empty val_loader
            
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device, non_blocking=True)
                    # HDF5 transition: targets may be float16 on disk, must be float32 for loss
                    targets = batch["target"].to(self.device, non_blocking=True).float()
                    psf_library = batch["psf_library"].to(self.device, non_blocking=True)
                else:
                    images, targets = batch
                    images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True).float()
                    psf_library = None
                
                # --- GPU-ACCELERATED LIVE NOISE INJECTION ---
                images_positive = torch.clamp(images, min=0.0) 
                images_noisy = torch.poisson(images_positive)
                images_noisy += torch.randn_like(images_noisy) * 5.0
                
                batch_medians = images_noisy.view(images_noisy.shape[0], -1).median(dim=1)[0]
                batch_medians = batch_medians.view(-1, 1, 1, 1)
                images_final = torch.asinh((images_noisy - batch_medians) / GLOBAL_STRETCH_SCALE)
                # --------------------------------------------
                    
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    preds = self.model(images_final)
                
                preds_fp32 = {k: v.float() for k, v in preds.items()}
                
                # FIX: Remove 'psf_library' argument and add 'pca_std'
                loss, _, _, _, _, _ = compute_grid_loss(
                    preds_fp32, targets, pca_std=self.model.pca_std, **self.loss_params
                )
                val_loss += loss.item()
        return val_loss / num_batches
