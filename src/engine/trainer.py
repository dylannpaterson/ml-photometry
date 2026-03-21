import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import re
from src.models.dense_grid import compute_grid_loss

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.start_epoch = 0
        
        # Extract loss parameters from config
        self.loss_params = config["data_params"].get("loss_params", {}).copy()
        self.loss_params["stretch_scale"] = config["data_params"].get("GLOBAL_STRETCH_SCALE", 10.0)

    def resume(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path, self.start_epoch = find_latest_checkpoint(prefix=self.checkpoint_prefix)
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path} (Epoch {self.start_epoch})")
            # Handle full checkpoint dict or state dict
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(ckpt)

    def train(self):
        print(f"Starting Training [{self.checkpoint_prefix}]: {self.epochs} epochs")
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train(); epoch_loss, start_time = 0, time.time()
            for i, batch in enumerate(self.train_loader):
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    targets = batch["target"].to(self.device)
                else:
                    images, targets = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                preds = self.model(images)
                loss, p_loss, po_loss, f_loss, s_loss, b_loss = compute_grid_loss(preds, targets, **self.loss_params)
                
                if torch.isnan(loss):
                    print(f"⚠️ NaN detected at step {i}")
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Step [{i}/{len(self.train_loader)}], Loss: {loss.item():.4f} (P:{p_loss.item():.4f}, Pos:{po_loss.item():.4f}, F:{f_loss.item():.4f}, S:{s_loss.item():.4f}, B:{b_loss.item():.4f})")
            
            avg_epoch_loss = epoch_loss/len(self.train_loader)
            print(f"==> Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f} | Time: {time.time()-start_time:.1f}s")
            val_loss = self.validate(); print(f"Validation Loss: {val_loss:.4f}")
            
            self.scheduler.step(val_loss)
            
            os.makedirs("checkpoints", exist_ok=True)
            # Save full checkpoint dict for easier resuming
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join("checkpoints", f"{self.checkpoint_prefix}_epoch_{epoch+1}.pth"))
        torch.save(self.model.state_dict(), os.path.join("checkpoints", f"{self.checkpoint_prefix}_final.pth"))

    def validate(self):
        self.model.eval(); val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    targets = batch["target"].to(self.device)
                else:
                    images, targets = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                    
                preds = self.model(images)
                loss, _, _, _, _, _ = compute_grid_loss(preds, targets, **self.loss_params)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)
