import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import re
from src.models.dense_grid import compute_grid_loss

def find_latest_checkpoint(checkpoint_dir="checkpoints", prefix="stage0"):
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # Matches prefix_epoch_N.pth
    pattern = re.compile(rf"{prefix}_epoch_(\d+)\.pth")
    
    latest_epoch = 0
    latest_file = None
    
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        for f in files:
            match = pattern.match(f)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_file = os.path.join(checkpoint_dir, f)
                
    return latest_file, latest_epoch

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, epochs=100, lr=0.0001, checkpoint_prefix="stage0"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_prefix = checkpoint_prefix
        self.epochs = epochs
        self.lr = lr
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.start_epoch = 0

    def resume(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path, self.start_epoch = find_latest_checkpoint(prefix=self.checkpoint_prefix)
        
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path} (Epoch {self.start_epoch})")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            print(f"No checkpoint found with prefix '{self.checkpoint_prefix}' to resume from.")

    def train(self):
        print(f"Starting Training [{self.checkpoint_prefix}]: {self.epochs} epochs, starting at {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            epoch_loss = 0
            start_time = time.time()
            
            for i, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                preds = self.model(images)
                
                loss, p_loss, r_loss, s_loss, b_loss = compute_grid_loss(preds, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Step [{i}/{len(self.train_loader)}], Loss: {loss.item():.4f} (P:{p_loss.item():.4f}, R:{r_loss.item():.4f}, S:{s_loss.item():.4f}, B:{b_loss.item():.4f})")

            avg_loss = epoch_loss / len(self.train_loader)
            duration = time.time() - start_time
            print(f"==> Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Time: {duration:.1f}s")

            # Validation
            val_loss = self.validate()
            print(f"Validation Loss: {val_loss:.4f}")

            # Save epoch checkpoint
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_name = f"{self.checkpoint_prefix}_epoch_{epoch+1}.pth"
            torch.save(self.model.state_dict(), os.path.join("checkpoints", ckpt_name))

        final_path = os.path.join("checkpoints", f"{self.checkpoint_prefix}_final.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"Final Model for stage saved to {final_path}")

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                preds = self.model(images)
                loss, _, _, _, _ = compute_grid_loss(preds, targets)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)
