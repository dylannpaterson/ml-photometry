import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import re

from scripts.generate_simple_synthetic_data import GaussianStarDataset
from scripts.pregenerate_data import PregeneratedDataset
from models.dense_grid_model import DenseGridModel, compute_loss

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    files = os.listdir(checkpoint_dir)
    pattern = re.compile(r"bulge_survey_epoch_(\d+)\.pth")
    
    latest_epoch = 0
    latest_file = None
    
    for f in files:
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = os.path.join(checkpoint_dir, f)
                
    return latest_file, latest_epoch

from scripts.config_utils import load_config

def train():
    # 1. Configuration
    config = load_config()
    run_cfg = config["run_config"]
    hyper_cfg = config["training_hyperparams"]
    
    batch_size = hyper_cfg["batch_size"]
    lr = hyper_cfg["learning_rate"]
    epochs = hyper_cfg["epochs"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Setup
    train_dir = "data/train"
    val_dir = "data/val"
    
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
        print(f"Loading PREGENERATED dataset from {train_dir}")
        train_dataset = PregeneratedDataset(train_dir)
        val_dataset = PregeneratedDataset(val_dir)
    else:
        print("Pregenerated data not found. Falling back to ON-THE-FLY generation.")
        data_cfg = config["data_params"]
        dataset_params = {
            "num_samples": data_cfg["num_train_samples"],
            "min_stars": data_cfg["min_stars"],
            "max_stars": data_cfg["max_stars"],
            "image_size": data_cfg["image_size"]
        }
        train_dataset = GaussianStarDataset(**dataset_params)
        val_dataset = GaussianStarDataset(num_samples=data_cfg["num_val_samples"], 
                                         min_stars=data_cfg["min_stars"], 
                                         max_stars=data_cfg["max_stars"], 
                                         image_size=data_cfg["image_size"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model, Optimizer
    model = DenseGridModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Resume from Checkpoint
    latest_ckpt, start_epoch = find_latest_checkpoint()
    resume = run_cfg["resume_from_checkpoint"]
    
    if latest_ckpt and resume:
        print(f"Resuming from checkpoint: {latest_ckpt} (Epoch {start_epoch})")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    elif not resume:
        print("Starting training from scratch (resume_from_checkpoint=false).")
        start_epoch = 0
    else:
        print("Starting training from scratch (no checkpoint found).")

    # 5. Training Loop
    print(f"Starting Training (Edge-to-Edge): {epochs} epochs, starting at {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            
            loss, p_loss, r_loss = compute_loss(preds, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        duration = time.time() - start_time
        print(f"==> Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Time: {duration:.1f}s")

        # 6. Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                loss, _, _ = compute_loss(preds, targets)
                val_loss += loss.item()
        
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/bulge_survey_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "checkpoints/bulge_survey_final.pth")
    print("Final Model saved to checkpoints/bulge_survey_final.pth")

if __name__ == "__main__":
    train()
