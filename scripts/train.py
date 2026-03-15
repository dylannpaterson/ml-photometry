import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from scripts.generate_simple_synthetic_data import GaussianStarDataset
from scripts.pregenerate_data import PregeneratedDataset
from models.dense_grid_model import DenseGridModel, compute_loss

def train():
    # 1. Hyperparameters
    batch_size = 16 
    lr = 1e-4      
    epochs = 50     
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
        train_dataset = GaussianStarDataset(num_samples=5000, min_stars=500, max_stars=1500, image_size=256)
        val_dataset = GaussianStarDataset(num_samples=500, min_stars=500, max_stars=1500, image_size=256)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model, Optimizer
    model = DenseGridModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    print(f"Starting Training (Edge-to-Edge): {epochs} epochs")
    
    for epoch in range(epochs):
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

        # 5. Validation
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
