import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from scripts.generate_simple_synthetic_data import GaussianStarDataset
from models.dense_grid_model import DenseGridModel, compute_loss

def train():
    # 1. Hyperparameters
    batch_size = 8 # Reduced batch size slightly due to much higher density
    lr = 5e-5      # Lowered learning rate for stability with Focal Loss
    epochs = 50     # Increased epochs for the harder realistic data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Setup
    # Bulge Survey settings: high density (up to 1500 stars/chunk)
    train_dataset = GaussianStarDataset(num_samples=5000, min_stars=500, max_stars=1500)
    val_dataset = GaussianStarDataset(num_samples=500, min_stars=500, max_stars=1500)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model, Optimizer
    model = DenseGridModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    print(f"Starting Training: {epochs} epochs, {len(train_loader)} steps per epoch")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_prob_loss = 0
        epoch_reg_loss = 0
        
        start_time = time.time()
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            
            # Using updated Focal Loss internally
            loss, p_loss, r_loss = compute_loss(preds, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_prob_loss += p_loss.item()
            epoch_reg_loss += r_loss.item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        avg_prob = epoch_prob_loss / len(train_loader)
        avg_reg = epoch_reg_loss / len(train_loader)
        
        duration = time.time() - start_time
        print(f"==> Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} (Prob: {avg_prob:.4f}, Reg: {avg_reg:.4f}) | Time: {duration:.1f}s")

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

    # 6. Save Final Model
    torch.save(model.state_dict(), "checkpoints/bulge_survey_final.pth")
    print("Final Model saved to checkpoints/bulge_survey_final.pth")

if __name__ == "__main__":
    train()