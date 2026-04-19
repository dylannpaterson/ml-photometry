import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import subprocess
from castor.cloud.config_utils import load_config
from castor.models.dense_grid import DenseGridModel
from castor.constants import GLOBAL_STRETCH_SCALE, DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE

class HandoverWrapper(nn.Module):
    """
    Bakes Preprocessing into the ONNX Graph for Pollux.
    Accepts raw linear images [Batch, 1, H, W] and an optional Confidence Prior [Batch, 1, H, W].
    Outputs standard catalogs.
    """
    def __init__(self, base_model, stretch_scale=GLOBAL_STRETCH_SCALE):
        super(HandoverWrapper, self).__init__()
        self.base_model = base_model
        self.register_buffer("stretch_scale", torch.tensor(stretch_scale))

    def forward(self, x, prior=None):
        # 1. Input Validation (Ensure Batch, Channel, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # 2. Per-image Median Subtraction (Baking preprocessing into graph)
        # Flatten H,W to compute median per image in batch
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        
        # ONNX-compatible median (Sort and take middle)
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        num_pixels = H * W
        
        # We use sort instead of median() as it has better ONNX support in some environments
        sorted_x, _ = torch.sort(x_flat, dim=1)
        medians = sorted_x[:, num_pixels // 2]
        medians = medians.view(B, 1, 1, 1)
        
        # 3. Arcsinh Stretch
        x_stretched = torch.arcsinh((x - medians) / self.stretch_scale)
        
        # 4. Handle Prior
        if prior is None:
            prior = torch.zeros_like(x)
        
        # 5. Base Model Inference
        preds = self.base_model(x_stretched, prior=prior)
        
        # 6. Return structured tuple for ONNX compatibility
        # stars: [Batch, Hg, Wg, K, 7] -> (p, dx, dy, flux, log_var_x, log_var_y, log_var_m)
        # background: [Batch, Hg, Wg, 1]
        return preds["stars"], preds["background"]

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

def main():
    parser = argparse.ArgumentParser(description="Export Castor Model to ONNX for Pollux Handover")
    parser.add_argument("--stage", type=int, default=0, help="Stage index")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default=None, help="Path to specific model checkpoint")
    parser.add_argument("--output_dir", default="artifacts", help="Where to save the ONNX and Parity files")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    device = torch.device("cpu") # Export on CPU for maximum compatibility
    
    # 1. Load Model
    data_cfg = config["data_params"]
    stage_key = f"stage{args.stage}"
    stage_cfg = config["curriculum"].get(stage_key, {})
    
    K = data_cfg.get("max_capacity_per_cell", MAX_CAPACITY_PER_CELL)
    S = data_cfg.get("shape_size", SHAPE_SIZE)
    cell_size = stage_cfg.get("cell_size", DEFAULT_CELL_SIZE)
    img_size = data_cfg.get("image_size", 256)
    
    base_model = DenseGridModel(K=K, shape_size=S, cell_size=cell_size).to(device)
    
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/stage{args.stage}_final.pth"
    
    if not os.path.exists(checkpoint_path):
        # Try latest epoch if final doesn't exist
        from castor.engine.trainer import find_latest_checkpoint
        latest, _ = find_latest_checkpoint(prefix=f"stage{args.stage}")
        if latest:
            checkpoint_path = latest
        else:
            print(f"❌ Error: No checkpoint found for stage {args.stage}")
            return

    print(f"📂 Loading weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        base_model.load_state_dict(ckpt['model_state_dict'])
    else:
        base_model.load_state_dict(ckpt)
    
    base_model.eval()
    
    # 2. Wrap with Preprocessing
    model = HandoverWrapper(base_model, stretch_scale=GLOBAL_STRETCH_SCALE)
    model.eval()
    
    # 3. Prepare Artifact Versioning
    git_hash = get_git_hash()
    os.makedirs(args.output_dir, exist_ok=True)
    onnx_filename = f"stage{args.stage}_{git_hash}.onnx"
    onnx_path = os.path.join(args.output_dir, onnx_filename)
    
    # 4. Generate Golden Master Parity Data
    print("💎 Generating Golden Master Parity Test data...")
    dummy_input = torch.randn(1, 1, img_size, img_size) * 50.0 + 100.0 # Random ADU-like data
    dummy_prior = torch.zeros(1, 1, img_size, img_size) # Default empty prior
    with torch.no_grad():
        pytorch_stars, pytorch_bg = model(dummy_input, prior=dummy_prior)
    
    np.save(os.path.join(args.output_dir, f"test_input_{git_hash}.npy"), dummy_input.numpy())
    np.save(os.path.join(args.output_dir, f"test_prior_{git_hash}.npy"), dummy_prior.numpy())
    np.save(os.path.join(args.output_dir, f"test_output_stars_{git_hash}.npy"), pytorch_stars.numpy())
    np.save(os.path.join(args.output_dir, f"test_output_bg_{git_hash}.npy"), pytorch_bg.numpy())
    
    # 5. Export to ONNX
    print(f"🚀 Exporting ONNX model to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_input, dummy_prior),
        onnx_path,
        export_params=True,
        opset_version=12, # Supports arcsinh and median
        do_constant_folding=True,
        input_names=['input_adu', 'prior_map'],
        output_names=['stars', 'background'],
        dynamic_axes={
            'input_adu': {0: 'batch_size'},
            'prior_map': {0: 'batch_size'},
            'stars': {0: 'batch_size'},
            'background': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Handover artifact ready: {onnx_path}")
    print(f"📦 Contents: ONNX Model + Parity Tensors (Input, Stars, Background)")

if __name__ == "__main__":
    main()
