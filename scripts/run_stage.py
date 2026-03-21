import argparse
import torch
import numpy as np
import os
import sys
from src.cloud.config_utils import load_config
from src.models.dense_grid import DenseGridModel
from src.data.dataset import PregeneratedDataset
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
# Removed top-level InferenceEngine import
from src.engine.analyzer import ThresholdAnalyzer
from src.data.stage0_gaussian import GaussianPretrainingProvider
from torch.utils.data import DataLoader

def get_stage_config(config, stage_idx):
    """Extracts configuration for a specific curriculum stage."""
    curriculum = config.get("curriculum", {})
    stage_key = f"stage{stage_idx}"
    if stage_key not in curriculum:
        print(f"❌ Error: Stage {stage_idx} not defined in config.")
        sys.exit(1)
    return curriculum[stage_key]

def load_stage_model(stage_idx, device, config, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = f"checkpoints/stage{stage_idx}_final.pth"
        
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
        return None
    
    data_cfg = config["data_params"]
    stage_key = f"stage{stage_idx}"
    stage_cfg = config["curriculum"].get(stage_key, {})
    
    K = data_cfg["max_capacity_per_cell"]
    S = data_cfg["shape_size"]
    # Get cell_size from stage config, default to 4
    cell_size = stage_cfg.get("cell_size", 4)
    
    model = DenseGridModel(K=K, shape_size=S, cell_size=cell_size).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model

def run_train(stage_idx, config, device):
    print(f"--- 🚀 Curriculum Stage {stage_idx}: Training ---")
    stage_key = f"stage{stage_idx}"
    stage_cfg = config["curriculum"].get(stage_key, {})
    data_cfg = config["data_params"]
    stage_prefix = f"stage{stage_idx}"
    
    # 1. Cleanup old checkpoints if not resuming
    resume_from_last = stage_cfg.get("resume_from_last_stage", False)
    resume_from_ckpt = config["run_config"].get("resume_from_checkpoint", False)
    force_gen = config["run_config"].get("force_regenerate_data", False)
    
    if not resume_from_last and not resume_from_ckpt:
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            print(f"🧹 Cleaning up old checkpoints for {stage_prefix}...")
            for f in os.listdir(checkpoint_dir):
                if f.startswith(stage_prefix) and f.endswith(".pth"):
                    os.remove(os.path.join(checkpoint_dir, f))

    # Data Setup
    K = data_cfg["max_capacity_per_cell"]
    S = data_cfg["shape_size"]
    cell_size = stage_cfg.get("cell_size", 4)
    stretch_scale = data_cfg.get("GLOBAL_STRETCH_SCALE", 10.0)

    if stage_idx == 0:
        mosaic_dir = os.path.join(stage_cfg["data_dir"], "mosaics")
        if force_gen or not os.path.exists(mosaic_dir) or not os.listdir(mosaic_dir):
            print("🛠️ Generating Mosaics for Stage 0...")
            cfg_path = config.get("config_path", "config/config.yaml")
            # Extract num_mosaics from config if available
            mos_cfg = stage_cfg.get("mosaic_params", {"num_mosaics": 5})
            num_mos = mos_cfg.get("num_mosaics", 5)
            # Use the correct config file path
            os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 scripts/generate_mosaics.py --num {num_mos} --stage {stage_idx} --config {cfg_path}")

        from src.data.stage0_gaussian import GaussianMosaicDataset
        print("🛠️ Using Mosaic Sampling for high-speed training & validation...")
        train_dataset = GaussianMosaicDataset(
            mosaic_dir,
            num_samples=data_cfg["num_train_samples"],
            image_size=data_cfg["image_size"],
            cell_size=cell_size,
            global_stretch_scale=stretch_scale
        )
        # Use the same mosaics for validation but with a fixed sample count
        val_dataset = GaussianMosaicDataset(
            mosaic_dir,
            num_samples=data_cfg["num_val_samples"],
            image_size=data_cfg["image_size"],
            cell_size=cell_size,
            global_stretch_scale=stretch_scale
        )
    else:
        data_dir = stage_cfg["data_dir"]
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")

        if not os.path.exists(train_dir) or not os.listdir(train_dir):
            print(f"❌ Error: Data not found in {train_dir}. Run 'gen' for stage {stage_idx} first.")
            return

        train_dataset = PregeneratedDataset(train_dir, K=K, shape_size=S)
        val_dataset = PregeneratedDataset(val_dir, K=K, shape_size=S)

    
    batch_size = stage_cfg["batch_size"]
    num_workers = stage_cfg.get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model Setup
    model = DenseGridModel(K=K, shape_size=S, cell_size=cell_size).to(device)
    
    # Custom Trainer Setup
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        config, 
        device, 
        epochs=stage_cfg["epochs"],
        lr=stage_cfg["learning_rate"],
        checkpoint_prefix=stage_prefix
    )
    
    if resume_from_last and stage_idx > 0:
        last_stage_model = f"checkpoints/stage{stage_idx-1}_final.pth"
        if os.path.exists(last_stage_model):
            print(f"📈 Resuming from Stage {stage_idx-1} weights...")
            model.load_state_dict(torch.load(last_stage_model, map_location=device))
    elif resume_from_ckpt:
        trainer.resume()
        
    trainer.train()
    print(f"✅ Stage {stage_idx} complete.")

def run_eval(stage_idx, config, device, checkpoint=None):
    print(f"--- 📊 Curriculum Stage {stage_idx}: Evaluation ---")
    model = load_stage_model(stage_idx, device, config, checkpoint)
    if not model: return

    if stage_idx == 0:
        evaluator = Evaluator(model, device, config)
        # Increased to 500 chunks for better statistical stability
        evaluator.run_evaluation(num_chunks=500)
    else:
        print(f"⚠️ Specialized evaluator for stage {stage_idx} not yet implemented.")

def run_infer(stage_idx, config, device, checkpoint=None):
    from src.engine.evaluator import match_stars
    from src.engine.inference import InferenceEngine
    print(f"--- 🛰️ Curriculum Stage {stage_idx}: Inference ---")
    model = load_stage_model(stage_idx, device, config, checkpoint)
    if not model: return

    engine = InferenceEngine(model, device, config)
    
    # Stage-specific provider
    if stage_idx == 0:
        data_cfg = config["data_params"]
        stretch_scale = data_cfg.get("GLOBAL_STRETCH_SCALE", 10.0)
        provider = GaussianPretrainingProvider(
            min_stars=data_cfg["min_stars"],
            max_stars=data_cfg["max_stars"],
            image_size=data_cfg["image_size"],
            max_capacity_per_cell=data_cfg["max_capacity_per_cell"],
            shape_size=data_cfg["shape_size"],
            global_stretch_scale=stretch_scale
        )
        
        # generate_chunk now returns a sparse dict
        sparse_sample = provider.generate_chunk()
        image_tensor = sparse_sample["image"]
        gt_bg_map = sparse_sample["background_map"].numpy()
        chunk_median = sparse_sample.get("chunk_median", 0.0)
        
        # Extract true stars from the target base_grid for visualization
        true_stars = []
        target_grid = sparse_sample["base_grid"].numpy()
        cell_size, grid_size = provider.cell_size, provider.grid_size
        K = provider.K
        for y in range(grid_size):
            for x in range(grid_size):
                for k in range(K):
                    slot = target_grid[y, x, k]
                    tp, tdx, tdy, tm, tc = slot
                    if tp == 1.0:
                        tgx = (x * cell_size) + tdx
                        tgy = (y * cell_size) + tdy
                        # m in target is log10(flux), convert to linear for visualize
                        true_stars.append((tgx, tgy, 10**tm, tc))
        
        predicted_stars, predicted_shapes, bg_map = engine.predict(image_tensor)
        
        # DEBUG: Print normalization stats
        matches, _, _ = match_stars(true_stars, predicted_stars)
        if matches:
            ratios = []
            print("\n--- Normalization Diagnostic ---")
            for i in range(len(matches)):
                t_idx, p_idx, _ = matches[i]
                t_flux = true_stars[t_idx][2]
                p_flux = predicted_stars[p_idx][2]
                ratios.append(p_flux / t_flux)
                if i < 5:
                    print(f"Star {i}: True Flux={t_flux:7.1f}, Pred Flux={p_flux:7.1f}, Ratio={ratios[-1]:.3f}")
            
            print(f"\nMean Ratio (Pred/True): {np.mean(ratios):.4f}")
            print(f"Median Ratio:           {np.median(ratios):.4f}")
            print(f"Std Dev of Ratio:       {np.std(ratios):.4f}")
        else:
            print("\n--- Normalization Diagnostic: No matches found ---")
            
        engine.visualize(image_tensor, true_stars, predicted_stars, predicted_shapes, bg_map, gt_bg_map, threshold=0.5, chunk_median=chunk_median)
    else:
        print(f"⚠️ Specialized inference for stage {stage_idx} not yet implemented.")

def run_analyze(stage_idx, config, device, checkpoint=None):
    print(f"--- 📈 Curriculum Stage {stage_idx}: Threshold Analysis ---")
    model = load_stage_model(stage_idx, device, config, checkpoint)
    if not model: return

    if stage_idx == 0:
        provider = GaussianPretrainingProvider(
            min_stars=config["data_params"]["min_stars"],
            max_stars=config["data_params"]["max_stars"],
            image_size=config["data_params"]["image_size"],
            global_stretch_scale=config["data_params"].get("GLOBAL_STRETCH_SCALE", 10.0)
        )
        analyzer = ThresholdAnalyzer(model, device, provider)
        analyzer.run_analysis(num_chunks=20)
    else:
        print(f"⚠️ Specialized analysis for stage {stage_idx} not yet implemented.")

def main():
    parser = argparse.ArgumentParser(description="Roman Point Source Curriculum Runner")
    parser.add_argument("stage", type=int, help="Curriculum stage index")
    parser.add_argument("action", choices=["train", "eval", "infer", "analyze"], help="Action to perform")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default=None, help="Path to specific model checkpoint")
    
    args = parser.parse_args()
    config = load_config(args.config)
    config["config_path"] = args.config # Store for sub-scripts
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    actions = {
        "train": run_train,
        "eval": run_eval,
        "infer": run_infer,
        "analyze": run_analyze
    }
    
    if args.action in ["eval", "infer", "analyze"]:
        actions[args.action](args.stage, config, device, args.checkpoint)
    else:
        actions[args.action](args.stage, config, device)

if __name__ == "__main__":
    main()
