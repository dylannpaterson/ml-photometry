import argparse
import torch
import numpy as np
import os
import sys
import shutil
from castor.cloud.config_utils import load_config
from castor.models.dense_grid import DenseGridModel
from castor.data.dataset import PregeneratedDataset
from castor.engine.trainer import Trainer
from castor.engine.evaluator import Evaluator
# Removed top-level InferenceEngine import
from castor.engine.analyzer import ThresholdAnalyzer
from castor.data.stage0_gaussian import GaussianPretrainingProvider
from torch.utils.data import DataLoader
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE

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
        return None, None
    
    data_cfg = config["data_params"]
    stage_key = f"stage{stage_idx}"
    stage_cfg = config["curriculum"].get(stage_key, {})
    
    K = data_cfg.get("max_capacity_per_cell", MAX_CAPACITY_PER_CELL)
    S = data_cfg.get("shape_size", SHAPE_SIZE)
    # Get cell_size from stage config, default to DEFAULT_CELL_SIZE
    cell_size = stage_cfg.get("cell_size", DEFAULT_CELL_SIZE)
    
    model = DenseGridModel(K=K, shape_size=S, cell_size=cell_size).to(device)
    
    # Handle full checkpoint dict or raw state dict
    ckpt = torch.load(checkpoint_path, map_location=device)
    psf_library = None
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        
        # Capture psf_library if it exists
        if 'psf_library' in ckpt:
            psf_library = ckpt['psf_library']
    else:
        model.load_state_dict(ckpt)
        
    return model, psf_library

def ensure_stage0_data(stage_cfg, data_cfg, config_path):
    """Checks for HDF5 data and generates a small amount if missing."""
    train_mos_dir = os.path.join(stage_cfg["data_dir"], "mosaics_train")
    val_mos_dir = os.path.join(stage_cfg["data_dir"], "mosaics_val")
    val_h5 = os.path.join(stage_cfg["data_dir"], "stage0_val.h5")
    
    if not os.path.exists(val_h5):
        print("🔍 Local Stage 0 data not found. Triggering small-scale generation for inference/analysis...")
        os.makedirs(train_mos_dir, exist_ok=True)
        os.makedirs(val_mos_dir, exist_ok=True)
        
        # Generate small sets for quick local testing
        os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 scripts/generate_mosaics.py --num 1 --stage 0 --config {config_path} --output_dir {train_mos_dir}")
        os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 scripts/generate_mosaics.py --num 1 --stage 0 --config {config_path} --output_dir {val_mos_dir}")
        
        # Convert to a small HDF5 (100 samples is plenty for a few inference visuals)
        os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 scripts/convert_to_hdf5.py --train_dir {train_mos_dir} --val_dir {val_mos_dir} --output_dir {stage_cfg['data_dir']} --train_samples 100 --val_samples 100")
        
        # Cleanup
        shutil.rmtree(train_mos_dir, ignore_errors=True)
        shutil.rmtree(val_mos_dir, ignore_errors=True)

        if not os.path.exists(val_h5):
            print("❌ Error: Failed to generate local data fallback.")
            return False
            
    return True

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
    K = data_cfg.get("max_capacity_per_cell", MAX_CAPACITY_PER_CELL)
    S = data_cfg.get("shape_size", SHAPE_SIZE)
    cell_size = stage_cfg.get("cell_size", DEFAULT_CELL_SIZE)
    stretch_scale = data_cfg.get("GLOBAL_STRETCH_SCALE", GLOBAL_STRETCH_SCALE)

    if stage_idx == 0:
        train_h5 = os.path.join(stage_cfg["data_dir"], "stage0_train.h5")
        val_h5 = os.path.join(stage_cfg["data_dir"], "stage0_val.h5")
        train_mos_dir = os.path.join(stage_cfg["data_dir"], "mosaics_train")
        val_mos_dir = os.path.join(stage_cfg["data_dir"], "mosaics_val")
        
        # 1. Check if we actually NEED to generate anything
        needs_gen = force_gen or not os.path.exists(train_h5) or not os.path.exists(val_h5)
        
        if needs_gen:
            # Only generate raw mosaics if they don't already exist
            if force_gen or not os.path.exists(train_mos_dir) or not os.listdir(train_mos_dir):
                print("🛠️ Generating Mosaics for Stage 0...")
                cfg_path = config.get("config_path", "config/config.yaml")
                mos_cfg = stage_cfg.get("mosaic_params", {"num_mosaics": 5, "val_mosaics": 2})
                num_mos = mos_cfg.get("num_mosaics", 5)
                num_val_mos = mos_cfg.get("val_mosaics", 2)
                
                os.makedirs(train_mos_dir, exist_ok=True)
                os.makedirs(val_mos_dir, exist_ok=True)
                
                os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 scripts/generate_mosaics.py --num {num_mos} --stage {stage_idx} --config {cfg_path} --output_dir {train_mos_dir}")
                os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 scripts/generate_mosaics.py --num {num_val_mos} --stage {stage_idx} --config {cfg_path} --output_dir {val_mos_dir}")
            
            print(f"🛠️ HDF5 dataset conversion triggered (force_gen={force_gen})...")
            # Clear old ones if force_gen is true to avoid h5py append/overlap confusion
            if force_gen:
                if os.path.exists(train_h5): os.remove(train_h5)
                if os.path.exists(val_h5): os.remove(val_h5)
            
            # The conversion script now handles incremental cleanup of raw files
            ret = os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 scripts/convert_to_hdf5.py --train_dir {train_mos_dir} --val_dir {val_mos_dir} --output_dir {stage_cfg['data_dir']} --train_samples {data_cfg['num_train_samples']} --val_samples {data_cfg['num_val_samples']}")
            
            # Final cleanup
            if ret == 0:
                print(f"🧹 Final cleanup of raw mosaic directories...")
                shutil.rmtree(train_mos_dir, ignore_errors=True)
                shutil.rmtree(val_mos_dir, ignore_errors=True)

        from castor.data.stage0_gaussian import HDF5MosaicDataset
        print(f"🛠️ Using HDF5 Dataset: {train_h5}")
        train_dataset = HDF5MosaicDataset(train_h5)
        val_dataset = HDF5MosaicDataset(val_h5)
    elif stage_idx == 1:
        # NEW: Stage 1 Multi-Telescope Foundation Dataset (Macro-Sparse)
        from castor.data.stage1_dataset import Stage1MacroSparseDataset
        mosaic_dir = "data/stage1_mosaics"
        
        if force_gen or not os.path.exists(mosaic_dir) or not os.listdir(mosaic_dir):
            print("🛠️ Generating Stage 1 High-Fidelity Mosaics...")
            os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 scripts/generate_stage1_mosaics.py")

        print("🛠️ Using Stage 1 Macro-Sparse Pipeline (Cached Physics, Live Noise)...")
        train_dataset = Stage1MacroSparseDataset(
            mosaic_dir,
            num_samples=data_cfg["num_train_samples"],
            image_size=data_cfg["image_size"],
            cell_size=cell_size,
            K=K,
            global_stretch_scale=stretch_scale
        )
        val_dataset = Stage1MacroSparseDataset(
            mosaic_dir,
            num_samples=data_cfg["num_val_samples"],
            image_size=data_cfg["image_size"],
            cell_size=cell_size,
            K=K,
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
    
    # Enable hardware optimizations if multiple workers are used
    use_optimizations = num_workers > 0
    prefetch_factor = (4 if stage_idx == 0 else 2) if use_optimizations else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=use_optimizations,
        persistent_workers=use_optimizations,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=use_optimizations,
        persistent_workers=use_optimizations,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )

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
        
    if os.environ.get("PROFILE") == "1":
        import cProfile, pstats
        print("📊 Profiling training run...")
        profiler = cProfile.Profile()
        profiler.enable()
        trainer.train()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')
        stats.print_stats(50)
        stats.dump_stats("profile_results.prof")
        print("✅ Profiling complete. Results saved to profile_results.prof")
    else:
        trainer.train()
    print(f"✅ Stage {stage_idx} complete.")

def run_eval(stage_idx, config, device, checkpoint=None):
    print(f"--- 📊 Curriculum Stage {stage_idx}: Evaluation ---")
    model, _ = load_stage_model(stage_idx, device, config, checkpoint)
    if not model: return

    if stage_idx == 0:
        # Ensure data exists (Stage 0 only for now)
        stage_cfg = config["curriculum"]["stage0"]
        data_cfg = config["data_params"]
        config_path = config.get("config_path", "config/config.yaml")
        if not ensure_stage0_data(stage_cfg, data_cfg, config_path):
            return
            
        evaluator = Evaluator(model, device, config)
        # Increased to 500 chunks for better statistical stability
        evaluator.run_evaluation(num_chunks=500)
    else:
        print(f"⚠️ Specialized evaluator for stage {stage_idx} not yet implemented.")

def run_infer(stage_idx, config, device, checkpoint=None):
    from castor.engine.evaluator import match_stars
    from castor.engine.inference import InferenceEngine
    print(f"--- 🛰️ Curriculum Stage {stage_idx}: Inference ---")
    model, psf_lib_ckpt = load_stage_model(stage_idx, device, config, checkpoint)
    if not model: return

    engine = InferenceEngine(model, device, config)
    
    # Stage-specific provider
    if stage_idx == 0:
        from castor.data.stage0_gaussian import HDF5MosaicDataset
        data_cfg = config["data_params"]
        stage_cfg = config["curriculum"]["stage0"]
        config_path = config.get("config_path", "config/config.yaml")
        
        # Ensure data exists fallback
        if not ensure_stage0_data(stage_cfg, data_cfg, config_path):
            return
            
        val_h5 = os.path.join(stage_cfg["data_dir"], "stage0_val.h5")
        dataset = HDF5MosaicDataset(val_h5)
        import random
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        
        image_tensor = sample["image"]
        target = sample["target"]
        
        # --- PSF Basis Loading ---
        # Priority: Checkpoint embedded library > Local .pt file > Dataset sample
        if psf_lib_ckpt is not None:
            print("📂 Using PSF basis embedded in model checkpoint...")
            psf_lib = psf_lib_ckpt.squeeze(0) # [N_PCA + 1, 961]
        elif os.path.exists("stage0_psf_basis.pt"):
            print(f"📂 Loading PSF basis from stage0_psf_basis.pt...")
            psf_lib = torch.load("stage0_psf_basis.pt", map_location="cpu")
        else:
            psf_lib = sample["psf_library"] # [N_PCA + 1, 961]
            
        # Extract basis and mean for reconstruction
        # psf_lib is expected to be [N_PCA + 1, 961]
        psf_basis = psf_lib[:-1, :]
        mean_psf = psf_lib[-1, :]
        # -------------------------
        
        # --- THE FIX: Apply Live Noise and Stretch ---
        stretch_scale = data_cfg.get("GLOBAL_STRETCH_SCALE", 10.0)
        
        img_pos = torch.clamp(image_tensor, min=0.0)
        img_noisy = torch.poisson(img_pos)
        img_noisy += torch.randn_like(img_noisy) * 5.0  # Read noise
        
        # Calculate the median of the NOISY image to center the stretch
        noisy_median = img_noisy.median().item()
        
        # Apply the Arcsinh stretch (Network Space)
        img_stretched = torch.arcsinh((img_noisy - noisy_median) / stretch_scale)
        # ---------------------------------------------
        
        # Extract true stars from the target grid for visualization
        true_stars = []
        cell_size = dataset.cell_size
        grid_size = dataset.grid_size
        K = dataset.K
        
        # target shape is (grid_size, grid_size, K*4 + 1)
        target_grid = target[:, :, :-1].view(grid_size, grid_size, K, -1).numpy()
        gt_bg_map = target[:, :, -1:].numpy()
        
        for y in range(grid_size):
            for x in range(grid_size):
                for k in range(K):
                    slot = target_grid[y, x, k]
                    tp = slot[0]
                    # NEW: For visualization, we keep stars with visible labels
                    if tp > 0.0:
                        tdx, tdy, raw_flux = slot[1], slot[2], slot[3]
                        
                        tgx = (x * cell_size) + tdx
                        tgy = (y * cell_size) + tdy
                        # [p, dx, dy, flux]
                        true_stars.append((tp, tgx, tgy, float(raw_flux)))
        
        print(f"DEBUG: Found {len(true_stars)} stars with non-zero objectness labels in the chunk.")
        # Pass PCA reconstruction components to predict
        # FIX: Pass raw linear noisy image, as predict() handles its own stretch
        predicted_stars, predicted_shapes, bg_map = engine.predict(
            img_noisy, 
            psf_basis=psf_basis.numpy(), 
            mean_psf=mean_psf.numpy()
        )
        
        # --- THE FIX: Extract Jitter for Visualization ---
        # Metadata: [exp_time, zp, sky_mag, s_jit, q_jit, theta_jit]
        # We need the last 3 values for the reconstruction to not look "blocky" (sharp)
        try:
            with h5py.File(val_h5, 'r') as f:
                # Get the metadata for the specific mosaic index
                # We need a way to link sample index to mosaic index
                # For now, we assume the dataset returns metadata or we check common filenames
                # Fallback: Check if metadata is available in the sample
                if "meta" in sample:
                    meta = sample["meta"]
                    jitter_params = (meta[3], meta[4], meta[5])
                else:
                    jitter_params = None
        except:
            jitter_params = None
        # -------------------------------------------------
        
        # DEBUG: Print normalization stats
        # Note: predicted_stars elements are (x, y, flux, p, sigmas)
        # match_stars expects (x, y, flux) for matching
        match_true = [(s[1], s[2], s[3]) for s in true_stars]
        match_pred = [(s[0], s[1], s[2]) for s in predicted_stars]
        matches, _, _ = match_stars(match_true, match_pred)
        if matches:
            ratios = []
            print("\n--- Normalization Diagnostic ---")
            for i in range(len(matches)):
                t_idx, p_idx, _ = matches[i]
                t_flux = true_stars[t_idx][3]
                p_flux = predicted_stars[p_idx][2]
                ratios.append(p_flux / t_flux)
                if i < 5:
                    print(f"Star {i}: True Flux={t_flux:7.1f}, Pred Flux={p_flux:7.1f}, Ratio={ratios[-1]:.3f}")
            
            print(f"\nMean Ratio (Pred/True): {np.mean(ratios):.4f}")
            print(f"Median Ratio:           {np.median(ratios):.4f}")
            print(f"Std Dev of Ratio:       {np.std(ratios):.4f}")
        else:
            print("\n--- Normalization Diagnostic: No matches found ---")
            
        engine.visualize(
            img_stretched, 
            true_stars, 
            predicted_stars, 
            predicted_shapes, 
            bg_map, 
            gt_bg_map, 
            threshold=0.5, 
            chunk_median=noisy_median,
            jitter_params=jitter_params,
            psf_basis=psf_basis.numpy(),
            mean_psf=mean_psf.numpy()
        )
    else:
        print(f"⚠️ Specialized inference for stage {stage_idx} not yet implemented.")

def run_analyze(stage_idx, config, device, checkpoint=None):
    print(f"--- 📈 Curriculum Stage {stage_idx}: Threshold Analysis ---")
    model, _ = load_stage_model(stage_idx, device, config, checkpoint)
    if not model: return

    if stage_idx == 0:
        from castor.data.stage0_gaussian import HDF5MosaicDataset
        stage_cfg = config["curriculum"]["stage0"]
        data_cfg = config["data_params"]
        config_path = config.get("config_path", "config/config.yaml")
        
        # Ensure data exists fallback
        if not ensure_stage0_data(stage_cfg, data_cfg, config_path):
            return
            
        val_h5 = os.path.join(stage_cfg["data_dir"], "stage0_val.h5")
        dataset = HDF5MosaicDataset(val_h5)
        analyzer = ThresholdAnalyzer(model, device, dataset)
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
