import argparse
import torch
import numpy as np
import os
import sys
import shutil
from castor.cloud.config_utils import load_config
from castor.models.dense_grid import DenseGridModel
from castor.engine.trainer import Trainer
from castor.engine.evaluator import Evaluator
# Removed top-level InferenceEngine import
from castor.engine.analyzer import ThresholdAnalyzer
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

def get_safe_batch_size(target_batch_size, device):
    """Detects VRAM and scales batch size to avoid OOM, return (micro_batch, accumulation_steps)."""
    if device.type != 'cuda':
        return target_batch_size, 1

    vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    # ResNet-34 + FPN on 256x256 is lightweight (~100MB per sample)
    if vram_gb < 8:
        micro_batch = 16
    elif vram_gb < 12:
        micro_batch = 32
    elif vram_gb < 16:
        micro_batch = 64
    else:
        micro_batch = 128

    micro_batch = min(micro_batch, target_batch_size)
    acc_steps = max(1, target_batch_size // micro_batch)
    return micro_batch, acc_steps
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
                
                # Check for existing library to ensure consistency
                lib_arg = "--psf_library master_psf_library.pt" if os.path.exists("master_psf_library.pt") else ""
                
                os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 scripts/generate_mosaics.py --num {num_mos} --stage {stage_idx} --config {cfg_path} --output_dir {train_mos_dir} {lib_arg}")
                os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 scripts/generate_mosaics.py --num {num_val_mos} --stage {stage_idx} --config {cfg_path} --output_dir {val_mos_dir} {lib_arg}")
            
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
        print(f"❌ Error: Stage {stage_idx} data loading via PregeneratedDataset is obsolete.")
        print(f"   Please implement HDF5MosaicDataset support for Stage {stage_idx}.")
        return

    
    batch_size = stage_cfg["batch_size"]
    num_workers = stage_cfg.get("num_workers", 0)
    
    # NEW: Memory-Aware Dynamic Batch Scaling
    micro_batch, acc_steps = get_safe_batch_size(batch_size, device)
    if micro_batch < batch_size:
        print(f"📟 Memory Safety: Scaling physical batch {batch_size} -> {micro_batch} (Steps: {acc_steps})")
    
    # Enable hardware optimizations if multiple workers are used
    use_optimizations = num_workers > 0
    prefetch_factor = (4 if stage_idx == 0 else 2) if use_optimizations else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=micro_batch, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=use_optimizations,
        persistent_workers=use_optimizations,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=micro_batch, 
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
    from castor.engine.inference import InferenceEngine, generate_custom_inference_prior
    import random
    import h5py
    print(f"--- 🛰️ Curriculum Stage {stage_idx}: Inference ---")
    model, psf_lib_ckpt = load_stage_model(stage_idx, device, config, checkpoint)
    if model is None: return

    engine = InferenceEngine(model, device, config)
    
    if stage_idx == 0:
        from castor.data.stage0_gaussian import HDF5MosaicDataset
        data_cfg = config["data_params"]
        stage_cfg = config["curriculum"]["stage0"]
        config_path = config.get("config_path", "config/config.yaml")
        
        if not ensure_stage0_data(stage_cfg, data_cfg, config_path):
            return
            
        val_h5 = os.path.join(stage_cfg["data_dir"], "stage0_val.h5")
        dataset = HDF5MosaicDataset(val_h5)
        
        # NEW: Respect CLI --num_chunks argument if provided
        num_chunks_to_infer = config.get("num_chunks_override", min(20, len(dataset)))
        print(f"🔭 Running Global Inference over {num_chunks_to_infer} chunks (Standard + Oracle)...")
        
        # Four sets of aggregations
        global_true = []
        global_pred_flat = []
        global_pred_oracle = []
        global_pred_oracle_p43 = []
        global_pred_oracle_p100 = []
        
        hero_data_flat = None
        hero_data_oracle = None
        hero_data_oracle_p43 = None
        hero_data_oracle_p100 = None
        
        # Robust PSF Extraction logic (shared for all chunks)
        master_mean_psf = None
        master_psf_basis = None

        for idx in range(num_chunks_to_infer):
            sample = dataset[idx]
            image_tensor = sample["image"]
            target = sample["target"]
            
            # --- One-time PSF extraction ---
            if master_mean_psf is None:
                psf_lib_data = psf_lib_ckpt if psf_lib_ckpt is not None else sample["psf_library"]
                if isinstance(psf_lib_data, dict):
                    master_mean_psf = psf_lib_data.get('mean_psf')
                    master_psf_basis = psf_lib_data.get('eigen_psfs')
                elif isinstance(psf_lib_data, (list, tuple)):
                    master_psf_basis = psf_lib_data[0]
                    master_mean_psf = psf_lib_data[2] if len(psf_lib_data) > 2 else psf_lib_data[-1]
                else:
                    data = psf_lib_data if isinstance(psf_lib_data, np.ndarray) else psf_lib_data.cpu().numpy()
                    data = np.squeeze(data)
                    if data.ndim == 2: master_mean_psf = data
                    elif data.ndim == 1: master_mean_psf = data.reshape(int(data.shape[0]**0.5), -1)
                    else:
                        master_psf_basis = data[:-1]
                        master_mean_psf = data[-1]
                        if master_mean_psf.ndim == 1:
                            s = int(master_mean_psf.shape[0]**0.5)
                            master_mean_psf = master_mean_psf.reshape(s, s)
                            master_psf_basis = master_psf_basis.reshape(-1, s, s)

                if torch.is_tensor(master_mean_psf): master_mean_psf = master_mean_psf.detach().cpu().numpy()
                S_full = master_mean_psf.shape[0]
                if S_full > SHAPE_SIZE:
                    O = S_full // SHAPE_SIZE
                    master_mean_psf = master_mean_psf.reshape(SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(1, 3))
                    if master_psf_basis is not None:
                        master_psf_basis = master_psf_basis.reshape(-1, S_full, S_full)
                        master_psf_basis = master_psf_basis.reshape(-1, SHAPE_SIZE, O, SHAPE_SIZE, O).mean(axis=(2, 4))
                master_mean_psf /= (np.sum(master_mean_psf) + 1e-9)
                if master_psf_basis is not None: master_psf_basis = master_psf_basis.reshape(-1, SHAPE_SIZE * SHAPE_SIZE)

            # --- Physical Prep ---
            stretch_scale = data_cfg.get("GLOBAL_STRETCH_SCALE", 10.0)
            img_pos = torch.clamp(image_tensor, min=0.0)
            img_noisy = torch.poisson(img_pos)
            img_noisy += torch.randn_like(img_noisy) * 5.0
            noisy_median = img_noisy.median().item()
            img_stretched = torch.arcsinh((img_noisy - noisy_median) / stretch_scale)

            # Extract Truth for this chunk
            true_stars_chunk = []
            cell_size, grid_size, K = dataset.cell_size, dataset.grid_size, dataset.K
            target_grid = target[:, :, :-1].view(grid_size, grid_size, K, -1).numpy()
            gt_bg_map = target[:, :, -1:].numpy()
            
            for y in range(grid_size):
                for x in range(grid_size):
                    for k in range(K):
                        slot = target_grid[y, x, k]
                        if slot[0] > 0.0:
                            true_stars_chunk.append((slot[0], (x * cell_size) + slot[1], (y * cell_size) + slot[2], float(slot[3])))

            # 1. Standard Prediction (Flat Prior)
            pred_stars_flat, _, bg_map_flat = engine.predict(img_noisy, threshold=0.0, psf_basis=master_psf_basis, mean_psf=master_mean_psf)
            
            # 2. Oracle Prediction (Perfect Prior - All Stars)
            perfect_catalog = [(s[1], s[2], s[0]) for s in true_stars_chunk]
            oracle_prior_map = generate_custom_inference_prior(perfect_catalog, img_size=image_tensor.shape[-1], sigma=1.5, device=device)
            pred_stars_oracle, _, bg_map_oracle = engine.predict(img_noisy, threshold=0.0, psf_basis=master_psf_basis, mean_psf=master_mean_psf, prior_map=oracle_prior_map)

            # 3. Oracle Prediction (p >= 0.43)
            perfect_catalog_p43 = [(s[1], s[2], s[0]) for s in true_stars_chunk if s[0] >= 0.43]
            oracle_prior_map_p43 = generate_custom_inference_prior(perfect_catalog_p43, img_size=image_tensor.shape[-1], sigma=1.5, device=device)
            pred_stars_oracle_p43, _, bg_map_oracle_p43 = engine.predict(img_noisy, threshold=0.0, psf_basis=master_psf_basis, mean_psf=master_mean_psf, prior_map=oracle_prior_map_p43)

            # 4. Oracle Prediction (p >= 1.0)
            perfect_catalog_p100 = [(s[1], s[2], s[0]) for s in true_stars_chunk if s[0] >= 1.0]
            oracle_prior_map_p100 = generate_custom_inference_prior(perfect_catalog_p100, img_size=image_tensor.shape[-1], sigma=1.5, device=device)
            pred_stars_oracle_p100, _, bg_map_oracle_p100 = engine.predict(img_noisy, threshold=0.0, psf_basis=master_psf_basis, mean_psf=master_mean_psf, prior_map=oracle_prior_map_p100)

            # Aggregate with Global Offset to prevent overlap during global matching
            offset = idx * 10000.0
            for s in true_stars_chunk:
                global_true.append((s[0], s[1] + offset, s[2] + offset, s[3]))
            for s in pred_stars_flat:
                global_pred_flat.append((s[0] + offset, s[1] + offset, s[2], s[3], s[4]))
            for s in pred_stars_oracle:
                global_pred_oracle.append((s[0] + offset, s[1] + offset, s[2], s[3], s[4]))
            for s in pred_stars_oracle_p43:
                global_pred_oracle_p43.append((s[0] + offset, s[1] + offset, s[2], s[3], s[4]))
            for s in pred_stars_oracle_p100:
                global_pred_oracle_p100.append((s[0] + offset, s[1] + offset, s[2], s[3], s[4]))

            # Capture Hero Data (first chunk)
            if idx == 0:
                jitter_params = None
                try:
                    if "meta" in sample:
                        meta = sample["meta"]
                        jitter_params = (meta[3], meta[4], meta[5])
                except: pass
                
                hero_data_flat = {
                    "image_stretched": img_stretched,
                    "true_stars": true_stars_chunk,
                    "pred_stars": pred_stars_flat,
                    "bg_map": bg_map_flat,
                    "gt_bg_map": gt_bg_map,
                    "chunk_median": noisy_median,
                    "jitter_params": jitter_params
                }
                hero_data_oracle = hero_data_flat.copy()
                hero_data_oracle["pred_stars"] = pred_stars_oracle
                hero_data_oracle["bg_map"] = bg_map_oracle
                hero_data_oracle["prior_map"] = oracle_prior_map

                hero_data_oracle_p43 = hero_data_flat.copy()
                hero_data_oracle_p43["pred_stars"] = pred_stars_oracle_p43
                hero_data_oracle_p43["bg_map"] = bg_map_oracle_p43
                hero_data_oracle_p43["prior_map"] = oracle_prior_map_p43

                hero_data_oracle_p100 = hero_data_flat.copy()
                hero_data_oracle_p100["pred_stars"] = pred_stars_oracle_p100
                hero_data_oracle_p100["bg_map"] = bg_map_oracle_p100
                hero_data_oracle_p100["prior_map"] = oracle_prior_map_p100

                print(f"📸 Captured Hero Sample with {len(true_stars_chunk)} stars.")

        # Final Global Visualizations
        engine.visualize(hero_data_flat, global_true, global_pred_flat, threshold=0.43, output_path="inference_comparison_flat.png", mean_psf=master_mean_psf, num_chunks=num_chunks_to_infer)
        engine.visualize(hero_data_oracle, global_true, global_pred_oracle, threshold=0.43, output_path="inference_comparison_oracle.png", mean_psf=master_mean_psf, num_chunks=num_chunks_to_infer)
        engine.visualize(hero_data_oracle_p43, global_true, global_pred_oracle_p43, threshold=0.43, output_path="inference_comparison_oracle_p43.png", mean_psf=master_mean_psf, num_chunks=num_chunks_to_infer)
        engine.visualize(hero_data_oracle_p100, global_true, global_pred_oracle_p100, threshold=0.43, output_path="inference_comparison_oracle_p100.png", mean_psf=master_mean_psf, num_chunks=num_chunks_to_infer)
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
        num_chunks = config.get("num_chunks_override", 20)
        analyzer = ThresholdAnalyzer(model, device, dataset)
        analyzer.run_analysis(num_chunks=num_chunks)
    else:
        print(f"⚠️ Specialized analysis for stage {stage_idx} not yet implemented.")

def main():
    parser = argparse.ArgumentParser(description="Roman Point Source Curriculum Runner")
    parser.add_argument("stage", type=int, help="Curriculum stage index")
    parser.add_argument("action", choices=["train", "eval", "infer", "analyze"], help="Action to perform")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default=None, help="Path to specific model checkpoint")
    parser.add_argument("--num_chunks", type=int, default=None, help="Number of chunks for inference/analysis")
    
    args = parser.parse_args()
    config = load_config(args.config)
    config["config_path"] = args.config # Store for sub-scripts
    if args.num_chunks:
        config["num_chunks_override"] = args.num_chunks
    
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
