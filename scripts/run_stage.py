"""
Roman Point Source Curriculum Runner script.

This script manages the execution of different stages in the point source 
analysis curriculum, supporting training, evaluation, inference, and analysis.
"""

import argparse
import torch
import numpy as np
import os
import sys
import shutil
import matplotlib.pyplot as plt
from castor.cloud.config_utils import load_config
from castor.models.dense_grid import DenseGridModel
from castor.engine.trainer import Trainer
from castor.engine.evaluator import Evaluator
# Removed top-level InferenceEngine import
from castor.engine.analyzer import ThresholdAnalyzer
from torch.utils.data import DataLoader, SubsetRandomSampler
from castor.constants import DEFAULT_CELL_SIZE, MAX_CAPACITY_PER_CELL, SHAPE_SIZE, GLOBAL_STRETCH_SCALE

def get_stage_config(config, stage_idx):
    """
    Extracts configuration for a specific curriculum stage.

    Parameters
    ----------
    config : dict
        The full configuration dictionary.
    stage_idx : int
        The curriculum stage index.

    Returns
    -------
    dict
        The configuration for the specified stage.
    """
    curriculum = config.get("curriculum", {})
    stage_key = f"stage{stage_idx}"
    if stage_key not in curriculum:
        print(f"❌ Error: Stage {stage_idx} not defined in config.")
        sys.exit(1)
    return curriculum[stage_key]

def load_stage_model(stage_idx, device, config, checkpoint_path=None):
    """
    Loads a model for a specific curriculum stage from a checkpoint.

    Parameters
    ----------
    stage_idx : int
        The curriculum stage index.
    device : torch.device
        The device to load the model onto.
    config : dict
        The full configuration dictionary.
    checkpoint_path : str, optional
        Path to a specific checkpoint. If None, it uses the default final 
        checkpoint for the stage.

    Returns
    -------
    tuple
        A tuple (model, psf_library) if successful, else (None, None).
    """
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

def ensure_stage0_data(stage_cfg, data_cfg, config):
    """
    Checks for HDF5 data and generates a small amount if missing using parallel logic.

    Parameters
    ----------
    stage_cfg : dict
        Configuration for the current stage.
    data_cfg : dict
        Data-specific configuration.
    config : dict
        The full configuration dictionary.

    Returns
    -------
    bool
        True if data is available or successfully generated, False otherwise.
    """
    val_h5 = os.path.join(stage_cfg["data_dir"], "stage0_val.h5")
    
    if not os.path.exists(val_h5):
        print("🔍 Local Stage 0 data not found. Triggering parallel generation for inference/analysis...")
        from castor.data.stage0_gaussian import run_stage0_parallel_generation
        
        # Override sample counts for quick local inference setup
        # 200 samples is plenty for a few diagnostic visuals
        orig_val_samples = config["data_params"]["num_val_samples"]
        config["data_params"]["num_val_samples"] = 200
        
        try:
            run_stage0_parallel_generation(config, split='val')
        finally:
            # Restore original config values
            config["data_params"]["num_val_samples"] = orig_val_samples

        if not os.path.exists(val_h5):
            print("❌ Error: Failed to generate local parallel data fallback.")
            return False
            
    return True

def get_safe_batch_size(target_batch_size, device):
    """
    Detects VRAM and scales batch size to avoid OOM.

    Parameters
    ----------
    target_batch_size : int
        The desired batch size from the configuration.
    device : torch.device
        The device being used for training.

    Returns
    -------
    tuple
        A tuple (micro_batch, accumulation_steps).
    """
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
    """
    Runs the training process for a specified stage.

    Parameters
    ----------
    stage_idx : int
        The curriculum stage index.
    config : dict
        The full configuration dictionary.
    device : torch.device
        The device to run training on.
    """
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
        data_h5 = os.path.join(stage_cfg["data_dir"], "stage0_data.h5")
        
        if force_gen or not os.path.exists(data_h5):
            print("🛠️ Generating Stage 0 Data (Unified Parallel HDF5)...")
            from castor.data.stage0_gaussian import run_stage0_parallel_generation
            run_stage0_parallel_generation(config)

        from castor.data.stage0_gaussian import HDF5ChunkDataset
        from torch.utils.data import Subset
        print(f"🛠️ Using Unified HDF5 Dataset: {data_h5}")
        full_dataset = HDF5ChunkDataset(data_h5)
        
        # 90/10 Train/Val Split
        num_total = len(full_dataset)
        num_train = int(0.9 * num_total)
        # Use fixed indices for reproducibility
        train_dataset = Subset(full_dataset, range(0, num_train))
        val_dataset = Subset(full_dataset, range(num_train, num_total))
        
    elif stage_idx == 1:
        # Stage 1 Multi-Telescope Foundation (Unified GalSim HDF5)
        data_h5 = os.path.join(stage_cfg["data_dir"], "stage1_data.h5")
        
        if force_gen or not os.path.exists(data_h5):
            print("🛠️ Generating Stage 1 High-Fidelity GalSim Data (Parallel)...")
            cfg_path = config.get("config_path", "config/config.yaml")
            ret = os.system(f"export PYTHONPATH=$PYTHONPATH:. && python3 castor/data/stage1_galsim.py --config {cfg_path}")
            if ret != 0 or not os.path.exists(data_h5):
                print("❌ Error: Stage 1 data generation failed."); return

        from castor.data.stage0_gaussian import HDF5ChunkDataset
        from torch.utils.data import Subset
        print(f"🛠️ Using Stage 1 Unified HDF5 Dataset: {data_h5}")
        full_dataset = HDF5ChunkDataset(data_h5)
        
        num_total = len(full_dataset)
        num_train = int(0.9 * num_total)
        train_dataset = Subset(full_dataset, range(0, num_train))
        val_dataset = Subset(full_dataset, range(num_train, num_total))
    else:
        print(f"❌ Error: Stage {stage_idx} data loading via PregeneratedDataset is obsolete.")
        print(f"   Please implement HDF5ChunkDataset support for Stage {stage_idx}.")
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
    
    # --- EPOCH SUBSETTING (Stage 0 Only) ---
    if stage_idx == 0:
        # For Stage 0, we'll wrap the DataLoader to re-subset every epoch
        # This is handled inside Trainer.train by re-initializing the iterator
        # but we need to pass a sampler that shuffles every epoch.
        # However, SubsetRandomSampler only shuffles once.
        # We'll use a custom Trainer hook or just re-initialize the train_loader each epoch.
        train_loader = None # Will be initialized inside Trainer
    else:
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

    # --- STAGE 0 EPOCH SUBSETTING HOOK ---
    if stage_idx == 0:
        def stage0_epoch_callback(epoch):
            num_total = len(train_dataset)
            indices = np.random.permutation(num_total)
            subset_indices = indices[:num_total // 2]
            sampler = SubsetRandomSampler(subset_indices)
            
            new_loader = DataLoader(
                train_dataset,
                batch_size=micro_batch,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=use_optimizations,
                persistent_workers=use_optimizations,
                prefetch_factor=prefetch_factor,
                drop_last=True
            )
            return new_loader
            
        trainer.epoch_loader_callback = stage0_epoch_callback

    
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
    """
    Runs the evaluation process for a specified stage.

    Parameters
    ----------
    stage_idx : int
        The curriculum stage index.
    config : dict
        The full configuration dictionary.
    device : torch.device
        The device to run evaluation on.
    checkpoint : str, optional
        Path to a specific model checkpoint.
    """
    print(f"--- 📊 Curriculum Stage {stage_idx}: Evaluation ---")
    model, _ = load_stage_model(stage_idx, device, config, checkpoint)
    if not model: return

    if stage_idx == 0:
        # Ensure data exists (Stage 0 only for now)
        stage_cfg = config["curriculum"]["stage0"]
        data_cfg = config["data_params"]
        config_path = config.get("config_path", "config/config.yaml")
        if not ensure_stage0_data(stage_cfg, data_cfg, config):
            return
            
        evaluator = Evaluator(model, device, config)
        # Increased to 500 chunks for better statistical stability
        evaluator.run_evaluation(num_chunks=500)
    else:
        print(f"⚠️ Specialized evaluator for stage {stage_idx} not yet implemented.")

def run_infer(stage_idx, config, device, checkpoint=None):
    """
    Runs the inference process for a specified stage.

    Parameters
    ----------
    stage_idx : int
        The curriculum stage index.
    config : dict
        The full configuration dictionary.
    device : torch.device
        The device to run inference on.
    checkpoint : str, optional
        Path to a specific model checkpoint.
    """
    from castor.engine.evaluator import match_stars
    from castor.engine.inference import InferenceEngine, generate_custom_inference_prior
    import random
    import h5py
    print(f"--- 🛰️ Curriculum Stage {stage_idx}: Inference ---")
    model, psf_lib_ckpt = load_stage_model(stage_idx, device, config, checkpoint)
    if model is None: return

    engine = InferenceEngine(model, device, config)
    
    if stage_idx == 0:
        from castor.data.stage0_gaussian import HDF5ChunkDataset
        data_cfg = config["data_params"]
        stage_cfg = config["curriculum"]["stage0"]
        config_path = config.get("config_path", "config/config.yaml")
        
        if not ensure_stage0_data(stage_cfg, data_cfg, config):
            return
            
        val_h5 = os.path.join(stage_cfg["data_dir"], "stage0_val.h5")
        dataset = HDF5ChunkDataset(val_h5)
        
        # Respect CLI --num_chunks argument if provided
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
        
        for idx in range(num_chunks_to_infer):
            sample = dataset[idx]
            image_tensor = sample["image"]
            target = sample["target"]
            
            # --- Physical Prep ---
            stretch_scale = data_cfg.get("GLOBAL_STRETCH_SCALE", 10.0)
            img_pos = torch.clamp(image_tensor, min=0.0)
            img_noisy = torch.poisson(img_pos)
            img_noisy += torch.randn_like(img_noisy) * 5.0
            # Robust Background Estimation: Use 10th percentile to avoid bright star contamination
            # This matches the new robust fallback in InferenceEngine.predict
            robust_median = float(torch.quantile(img_noisy.view(-1), 0.10))
            img_stretched = torch.arcsinh((img_noisy - robust_median) / stretch_scale)

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
            # Use threshold=0.0 to get all candidates, filter later for metrics
            pred_stars_flat, bg_map_flat = engine.predict(img_noisy, threshold=0.0, chunk_median=robust_median)
            
            # 2. Oracle Prediction (Perfect Prior - All Stars)
            # Use ground truth p value (s[0]) for amplitude and sigma=0.405 to match training
            perfect_catalog = [(s[1], s[2], s[0]) for s in true_stars_chunk]
            oracle_prior_map = generate_custom_inference_prior(perfect_catalog, img_size=image_tensor.shape[-1], sigma=0.405, device=device)
            pred_stars_oracle, bg_map_oracle = engine.predict(img_noisy, threshold=0.0, prior_map=oracle_prior_map, chunk_median=robust_median)

            # 3. Oracle Prediction (p >= 0.43)
            perfect_catalog_p43 = [(s[1], s[2], s[0]) for s in true_stars_chunk if s[0] >= 0.43]
            oracle_prior_map_p43 = generate_custom_inference_prior(perfect_catalog_p43, img_size=image_tensor.shape[-1], sigma=0.405, device=device)
            pred_stars_oracle_p43, bg_map_oracle_p43 = engine.predict(img_noisy, threshold=0.0, prior_map=oracle_prior_map_p43, chunk_median=robust_median)

            # 4. Oracle Prediction (p >= 1.0)
            perfect_catalog_p100 = [(s[1], s[2], s[0]) for s in true_stars_chunk if s[0] >= 1.0]
            oracle_prior_map_p100 = generate_custom_inference_prior(perfect_catalog_p100, img_size=image_tensor.shape[-1], sigma=0.405, device=device)
            pred_stars_oracle_p100, bg_map_oracle_p100 = engine.predict(img_noisy, threshold=0.0, prior_map=oracle_prior_map_p100, chunk_median=robust_median)

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
                hero_data_flat = {
                    "image_stretched": img_stretched,
                    "true_stars": true_stars_chunk,
                    "pred_stars": pred_stars_flat,
                    "bg_map": bg_map_flat,
                    "gt_bg_map": gt_bg_map,
                    "chunk_median": robust_median
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

                # Standalone Prior Images
                plt.imsave("hero_prior_oracle.png", oracle_prior_map, cmap='magma', origin='lower')
                plt.imsave("hero_prior_oracle_p43.png", oracle_prior_map_p43, cmap='magma', origin='lower')
                plt.imsave("hero_prior_oracle_p100.png", oracle_prior_map_p100, cmap='magma', origin='lower')

                print(f"📸 Captured Hero Sample with {len(true_stars_chunk)} stars.")

        # Final Global Visualizations
        engine.visualize(hero_data_flat, global_true, global_pred_flat, threshold=0.43, output_path="inference_comparison_flat.png", num_chunks=num_chunks_to_infer)
        
        # 🚀 NEW: Use p=0.9 threshold for informed prior visualizations
        engine.visualize(hero_data_oracle, global_true, global_pred_oracle, threshold=0.9, output_path="inference_comparison_oracle.png", num_chunks=num_chunks_to_infer)
        engine.visualize(hero_data_oracle_p43, global_true, global_pred_oracle_p43, threshold=0.9, output_path="inference_comparison_oracle_p43.png", num_chunks=num_chunks_to_infer)
        engine.visualize(hero_data_oracle_p100, global_true, global_pred_oracle_p100, threshold=0.9, output_path="inference_comparison_oracle_p100.png", num_chunks=num_chunks_to_infer)
    else:
        print(f"⚠️ Specialized inference for stage {stage_idx} not yet implemented.")

def run_analyze(stage_idx, config, device, checkpoint=None):
    """
    Runs the threshold analysis process for a specified stage.

    Parameters
    ----------
    stage_idx : int
        The curriculum stage index.
    config : dict
        The full configuration dictionary.
    device : torch.device
        The device to run analysis on.
    checkpoint : str, optional
        Path to a specific model checkpoint.
    """
    print(f"--- 📈 Curriculum Stage {stage_idx}: Threshold Analysis ---")
    model, _ = load_stage_model(stage_idx, device, config, checkpoint)
    if not model: return

    if stage_idx == 0:
        from castor.data.stage0_gaussian import HDF5ChunkDataset
        stage_cfg = config["curriculum"]["stage0"]
        data_cfg = config["data_params"]
        config_path = config.get("config_path", "config/config.yaml")
        
        # Ensure data exists fallback
        if not ensure_stage0_data(stage_cfg, data_cfg, config):
            return
            
        val_h5 = os.path.join(stage_cfg["data_dir"], "stage0_val.h5")
        dataset = HDF5ChunkDataset(val_h5)
        num_chunks = config.get("num_chunks_override", 20)
        analyzer = ThresholdAnalyzer(model, device, dataset)
        analyzer.run_analysis(num_chunks=num_chunks)
    else:
        print(f"⚠️ Specialized analysis for stage {stage_idx} not yet implemented.")

def main():
    """
    Main entry point for the Roman Point Source Curriculum Runner.
    """
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
