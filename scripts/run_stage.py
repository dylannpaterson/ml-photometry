import argparse
import torch
import os
import sys
from src.cloud.config_utils import load_config
from src.models.dense_grid import DenseGridModel
from src.data.dataset import PregeneratedDataset
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.engine.inference import InferenceEngine
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

def load_stage_model(stage_idx, device):
    model_path = f"checkpoints/stage{stage_idx}_final.pth"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model for stage {stage_idx} not found at {model_path}")
        return None
    model = DenseGridModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def run_train(stage_idx, config, device):
    print(f"--- 🚀 Curriculum Stage {stage_idx}: Training ---")
    stage_cfg = get_stage_config(config, stage_idx)
    
    # Data Setup
    data_dir = stage_cfg["data_dir"]
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print(f"❌ Error: Data not found in {train_dir}. Run 'gen' for stage {stage_idx} first.")
        return

    train_dataset = PregeneratedDataset(train_dir)
    val_dataset = PregeneratedDataset(val_dir)
    
    batch_size = stage_cfg["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model Setup
    model = DenseGridModel().to(device)
    
    # Custom Trainer Setup
    stage_prefix = f"stage{stage_idx}"
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
    
    if stage_cfg.get("resume_from_last_stage", False) and stage_idx > 0:
        last_stage_model = f"checkpoints/stage{stage_idx-1}_final.pth"
        if os.path.exists(last_stage_model):
            print(f"📈 Resuming from Stage {stage_idx-1} weights...")
            model.load_state_dict(torch.load(last_stage_model, map_location=device))
    elif config["run_config"].get("resume_from_checkpoint", False):
        trainer.resume()
        
    trainer.train()
    print(f"✅ Stage {stage_idx} complete.")

def run_eval(stage_idx, config, device):
    print(f"--- 📊 Curriculum Stage {stage_idx}: Evaluation ---")
    model = load_stage_model(stage_idx, device)
    if not model: return

    if stage_idx == 0:
        evaluator = Evaluator(model, device, config)
        evaluator.run_evaluation()
    else:
        print(f"⚠️ Specialized evaluator for stage {stage_idx} not yet implemented.")

def run_infer(stage_idx, config, device):
    print(f"--- 🛰️ Curriculum Stage {stage_idx}: Inference ---")
    model = load_stage_model(stage_idx, device)
    if not model: return

    engine = InferenceEngine(model, device, config)
    
    # Stage-specific provider
    if stage_idx == 0:
        provider = GaussianPretrainingProvider(
            min_stars=config["data_params"]["min_stars"],
            max_stars=config["data_params"]["max_stars"],
            image_size=config["data_params"]["image_size"]
        )
        image_tensor, _, true_catalogue = provider.generate_chunk()
        predicted = engine.predict(image_tensor)
        engine.visualize(image_tensor, true_catalogue, predicted, threshold=0.5)
    else:
        print(f"⚠️ Specialized inference for stage {stage_idx} not yet implemented.")

def run_analyze(stage_idx, config, device):
    print(f"--- 📈 Curriculum Stage {stage_idx}: Threshold Analysis ---")
    model = load_stage_model(stage_idx, device)
    if not model: return

    if stage_idx == 0:
        provider = GaussianPretrainingProvider(
            min_stars=config["data_params"]["min_stars"],
            max_stars=config["data_params"]["max_stars"],
            image_size=config["data_params"]["image_size"]
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
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    actions = {
        "train": run_train,
        "eval": run_eval,
        "infer": run_infer,
        "analyze": run_analyze
    }
    
    actions[args.action](args.stage, config, device)

if __name__ == "__main__":
    main()
