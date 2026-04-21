import yaml
import os
from castor.data.stage0_gaussian import run_stage0_parallel_generation

def main():
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Override for 10 samples
    config["data_params"]["num_train_samples"] = 10
    config["data_params"]["num_val_samples"] = 2
    config["curriculum"]["stage0"]["data_dir"] = "data/stage0_test_batch"
    
    print(f"🚀 Generating 10 Stage 0 samples in {config['curriculum']['stage0']['data_dir']}...")
    run_stage0_parallel_generation(config, split='train')
    print("✅ Generation complete.")

if __name__ == "__main__":
    main()
