import yaml
import os

REQUIRED_FIELDS = {
    "run_config": ["resume_from_checkpoint", "force_regenerate_data"],
    "training_hyperparams": ["epochs", "batch_size", "learning_rate"],
    "data_params": [
        "num_train_samples",
        "num_val_samples",
        "min_stars",
        "max_stars",
        "image_size",
        "max_capacity_per_cell"
    ]
}

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Strict validation
    for section, fields in REQUIRED_FIELDS.items():
        if section not in config:
            raise ValueError(f"Missing required section in config: '{section}'")
        
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing required field in config: '{section}.{field}'")
            if config[section][field] is None:
                raise ValueError(f"Field cannot be null: '{section}.{field}'")

    return config

if __name__ == "__main__":
    # Test loading
    try:
        cfg = load_config()
        print("Config loaded successfully!")
        print(cfg)
    except Exception as e:
        print(f"Error: {e}")
