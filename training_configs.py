# train_configs_tf.py

def get_config(name):
    configs = {
        "base": {
            "epochs": 10,
            "lr": 3e-4,
            "batch_size": 8,
            "patience": 5,
            "val_freq": 1
        },
        "optim": {
            "epochs": 40,
            "lr": 1e-3,
            "batch_size": 8,
            "patience": 5,
            "val_freq": 1,
            "scheduler": {
                "type": "step",
                "step_size": 5,
                "gamma": 0.5
            }
        },
        "merged": {
            "epochs": 30,
            "lr": 5e-4,
            "batch_size": 16,
            "patience": 7,
            "val_freq": 5,
            "scheduler": {
                "type": "step",
                "step_size": 8,
                "gamma": 0.5
            }
        },
        "depthwise": {
            "epochs": 20,
            "lr": 2e-3,
            "batch_size": 8,
            "patience": 5,
            "val_freq": 1,
            "scheduler": {
                "type": "step",
                "step_size": 5,
                "gamma": 0.5
            }
        }
        # Extend with more configs if needed
    }
    return configs[name]
