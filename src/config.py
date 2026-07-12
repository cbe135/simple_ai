import os
import yaml
from datetime import datetime


def load_config(args, config_path=None):
    """Load the latest config YAML or the specified one."""
    if config_path:
        with open(config_path, "r") as fp:
            return yaml.safe_load(fp)

    config_file = args["environ"]["config_file"]
    configs = [
        a for a in os.listdir() if config_file in a and a.endswith(".yaml")
    ]
    configs = sorted(
        configs,
        key=lambda x: datetime.strptime(
            x.replace(f"_{config_file}", ""), "%m%d_%H%M%S"
        ),
    )
    with open(configs[-1], "r") as fp:
        return yaml.safe_load(fp)


def save_config(args, run_dir=None):
    """Save the current args to a YAML file (``config.yaml`` in run_dir)."""
    config_file = args["environ"].get("config_file", "config.yaml")
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        filename = os.path.join(run_dir, "config.yaml")
    else:
        filename = config_file
    with open(filename, "w") as fp:
        yaml.dump(args, fp)
    return filename
