import json

import torch

from trainer import train_loop


def main():
    # Load the pipeline configuration file
    with open("config.json", "r", encoding="utf8") as f:
        config = json.load(f)

    use_gpu = config["use_gpu"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    train_loop(config, device)


if __name__ == "__main__":
    main()