import json
import os
from utils import create_vocabulary, download_dataset, extract_embeddings
import datasets

if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # train_set, _, _ = download_dataset(config["dataset_dir"])
    train_set = datasets.load_dataset("conll2003")['train']

    vocab = create_vocabulary(train_set, config["vocab_size"])
    os.makedirs('dataset', exist_ok=True)
    with open('dataset/vocab.json', 'w') as f:
        f.write(json.dumps(vocab))
    # Extract GloVe embeddings for tokens present in the training set vocab
    # extract_embeddings(config, vocab,)