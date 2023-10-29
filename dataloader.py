import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class CoNLLDataset(Dataset):
    """Implements CoNLL2003 dataset consumption class.

       Data is saved in .txt format.
       Each sample is in a distinct line in the following format:
        - sample_length[TAB]input_tokens[TAB]ner_labels_per_token
        - @input_tokens and @ner_labels_per_token are also separated by [TAB]
    """
    def __init__(self, config, data, separator="\t"):

        self.data = data

        # self.data = [sample.replace("\n", "") for sample in self.data]

        # Load the vocabulary mappings
        with open(config["word2idx_path"], "r", encoding="utf8") as f:
            self._word2idx = json.load(f)
        self._idx2word = {str(idx): word for word, idx in self._word2idx.items()}
        self._word2idx = {word: int(idx)  for word, idx in self._word2idx.items()}
        # Set the default value for the OOV tokens
        # self._word2idx = defaultdict(
        #     lambda: self._word2idx[config["OOV_token"]],
        #     self._word2idx
        # )

        self._separator = separator
        self._PAD_token = config["PAD_token"]
        self._PAD_label = config["PAD_label"]
        self._OOV_token = config["OOV_token"]
        self._max_len = config["max_len"]

        self._dataset_size = len(self.data)
        self.process_data()

    def __len__(self):
        return self._dataset_size

    def process_data(self):
        self.all_tokens = []
        self.all_labels = []
        self.all_padding_masks = []
        for sample in self.data:
            tokens = sample['tokens']
            labels = sample['ner_tags']
            
            tokens = tokens[:self._max_len]
            labels = labels[:self._max_len]

            sample_size = len(tokens)
            padding_size = self._max_len - sample_size
            if padding_size > 0:
                tokens += [self._PAD_token for _ in range(padding_size)]
                labels += [self._PAD_label for _ in range(padding_size)]
            tokens = [token.strip().lower() for token in tokens]
            tokens = [self._word2idx[token] if token in self._word2idx.keys() else self._word2idx[self._OOV_token] \
                      for token in tokens]
            tokens = torch.Tensor(tokens).long()

            # Adapt labels for PyTorch consumption
            labels = [int(label) for label in labels]
            labels = torch.Tensor(labels).long()

            # Define the padding mask
            padding_mask = torch.ones([self._max_len, ])
            padding_mask[:sample_size] = 0.0

            self.all_tokens.append(tokens)
            self.all_labels.append(labels)
            self.all_padding_masks.append(padding_mask)
        
    def __getitem__(self, index):
        tokens = self.all_tokens[index]
        labels = self.all_labels[index]
        padding_mask = self.all_padding_masks[index]
        return tokens, labels, padding_mask