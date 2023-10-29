import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings
        x = self.dropout(x)
        return x
    
class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super(TokenEmbedding, self).__init__()
    self.weight = nn.Parameter(torch.zeros((vocab_size, embed_dim), dtype=torch.float32))
    nn.init.uniform_(self.weight, -0.10, +0.10)
    # T.nn.init.normal_(self.weight)  # mean = 0, stddev = 1

  def forward(self, x):
    return self.weight[x]

class NERClassifier(nn.Module):
    """Represents model which classifies named entities in the given body of text."""

    def __init__(self, config):
        """Initializes the module."""
        super(NERClassifier, self).__init__()
        num_classes = len(config["class_mapping"])
        # embedding_dim = config["embeddings"]["size"]
        num_of_transformer_layers = config["num_of_transformer_layers"]
        transformer_embedding_dim = config["transformer_embedding_dim"]
        attention_heads = config["attention_heads"]
        ff_dim = config["transformer_ff_dim"]
        dropout = config["dropout"]
        ntoken = config["ntoken"]
        self.d_model = transformer_embedding_dim
        # Load pretrained word embeddings
        # word_embeddings = torch.Tensor(np.loadtxt(config["embeddings"]["path"]))
        # self.embedding_layer = nn.Embedding.from_pretrained(
        #     word_embeddings,
        #     freeze=True,
        #     padding_idx=config["PAD_idx"]
        # )
        self.embedding_layer = TokenEmbedding(ntoken, transformer_embedding_dim)

        # self.entry_mapping = nn.Linear(embedding_dim, transformer_embedding_dim)
        self.positional_encodings = PositionalEncodings(
            config["max_len"],
            transformer_embedding_dim,
            dropout
        )

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(transformer_embedding_dim, attention_heads, ff_dim, dropout, batch_first=True),
            num_of_transformer_layers,
        )
        self.classifier = nn.Linear(transformer_embedding_dim, num_classes)

    def forward(self, x, padding_mask):
        """Performs forward pass of the module."""
        # Get token embeddings for each word in a sequence
        x = self.embedding_layer(x) * math.sqrt(self.d_model)

        # Map input tokens to the transformer embedding dim
        # x = self.entry_mapping(x)
        # x = F.leaky_relu(x)
        # Leverage the self-attention mechanism on the input sequence
        x = self.positional_encodings(x)
        # x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, padding_mask)
        # x = x.permute(1, 0, 2)

        y_pred = self.classifier(x)
        return y_pred
    

