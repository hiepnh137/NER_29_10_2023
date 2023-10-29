import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    super(TokenEmbedding, self).__init__()
    self.weight = nn.Parameter(torch.zeros((vocab_size, embed_dim), dtype=torch.float32))
    nn.init.uniform_(self.weight, -0.10, +0.10)
    self.vocab_size = vocab_size
    # T.nn.init.normal_(self.weight)  # mean = 0, stddev = 1

  def forward(self, x):
    x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size)
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
        vocab_size = config["vocab_size"]
        self.d_model = transformer_embedding_dim
        # Load pretrained word embeddings
        # word_embeddings = torch.Tensor(np.loadtxt(config["embeddings"]["path"]))
        # self.embedding_layer = nn.Embedding.from_pretrained(
        #     word_embeddings,
        #     freeze=True,
        #     padding_idx=config["PAD_idx"]
        # )
        # self.embedding_layer = TokenEmbedding(vocab_size, transformer_embedding_dim)
        self.embedding_layer = nn.Embedding(vocab_size, transformer_embedding_dim)

        # self.entry_mapping = nn.Linear(embedding_dim, transformer_embedding_dim)
        self.positional_encodings = PositionalEncoding(
            transformer_embedding_dim,
            dropout,
            config["max_len"]
        )

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(transformer_embedding_dim, attention_heads, ff_dim, dropout, batch_first=False),
            num_of_transformer_layers,
        )
        self.classifier = nn.Linear(transformer_embedding_dim, num_classes)
        
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, padding_mask):
        """Performs forward pass of the module."""
        # Get token embeddings for each word in a sequence
        x = self.embedding_layer(x) * math.sqrt(self.d_model)
        print('x: ', x.shape)
        # Map input tokens to the transformer embedding dim
        # x = self.entry_mapping(x)
        # x = F.leaky_relu(x)
        # Leverage the self-attention mechanism on the input sequence
        print('padding_mask: ', padding_mask.shape)
        # padding_mask = padding_mask.permute(1,0)
        # x = x.permute(1, 0, 2)
        x = self.positional_encodings(x)
        print('x2: ', x.shape)
        x = self.transformer_encoder(x, padding_mask)
        x = x.permute(1, 0, 2)

        y_pred = self.classifier(x)
        return y_pred
    

