import torch
import torch.nn as nn
import torch.nn.functional as F

import vocab
import data


vocab_map = vocab.VocabMap()
vocab_size = vocab_map.vocab_size

SPATIAL_SIZE = 8

class GuesserNet(nn.Module):
    def __init__(self, hidden_dim=128, token_embed_dim=64, category_embed_dim=16,
                 vocab_size=vocab_size):
        super(GuesserNet, self).__init__()
        
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=token_embed_dim
        )
        
        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=category_embed_dim
        )
        
        self.dialogue_encoder = nn.GRU(
            input_size=token_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(category_embed_dim + SPATIAL_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
    def forward(self, dialogues, all_cats, all_spatial):
        pass
    