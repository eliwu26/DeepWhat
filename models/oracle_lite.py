import torch
import torch.nn as nn
import torch.nn.functional as F

import vocab
import data


vocab_map = vocab.VocabMap()
vocab_size = vocab_map.vocab_size

RESNET_FEATURE_SIZE = 2048
SPATIAL_SIZE = 8

class OracleLiteNet(nn.Module):
    def __init__(self, question_hidden_dim=128, token_embed_dim=64, category_embed_dim=32,
                 vocab_size=vocab_size, num_categories=data.NUM_CATEGORIES):
        super(OracleLiteNet, self).__init__()
        
        self.question_hidden_dim = question_hidden_dim
        
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=token_embed_dim
        )
        
        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=category_embed_dim
        )
        
        self.question_encoder = nn.GRU(
            input_size=token_embed_dim,
            hidden_size=question_hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        fc_in_dim = RESNET_FEATURE_SIZE + SPATIAL_SIZE + question_hidden_dim + category_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(fc_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, tokens, question_lens, features, categories):
        embed_tokens = self.token_embedding(tokens)
        output, h_n = self.question_encoder(embed_tokens)
        gather_index = (question_lens-1).repeat(self.question_hidden_dim, 1, 1).transpose(0, 2)
        question_encodings = output.gather(dim=1, index=gather_index).squeeze()
        
        embed_category = self.category_embedding(categories)

        fc_in = torch.cat([question_encodings, features, embed_category], 1)
        
        return self.mlp(fc_in)
        
    def train_step(self, tokens, q_lens, features, cats, answers):
        scores = self(tokens, q_lens, features, cats)
        loss = self.loss_fn(scores, answers)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss