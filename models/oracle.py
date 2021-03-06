import torch
import torch.nn as nn
import torch.nn.functional as F

import vocab
import data


vocab_map = vocab.VocabMap()
vocab_size = vocab_map.vocab_size

RESNET_FEATURE_SIZE = 2048
SPATIAL_SIZE = 8

class OracleNet(nn.Module):
    def __init__(self, question_hidden_dim=128, token_embed_dim=64, category_embed_dim=32,
                 vocab_size=vocab_size, num_categories=data.NUM_CATEGORIES):
        super(OracleNet, self).__init__()
        
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
        
        fc_in_dim = 2 * RESNET_FEATURE_SIZE + SPATIAL_SIZE + question_hidden_dim + category_embed_dim
        
        self.fc1 = nn.Linear(fc_in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        #self.dout = nn.Dropout()
        #self.fc2 = nn.Linear(128,3)
        
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, tokens, question_lens, features, categories):
        '''
        tokens: [batch_size, max_len]
        question_lens: [batch_size]
        features: [batch_size, 2 * RESNET_FEATURE_SIZE + SPATIAL_SIZE == 4104]
        categories: [batch_size]
        '''
        embed_tokens = self.token_embedding(tokens)
        output, h_n = self.question_encoder(embed_tokens)
        gather_index = (question_lens-1).repeat(self.question_hidden_dim, 1, 1).transpose(0, 2)
        question_encodings = output.gather(dim=1, index=gather_index).squeeze(1)
        
        embed_category = self.category_embedding(categories)

        fc1_in = torch.cat([question_encodings, features, embed_category], 1)
        h1 = F.relu(self.fc1(fc1_in))
        #hdrop = self.dout(h1)
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)
        #return self.fc2(h1)
        #return self.fc2(hdrop)
        #return self.fc3(hdrop)
        
    def train_step(self, tokens, q_lens, features, cats, answers):
        scores = self(tokens, q_lens, features, cats)
        loss = self.loss_fn(scores, answers)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
