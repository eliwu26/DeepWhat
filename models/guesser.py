import torch
import torch.nn as nn
import torch.nn.functional as F

import vocab
import data
import packed_sequence_utils


vocab_map = vocab.VocabMap()
vocab_size = vocab_map.vocab_size

SPATIAL_SIZE = 8

class GuesserNet(nn.Module):
    def __init__(self, hidden_dim=256, token_embed_dim=64, category_embed_dim=16,
                 vocab_size=vocab_size, num_categories=data.NUM_CATEGORIES):
        super(GuesserNet, self).__init__()
        
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=token_embed_dim
        )
        
        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=category_embed_dim
        )
        
        #self.dialogue_encoder = nn.LSTM(
        self.dialogue_encoder = nn.GRU(
            input_size=token_embed_dim,
            hidden_size=hidden_dim,
            #num_layers=1,
            num_layers=2,
            #num_layers=3,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(category_embed_dim + SPATIAL_SIZE, 64),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(64, hidden_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, dialogues, dialogue_lens, all_cats, all_spatial):
        '''
        dialogues: [batch_size, max_len]
        dialogue_lens: list[int] of len batch_size
        all_cats: [batch_size, num_objs]
        all_spatial: [batch_size, num_objs, SPATIAL_SIZE == 8]
        '''
        embed_dialogue_padded = self.token_embedding(dialogues)
        embed_dialogue_packed = nn.utils.rnn.pack_padded_sequence(
            embed_dialogue_padded, dialogue_lens, batch_first=True
        )
        
        output, h_n = self.dialogue_encoder(embed_dialogue_packed)
        dialogue_encodings = packed_sequence_utils.get_last_step_tensor(
            output,
            [l - 1 for l in dialogue_lens] # might be a bug but crashes without -1
        )
        
        embed_categories = self.category_embedding(all_cats)
        
        obj_scores_list = []
        for i in range(embed_categories.size(1)): # iterate over each object
            obj_features = torch.cat(
                [embed_categories[:, i, :], all_spatial[:, i, :]],
                1
            )
            
            obj_embedding = self.mlp(obj_features)
            obj_score = (obj_embedding * dialogue_encodings).sum(dim=1)
            obj_scores_list.append(obj_score)
            
        obj_scores = torch.cat(obj_scores_list, 1)
        return obj_scores
    
    def train_step(self, dialogues, dialogue_lens, all_cats, all_spatial, correct_objs):
        scores = self(dialogues, dialogue_lens, all_cats, all_spatial)
        loss = self.loss_fn(scores, correct_objs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss