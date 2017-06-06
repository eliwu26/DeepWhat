import torch
import torch.nn as nn
import torch.nn.functional as F

import vocab


vocab_map = vocab.VocabMap()
vocab_size = vocab_map.vocab_size

RESNET_FEATURE_SIZE = 2048

class QuestionerNet(nn.Module):
    def __init__(self, vocab_size=vocab_size, token_embed_dim=64):
        super(QuestionerNet, self).__init__()
        
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=token_embed_dim
        )
        
        self.encoder = nn.LSTM(
            input_size=RESNET_FEATURE_SIZE + token_embed_dim,
            hidden_size=vocab_size,
            num_layers=1,
            batch_first=True
        )
        
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, mode='beamsearch'):
        '''
        Eval time sample
        '''
        pass
    
    def train_step(self, features, in_seq, out_seq, seq_mask):
        in_embed = self.token_embedding(in_seq)
        
        features = features.unsqueeze(1)
        features_repeated = features.repeat(1, in_seq.size(1), 1)
        
        encoder_inputs = torch.cat([in_embed, features_repeated], 2)
        
        logits, h_n = self.encoder(encoder_inputs)
        
        # see: https://github.com/pytorch/pytorch/issues/764
        # and https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
        logits_flat = logits.contiguous().view(-1, logits.size(-1))
        out_seq_flat = out_seq.view(-1, 1)
        
        log_probs_flat = F.log_softmax(logits_flat)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=out_seq_flat)
        
        losses = losses_flat.view(*out_seq.size())
        loss = losses.sum() / seq_mask.sum()
        
        return loss
        