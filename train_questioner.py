import pickle

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from logging_utils import start_log, log_print
from models.questioner import QuestionerNet
import data


class QuestionerDataset(Dataset):
    def __init__(self, features, in_seqs, out_seqs, seq_lens):
        assert features.shape[0] == in_seqs.shape[0] == out_seqs.shape[0] == seq_lens.shape[0]
        
        # hack, made a bug in old generate_questioner_datasets.py
        in_seqs = in_seqs.astype(np.int)
        out_seqs = out_seqs.astype(np.int)
        
        self.features = torch.from_numpy(features)
        self.in_seqs = torch.from_numpy(in_seqs)
        self.out_seqs = torch.from_numpy(out_seqs)
        self.seq_lens = seq_lens
        self.max_len = max(seq_lens)
        
    def __len__(self):
        return self.features.size(0)
    
    def make_mask(self, seq_len):
        mask = np.zeros(self.max_len)
        mask[:seq_len] = 1
        return torch.FloatTensor(mask)
    
    def __getitem__(self, i):
        return (
            self.features[i],
            self.in_seqs[i],
            self.out_seqs[i],
            self.make_mask(self.seq_lens[i])
        )

def load_dataset(split, small):
    with open(data.get_processed_file('questioner', split, small), 'rb') as f:
        return pickle.load(f)

def get_data_loader(split, small):
    return DataLoader(
        QuestionerDataset(*load_dataset(split, small)),
        batch_size=64,
        shuffle=True,
        num_workers=1
    )

def train(model, descriptor, loader_train, num_epochs, print_every=100):
    for epoch in range(num_epochs):
        tqdm.write('Starting epoch {} / {}'.format(epoch + 1, num_epochs))
        model.train()
        
        for t, (features, in_seqs, out_seqs, seq_masks) in enumerate(loader_train):
            features_var = Variable(features.cuda(), requires_grad=False)
            in_seqs_var = Variable(in_seqs.cuda(), requires_grad=False)
            out_seqs_var = Variable(out_seqs.cuda(), requires_grad=False)
            seq_masks_var = Variable(seq_masks.cuda(), requires_grad=False)
            
            loss = model.train_step(
                features_var[:2], in_seqs_var[:2], out_seqs_var[:2], seq_masks_var[:2]
            )
            
            if t % print_every == 0:
                log_print(descriptor, 't = {}, loss = {:.4}'.format(t + 1, loss.data[0]))
        torch.save(model.state_dict(), data.get_saved_model(descriptor))

def main():
    file_descriptor = 'questioner_lstm1_fc2'
    small = False
    loader_train = get_data_loader('train', small)
    # loader_valid = get_data_loader('valid', small)
    # loader_test = get_data_loader('test', small)

    questioner_net = QuestionerNet().cuda()
    train(questioner_net, file_descriptor, loader_train, num_epochs=20)

if __name__ == '__main__':
    main()
    