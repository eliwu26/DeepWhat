from collections import defaultdict
import pickle

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

import data
from models.guesser import GuesserNet


class GuesserDataset(Dataset):
    def __init__(self, dialogues, all_cats, all_spatial, correct_objs):
        assert len(dialogues) == len(all_cats) == len(all_spatial) == len(correct_objs)
        
        self.dialogues = dialogues
        self.all_cats = all_cats
        self.all_spatial = all_spatial
        self.correct_objs = correct_objs
        
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, i):
        return (self.dialogues[i], self.all_cats[i],
                self.all_spatial[i], self.correct_objs[i])

class CurriculumRandomSampler(Sampler):
    '''
    Samples elements in a GuesserDataset randomly, in increasing order of number
    of objects. 
    
    Arguments:
        data_source (GuesserDataset): dataset to sample from
        batch_size: must match batch size for DataLoader for all samples in a batch
                    to have same number of objects
    '''
    def __init__(self, data_source, batch_size=64):
        self.data_source = data_source
        self.batch_size = 64
        
        self.num_objs_to_idx = defaultdict(list)
        for i, objs in enumerate(data_source.all_cats):
            self.num_objs_to_idx[len(objs)].append(i)
            
    def __iter__(self):
        for num_objs in sorted(self.num_objs_to_idx):
            num_examples = len(self.num_objs_to_idx[num_objs])
            
            perm = torch.randperm(num_examples)
            for p in perm:
                yield self.num_objs_to_idx[num_objs][p]
            
            # return indices with same number of objects for the rest of the batch
            for i in range(-num_examples % self.batch_size):
                yield self.num_objs_to_idx[num_objs][perm[i % num_examples]]
    
    def __len__(self):
        return len(self.data_source)
    
def load_dataset(split, small):
    with open(data.get_processed_file('guesser', split, small), 'rb') as f:
        return pickle.load(f)

def collate_desc_dialogue_len(batch):
    batch.sort(
        key=lambda x: len(x[0]), # dialogue length
        reverse=True
    )
    return list(zip(*batch))
    
def get_data_loader(split, small):
    dataset = GuesserDataset(*load_dataset(split, small))
    
    return DataLoader(
        dataset,
        batch_size=64,
        sampler=CurriculumRandomSampler(dataset),
        collate_fn=collate_desc_dialogue_len
    )

def make_vars(dialogues, all_cats, all_spatial, correct_objs, **kwargs):
    dialogue_lens = list(map(len, dialogues))
    
    dialogues_padded = np.zeros([len(dialogues), dialogue_lens[0]], dtype=int)
    for i, dialogue in enumerate(dialogues):
        dialogues_padded[i, :dialogue_lens[i]] = dialogue
    
    dialogues_var = Variable(torch.from_numpy(dialogues_padded).cuda(), **kwargs)
    all_cats_var = Variable(torch.LongTensor(all_cats).cuda(), **kwargs)
    all_spatial_var = Variable(torch.FloatTensor(all_spatial).cuda(), **kwargs)
    correct_objs_var = Variable(torch.LongTensor(correct_objs).cuda(), **kwargs)
    
    return (dialogues_var, all_cats_var, all_spatial_var, correct_objs_var, dialogue_lens)

def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    
    for dialogues, all_cats, all_spatial, correct_objs in loader:
        dialogues_var, all_cats_var, all_spatial_var, correct_objs_var, dialogue_lens = \
            make_vars(dialogues, all_cats, all_spatial, correct_objs, volatile=True)

        scores = model(dialogues_var, dialogue_lens, all_cats_var, all_spatial_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == torch.LongTensor(correct_objs)).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    tqdm.write('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train(model, num_epochs, print_every=1000):
    tqdm.write('Getting accuracy on validation set')
    check_accuracy(model, loader_valid)
    
    for epoch in range(num_epochs):
        tqdm.write('Starting epoch {} / {}'.format(epoch + 1, num_epochs))
        model.train()

        for t, (dialogues, all_cats, all_spatial, correct_objs) in tqdm(enumerate(loader_train)):
            dialogues_var, all_cats_var, all_spatial_var, correct_objs_var, dialogue_lens = \
            make_vars(dialogues, all_cats, all_spatial, correct_objs, requires_grad=False)

            loss = model.train_step(
                dialogues_var, dialogue_lens, all_cats_var, all_spatial_var, correct_objs_var
            )

            if t % print_every == 0:
                tqdm.write('t = {}, loss = {:.4}'.format(t + 1, loss.data[0]))

        tqdm.write('Getting accuracy on training set')
        check_accuracy(model, loader_train)
        tqdm.write('Getting accuracy on validation set')
        check_accuracy(model, loader_valid)
        
    tqdm.write('Getting accuracy on training set')
    check_accuracy(model, loader_train)
    tqdm.write('Getting accuracy on test set')
    check_accuracy(model, loader_test)


small = True
loader_train = get_data_loader('train', small)
loader_valid = get_data_loader('valid', small)
loader_test = get_data_loader('valid', small)

guesser_net = GuesserNet().cuda()
train(guesser_net, num_epochs=100)
check_accuracy(guesser_net, loader_valid)