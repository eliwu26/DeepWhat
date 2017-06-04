from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import pickle
from tqdm import tqdm

import data
# from models.guesser import GuesserNet


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
        for i, objs in enumerate(data_source.all_objs):
            self.num_objs_to_idx[len(objs)].append(i)
            
    def __iter__(self):
        for num_objs in sorted(self.num_objs_to_idx):
            num_examples = len(self.num_objs_to_idx[num_objs])
            print(num_objs, num_examples)
            
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

def get_data_loader(split, small):
    dataset = GuesserDataset(*load_dataset(split, small))
    
    return DataLoader(
        dataset,
        batch_size=64,
        sampler=CurriculumRandomSampler(dataset),
        collate_fn=lambda batch: batch
    )


small = False
loader_train = get_data_loader('train', small)
loader_valid = get_data_loader('valid', small)
loader_test = get_data_loader('valid', small)