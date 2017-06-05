from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler


# does not work - samples a skewed dataset, which is bad for validation
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