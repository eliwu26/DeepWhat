from collections import defaultdict
import pickle

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm

import data
from logging_utils import start_log, log_print
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

class GuesserCurriculumDataLoader(object):
    '''
    Loads batches with examples with the same number of objects,
    in increasing order of number of objects. 
    
    If there are fewer than batch_size remaining examples for the
    current number of objects, the returned batch may be smaller. 
    
    Arguments:
        data_source (GuesserDataset): dataset to sample from
        batch_size
    '''
    def __init__(self, dataset, batch_size=64):
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.num_objs_to_idx = defaultdict(list)
        for i, objs in enumerate(dataset.all_cats):
            self.num_objs_to_idx[len(objs)].append(i)
    
    def collate(self, batch):
        batch.sort(
            key=lambda x: len(x[0]), # dialogue length
            reverse=True
        )
        return list(zip(*batch))
    
    def __iter__(self):
        for num_objs in sorted(self.num_objs_to_idx):
            num_examples = len(self.num_objs_to_idx[num_objs])
            
            perm = torch.randperm(num_examples)
            for p in range(0, num_examples, self.batch_size):
                yield self.collate([
                    self.dataset[perm[p]]
                    for i in range(p, p + self.batch_size)
                ])
    
    def __len__(self):
        return len(self.dialogues)

def load_dataset(split, small):
    with open(data.get_processed_file('guesser', split, small), 'rb') as f:
        return pickle.load(f)
    
def get_data_loader(split, small):
    dataset = GuesserDataset(*load_dataset(split, small))
    return GuesserCurriculumDataLoader(dataset)

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
        
def check_accuracy(model, descriptor, loader):
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
    log_print(descriptor, 'Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

def train(model, descriptor, loader_valid_local, loader_train_local, loader_test_local, num_epochs, print_every=500):
    log_print(descriptor, 'Getting accuracy on validation set')
    check_accuracy(model, descriptor, loader_valid_local)
    current_max_val_acc = 0;
    for epoch in range(num_epochs):
        log_print(descriptor, 'Starting epoch {} / {}'.format(epoch + 1, num_epochs))
        model.train()

        for t, (dialogues, all_cats, all_spatial, correct_objs) in tqdm(enumerate(loader_train_local)):
            dialogues_var, all_cats_var, all_spatial_var, correct_objs_var, dialogue_lens = \
            make_vars(dialogues, all_cats, all_spatial, correct_objs, requires_grad=False)

            loss = model.train_step(
                dialogues_var, dialogue_lens, all_cats_var, all_spatial_var, correct_objs_var
            )

            if t % print_every == 0:
                log_print(descriptor, 't = {}, loss = {:.4}'.format(t + 1, loss.data[0]))

        log_print(descriptor, 'Getting accuracy on validation set')
        accuracy = check_accuracy(model, descriptor, loader_valid_local)
        if accuracy > current_max_val_acc:
            current_max_val_acc = accuracy
            torch.save(model.state_dict(), data.get_saved_model(descriptor))
        print(current_max_val_acc)
        
    best_model = GuesserNet().cuda()
    best_model.load_state_dict(torch.load(data.get_saved_model(descriptor)))
    
    print(current_max_val_acc)
    log_print(descriptor, 'Getting accuracy on training set')
    check_accuracy(best_model, descriptor, loader_train_local)
    log_print(descriptor, 'Getting accuracy on validation set')
    check_accuracy(best_model, descriptor, loader_valid_local)
    log_print(descriptor, 'Getting accuracy on test set')
    check_accuracy(best_model, descriptor, loader_test_local)

def main():
    file_descriptor = 'guesser_gru3_fc2_cat16_h256_we64_longtrain'
    small = False
    loader_train = get_data_loader('train', small)
    loader_valid = get_data_loader('valid', small)
    loader_test = get_data_loader('test', small)

    guesser_net = GuesserNet().cuda()
    train(guesser_net, file_descriptor, loader_valid, loader_train, loader_test, num_epochs=75)
    
if __name__ == '__main__':
    main()
    