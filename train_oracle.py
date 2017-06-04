import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm

import data
from models.oracle import OracleNet


class OracleDataset(Dataset):
    def __init__(self, tokens, question_lengths, features, categories, answers):
        assert tokens.shape[0] == question_lengths.shape[0] == features.shape[0] \
            == categories.shape[0] == answers.shape[0]
        
        self.tokens = torch.from_numpy(tokens)
        self.question_lengths = torch.from_numpy(question_lengths)
        self.features = torch.from_numpy(features)
        self.categories = torch.from_numpy(categories)
        self.answers = torch.from_numpy(answers)
        
    def __len__(self):
        return self.tokens.size(0)
    
    def __getitem__(self, i):
        return (self.tokens[i], self.question_lengths[i],
                self.features[i], self.categories[i], self.answers[i])

def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    
    for tokens, q_lens, features, cats, answers in loader:
        tokens_var = Variable(tokens.cuda(), volatile=False)
        q_lens_var = Variable(q_lens.cuda(), volatile=False)
        features_var = Variable(features.cuda(), volatile=False)
        cats_var = Variable(cats.cuda(), volatile=False)

        scores = model(tokens_var, q_lens_var, features_var, cats_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == answers).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    tqdm.write('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train(model, num_epochs, print_every=1000):
    tqdm.write('Getting accuracy on validation set')
    check_accuracy(model, loader_valid)
    
    for epoch in range(num_epochs):
        tqdm.write('Starting epoch {} / {}'.format(epoch + 1, num_epochs))
        model.train()

        for t, (tokens, q_lens, features, cats, answers) in tqdm(enumerate(loader_train)):
            tokens_var = Variable(tokens.cuda(), requires_grad=False)
            q_lens_var = Variable(q_lens.cuda(), requires_grad=False)
            features_var = Variable(features.cuda(), requires_grad=False)
            cats_var = Variable(cats.cuda(), requires_grad=False)
            answers_var = Variable(answers.cuda(), requires_grad=False)

            loss = model.train_step(
                tokens_var, q_lens_var, features_var, cats_var, answers_var
            )

            if t % print_every == 0:
                tqdm.write('t = {}, loss = {:.4}'.format(t + 1, loss.data[0]))

        tqdm.write('Getting accuracy on validation set')
        check_accuracy(model, loader_valid)
        
    tqdm.write('Getting accuracy on training set')
    check_accuracy(model, loader_train)
    tqdm.write('Getting accuracy on test set')
    check_accuracy(model, loader_test)

def load_dataset(split, small):
    with open(data.get_processed_file('oracle', split, small), 'rb') as f:
        return pickle.load(f)

def get_data_loader(split, small):
    return DataLoader(
        OracleDataset(*load_dataset(split, small)),
        batch_size=64,
        shuffle=True,
        num_workers=1
    )


small = False
loader_train = get_data_loader('train', small)
loader_valid = get_data_loader('valid', small)
loader_test = get_data_loader('valid', small)
        
oracle_net = OracleNet().cuda()
train(oracle_net, num_epochs=15)
check_accuracy(oracle_net, loader_valid)