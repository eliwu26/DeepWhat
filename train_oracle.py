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

def check_accuracy(model, descriptor, loader):
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    
    for tokens, q_lens, features, cats, answers in loader:
        tokens_var = Variable(tokens.cuda(), volatile=True)
        q_lens_var = Variable(q_lens.cuda(), volatile=True)
        features_var = Variable(features.cuda(), volatile=True)
        cats_var = Variable(cats.cuda(), volatile=True)

        scores = model(tokens_var, q_lens_var, features_var, cats_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == answers).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    log_print(descriptor,'Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

def start_log(filename):
    with open(data.get_log_file(filename), 'w') as f:
        f.write("")
        
def log_print(filename, message):
    tqdm.write(message)
    with open(data.get_log_file(filename), 'a') as f:
        f.write(message + '\n')
        
def train(model, descriptor, loader_valid_local, loader_train_local, loader_test_local, num_epochs, print_every=1000):
    start_log(descriptor)
    log_print(descriptor, 'Getting accuracy on validation set')
    check_accuracy(model, descriptor, loader_valid_local)
    current_max_val_acc = 0;
    for epoch in range(num_epochs):
        log_print(descriptor, 'Starting epoch {} / {}'.format(epoch + 1, num_epochs))
        model.train()

        for t, (tokens, q_lens, features, cats, answers) in tqdm(enumerate(loader_train_local)):
            tokens_var = Variable(tokens.cuda(), requires_grad=False)
            q_lens_var = Variable(q_lens.cuda(), requires_grad=False)
            features_var = Variable(features.cuda(), requires_grad=False)
            cats_var = Variable(cats.cuda(), requires_grad=False)
            answers_var = Variable(answers.cuda(), requires_grad=False)

            loss = model.train_step(
                tokens_var, q_lens_var, features_var, cats_var, answers_var
            )

            if t % print_every == 0:
                log_print(descriptor, 't = {}, loss = {:.4}'.format(t + 1, loss.data[0]))

        log_print(descriptor, 'Getting accuracy on validation set')
        accuracy = check_accuracy(model, descriptor, loader_valid_local)
        if(accuracy > current_max_val_acc):
            current_max_val_acc = accuracy
            torch.save(model.state_dict(), data.get_saved_model(descriptor))
        
    log_print(descriptor, 'Getting accuracy on training set')
    check_accuracy(model, descriptor, loader_train_local)
    log_print(descriptor, 'Getting accuracy on test set')
    check_accuracy(model, descriptor, loader_test_local)

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

def main():
    file_descriptor = 'GRU_3FC'
    small = False
    loader_train = get_data_loader('train', small)
    loader_valid = get_data_loader('valid', small)
    loader_test = get_data_loader('test', small)

    oracle_net = OracleNet().cuda()
    train(oracle_net, file_descriptor, loader_valid, loader_train, loader_test, num_epochs=15)
    
if __name__ == '__main__':
    main()
    