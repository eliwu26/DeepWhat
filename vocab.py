import pickle

import data

def get_tokens(question):
    return question.lower().replace('?', '').replace('/', ' ').replace('.', '').replace(',', '').replace('\\', '').replace('"', '').replace('(', '').replace(')', '').replace("'", '').replace('!', '').replace(':', '').replace(';', '').replace('$', '$ ').replace('%', '% ').replace('>', '').replace('<', '').strip().split()

class VocabMap(object):
    def __init__(self):
        with open(data.VOCAB_MAP, 'rb') as f:
            self.token_to_id = pickle.load(f)
            
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.unk = self.token_to_id['<unk>']
        self.qmark = self.token_to_id['<?>']
        self.vocab_size = max(self.id_to_token) + 1
            
    def get_id_from_token(self, token):
        return self.token_to_id.get(token, self.unk)
    
    def get_token_from_id(self, i):
        return self.id_to_token[i]