import pickle

import data

def get_tokens(question):
    return question.lower().replace('?', '').replace('/', ' ').replace('.', '').replace(',', '').replace('\\', '').replace('"', '').replace('(', '').replace(')', '').replace("'", '').replace('!', '').replace(':', '').replace(';', '').replace('$', '$ ').replace('%', '% ').replace('>', '').replace('<', '').strip().split()

class VocabMap(object):
    def __init__(self):
        with open(data.VOCAB_MAP, 'rb') as f:
            self.token_to_id = pickle.load(f)
            self.id_to_token = {i: token for token, i in self.vocab_to_id.items()}
            self.unk = self.token_to_id['<unk>']
            
    def token_to_id(self, token):
        return self.token_to_id.get(token, self.unk)
    
    def id_to_token(self, i):
        return self.id_to_token[i]