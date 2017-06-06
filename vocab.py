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
        self.stop = self.token_to_id['<stop>']
        self.start = self.token_to_id['<start>']
        
        self.yes = self.token_to_id['<Yes>']
        self.no = self.token_to_id['<No>']
        self.na = self.token_to_id['<N/A>']
        self.vocab_size = len(self.id_to_token)

    def get_id_from_token(self, token):
        return self.token_to_id.get(token, self.unk)
    
    def get_token_from_id(self, i):
        return self.id_to_token[i]
    
class VocabTagger(object):
    def __init__(self):
        self.vocab_map = VocabMap()
        
    def get_dialogue_tokens(self, qas):
        dialogue_tokens = []
        
        dialogue_tokens.append(self.vocab_map.start)
        
        for qa in qas:
            question_tokens = get_tokens(qa['question'])
            dialogue_tokens.extend(
                self.vocab_map.get_id_from_token(token) for token in question_tokens
            )
            dialogue_tokens.append(self.vocab_map.qmark)

            if qa['answer'] == 'Yes':
                dialogue_tokens.append(self.vocab_map.yes)
            elif qa['answer'] == 'No':
                dialogue_tokens.append(self.vocab_map.no)
            elif qa['answer'] == 'N/A':
                dialogue_tokens.append(self.vocab_map.na)
            else:
                raise ValueError()
         
        dialogue_tokens.append(self.vocab_map.stop)
        
        return dialogue_tokens