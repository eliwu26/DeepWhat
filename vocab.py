import pickle

import data

def tokenize(question):
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
    
    def get_answer_id(self, answer):
        if answer == 'Yes':
            return self.vocab_map.yes
        elif answer == 'No':
            return self.vocab_map.no
        elif answer == 'N/A':
            return self.vocab_map.na
        else:
            raise ValueError('invalid answer')
            
    def get_answer(self, answer_id):
        if answer_id == self.vocab_map.yes:
            return 'Yes'
        elif answer_id == self.vocab_map.no:
            return 'No'
        elif answer_id == self.vocab_map.na:
            return 'N/A'
        else:
            raise ValueError('invalid answer ID')
    
    def get_question_ids(self, question_tokens, qmark=False):
        ids = [self.vocab_map.get_id_from_token(token)
               for token in tokenize(question_tokens)]
        if qmark:
            ids.append(self.vocab_map.qmark)
        return ids
    
    def get_dialogue_ids(self, qas):
        '''
        For use in generating guesser dataset.
        '''
        dialogue_tokens = []
        
        dialogue_tokens.append(self.vocab_map.start)
        
        for qa in qas:
            dialogue_tokens.extend(
                self.get_question_ids(qa['question'])
            )
            dialogue_tokens.append(self.vocab_map.qmark)

        dialogue_tokens.append(self.get_answer_id(qa['answer']))
         
        dialogue_tokens.append(self.vocab_map.stop)
        
        return dialogue_tokens
    
    def get_question_tokens(self, question_ids):
        return [self.vocab_map.get_token_from_id(i) for i in question_ids]