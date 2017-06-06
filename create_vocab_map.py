import json
import pickle
from collections import Counter

import data
import vocab

if __name__ == '__main__':
    counts = Counter()
    
    with open(data.get_gw_file('train'), 'r') as f:
        i = 0
        
        for line in f:
            i += 1
            if i % 1000 == 0:
                print(i)
            
            example = json.loads(line)
            for qa in example['qas']:
                question = qa['question']
                counts.update(vocab.get_tokens(question))
                
    vocab = ['<start>', '<unk>', '<?>', '<stop>']
    
    print('Total tokens: {}'.format(len(counts)))
    filtered_tokens = [token for token, count in counts.most_common() if count >= 10]
    print('Selected tokens: {}'.format(len(filtered_tokens)))
    
    vocab += filtered_tokens
    vocab += ['<Yes>', '<No>', '<N/A>']
    
    with open(data.VOCAB_LIST, 'w') as f:
        for word in vocab:
            print(word, file=f)
    
    token_to_id = {token: i+1 for i, token in enumerate(vocab)}
    
    with open(data.VOCAB_MAP, 'wb') as f:
        pickle.dump(token_to_id, f)