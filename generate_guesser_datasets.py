import pickle
import json

import numpy as np

import data
import data_utils
from vocab import VocabTagger


def make_dataset(split, small=False):
    data_dialogues = []
    data_all_spatial = []
    data_all_cats = []
    data_correct_objs = []
    
    i = 0
    with open(data.get_gw_file(split), 'r') as f:
        for line in f:
            i += 1
            if i % 1000 == 0:
                print(i)
                if small and i == 1000:
                    break
            
            example = json.loads(line)
            
            correct_obj_id = example['object_id']
            correct_obj_idx = [i for i, o in enumerate(example['objects'])
                               if o['id'] == correct_obj_id][0]
            
            all_cats = [o['category_id'] for o in example['objects']]
            all_spatial = [data_utils.get_spatial_features(example, o) for o in example['objects']]
            
            dialogue_tokens = vocab_tagger.get_dialogue_ids(example['qas'])
            
            data_dialogues.append(dialogue_tokens)
            data_all_cats.append(all_cats)
            data_all_spatial.append(all_spatial)
            data_correct_objs.append(correct_obj_idx)
    
    with open(data.get_processed_file('guesser', split, small), 'wb') as f:
        pickle.dump((data_dialogues, data_all_cats, data_all_spatial, data_correct_objs),
                    f, protocol=4)

if __name__ == '__main__':
    vocab_tagger = VocabTagger()
    
    for small in (True, False):
        for split in ('train', 'valid', 'test'):
            print('================== {}, small = {} =================='.format(split, small))
            make_dataset(split, small)
