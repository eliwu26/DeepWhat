import pickle
import json

import numpy as np

import data
import vocab

# add functions to get processed features here
def get_spatial_features(example, obj):
    img_width, img_height = example['image']['width'], example['image']['height']
    
    x, y, bbox_width, bbox_height = obj['bbox']
    x = 2 * (x / img_width) - 1
    y = 2 * (y / img_height) - 1
    bbox_width = 2 * bbox_width / img_width
    bbox_height = 2 * bbox_height / img_height

    return (x, y, x + bbox_width, y + bbox_height, x + bbox_width / 2, y + bbox_width / 2, bbox_width, bbox_height)

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
            all_spatial = [get_spatial_features(example, o) for o in example['objects']]
            
            dialogue_tokens = []
            for qa in example['qas']:
                question_tokens = vocab.get_tokens(qa['question'])
                dialogue_tokens.extend(
                    vocab_map.get_id_from_token(token) for token in question_tokens
                )
                dialogue_tokens.append(vocab_map.qmark)
                
                assert qa['answer'] in ('Yes', 'No', 'N/A')
                dialogue_tokens.append(vocab_map.get_id_from_token(
                    '<{}>'.format(qa['answer'])
                ))
                
            data_dialogues.append(dialogue_tokens)
            data_all_cats.append(all_cats)
            data_all_spatial.append(all_spatial)
            data_correct_objs.append(correct_obj_idx)
    
    with open(data.get_processed_file('guesser', split, small), 'wb') as f:
        pickle.dump((data_dialogues, data_all_cats, data_all_spatial, data_correct_objs),
                    f, protocol=4)

if __name__ == '__main__':
    vocab_map = vocab.VocabMap()
    
    for small in (True, False):
        for split in ('train', 'valid', 'test'):
            print('================== {}, small = {} =================='.format(split, small))
            make_dataset(split, small)
