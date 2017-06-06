import pickle
import json

import numpy as np
from PIL import Image

import data
import vocab
from resnet_feature_extractor import ResnetFeatureExtractor

# add functions to get processed features here
def get_spatial_features(example, obj):
    img_width, img_height = example['image']['width'], example['image']['height']
    
    x, y, bbox_width, bbox_height = obj['bbox']
    x = 2 * (x / img_width) - 1
    y = 2 * (y / img_height) - 1
    bbox_width = 2 * bbox_width / img_width
    bbox_height = 2 * bbox_height / img_height

    return [x, y, x + bbox_width, y + bbox_height, x + bbox_width / 2, y + bbox_width / 2, bbox_width, bbox_height]

def make_dataset(split, small=False):
    data_tokens = []
    data_question_lengths = []
    data_features = [] # concatenation of image features, crop features, spatial information
    data_categories = []
    data_answers = []
    
    i = 0
    with open(data.get_gw_file(split), 'r') as f:
        for line in f:
            i += 1
            if i % 100 == 0:
                print(i)
                if small and i == 1000:
                    break
            
            example = json.loads(line)
            object_id = example['object_id']
            obj = [o for o in example['objects'] if o['id'] == object_id][0]
            
            img_path = data.get_coco_file(example['image']['file_name'])
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            x, y, bbox_width, bbox_height = obj['bbox']
            area = map(int, [x, y, x + bbox_width, y + bbox_height])
            img_crop = img.crop(area)
            
            features = np.concatenate([
                resnet_feature_extractor.get_image_features(img),
                resnet_feature_extractor.get_image_features(img_crop),
                get_spatial_features(example, obj)
            ])
            
            for qa in example['qas']:
                question_tokens = vocab.get_tokens(qa['question'])
                token_ids = [vocab_map.get_id_from_token(token) for token in question_tokens]
                token_ids.append(vocab_map.qmark)
                
                np_token_ids = np.zeros(data.MAX_TOKENS_PER_QUESTION, dtype=int)
                len_tokens = min(len(token_ids), data.MAX_TOKENS_PER_QUESTION)
                np_token_ids[:len_tokens] = token_ids[:len_tokens]
                
                data_tokens.append(np_token_ids)
                data_question_lengths.append(len_tokens)
                data_features.append(features)
                data_categories.append(obj['category_id'])
                data_answers.append(data.get_answer_id(qa['answer']))
        
    np_tokens = np.vstack(data_tokens)
    np_question_lengths = np.array(data_question_lengths, dtype=int)
    np_features = np.vstack(data_features, dtype=np.float32)
    np_categories = np.array(data_categories, dtype=int)
    np_answers = np.array(data_answers, dtype=int)
    
    with open(data.get_processed_file('oracle', split, small), 'wb') as f:
        pickle.dump((np_tokens, np_question_lengths, np_features, np_categories, np_answers), f, protocol=4)

if __name__ == '__main__':
    resnet_feature_extractor = ResnetFeatureExtractor()
    
    vocab_map = vocab.VocabMap()
    
    for small in (True, False):
        for split in ('train', 'valid', 'test'):
            print('================== {}, small = {} =================='.format(split, small))
            make_dataset(split, small)
