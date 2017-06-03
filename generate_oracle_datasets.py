import pickle
import json

# load pytorch etc here
from PIL import Image

import data
import vocab

# add functions to get processed features here
def get_spatial_features(example):
    img_width, img_height = example['image']['width'], example['image']['height']
    object_id = example['object_id']
    
    x, y, bbox_width, bbox_height = [o['bbox'] for o in example['objects'] if o['id'] == object_id][0]
    x = 2 * (x / img_width) - 1
    y = 2 * (y / img_height) - 1
    bbox_width = 2 * bbox_width / img_width
    bbox_height = 2 * bbox_height / img_height

    return [x, y, x + bbox_width, y + bbox_height, x + bbox_width / 2, y + bbox_width / 2, bbox_width, bbox_height]

if __name__ == '__main__':
    # load pretrained model here
    
    vocab_map = data.VocabMap()
    
    data_tokens = []
    data_features = [] # concatenation of image features, crop features, spatial information
    data_categories = []
    
    for split in ('train', 'valid', 'test'):
        with open(data.get_gw_file(split), 'r') as f:
            for line in f:
                example = json.loads(line)
                
                img_path = data.get_coco_file(example['image']['file_name'])
                img = Image.open(img_path)
                
                img_resized = img.resize((224, 224), Image.ANTIALIAS)
                
                spatial_features = get_spatial_features(example)
                
                for qa in example['qas']:
                    question_tokens = vocab.get_tokens(qa['question'])
                    question_token_ids = [vocab_map.token_to_id(token) for token in question_tokens]
                    data_tokens.append(question_token_ids)
                    
                    