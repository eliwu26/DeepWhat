import pickle
import json

# load pytorch etc here
from PIL import Image

import data
import vocab

# add functions to get processed features here

if __name__ == '__main__':
    # load pretrained model here
    
    vocab_map = data.VocabMap()
    
    for split in ('train', 'valid', 'test'):
        with open(data.get_gw_file(split), 'r') as f:
            for line in f:
                example = json.loads(line)
                
                img_path = data.get_coco_file(example['image']['file_name'])
                img = Image.open(img_path)
                img_resized = img.resize((224, 224), Image.ANTIALIAS)
                
                for qa in example['qas']:
                    question_tokens = vocab.get_tokens(qa['question'])
                    question_token_ids = [vocab_map.token_to_id(token) for token in question_tokens]