import pickle
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable

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

    return [x, y, x + bbox_width, y + bbox_height, x + bbox_width / 2, y + bbox_width / 2, bbox_width, bbox_height]

def get_image_features(img):
    img_tensor = preprocess(img).type(torch.cuda.FloatTensor)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)

    return model(img_variable).cpu().data.numpy().squeeze()

if __name__ == '__main__':
    model = models.resnet50(pretrained=True).cuda()

    # remove last fully-connected layer
    new_model = nn.Sequential(*list(model.children())[:-1])
    model = new_model

    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Scale(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       normalize
    ])
    
    vocab_map = vocab.VocabMap()
    
    data_tokens = []
    data_token_masks = []
    data_features = [] # concatenation of image features, crop features, spatial information
    data_categories = []
    data_answers = []
    
    for split in ('train', 'valid', 'test'):
        print('================== {} =================='.format(split))
        
        i = 0
        with open(data.get_gw_file(split), 'r') as f:
            for line in f:
                i += 1
                if i % 100 == 0:
                    print(i)
                
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
                
                features = np.concatenate([get_image_features(img), get_image_features(img_crop), get_spatial_features(example, obj)])
                
                for qa in example['qas']:
                    question_tokens = vocab.get_tokens(qa['question'])
                    token_ids = [vocab_map.get_id_from_token(token) for token in question_tokens]
                    token_ids.append(vocab_map.qmark)
                    
                    np_token_ids = np.zeros(data.MAX_TOKENS_PER_QUESTION, dtype=int)
                    len_tokens = min(len(token_ids), data.MAX_TOKENS_PER_QUESTION)
                    np_token_ids[:len_tokens] = token_ids[:len_tokens]
                    np_token_mask = np.zeros(data.MAX_TOKENS_PER_QUESTION, dtype=bool)
                    np_token_mask[:len_tokens] = True
                    
                    data_tokens.append(np_token_ids)
                    data_token_masks.append(np_token_mask)
                    data_features.append(features)
                    data_categories.append(obj['category_id'])
                    data_answers.append(data.get_answer_id(qa['answer']))
        
        np_tokens = np.vstack(data_tokens)
        np_token_masks = np.vstack(data_token_masks)
        np_features = np.vstack(data_features)
        np_categories = np.array(data_categories, dtype=int)
        np_answers = np.array(data_answers, dtype=int)
        
        with open(data.get_processed_file('oracle', split), 'wb') as f:
            pickle.dump((np_tokens, np_token_masks, np_features, np_categories, np_answers), f)