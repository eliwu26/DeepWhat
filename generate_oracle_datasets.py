import pickle
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable

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

model = models.resnet50(pretrained=True)

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

def get_image_features(img):
    img_resized = img.resize((224, 224), Image.ANTIALIAS)
    img_tensor = preprocess(img_resized)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)

    return model(img_variable).data.numpy().squeeze()

if __name__ == '__main__':
    # load pretrained model here
    
    vocab_map = vocab.VocabMap()
    
    data_tokens = []
    data_features = [] # concatenation of image features, crop features, spatial information
    data_categories = []
    
    for split in ('train', 'valid', 'test'):
        with open(data.get_gw_file(split), 'r') as f:
            
            for line in f:
                example = json.loads(line)
                img_path = data.get_coco_file(example['image']['file_name'])
                img = Image.open(img_path)
                
                img_features = get_image_features(img)

                spatial_features = get_spatial_features(example)
                
                for qa in example['qas']:
                    question_tokens = vocab.get_tokens(qa['question'])
                    question_token_ids = [vocab_map.token_to_id(token) for token in question_tokens]
                    data_tokens.append(question_token_ids)