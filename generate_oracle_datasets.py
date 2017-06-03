import pickle
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable

from PIL import Image

import data

# add functions to get processed features here

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


if __name__ == '__main__':
    # load pretrained model here
    
    for split in ('train', 'valid', 'test'):
        with open(data.get_gw_file(split), 'r') as f:
            
            for line in f:
                example = json.loads(line)
                img_path = data.get_coco_file(example['image']['file_name'])
                print(img_path)
                img = Image.open(img_path)
                img_resized = img.resize((224, 224), Image.ANTIALIAS)
                img_tensor = preprocess(img_resized)
                img_tensor.unsqueeze_(0)
                img_variable = Variable(img_tensor)
                
                img_features = model(img_variable).data.numpy().squeeze()
                print(img_features)