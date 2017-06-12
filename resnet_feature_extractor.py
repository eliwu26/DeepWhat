import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class ResnetFeatureExtractor(object):
    def __init__(self):
        model = models.resnet50(pretrained=True).cuda()

        # remove last fully-connected layer
        self.model = nn.Sequential(*list(model.children())[:-1])

        self.preprocess = transforms.Compose([
           transforms.Scale(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
            )
        ])
        
    def get_image_features(self, img):
        img = Image.fromarray(img)
        img_tensor = self.preprocess(img).type(torch.cuda.FloatTensor)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)

        return self.model(img_variable).cpu().data.numpy().squeeze()
 