import pickle
import json

import numpy as np

import data
import data_utils
from resnet_feature_extractor import ResnetFeatureExtractor


def make_dataset(split):
    data_all_spatial = []
    data_all_cats = []
    data_img_feats = []
    
    i = 0
    with open(data.get_gw_file(split), 'r') as f:
        for line in f:
            i += 1
            if i % 1000 == 0:
                print(i)
            
            example = json.loads(line)
            
            all_cats = [o['category_id'] for o in example['objects']]
            all_spatial = [data_utils.get_spatial_features(example, o) for o in example['objects']]
            
            img_path = data.get_coco_file(example['image']['file_name'])
            img = data_utils.img_from_path(img_path)
            
            data_all_cats.append(all_cats)
            data_all_spatial.append(all_spatial)
            data_img_feats.append(resnet_feature_extractor.get_image_features(img))
    
    with open(data.get_processed_file('game', split, small), 'wb') as f:
        pickle.dump((data_all_cats, data_all_spatial, data_img_feats), f, protocol=4)

if __name__ == '__main__':    
    resnet_feature_extractor = ResnetFeatureExtractor()
    
    for split in ('valid', 'test'):
            print('================== {} =================='.format(split))
            make_dataset(split)
