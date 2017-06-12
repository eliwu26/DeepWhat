import pickle
import json

import numpy as np

import data
import data_utils


def make_dataset(split, small=False):
    data_imgs = []
    data_raw_objs = []
    data_all_spatial = []
    data_all_cats = []
    data_correct_obj = []
    
    i = 0
    with open(data.get_gw_file(split), 'r') as f:
        for line in f:
            i += 1
            if i % 1000 == 0:
                print(i)
                if small and i == 1000:
                    break
            
            example = json.loads(line)
            
            all_cats = [o['category_id'] for o in example['objects']]
            all_spatial = [data_utils.get_spatial_features(example, o) for o in example['objects']]
            
            img_path = data.get_coco_file(example['image']['file_name'])
            img = data_utils.img_from_path(img_path)
            
            correct_obj_id = example['object_id']
            correct_obj_idx = [i for i, o in enumerate(example['objects'])
                               if o['id'] == correct_obj_id][0]
            
            data_imgs.append(img)
            data_raw_objs.append(example['objects'])
            data_all_cats.append(all_cats)
            data_all_spatial.append(all_spatial)
            data_correct_obj.append(correct_obj_idx)
    
    with open(data.get_processed_file('game', split, small), 'wb') as f:
        pickle.dump(
            (data_imgs, data_raw_objs, data_all_cats, data_all_spatial, data_correct_obj),
            f, protocol=4
        )

if __name__ == '__main__':    
    for small in (True, False):
        for split in ('train', 'valid', 'test'):
                print('================== {}, small = {} =================='.format(split, small))
                make_dataset(split, small)
