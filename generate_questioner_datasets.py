import pickle
import json

import numpy as np
from PIL import Image

import data
from vocab import VocabTagger
from resnet_feature_extractor import ResnetFeatureExtractor


def make_padded_ndarray(seqs):
    max_len = max(map(len, seqs))
    result = np.zeros([len(seqs), max_len])
    for i, seq in enumerate(seqs):
        result[i, :len(seq)] = seq
    return result

def make_dataset(split, small=False):
    data_features = [] # resnet features
    data_input_seqs = []
    data_output_seqs = []
    
    i = 0
    with open(data.get_gw_file(split), 'r') as f:
        for line in f:
            i += 1
            if i % 100 == 0:
                print(i)
                if small and i == 1000:
                    break
            
            example = json.loads(line)
            img_path = data.get_coco_file(example['image']['file_name'])
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            dialogue_tokens = vocab_tagger.get_dialogue_tokens(example['qas'])
            
            data_features.append(resnet_feature_extractor.get_image_features(img))
            data_input_seqs.append(dialogue_tokens[:-1])
            data_output_seqs.append(dialogue_tokens[1:])
    
    np_features = np.array(data_features)
    np_input_seqs = make_padded_ndarray(data_input_seqs)
    np_output_seqs = make_padded_ndarray(data_output_seqs)
    
    with open(data.get_processed_file('questioner', split, small), 'wb') as f:
        pickle.dump((np_features, np_input_seqs, np_output_seqs), f, protocol=4)

if __name__ == '__main__':
    resnet_feature_extractor = ResnetFeatureExtractor()
    vocab_tagger = VocabTagger()
    
    for small in (True, False):
        for split in ('train', 'valid', 'test'):
            print('================== {}, small = {} =================='.format(split, small))
            make_dataset(split, small)
