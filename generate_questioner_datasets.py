import pickle
import json

import numpy as np
from PIL import Image

import data
from vocab import VocabTagger
from resnet_feature_extractor import ResnetFeatureExtractor


def make_padded_ndarray(seqs):
    max_len = max(map(len, seqs))
    result = np.zeros([len(seqs), max_len], dtype=int)
    for i, seq in enumerate(seqs):
        result[i, :len(seq)] = seq
    return result

def make_dataset(split, small=False):
    data_features = [] # resnet features
    data_input_seqs = []
    data_output_seqs = []
    data_seq_lens = []
    
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
            
            input_seq = [vocab_tagger.vocab_map.start]
            output_seq = []
            for qa in example['qas']:
                question_tokens = vocab_tagger.get_question_ids(qa['question'], qmark=False)
                answer_token = vocab_tagger.get_answer_id(qa['answer'])
                
                input_seq.extend(question_tokens)
                input_seq.append(answer_token)
                output_seq.extend(question_tokens)
                output_seq.append(vocab_tagger.vocab_map.qmark)
            output_seq.append(vocab_tagger.vocab_map.stop)
            
            if len(input_seq) <= 100 or split != 'train':
                data_features.append(resnet_feature_extractor.get_image_features(img))
                data_input_seqs.append(input_seq)
                data_output_seqs.append(output_seq)
                data_seq_lens.append(len(input_seq))
    
    np_features = np.array(data_features)
    np_input_seqs = make_padded_ndarray(data_input_seqs)
    np_output_seqs = make_padded_ndarray(data_output_seqs)
    
    with open(data.get_processed_file('questioner', split, small), 'wb') as f:
        pickle.dump((np_features, np_input_seqs, np_output_seqs, data_seq_lens),
                    f, protocol=4)

if __name__ == '__main__':
    resnet_feature_extractor = ResnetFeatureExtractor()
    vocab_tagger = VocabTagger()
    
    for small in (True, False):
        for split in ('train', 'valid', 'test'):
            print('================== {}, small = {} =================='.format(split, small))
            make_dataset(split, small)
