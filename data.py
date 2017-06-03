import os

DATASETS_DIR = '/datasets'
COCO_DIR = os.path.join(DATASETS_DIR, 'coco')

TRAIN_FILE = os.path.join(DATASETS_DIR, 'guesswhat.train.jsonl')
VALID_FILE = os.path.join(DATASETS_DIR, 'guesswhat.valid.jsonl')
TEST_FILE = os.path.join(DATASETS_DIR, 'guesswhat.test.jsonl')

PROCESSED_DIR = '/processed'

VOCAB_LIST = os.path.join(PROCESSED_DIR, 'vocab.txt')
VOCAB_MAP = os.path.join(PROCESSED_DIR, 'vocab.pickle')

def get_gw_file(split):
    if split == 'train':
        return TRAIN_FILE
    elif split == 'valid':
        return VALID_FILE
    elif split == 'test':
        return TEST_FILE
    
def get_coco_file(filename):
    return os.path.join(COCO_DIR, filename)