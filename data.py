import os

DATASETS_DIR = '/datasets'
COCO_DIR = os.path.join(DATASETS_DIR, 'coco')

TRAIN_FILE = os.path.join(DATASETS_DIR, 'guesswhat.train.jsonl')
VALID_FILE = os.path.join(DATASETS_DIR, 'guesswhat.valid.jsonl')
TEST_FILE = os.path.join(DATASETS_DIR, 'guesswhat.test.jsonl')

PROCESSED_DIR = '/processed'

VOCAB_LIST = os.path.join(PROCESSED_DIR, 'vocab.txt')
VOCAB_MAP = os.path.join(PROCESSED_DIR, 'vocab.pickle')

MAX_TOKENS_PER_QUESTION = 20

def get_gw_file(split):
    if split == 'train':
        return TRAIN_FILE
    elif split == 'valid':
        return VALID_FILE
    elif split == 'test':
        return TEST_FILE
    
def get_processed_file(model, split, small):
    return os.path.join(PROCESSED_DIR, '{}_{}{}.pickle'.format(model, split, '_small' if small else ''))
    
def get_coco_file(filename):
    return os.path.join(COCO_DIR, filename)

def get_answer_id(answer):
    if answer == 'Yes':
        return 0
    elif answer == 'No':
        return 1
    elif answer == 'N/A':
        return 2
    else:
        raise ValueError('invalid answer')