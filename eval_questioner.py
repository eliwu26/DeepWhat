import pickle
import random

import data
from vocab import VocabTagger
from game import GuessWhatGame


def get_example(i):
    return data_imgs[i], data_raw_objs[i], data_all_cats[i], data_all_spatial[i]

def play_game(i):
    img, raw_objs, obj_cats, obj_spatial = get_example(i)

    num_objs = len(obj_cats)
    correct_obj = random.randint(0, num_objs - 1)
    
    game = GuessWhatGame(img, obj_cats, obj_spatial)
    
    answer = None
    for i in range(20):
        question_ids = game.question(answer, mode='sample')
        if question_ids[0] == vocab_tagger.vocab_map.stop:
            break

        answer_id = game.answer(question_ids)
        answer = vocab_tagger.get_answer(answer_id)

    pred_idx = game.guess()
    
    return pred_idx == correct_obj

vocab_tagger = VocabTagger()
small = True

for split in ('train', 'valid'):
    print('Playing game with images in: {}'.format(split))
    
    with open(data.get_processed_file('game', split, small), 'rb') as f:
        data_imgs, data_raw_objs, data_all_cats, data_all_spatial = pickle.load(f)

    num_correct = 0
    for i in range(len(data_imgs)):
        if i % 100 == 0:
            print(i)
        
        num_correct += play_game(i)
    
    print('Accuracy: {} / {} = {}'.format(num_correct, len(data_imgs), num_correct / len(data_imgs)))