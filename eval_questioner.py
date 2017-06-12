import pickle
import random

import data
from vocab import VocabTagger
from game import GuessWhatGame, GuessWhatAgents
from logging_utils import start_log, log_print


def get_example(i):
    return data_img_names[i], data_raw_objs[i], data_all_cats[i], data_all_spatial[i], data_correct_obj[i]

def play_game(i, seen_obj):
    '''
    seen_obj: use obj seen during training if True, else random obj
    '''
    img_name, raw_objs, obj_cats, obj_spatial, correct_obj = get_example(i)

    num_objs = len(obj_cats)
    if not seen_obj:
        correct_obj = random.randint(0, num_objs - 1)
    
    game = GuessWhatGame(agents, img_name, obj_cats, obj_spatial)
    
    answer = None
    for i in range(20):
        question_ids = game.question(answer, mode='sample')
        if question_ids[0] == vocab_tagger.vocab_map.stop:
            break

        answer_id = game.answer(question_ids)
        answer = vocab_tagger.get_answer(answer_id)

    pred_idx = game.guess()
    
    return pred_idx == correct_obj

descriptor = 'eval_questioner_lstm1_fc2'

vocab_tagger = VocabTagger()
agents = GuessWhatAgents()
small = True

for split in ('train', 'valid'):
    for seen_obj in (True, False):
        if split == 'valid' and seen_obj == True:
            continue
        
        log_print(descriptor, 'Playing game with images in: {} | seen objects: {}'.format(split, seen_obj))

        with open(data.get_processed_file('game', split, small), 'rb') as f:
            data_img_names, data_raw_objs, data_all_cats, data_all_spatial, data_correct_obj = pickle.load(f)

        num_correct = 0
        for i in range(len(data_img_names)):
            if i % 100 == 0:
                print(i)

            num_correct += play_game(i, seen_obj)

        log_print(descriptor, 'Accuracy: {} / {} = {}'.format(num_correct, len(data_imgs), num_correct / len(data_imgs)))