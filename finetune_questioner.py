import pickle
import random

import torch
from torch.autograd import Variable
from tqdm import tqdm

import data
from vocab import VocabTagger
from logging_utils import start_log, log_print
from game import GuessWhatAgents, GuessWhatGame


BASELINE_ALPHA = 0.01
REINFORCE_RATE = 0.001

descriptor = 'questioner_reinforce_lstm1_fc2'

class GuessWhatReinforceGame(GuessWhatGame):
    def question(self, answer=None, mode='sample'):
        if answer is not None:
            answer_id = vocab_tagger.get_answer_id(answer) if answer is not None else None
            answer = Variable(
                torch.LongTensor([answer_id]).unsqueeze(0).cuda(),
                requires_grad=False
            )
        
        question_ids, self.questioner_h, probs, outputs = self.agents.questioner_net.sample(
            self.img_features,
            h_0=self.questioner_h,
            x_0=answer,
            mode=mode,
            reinforce=True
        )
        
        self.num_questions += 1
        self.dialogue.extend(question_ids)
        
        return question_ids, probs, outputs


split = 'train'
small = True

with open(data.get_processed_file('game', split, small), 'rb') as f:
    data_img_names, data_raw_objs, data_all_cats, data_all_spatial, data_correct_obj = pickle.load(f)

def get_example(i):
    return data_img_names[i], data_raw_objs[i], data_all_cats[i], data_all_spatial[i], data_correct_obj[i]

vocab_tagger = VocabTagger()
agents = GuessWhatAgents()
baseline = 0

optimizer = torch.optim.SGD(agents.questioner_net.parameters(), lr=REINFORCE_RATE)

for i in tqdm(range(len(data_img_names))):
    img_name, raw_objs, obj_cats, obj_spatial = get_example(i)
    num_objs = len(obj_cats)
    correct_obj = random.randint(0, num_objs - 1)
    
    game = GuessWhatReinforceGame(agents, img_name, obj_cats, obj_spatial,
                                  kwargs={'requires_grad': False})
    dialogue_probs = []
    dialogue_outputs = []
    
    answer = None
    for q in range(20):
        question_ids, probs, outputs = game.question(answer, mode='sample')
        
        if question_ids[0] == vocab_tagger.vocab_map.stop:
            dialogue_probs.append(probs[0])
            dialogue_outputs.append(outputs[0])
            break
            
        dialogue_probs.extend(probs)
        dialogue_outputs.extend(outputs)

        answer_id = game.answer(question_ids)
        answer = vocab_tagger.get_answer(answer_id)

    pred_idx = game.guess()
    reward = float(pred_idx == correct_obj)
    
    log_print(descriptor, 'i = {} | b = {} | r = {}'.format(i, baseline, reward))
    
    dialogue_log_probs = torch.log(torch.cat(dialogue_probs))
    adjusted_reward = reward - baseline
    
    J = torch.sum(torch.mul(dialogue_log_probs, adjusted_reward))
    
    for output in dialogue_outputs:
        output.reinforce(adjusted_reward)
    
    optimizer.zero_grad()
    J.backward()
    optimizer.step()
    
    baseline = BASELINE_ALPHA * reward + (1 - BASELINE_ALPHA) * baseline

torch.save(model.state_dict(), data.get_saved_model(descriptor))
