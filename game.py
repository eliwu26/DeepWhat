import random

import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image

import data
import data_utils
from resnet_feature_extractor import ResnetFeatureExtractor
from vocab import VocabTagger
from models.questioner import QuestionerNet
from models.oracle_lite import OracleLiteNet
from models.guesser import GuesserNet

vocab_tagger = VocabTagger()

class GuessWhatAgents(object):
    def __init__(self, questioner='questioner_lstm1_fc2'):
        self.resnet_feature_extractor = ResnetFeatureExtractor()

        self.questioner_net = QuestionerNet().cuda()
        self.oracle_net = OracleLiteNet().cuda()
        self.guesser_net = GuesserNet().cuda()

        self.oracle_net.load_state_dict(
            torch.load(data.get_saved_model('oraclelite_gru2_fc2_cat32_h128_we64')))
        self.guesser_net.load_state_dict(
            torch.load(data.get_saved_model('guesser_gru2_fc2_cat16_h256_we64')))
        self.questioner_net.load_state_dict(
            torch.load(data.get_saved_model(questioner)))
        

class GuessWhatGame(object):
    def __init__(self, agents, img_name, obj_cats, obj_spatial, correct_obj=None,
                 kwargs={'volatile': True}):
        '''
        kwargs: {'volatile': True} when evaluating,
                {'requires_grad': False} in reinforcement learning setting
        '''
        self.agents = agents
        self.kwargs = kwargs
        
        assert len(obj_cats) == len(obj_spatial)
        self.num_objs = len(obj_cats)
        
        if correct_obj is None:
            correct_obj = random.randint(0, self.num_objs - 1)

        assert 0 <= correct_obj < len(obj_cats)
        
        img_path = data.get_coco_file(img_name)
        img = data_utils.img_from_path(img_path)
        
        self.img_features = Variable(torch.from_numpy(
            self.agents.resnet_feature_extractor.get_image_features(img)
        ).unsqueeze(0).cuda(), **self.kwargs)
        
        self.obj_cats = Variable(
            torch.LongTensor(obj_cats).unsqueeze(0).cuda(),
            **self.kwargs
        )
        self.obj_spatial = Variable(
            torch.FloatTensor(obj_spatial).unsqueeze(0).cuda(),
            **self.kwargs
        )
        
        self.correct_obj = correct_obj
        
        self.oracle_features = torch.cat([
            self.img_features,
            self.obj_spatial[:, self.correct_obj, :]
        ], 1)
        
        self.reset_state()
        
    def reset_state(self):
        self.questioner_h = None
        self.num_questions = 0
        self.dialogue = []
        
    def question(self, answer=None, mode='sample'):
        if answer is not None:
            answer_id = vocab_tagger.get_answer_id(answer) if answer is not None else None
            answer = Variable(
                torch.LongTensor([answer_id]).unsqueeze(0).cuda(),
                **self.kwargs
            )
        
        question_ids, self.questioner_h = self.agents.questioner_net.sample(
            self.img_features,
            h_0=self.questioner_h,
            x_0=answer,
            mode=mode
        )
        
        self.num_questions += 1
        self.dialogue.extend(question_ids)
        
        return question_ids
    
    def answer(self, question_ids):
        question_len = len(question_ids)
        cat_var = self.obj_cats[:, self.correct_obj]
        question_ids = torch.LongTensor(question_ids).unsqueeze(0)
        q_lens = torch.LongTensor([question_len])
        
        question_var = Variable(question_ids.cuda(), **self.kwargs)
        question_len_var = Variable(q_lens.cuda(), **self.kwargs)
        
        scores = self.agents.oracle_net(question_var, question_len_var, self.oracle_features, cat_var)
        _, pred = scores.data.cpu().max(1)
        
        answer_idx = pred.squeeze().numpy()[0]
        answer = data.get_answer_from_idx(answer_idx)
        answer_id = vocab_tagger.get_answer_id(answer)
        
        self.dialogue.append(answer_id)
        
        return answer_id
    
    def guess(self):
        dialogue_var = Variable(
            torch.LongTensor(self.dialogue).unsqueeze(0).cuda(),
            **self.kwargs
        )
        dialogue_len = [len(self.dialogue)]
        
        scores = self.agents.guesser_net(dialogue_var, dialogue_len, self.obj_cats, self.obj_spatial)
        _, pred = scores.data.cpu().max(1)
        
        return pred.squeeze().numpy()[0]