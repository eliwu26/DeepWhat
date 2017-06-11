import random

import numpy as np
import torch
from torch.autograd import Variable

import data
from resnet_feature_extractor import ResnetFeatureExtractor
from vocab import VocabTagger
from models.questioner import QuestionerNet
from models.oracle_lite import OracleLiteNet
from models.guesser import GuesserNet


resnet_feature_extractor = ResnetFeatureExtractor()
vocab_tagger = VocabTagger()

questioner_net = QuestionerNet().cuda()
oracle_net = OracleLiteNet().cuda()
guesser_net = GuesserNet().cuda()

oracle_net.load_state_dict(
    torch.load(data.get_saved_model('oraclelite_gru2_fc2_cat32_h128_we64')))
guesser_net.load_state_dict(
    torch.load(data.get_saved_model('guesser_gru2_fc2_cat16_h256_we64')))
questioner_net.load_state_dict(
    torch.load(data.get_saved_model('questioner_lstm1_fc2')))
        

class GuessWhatGame(object):
    def __init__(self, img, obj_cats, obj_spatial, correct_obj=None):
        assert len(obj_cats) == len(obj_spatial)
        self.num_objs = len(obj_cats)
        
        if correct_obj is None:
            correct_obj = random.randint(0, self.num_objs - 1)

        assert 0 <= correct_obj < len(obj_cats)
        
        self.img = img
        self.img_features = Variable(torch.from_numpy(
            resnet_feature_extractor.get_image_features(img)
        ).unsqueeze(0).cuda(), volatile=True)
        
        self.obj_cats = Variable(torch.LongTensor(obj_cats).unsqueeze(0).cuda(), volatile=True)
        self.obj_spatial = Variable(torch.FloatTensor(obj_spatial).unsqueeze(0).cuda(), volatile=True)
        
        self.correct_obj = correct_obj
        
        self.oracle_features = torch.cat([
            self.img_features,
            self.obj_spatial[:, self.correct_obj, :]
        ], 1)
        
        self.questioner_h = None
        self.num_questions = 0
        
    def question(self, answer=None, mode='sample'):
        if answer is not None:
            answer_id = vocab_tagger.get_answer_id(answer) if answer is not None else None
            answer = Variable(torch.LongTensor(answer_id).unsqueeze(0).cuda(), volatile=True)
        
        question_ids, self.questioner_h = questioner_net.sample(
            self.img_features,
            h_0=self.questioner_h,
            x_0=answer,
            mode=mode)
        
        return question_ids
    
    def answer(self, question_ids):
        question_len = len(question_ids)
        cat_var = self.obj_cats[:, self.correct_obj]
        question_ids = torch.LongTensor(question_ids).unsqueeze(0)
        q_lens = torch.LongTensor([question_len])
        
        question_var = Variable(question_ids.cuda(), volatile=True)
        question_len_var = Variable(q_lens.cuda(), volatile=True)
        
        scores = oracle_net(question_var, question_len_var, self.oracle_features, cat_var)
        _, pred = scores.data.cpu().max(1)
        
        answer_idx = pred.squeeze().numpy()[0]
        return data.get_answer_from_idx(answer_idx)
