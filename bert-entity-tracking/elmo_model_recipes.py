# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from math import exp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class elmo_ncrf_recipes(nn.Module):

    def __init__(self, hidden_dim, hidden_dim2, tag_to_ix):
        super(elmo_ncrf_recipes, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.tagset_size = len(tag_to_ix)
        self.tag_to_ix = tag_to_ix
        self.criterion = nn.NLLLoss()
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
        self.lstm_sent = nn.LSTM(1024, hidden_dim // 2,
                            num_layers=1, bidirectional =True)
        self.lstm_entity = nn.LSTM(self.hidden_dim, self.hidden_dim2//2, num_layers =1, bidirectional = True)
        self.ing_sent = nn.Linear(1024, self.hidden_dim)
        self.bi_attention = nn.Linear(1024, self.hidden_dim)
        self.ing_red = nn.Linear(1024, self.hidden_dim//2)
        self.hid2tag = nn.Linear(self.hidden_dim2, self.tagset_size)
        self.hid2logits = nn.Linear(self.hidden_dim2, 2)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.CrossEntropyLoss()
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        
        

    def init_hidden(self):
        
        #return torch.randn(2,1,self.hidden_dim)
        
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def init_hidden_entity_lstm(self, ing_embed):
        
        #return torch.randn(2,1,self.hidden_dim)
        ing_red = self.ing_red(ing_embed)
        ing_red = ing_red.view(1,1,-1)
        ing_red = ing_red.expand(2,-1,-1)

        #return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
        #        ing_red.contiguous())

        return (torch.randn(2, 1, self.hidden_dim2 // 2).to(device),
                torch.randn(2, 1, self.hidden_dim2 // 2).to(device))

    def _get_elmo_features(self, sentence):


        sentences = [sentence]
        character_ids = batch_to_ids(sentences).to(device)
        embeddings = self.elmo(character_ids)
        embeds = embeddings['elmo_representations'][0].view(len(sentence), -1)
        return  embeds



        
    def _get_ing_elmo_features(self, ing_list):
        
        ings = []
        for ing in ing_list:
            ings.append(torch.mean(self._get_elmo_features(ing.split()),0))

        return torch.stack(ings)


    def _get_scores(self, sent_embeds, ing_embeds):


        
        lstm_out = sent_embeds

        ing_embeds_red = self.ing_red(ing_embeds)

        '''
        attn_ing = self.bi_attention(ing_embeds)
        attn_scores = torch.mm(lstm_out, torch.t(attn_ing))
        attn_scores =  F.softmax(attn_scores,1)
        attn_sent = torch.mm(torch.t(attn_scores), lstm_out)
        '''
        ing_max_pool, _ = torch.max(lstm_out, 0)
        concat_max_pool = torch.cat((ing_max_pool.view(1,-1),ing_embeds_red),1)
        tag_scores = self.hid2tag(concat_max_pool)


        return tag_scores[0]

    def _score_para(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score


    def neg_log_likelihood(self, sentence, indices, ings, targets):


        elmo_sent = self._get_elmo_features(sentence)
        assert elmo_sent.size(0) == indices[-1]+1

  
        self.hidden_loc = self.init_hidden()
        lstm_out, self.hidden_loc = self.lstm_sent(elmo_sent.view(len(elmo_sent),1,-1), self.hidden_loc)
        lstm_out = lstm_out.view(len(elmo_sent),-1)

        

        total_loss = torch.zeros(1).to(device)
        elmo_ings  =self._get_ing_elmo_features(ings)
 
        # print(elmo_ings.shape)
        # print(elmo_ings[:10,:])
        # print('\n')

        step_features = []

        for step_num in range(len(targets)):
            attn_ing = self.bi_attention(elmo_ings)
            attn_scores = torch.mm(lstm_out[indices[step_num]+1:indices[step_num+1]+1], torch.t(attn_ing))
            # print(attn_scores)
            # print('\n')
            attn_scores =  F.softmax(attn_scores,0)
            # print(attn_scores)
            # print('\n')
            # print(torch.t(attn_scores))
            # print('\n')
            attn_sent = torch.mm(torch.t(attn_scores), lstm_out[indices[step_num]+1:indices[step_num+1]+1])
            step_features.append(attn_sent[0])
        #print(step_features)
        #print('\n')
        step_features = torch.stack(step_features).contiguous()

        self.hidden_loc_entity_lstm = self.init_hidden_entity_lstm(elmo_ings)
        entity_lstm_out, self.hidden_loc_entity_lstm = self.lstm_entity(step_features.view(len(step_features),1,-1), self.hidden_loc_entity_lstm)


        #tag_scores = torch.stack(tag_scores)

        logits = self.hid2logits(entity_lstm_out.view(len(targets),-1))
        #logits = self.hid2logits(step_features)
        loss = self.bce_loss(logits, targets)
        return loss, logits

    def forward(self, sentence, indices, ings):
        elmo_sent = self._get_elmo_features(sentence)
        assert elmo_sent.size(0) == indices[-1]+1
        
        self.hidden_loc = self.init_hidden()
        lstm_out, self.hidden_loc = self.lstm_sent(elmo_sent.view(len(elmo_sent),1,-1), self.hidden_loc)
        lstm_out = lstm_out.view(len(elmo_sent),-1)

        
        #lstm_out = elmo_sent
        total_loss = torch.zeros(1).to(device)
        elmo_ings  =self._get_ing_elmo_features(ings)


        step_features = []

        for step_num in range(len(indices)-1):

            #curr_feature, _  = torch.max(lstm_out[indices[step_num]+1:indices[step_num+1]+1], 0)


            attn_ing = self.bi_attention(elmo_ings)
            attn_scores = torch.mm(lstm_out[indices[step_num]+1:indices[step_num+1]+1], torch.t(attn_ing))
            #print(attn_scores)
            attn_scores =  F.softmax(attn_scores,0)
            #print(attn_scores)
            attn_sent = torch.mm(torch.t(attn_scores), lstm_out[indices[step_num]+1:indices[step_num+1]+1])
            step_features.append(attn_sent[0])

        step_features = torch.stack(step_features).contiguous()

        self.hidden_loc_entity_lstm = self.init_hidden_entity_lstm(elmo_ings)
        entity_lstm_out, self.hidden_loc_entity_lstm = self.lstm_entity(step_features.view(len(step_features),1,-1), self.hidden_loc_entity_lstm)


        #tag_scores = torch.stack(tag_scores)
        logits = self.hid2logits(entity_lstm_out.view(len(targets),-1))
        #logits = self.hid2logits(step_features)
        #loss = self.bce_loss(logits, targets)
        return logits







    

                


