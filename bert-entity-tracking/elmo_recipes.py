from elmo_model_recipes import *
from tqdm import tqdm, trange
import bcolz, pickle, json
import numpy as np
import json
import time
from random import shuffle
import os
def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f

class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log = open(make_path(path), 'w')
        self.f_log.write(json.dumps(kwargs)+'\n')

    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log.write(json.dumps(kwargs)+'\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {"N":0, "NP": 1, "P": 2, START_TAG: 3, STOP_TAG: 4}

recipes_data = json.load(open('/scratch/cluster/agupta/recipes_elmo.json','r'))
model =elmo_ncrf_recipes(300,  150, tag_to_ix)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_data = []
val_data = []
test_data = []

for data in recipes_data:
    recipes_data[data]['para'] = []
    recipes_data[data]['targets'] = np.zeros((len(recipes_data[data]['text']),len(recipes_data[data]['ingredient_list'])))

    for step_num in range(len(recipes_data[data]['text'])):
        recipes_data[data]['para']+=recipes_data[data]['text'][str(step_num)]
    
    for step_num in recipes_data[data]['ingredients']:
        for ing in recipes_data[data]['ingredients'][step_num]:
            recipes_data[data]['targets'][int(step_num)][ing] = 1



for data in recipes_data:
    if len(recipes_data[data]['ingredient_list'])!=0 and len(recipes_data[data]['para'])!=0:
        if recipes_data[data]['split'] == 'train':
            train_data.append(recipes_data[data])
        elif recipes_data[data]['split'] == 'dev':
            val_data.append(recipes_data[data])
        else:
            test_data.append(recipes_data[data])



new_train_data = []
new_val_data = [] 


curr_data = {}

for data in test_data:
    for idx, ing in enumerate(data['ingredient_list']):
        curr_data['text'] = data['para']
        curr_data['lens'] = data['lens']
        curr_data['ing'] = " ".join(ing.split('_'))
        curr_data['gold'] = data['targets'][:,idx].tolist()
        new_train_data.append(curr_data)
        curr_data = {}

curr_data = {}

for data in val_data:
    for idx, ing in enumerate(data['ingredient_list']):
        curr_data['text'] = data['para']
        curr_data['lens'] = data['lens']
        curr_data['ing'] = " ".join(ing.split('_'))
        curr_data['gold'] = data['targets'][:,idx].tolist()
        new_val_data.append(curr_data)
        curr_data = {}



train_data = new_train_data
val_data = new_val_data
#print(new_train_data[:5])

shuffle(train_data)
print("len of train_data: %d, val_data: %d"%(len(train_data), len(val_data)))
data_points_processed = 0
# Make sure prepare_sequence from earlier in the LSTM section is loaded

logger_dict = {}

logger = ResultLogger('./elmo_entity_binary_300_150.jsonl', **logger_dict)

print(len(train_data))
for epoch in trange(30, desc="Epoch"):  # again, normally you would NOT do 300 epochs, it is toy data
    total_loss=0
    tqdm_bar = tqdm(train_data, desc="Training")
    model.train()
    train_loss = 0.0
    for _, ins in enumerate(tqdm_bar):

        #print(ins)
        model.zero_grad()
        gold_tags = torch.tensor(ins['gold'], dtype=torch.long).to(device)
        loss, logits = model.neg_log_likelihood(ins['text'], torch.tensor(ins['lens']).to(device), [ins['ing']], gold_tags)
        train_loss+=loss
        loss.backward()
        optimizer.step()


    model.eval()
    val_loss = 0.0
    for ins in val_data:
        ins['predicted_tags'] = []
        gold_tags = torch.tensor(ins['gold'], dtype=torch.long).to(device)
        loss, logits = model.neg_log_likelihood(ins['text'], torch.tensor(ins['lens']).to(device), [ins['ing']], gold_tags)
        ins['scores'] = logits.tolist()
        val_loss+=loss

    logger.log(train_loss=train_loss.to("cpu").detach().numpy().tolist(), val_loss =val_loss.to("cpu").detach().numpy().tolist())
    print("train_loss: %f, val_loss: %f"%(train_loss, val_loss))
    


    with open('./elmo_entity_binary_300_150'+str(epoch)+'.json', "w") as write_file:
        json.dump(val_data, write_file)
