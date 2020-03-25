from elmo_model_recipes import *
from tqdm import tqdm, trange
import bcolz, pickle, json
import numpy as np
import json

from random import shuffle
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


'''
new_train_data = []
new_val_data = [] 


curr_data = {}
count=0
for data in test_data:
    count+=1
    sorted_text = []
    for step_num in range(len(data['text'])):
        sorted_text.append(" ".join(data['text'][str(step_num)]))

    ings = [" ".join(ing.split('_')) for ing in  data['ingredient_list']]
    for idx, ing in enumerate(data['ingredient_list']):
        for step in range(len(sorted_text)):
            if step!=0:
                curr_data['text'] = " ".join(sorted_text[:step-1]).split()
            else:
                curr_data['text'] = []
            curr_data['ing'] = " ".join(ing.split('_')).split()
            curr_data['context'] = sorted_text[step].split()
            curr_data['gold'] = int(data['targets'][step,idx])
            curr_data['id'] = str(count) + '_' + str(idx) + ' ' + str(step)
            curr_data['all_ings'] = ings
            new_train_data.append(curr_data)
            curr_data = {}

curr_data = {}
count=0
for data in val_data:
    count+=1
    sorted_text = []
    for step_num in range(len(data['text'])):
        sorted_text.append(" ".join(data['text'][str(step_num)]))

    ings = [" ".join(ing.split('_')) for ing in  data['ingredient_list']]
    for idx, ing in enumerate(data['ingredient_list']):
        for step in range(len(sorted_text)):
            if step!=0:
                curr_data['text'] = " ".join(sorted_text[:step-1]).split()
            else:
                curr_data['text'] = []
            curr_data['ing'] = " ".join(ing.split('_')).split()
            curr_data['context'] = sorted_text[step].split()
            curr_data['gold'] = int(data['targets'][step,idx])
            curr_data['id'] = str(count) + '_' + str(idx) + ' ' + str(step)
            curr_data['all_ings'] = ings
            new_val_data.append(curr_data)
            curr_data = {}


print(len(new_train_data))
print(len(new_val_data))
json.dump(new_train_data, open('./train_gpt_ss_ai.json','w'))
json.dump(new_val_data, open('./val_gpt_ss_ai.json','w'))

'''

new_train_data = []
new_val_data = [] 


curr_data = {}
count=0
for data in test_data:
    count+=1
    sorted_text = []
    for step_num in range(len(data['text'])):
        sorted_text.append(" ".join(data['text'][str(step_num)]))

    ings = [" ".join(ing.split('_')) for ing in  data['ingredient_list']]
    for idx, ing in enumerate(data['ingredient_list']):
        for step in range(len(sorted_text)):
            curr_data['text'] = sorted_text
            curr_data['ing'] = " ".join(ing.split('_')).split()
            curr_data['context'] = step
            curr_data['gold'] = int(data['targets'][step,idx])
            curr_data['id'] = str(count) + '_' + str(idx) + ' ' + str(step)
            #curr_data['all_ings'] = ings
            new_train_data.append(curr_data)
            curr_data = {}

curr_data = {}
count=0
for data in val_data:
    count+=1
    sorted_text = []
    for step_num in range(len(data['text'])):
        sorted_text.append(" ".join(data['text'][str(step_num)]))

    ings = [" ".join(ing.split('_')) for ing in  data['ingredient_list']]
    for idx, ing in enumerate(data['ingredient_list']):
        for step in range(len(sorted_text)):
            #if step!=0:
            curr_data['text'] = sorted_text
            curr_data['ing'] = " ".join(ing.split('_')).split()
            curr_data['context'] = step
            curr_data['gold'] = int(data['targets'][step,idx])
            curr_data['id'] = str(count) + '_' + str(idx) + ' ' + str(step)
            #curr_data['all_ings'] = ings
            new_val_data.append(curr_data)
            curr_data = {}


print(len(new_train_data))
print(len(new_val_data))
json.dump(new_train_data, open('./train_bert_whole.json','w'))
json.dump(new_val_data, open('./val_bert_whole.json','w'))

'''
new_train_data = []
new_val_data = [] 


curr_data = {}

for data in test_data:
    for idx, ing in enumerate(data['ingredient_list']):
        curr_data['text'] = data['para']
        curr_data['lens'] = data['lens']
        curr_data['ing'] = " ".join(ing.split('_'))
        curr_data['gold'] = data['targets'][:,idx].tolist()
        #curr_data['id'] = str(count) + '_' + str(idx)
        #print(curr_data)
        new_train_data.append(curr_data)
        curr_data = {}

curr_data = {}

for data in val_data:
    for idx, ing in enumerate(data['ingredient_list']):
        curr_data['text'] = data['para']
        curr_data['lens'] = data['lens']
        curr_data['ing'] = " ".join(ing.split('_'))
        curr_data['gold'] = data['targets'][:,idx].tolist()
        #curr_data['id'] = str(count) + '_' + str(idx)
        #print(curr_data)
        new_val_data.append(curr_data)
        curr_data = {}



train_data = new_train_data
val_data = new_val_data
#print(new_train_data[:5])

shuffle(train_data)
print("len of train_data: %d, val_data: %d"%(len(train_data), len(val_data)))
data_points_processed = 0
# Make sure prepare_sequence from earlier in the LSTM section is loaded

print(len(train_data))
for epoch in trange(20, desc="Epoch"):  # again, normally you would NOT do 300 epochs, it is toy data
    total_loss=0
    tqdm_bar = tqdm(train_data, desc="Training")
    model.train()
    train_loss = 0.0
    for step, ins in enumerate(tqdm_bar):
        #print(ins)

        model.zero_grad()
        gold_tags = torch.tensor(ins['gold'], dtype=torch.long).to(device)
        loss, logits = model.neg_log_likelihood(ins['text'], torch.tensor(ins['lens']).to(device), [ins['ing']], gold_tags)
        train_loss+=loss

        #print(loss)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    for ins in val_data:
        #print('done!')
        ins['predicted_tags'] = []
        #ins['targets'] = ins['targets'].tolist()

        gold_tags = torch.tensor(ins['gold'], dtype=torch.long).to(device)
        loss, logits = model.neg_log_likelihood(ins['text'], torch.tensor(ins['lens']).to(device), [ins['ing']], gold_tags)
        ins['scores'] = logits.tolist()
        val_loss+=loss
        #ins['predicted_tags'].append(tag_seq)
    print("train_loss: %f, val_loss: %f"%(train_loss, val_loss))
    with open('./elmo_val_attention_binary_'+str(epoch)+'.json', "w") as write_file:
        json.dump(val_data, write_file)
'''
'''
expanded_val_data = []
curr_data = {}
count = 0
for data in val_data:
    #print(data)
    count+=1
    sorted_text = []
    data['targets'] = data['targets'].astype(int)
    for step_num in range(len(data['text'])):
        sorted_text.append(" ".join(data['text'][str(step_num)]))

    ings = [" ".join(ing.split('_')) for ing in  data['ingredient_list']]


    for idx, ing in enumerate(data['ingredient_list']):
        curr_data['text'] = sorted_text
        curr_data['all_ings'] = ings
        curr_data['ing'] = " ".join(ing.split('_'))
        curr_data['gold'] = data['targets'][:,idx].tolist()
        curr_data['id'] = str(count) + '_' + str(idx)
        #print(curr_data)
        expanded_val_data.append(curr_data)
        curr_data = {}
json.dump(expanded_val_data, open('./val_gpt_whole.json','w'))
print(len(expanded_val_data))
count = 0

print(expanded_val_data[:20])
expanded_val_data = []
curr_data = {}
for data in test_data:
    #print(data)
    count+=1
    sorted_text = []
    data['targets'] = data['targets'].astype(int)

    for step_num in range(len(data['text'])):
        sorted_text.append(" ".join(data['text'][str(step_num)]))

    ings = [" ".join(ing.split('_')) for ing in  data['ingredient_list']]

    for idx, ing in enumerate(data['ingredient_list']):
        curr_data['text'] = sorted_text
        curr_data['all_ings'] = ings
        curr_data['ing'] = " ".join(ing.split('_'))
        curr_data['gold'] = data['targets'][:,idx].tolist()
        curr_data['id'] = str(count) + '_' + str(idx)
        #print(curr_data)
        expanded_val_data.append(curr_data)
        curr_data = {}
json.dump(expanded_val_data, open('./test_gpt_whole.json','w'))
print(len(expanded_val_data))
'''
'''
with open('./bert_tagging_train_data.json', "w") as write_file:
    json.dump(train_data, write_file)
with open('./bert_tagging_val_data.json', "w") as write_file:
    json.dump(val_data, write_file)

print("done dumping")
'''
#create crf labels
'''
for ins in test_data:
    ins['targets'] = ins['targets'].tolist()
    ins['crf_labels'] = [[] for _ in range(len(ins['ingredient_list']))]
    for idx, ing in enumerate(ins['ingredient_list']):
        introduced  = 0
        for step_num in range(len(ins['text'])):
            if not introduced:
                if ins['targets'][step_num][idx] == 1:
                    ins['crf_labels'][idx].append(1)
                    introduced = 1
                else:
                    ins['crf_labels'][idx].append(0)

            else:
                if ins['targets'][step_num][idx] == 1:
                    ins['crf_labels'][idx].append(2)
                else:
                    ins['crf_labels'][idx].append(3)

for ins in val_data:
    ins['targets'] = ins['targets'].tolist()
    ins['crf_labels'] = [[] for _ in range(len(ins['ingredient_list']))]
    for idx, ing in enumerate(ins['ingredient_list']):
        introduced  = 0
        for step_num in range(len(ins['text'])):
            if not introduced:
                if ins['targets'][step_num][idx] == 1:
                    ins['crf_labels'][idx].append(1)
                    introduced = 1
                else:
                    ins['crf_labels'][idx].append(0)

            else:
                if ins['targets'][step_num][idx] == 1:
                    ins['crf_labels'][idx].append(2)
                else:
                    ins['crf_labels'][idx].append(3)


'''
'''
train_data = test_data
print(train_data[:5])

data_points_processed = 0
# Make sure prepare_sequence from earlier in the LSTM section is loaded

print(len(train_data))
for _ in trange(20, desc="Epoch"):  # again, normally you would NOT do 300 epochs, it is toy data
    total_loss=0
    tqdm_bar = tqdm(train_data, desc="Training")
    for step, ins in enumerate(tqdm_bar):
        #print(ins)
        for idx, ing in enumerate(ins['ingredient_list']):
            model.zero_grad()
            data_points_processed+=1
            
            if data_points_processed%100==0:
                print(loss)
            
            gold_tags = torch.tensor([tag_to_ix[t] for t in ins['crf_labels'][idx]], dtype=torch.long).to(device)
            loss = model.neg_log_likelihood(ins['para'], torch.tensor(ins['lens']).to(device), [ing], gold_tags)
            total_loss+=loss
            loss.backward()
            optimizer.step()

model.eval()
for ins in val_data:
    ins['predicted_tags'] = []
    ins['scores'] = []
    ins['targets'] = ins['targets'].tolist()
    for idx, ing in enumerate(ins['ingredient_list']):

        score, tag_seq = model(ins['para'], torch.tensor(ins['lens']).to(device), [ing])
        ins['scores'].append(score.tolist())
        ins['predicted_tags'].append(tag_seq)

with open('./ncrf_val_data_max_pooling_entity_lstm.json', "w") as write_file:
    json.dump(val_data, write_file)
'''

'''
expanded_val_data = []
curr_data = {}
count = 0
for data in val_data:
    #print(data)
    count+=1
    for step_num in range(len(data['text'])):
        for idx, ing in enumerate(data['ingredient_list']):
            curr_data['text'] = data['para'][:data['lens'][step_num]+1]
            curr_data['ing'] = ing.split('_')
            curr_data['context'] = data['para'][data['lens'][step_num]+1:data['lens'][step_num+1]+1]
            curr_data['gold'] = data['crf_labels'][idx][step_num]
            curr_data['id'] = str(count) + '_' + str(idx) + '_' + str(step_num)
            #print(curr_data)
            expanded_val_data.append(curr_data)
            curr_data = {}
json.dump(expanded_val_data, open('./val_gpt_4_tags.json','w'))
print(len(expanded_val_data))
count = 0
expanded_val_data = []
curr_data = {}
for data in test_data:
    #print(data)
    count+=1
    for step_num in range(len(data['text'])):
        for idx, ing in enumerate(data['ingredient_list']):
            curr_data['text'] = data['para'][:data['lens'][step_num]+1]
            curr_data['ing'] = ing.split('_')
            curr_data['context'] = data['para'][data['lens'][step_num]+1:data['lens'][step_num+1]+1]
            curr_data['gold'] = data['crf_labels'][idx][step_num]
            curr_data['id'] = str(count) + '_' + str(idx) + '_' + str(step_num)
            #print(curr_data)
            expanded_val_data.append(curr_data)
            curr_data = {}
json.dump(expanded_val_data, open('./test_gpt_4_tags.json','w'))
print(len(expanded_val_data))







'''
'''
expanded_val_data = []
curr_data = {}
count = 0
for data in train_data:
    #print(data)
    count+=1
    sorted_text = []
    data['targets'] = data['targets'].astype(int)
    for step_num in range(len(data['text'])):
        sorted_text.append(" ".join(data['text'][str(step_num)]))



    #for idx, ing in enumerate(data['ingredient_list']):
    idx = 0
    curr_data['text'] = sorted_text
    curr_data['ing'] =  "placeholder"
    curr_data['gold'] = data['targets'][:,idx].tolist()
    curr_data['id'] = str(count) + '_' + str(idx)
    #print(curr_data)
    expanded_val_data.append(curr_data)
    curr_data = {}
json.dump(expanded_val_data, open('./train_gpt_whole_just_recipes.json','w'))
print(len(expanded_val_data))
count = 0
'''

