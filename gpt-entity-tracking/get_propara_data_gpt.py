import json
import numpy as np
import json

train_data = json.load(open('../../propara_train_full.json','r'))
val_data_t2 = json.load(open('../../propara_val_locs_till_now.json','r'))
val_data = json.load(open('../../propara_val_sent_4.json','r'))
test_data = json.load(open('../../propara_test_4.json','r'))
test_data_sent = json.load(open('../../propara_test_sent_4.json','r'))



print(len(train_data[0]['para']))
print(train_data[0])

tag2ix = {"N": 0, "N_d":1, "E":2, "C":3, "D":4, "M":5}
#tag2ix = {"N": 0, "N_d":0, "E":1, "C":2, "D":3, "M":4}
'''
expanded_train_data = []
curr_data = {}
for ins in train_data:
    ins['entity'] = ins['entity'].replace(";"," ")
    for step in range(len(ins['indices'])-1):
        curr_data['context'] = ins['para'][ins['indices'][step]+1:ins['indices'][step+1]+1]
        curr_data['text'] = ins['para'][:ins['indices'][step]+1]
        curr_data['gold'] = tag2ix[ins['tags'][step]]
        curr_data['ing'] = ins['entity']
        expanded_train_data.append(curr_data)
        curr_data = {}
json.dump(expanded_train_data, open('./train_propara_gpt.json','w'))
print(len(expanded_train_data))
print(expanded_train_data[:10])
print("\n")
expanded_val_data = []
curr_data = {}

print(val_data[0])

for elem in val_data:
    elem['num_sents'] = len(elem['indices'])-1
    elem['tags'] = ['E' for idx in range(elem['num_sents'])]

    if elem['destroy_step']!=0:
        elem['tags'][elem['destroy_step']-1]='D'
        for j in range(elem['destroy_step'],elem['num_sents']):
            elem['tags'][j] = 'N_d'
    if elem['create_step']!=0:
        for j in range(elem['create_step']-1):
            elem['tags'][j] = 'N' 
        elem['tags'][elem['create_step']-1] = 'C'

    for idx in elem['move_steps']:
        elem['tags'][idx-1]="M"
for ins in val_data:
    ins['entity'] = ins['entity'].replace(";"," ")
    for step in range(len(ins['indices'])-1):
        curr_data['context'] = ins['para'][ins['indices'][step]+1:ins['indices'][step+1]+1]
        curr_data['text'] = ins['para'][:ins['indices'][step]+1]
        curr_data['gold'] = tag2ix[ins['tags'][step]]
        curr_data['ing'] = ins['entity']
        curr_data['id'] = str(ins['pid']) + '_' + str(step)
        expanded_val_data.append(curr_data)
        curr_data = {}
json.dump(expanded_val_data, open('./val_propara_gpt.json','w'))
print(len(expanded_val_data))
print(expanded_val_data[:10])
'''


expanded_train_data = []
curr_data = {}
pid_ents = {}
for ins in train_data:
    ins['entity'] = ins['entity'].replace(";"," ")
    curr_data['ing'] = ins['entity']
    curr_data['context'] = []
    curr_data['gold'] = []
    curr_data['pid'] = ins['pid']
    if curr_data['pid'] not in pid_ents:
        pid_ents[curr_data['pid']] = [ins['entity'].replace(";"," ")]
    else:
        pid_ents[curr_data['pid']].append(ins['entity'].replace(";"," "))
    for step in range(len(ins['indices'])-1):
        curr_data['context'].append(" ".join(ins['para'][ins['indices'][step]+1:ins['indices'][step+1]+1]))
        curr_data['gold'].append(tag2ix[ins['tags'][step]])
    expanded_train_data.append(curr_data)
    curr_data = {}

for ins in expanded_train_data:
    ins['all_ents'] = pid_ents[ins['pid']]
json.dump(expanded_train_data, open('./train_propara_gpt_whole.json','w'))
print(len(expanded_train_data))
print(expanded_train_data[:10])
print("\n")
expanded_val_data = []
curr_data = {}

print(val_data[0])

for elem in val_data:
    elem['num_sents'] = len(elem['indices'])-1
    elem['tags'] = ['E' for idx in range(elem['num_sents'])]

    if elem['destroy_step']!=0:
        elem['tags'][elem['destroy_step']-1]='D'
        for j in range(elem['destroy_step'],elem['num_sents']):
            elem['tags'][j] = 'N_d'
    if elem['create_step']!=0:
        for j in range(elem['create_step']-1):
            elem['tags'][j] = 'N' 
        elem['tags'][elem['create_step']-1] = 'C'

    for idx in elem['move_steps']:
        elem['tags'][idx-1]="M"

for elem in val_data_t2:
    elem['num_sents'] = len(elem['indices'])-1
    elem['tags'] = ['E' for idx in range(elem['num_sents'])]

    if elem['destroy_step']!=0:
        elem['tags'][elem['destroy_step']-1]='D'
        for j in range(elem['destroy_step'],elem['num_sents']):
            elem['tags'][j] = 'N_d'
    if elem['create_step']!=0:
        for j in range(elem['create_step']-1):
            elem['tags'][j] = 'N' 
        elem['tags'][elem['create_step']-1] = 'C'

    for idx in elem['move_steps']:
        elem['tags'][idx-1]="M"
pid_ents = {}
for ins in val_data:
    ins['entity'] = ins['entity'].replace(";"," ")
    curr_data['ing'] = ins['entity']
    curr_data['context'] = []
    curr_data['gold'] = []
    curr_data['pid'] = ins['pid']
    if curr_data['pid'] not in pid_ents:
        pid_ents[curr_data['pid']] = [ins['entity'].replace(";"," ")]
    else:
        pid_ents[curr_data['pid']].append(ins['entity'].replace(";"," "))
    for step in range(len(ins['indices'])-1):
        curr_data['context'].append(" ".join(ins['para'][ins['indices'][step]+1:ins['indices'][step+1]+1]))
        curr_data['gold'].append(tag2ix[ins['tags'][step]])
    expanded_val_data.append(curr_data)
    curr_data = {}
for ins in expanded_val_data:
    ins['all_ents'] = pid_ents[ins['pid']]
json.dump(expanded_val_data, open('./val_propara_gpt_whole.json','w'))
print(len(expanded_val_data))
print(expanded_val_data[:10])
print('\n\n')
#print(val_data[:10])



pid_ents = {}
expanded_val_data = []
curr_data = {}
for ins in val_data_t2:
    ins['entity'] = ins['entity'].replace(";"," ")
    curr_data['ing'] = ins['entity']
    curr_data['context'] = []
    curr_data['gold'] = []
    curr_data['pid'] = ins['pid']
    if curr_data['pid'] not in pid_ents:
        pid_ents[curr_data['pid']] = [ins['entity'].replace(";"," ")]
    else:
        pid_ents[curr_data['pid']].append(ins['entity'].replace(";"," "))
    for step in range(len(ins['indices'])-1):
        curr_data['context'].append(" ".join(ins['para'][ins['indices'][step]+1:ins['indices'][step+1]+1]))
        curr_data['gold'].append(tag2ix[ins['tags'][step]])
    expanded_val_data.append(curr_data)
    curr_data = {}

for ins in expanded_val_data:
    ins['all_ents'] = pid_ents[ins['pid']]
json.dump(expanded_val_data, open('./val_propara_gpt_whole_t2.json','w'))
print(len(expanded_val_data))
print(expanded_val_data[:10])
print('\n\n')
#print(val_data[:10])
pid_ents = {}
expanded_val_data = []
curr_data = {}
for ins in test_data_sent:
    ins['entity'] = ins['entity'].replace(";"," ")
    curr_data['ing'] = ins['entity']
    curr_data['context'] = []
    curr_data['gold'] = []
    curr_data['pid'] = ins['pid']
    if curr_data['pid'] not in pid_ents:
        pid_ents[curr_data['pid']] = [ins['entity'].replace(";"," ")]
    else:
        pid_ents[curr_data['pid']].append(ins['entity'].replace(";"," "))
    for step in range(len(ins['indices'])-1):
        curr_data['context'].append(" ".join(ins['para'][ins['indices'][step]+1:ins['indices'][step+1]+1]))
        #curr_data['gold'].append(tag2ix[ins['tags'][step]])
    expanded_val_data.append(curr_data)
    curr_data = {}
for ins in expanded_val_data:
    ins['all_ents'] = pid_ents[ins['pid']]
json.dump(expanded_val_data, open('./test_propara_gpt_whole_t2.json','w'))
print(len(expanded_val_data))
print(expanded_val_data[:10])
print('\n\n')
#print(val_data[:10])



pid_ents = {}
expanded_val_data = []
curr_data = {}
for ins in test_data:
    ins['entity'] = ins['entity'].replace(";"," ")
    curr_data['ing'] = ins['entity']
    curr_data['context'] = []
    curr_data['gold'] = []
    curr_data['pid'] = ins['pid']
    if curr_data['pid'] not in pid_ents:
        pid_ents[curr_data['pid']] = [ins['entity'].replace(";"," ")]
    else:
        pid_ents[curr_data['pid']].append(ins['entity'].replace(";"," "))
    for step in range(len(ins['indices'])-1):
        curr_data['context'].append(" ".join(ins['para'][ins['indices'][step]+1:ins['indices'][step+1]+1]))
        #curr_data['gold'].append(tag2ix[ins['tags'][step]])
    expanded_val_data.append(curr_data)
    curr_data = {}
for ins in expanded_val_data:
    ins['all_ents'] = pid_ents[ins['pid']]
json.dump(expanded_val_data, open('./test_propara_gpt_whole.json','w'))
print(len(expanded_val_data))
print(expanded_val_data[:10])
print('\n\n')
#print(val_data[:10])