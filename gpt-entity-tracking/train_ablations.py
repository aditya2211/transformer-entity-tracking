import argparse
import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from analysis import rocstories as rocstories_analysis
from datasets import rocstories
from model_pytorch import DoubleHeadModelModified, load_openai_pretrained_model
#from ..pytorch_pretrained_bert.modeling_openai import OpenAIGPTDoubleLMHeadModel, OpenAIGPTConfig

from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, iter_data,
                   ResultLogger, make_path)
from loss import ClassificationLossCompute, ClassificationLossComputeLM

def transform_recipe(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    xmb2 = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb2 = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        #x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [delimiter] + x3[:max_len]+ [clf_token]
        x12 = [start] + x1 + [delimiter]  + x2 +  [clf_token]
        #x12 = [start]  + x2 + [delimiter] + x3+ [clf_token]
        #x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        if l12 == 0:
            print('O length train para\n')
            continue
        if l12 > 512:
            continue
        #l13 = len(x13)
        xmb[i, :l12, 0] = x12
        #xmb[i, 1, :l13, 0] = x13
        mmb[i, :l12] = 1


        xing = [start] + x3 + [extra_token]
        ling = len(xing)

        xmb2[i,:ling,0] = xing
        mmb2[i,:ling] = 1 
        #mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    xmb2[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb, xmb2, mmb2

def transform_recipe_additional(X1, X2, X3):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 3), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        #x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [delimiter] + x3[:max_len]+ [clf_token]
        x12 = [start] + x1 + [delimiter] + x2 + [delimiter] + x3+ [clf_token]
        #x12 = [start]  + x2 + [delimiter] + x3+ [clf_token]
        #x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        if l12 == 0:
            print('O length train para\n')
            continue
        if l12 > 512:
            continue
        #l13 = len(x13)
        xmb[i, :l12, 0] = x12
        #xmb[i, 1, :l13, 0] = x13
        mmb[i, :l12] = 1
        #mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    xmb[:,: len(x1)+2,2] = encoder['_extra1_']
    xmb[:, len(x1)+2: len(x1)+2 + len(x2)+1,2] = encoder['_extra2_']
    xmb[:, len(x1)+2 + len(x2)+1:len(x1)+2 + len(x2)+1 + len(x3)+1,2] = encoder['_extra3_']
    return xmb, mmb

def transform_recipe3(X1, X2, X3, X1_helper, X2_helper):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 4), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3, x4, x5), in enumerate(zip(X1, X2, X3, X1_helper, X2_helper)):
        #x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [delimiter] + x3[:max_len]+ [clf_token]
        x12 = [start] + x1 + [delimiter] + x2 + [delimiter] + x3+ [clf_token]
        x14 = [ing_not_present_token] + x4 + [ing_not_present_token] + x5 + [ing_not_present_token] + [ing_present_token]*len(x3) + [ing_not_present_token]

        assert len(x1) == len(x4)
        assert len(x2) == len(x5)
        #x12 = [start]  + x2 + [delimiter] + x3+ [clf_token]
        #x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        if l12 == 0:
            print('O length train para\n')
            continue
        if l12 > 512:
            continue
        #l13 = len(x13)
        xmb[i, :l12, 0] = x12
        xmb[i, :l12, 3] = x14
        #xmb[i, 1, :l13, 0] = x13
        mmb[i, :l12] = 1
        #mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    xmb[:,: len(x1)+2,2] = encoder['_extra1_']
    xmb[:, len(x1)+2: len(x1)+2 + len(x2)+1,2] = encoder['_extra2_']
    xmb[:, len(x1)+2 + len(x2)+1:len(x1)+2 + len(x2)+1 + len(x3)+1,2] = encoder['_extra3_']
    return xmb, mmb
def transform_recipe_stories(X1):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, x1 in enumerate(X1):
        #x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [delimiter] + x3[:max_len]+ [clf_token]
        x12 = [start] + x1 + [clf_token]
        #x12 = [start]  + x2 + [delimiter] + x3+ [clf_token]
        #x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        if l12 == 0:
            print('O length train para\n')
            continue
        if l12 > 512:
            continue
        #l13 = len(x13)
        xmb[i, :l12, 0] = x12
        #xmb[i, 1, :l13, 0] = x13
        mmb[i, :l12-1] = 1
        #mmb[i, 1, :l13] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


def iter_apply(Xs, Ms, Ys, X2s, M2s):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb, xmb2, mmb2 in iter_data(Xs, Ms, Ys, X2s, M2s, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            XMB2 = torch.tensor(xmb2, dtype=torch.long).to(device)
            MMB2 = torch.tensor(mmb2).to(device)

    
            _, clf_logits = dh_model(XMB, XMB2)
            #print("+"*80)
            #print(clf_logits)
            #print("="*80)
            clf_logits *= n
            #print(clf_logits)
            #print("+"*80)
            
            
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost


def iter_predict(Xs, Ms):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb in iter_data(Xs, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)
    return logits


def log(save_dir, desc):
    global best_score
    print("Logging")
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid], trX2[:n_valid], trM2[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY, vaX2, vaM2)
    tr_cost = tr_cost / len(trY[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))
            np.save('./va_logits_nopre_lesslr.npy', va_logits)

def predict(dataset, submission_dir):
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))


def run_epoch():
    for xmb, mmb, xmb2, mmb2, ymb in iter_data(*shuffle(trX, trM, trX2, trM2,trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        XMB2 = torch.tensor(xmb2, dtype=torch.long).to(device)
        MMB2 = torch.tensor(mmb2).to(device).sum(dim=1)
        #print(xmb2[:,:10,:])
        lm_logits, clf_logits = dh_model(XMB, XMB2)
        compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        
        n_updates += 1
        if n_updates in [ 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)
        
def run_epoch_lm():

    for xmb, mmb in iter_data(*shuffle(trlmX, trlmM, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):

        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, _ = dh_model(XMB)
        compute_loss_fct(XMB, MMB, lm_logits)
        
        n_updates += 1
        '''
        if n_updates in [ 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)
        '''
argmax = lambda x: np.argmax(x, 1)

pred_fns = {
    'rocstories': argmax,
}

filenames = {
    'rocstories': 'ROCStories.tsv',
}

label_decoders = {
    'rocstories': None,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--n_batch', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-6)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    submit = args.submit
    dataset = args.dataset
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir
    log_dir = args.log_dir
    submission_dir = args.submission_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)


    '''
    print("Encoding dataset...")
    ((trX1, trX2, trX3, trY),
     (vaX1, vaX2, vaX3, vaY),
     (teX1, teX2, teX3)) = encode_dataset(*rocstories(data_dir, n_valid=args.n_valid),
                                          encoder=text_encoder)


    '''


    #LM-pretraining

    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    encoder['_extra1_'] = len(encoder)
    encoder['_extra2_'] = len(encoder)
    encoder['_extra3_'] = len(encoder)
    encoder['_ing_present_'] = len(encoder)
    encoder['_ing_not_present_'] = len(encoder)

    clf_token = encoder['_classify_']
    extra_token = encoder['_extra1_']
    ing_present_token = encoder['_ing_present_']
    ing_not_present_token = encoder['_ing_not_present_']
    n_special = 8

    '''
    train_file = json.load(open('../train_recipes.json','r'))
    val_file = json.load(open('../val_recipes.json','r'))


    t_passage = []
    v_passage = []
    for ins in train_file:
        t_passage.append(ins['story'])
    for ins in val_file:
        v_passage.append(ins['story'])

    a = (t_passage),(v_passage)
    ((trX1),(vaX1)) = encode_dataset(*a,encoder = text_encoder)

    '''
    train_lm_file = json.load(open('../train_recipes.json','r'))
    train_file = json.load(open('./test_gpt_with_ing_indicator.json','r'))
    val_file = json.load(open('./val_gpt_with_ing_indicator.json','r'))

    print(train_file[99])
    t_passage = []
    t_context = []
    t_ing = []
    t_gold = []
    tlm_passage = []
    tlm_context = []
    tlm_ing = []
    tlm_gold = []
    v_passage = []
    v_context = []
    v_ing = []
    v_gold = []

    def convert_ing_indicator(ind_sequence):
        new_seq = []
        for ind in ind_sequence:
            if ind==1:
                new_seq.append(encoder['_ing_present_'])
                #print('check')
            else:
                new_seq.append(encoder['_ing_not_present_'])
                #print('check2')

        return new_seq
    #shuffle(train_file)
    trX1helper = []
    trX2helper = []
    for ins in train_lm_file[:10]:
        tlm_passage.append(ins['story'])
        #tlm_context.append(" ".join(ins['context']))
        #tlm_ing.append(" ".join(ins['ing']))
        #tlm_gold.append(int(ins['gold']))

    gold_labels = {'0':0, '1':0,'2':0,'3':0}
    for ins in train_file:
        t_passage.append(" ".join(ins['text']))
        t_context.append(" ".join(ins['context']))
        t_ing.append(" ".join(ins['ing']))
        t_gold.append(int(ins['gold']))
        gold_labels[str(int(ins['gold']))]+=1

        trX1helper.append(convert_ing_indicator(ins['text_ing_ind']))
        trX2helper.append(convert_ing_indicator(ins['context_ing_ind']))
        #print(ins['gold'])
    print(gold_labels)

    vaX1helper = []
    vaX2helper = []
    gold_labels = {'0':0, '1':0,'2':0,'3':0}
    for ins in val_file:
        v_passage.append(" ".join(ins['text']))
        v_context.append(" ".join(ins['context']))
        v_ing.append(" ".join(ins['ing']))
        v_gold.append(int(ins['gold']))
        gold_labels[str(int(ins['gold']))]+=1
        vaX1helper.append(convert_ing_indicator(ins['text_ing_ind']))
        vaX2helper.append(convert_ing_indicator(ins['context_ing_ind']))
    print(gold_labels)
    print(tlm_passage[0])
    a = (tlm_passage,), (t_passage,t_context,t_ing,t_gold),(v_passage,v_context, v_ing,v_gold)

    ((trlmX1,),(trX1, trX2, trX3, trY),(vaX1, vaX2, vaX3, vaY)) = encode_dataset(*a,encoder = text_encoder)


    print(trlmX1[0])

    '''
    max_len = n_ctx // 3 - 3


    n_ctx = min(max(
        [len(x1[:max_len]) + len(x2[:max_len]) + len(x3[:max_len]) for x1, x2, x3 in zip(trX1, trX2, trX3)]
        + [len(x1[:max_len]) + len(x2[:max_len]) + len(x3[:max_len]) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]
        ) + 4, n_ctx)
    '''

    n_ctx = min(max([len(x1) + len(x2) + len(x3) for x1, x2, x3 in zip(trX1, trX2, trX3)]+[len(x1) + len(x2) + len(x3) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]+ [len(x1) for x1 in trlmX1])+4,n_ctx)
    #n_ctx = min(max([len(x2) + len(x3) for x1, x2, x3 in zip(trX1, trX2, trX3)]+[len(x2) + len(x3) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)])+3,n_ctx)
    
    print(n_ctx)
    vocab = n_vocab + n_special + n_ctx
    trX, trM, trX2, trM2 = transform_recipe(trX1, trX2, trX3)
    vaX, vaM, vaX2, vaM2 = transform_recipe(vaX1, vaX2, vaX3)

    print(vaX3[:4])
    print(vaX2[:4,:10:,0])
    trlmX, trlmM = transform_recipe_stories(trlmX1)
    n_train_lm = len(trlmX)
    n_train = len(trY)
    n_valid = len(vaY)

    print(len(trlmX))
    print(trlmM[0])

    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train_lm // n_batch_train) * args.n_iter

    dh_model = DoubleHeadModelModified(args, clf_token, extra_token, 'custom', vocab, n_ctx)

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    compute_loss_fct = ClassificationLossComputeLM(criterion,
                                                 args.lm_coef,
                                                 model_opt)
    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)

    dh_model.to(device)
    dh_model = nn.DataParallel(dh_model)

    '''
    trlmX = trX
    trlmM = trM
    '''

    n_updates = 0
    n_epochs = 0
    if dataset != 'stsb':
        trYt = trY
    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        torch.save(dh_model.state_dict(), make_path(path))
    best_score = 0
    '''
    for i in range(args.n_iter):

        print("running epoch lm only", i)
        run_epoch_lm()
        n_epochs += 1
        #log(save_dir, desc)
    '''
    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter
    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    compute_loss_fct = ClassificationLossCompute(criterion,
                                                 criterion,
                                                 args.lm_coef,
                                                 model_opt)

    for i in range(args.n_iter):

        print("running epoch", i)
        run_epoch()
        n_epochs += 1
        log(save_dir, desc)

    if submit:
        path = os.path.join(save_dir, desc, 'best_params')
        dh_model.load_state_dict(torch.load(path))
        '''
        predict(dataset, args.submission_dir)
        if args.analysis:
            rocstories_analysis(data_dir, os.path.join(args.submission_dir, 'ROCStories.tsv'),
                                os.path.join(log_dir, 'rocstories.jsonl'))
        '''