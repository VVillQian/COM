import torch
import pickle
import random
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim 
from losses import SupConLoss
from torch.utils.data import DataLoader
from model import MLP, Model, ModelConfig, ReverseLayerF
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from utils import Bank, IncompleteMultiViewDataset, read_data, prepare_data, get_mask

#from thop import profile


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def initialize(config):
    traindata, testdata = read_data(config['path'],config['splitrate'],config['norm'])
    trainmask, testmask = get_mask(len(traindata[0]),len(traindata[1]),config['training']['missing_rate']), get_mask(len(testdata[0]),len(testdata[1]),config['training']['missing_rate'])
    trainset,  testset  = IncompleteMultiViewDataset(traindata,trainmask), IncompleteMultiViewDataset(testdata,testmask)
    trainloader = DataLoader(trainset,batch_size=config['training']['batch_size'],shuffle=True)
    testloader  = DataLoader(testset, batch_size=config['training']['batch_size'])
    bank        = Bank(traindata[0],trainmask)
    modelconfig = ModelConfig(config)
    model       = Model(modelconfig).cuda()
    optimizer   = optim.Adam(model.parameters(),lr=config['training']['lr'])
    return model, optimizer, trainloader, testloader, bank


def train(config, model, optimizer, train_loader, bank, device='cuda'):
    model_path = './models/'+config['name']+'/'+str(round(config['training']['missing_rate'],1))+'tmp'+'.pt'#+str(config['seed'])
    best_acc   = 0
    view_num   = config['view_num']
    criterion  = nn.CrossEntropyLoss()
    adloss_fn  = nn.BCEWithLogitsLoss()
    supcon_fn  = SupConLoss()

    ###################################
    training_loss = {'cls':[], 'adv':[], 'con1':[], 'con2':[]}
    ###################################

    discriminators = [MLP([model.config.dims[i]]+[1024,1024,512,1],batchnorm=True).to(device=device) for i in range(view_num)]#
    for epoch in range(1, config['training']['epoch'] + 1):
        ################################
        CLSLoss  = 0
        ADVLoss  = 0
        CON1Loss = 0
        CON2Loss = 0
        ################################
        for data, labels, mask in train_loader:
            bsz = labels.shape[0]
            data = list(map(lambda x:x.to(dtype=torch.float,device=device), data))
            labels, mask = (labels-1).to(device = device), mask.to(device=device)
            x,fake,x_m,features, before_fusion, all_attentions = model(data,mask)##################################
            
            '''#
            print(sum([p.numel() for p in model.parameters()]))
            flops, params = profile(model, inputs=(data,mask))
            print('FLOPs: {}'.format(flops))
            print('Model Size: {}'.format(params))
            assert False
            #'''

            clsloss = criterion(x, labels)
            real    = bank.sample(bsz)
            real    = [ReverseLayerF.apply(e.to(dtype=torch.float,device=device),1.0) for e in real]
            mixture = [torch.cat([fake[i], real[i]],dim=0) for i in range(view_num)]
            preds   = [discriminators[i](mixture[i]).squeeze(1) for i in range(view_num)]
            target  = torch.cat([torch.zeros(bsz),torch.ones(bsz)]).to(dtype=torch.float,device=device)
            adloss  = sum([adloss_fn(preds[i],target) for i in range(view_num)])/view_num
            conloss1 = sum([supcon_fn(x_m[i],labels) for i in range(view_num)])/view_num
            conloss2 = supcon_fn(features)
            loss = clsloss + config['training']['lambda1']*adloss + config['training']['lambda2']*conloss1 + config['training']['lambda3']*conloss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ###################################################
            CLSLoss += clsloss.item()*bsz
            ADVLoss += adloss.item()*bsz
            CON1Loss += conloss1.item()*bsz
            CON2Loss += conloss2.item()*bsz
            

        training_loss['cls'].append(CLSLoss/len(train_loader))
        training_loss['adv'].append(ADVLoss/len(train_loader))
        training_loss['con1'].append(CON1Loss/len(train_loader))
        training_loss['con2'].append(CON2Loss/len(train_loader))
        ################################################

        scores, _, _, _, _ = test(model, train_loader, device)###
        #training_loss.update(scores)
        if scores['acc'] > best_acc:
            best_acc = scores['acc']#animal,KS 
            torch.save(model, model_path)

        if epoch % 10 == 0:
            print('{}, Epoch: {:>4}, Loss: {}'.format(datetime.datetime.now(), epoch, loss))

    return torch.load(model_path), training_loss############################


def test(model, loader, device = 'cuda'):
    
    pred, true, scores = [],[],{}

    #############
    all_labels, all_features, all_attentions, all_masks = [], [], [], []
    #############
    
    for data, labels, mask in loader:
        data = list(map(lambda x:x.to(dtype=torch.float,device=device), data))
        labels, mask = (labels-1).to(device=device), mask.to(device=device)
        x,_,_,_,features, attentions = model(data, mask)
        y_pred = torch.argmax(x,dim=1)
        pred.append(y_pred)
        true.append(labels)

        #############
        all_labels.append(labels)
        all_features.append(features)
        all_attentions.append(attentions[-1]) # the last layer attention value
        all_masks.append(mask)


    all_labels     = torch.cat(all_labels,     dim=0).cpu().detach()
    all_features   = torch.cat(all_features,   dim=0).cpu().detach()
    all_attentions = torch.cat(all_attentions, dim=0).cpu().detach()
    all_masks      = torch.cat(all_masks,      dim=0).cpu().detach()
        #############
    
    pred = torch.cat(pred).cpu()
    true = torch.cat(true).cpu()
    
    scores['acc']     = accuracy_score(true, pred)
    scores['macro_P'] = precision_score(true, pred, average='macro')
    scores['micro_P'] = precision_score(true, pred, average='micro')
    scores['macro_R'] = recall_score(true, pred, average='macro')
    scores['micro_R'] = recall_score(true, pred, average='micro')
    scores['macro_F'] = f1_score(true, pred,average='macro')
    scores['micro_F'] = f1_score(true, pred,average='micro')

    return scores, all_labels, all_features, all_attentions, all_masks#######


def train_and_eval(config, seed = 0):
    setup_seed(seed)
    config['seed'] = seed
    model, optimizer, trainloader, testloader, bank = initialize(config)
    model, _ = train(config, model, optimizer, trainloader, bank)   ##########
    scores, _, _, _, _ = test(model, testloader)                       ##########
    return scores

