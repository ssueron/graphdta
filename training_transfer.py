import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *

def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[1])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>2:
    cuda_name = "cuda:" + str(int(sys.argv[2]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Transfer Learning: chembl_pretraining -> pkis2_finetuning')
print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

pretrain_dataset = 'chembl_pretraining'
finetune_dataset = 'pkis2_finetuning'

device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = modeling().to(device)

pretrain_model_file = 'model_' + model_st + '_' + pretrain_dataset + '.model'
if os.path.isfile(pretrain_model_file):
    print('\nLoading pretrained drug-encoding layers from', pretrain_model_file)
    pretrained_dict = torch.load(pretrain_model_file, map_location=device)
    model_dict = model.state_dict()
    drug_layers = {k: v for k, v in pretrained_dict.items() if k.startswith(('conv', 'bn', 'fc1_xd'))}
    model_dict.update(drug_layers)
    model.load_state_dict(model_dict)
    print(f'Loaded {len(drug_layers)} drug-encoding layers')
else:
    print('\nWarning: Pretrained model not found:', pretrain_model_file)
    print('Training from scratch on', finetune_dataset)

print('\nFinetuning on', finetune_dataset)
train_data = TestbedDataset(root='data', dataset=finetune_dataset+'_train')
test_data = TestbedDataset(root='data', dataset=finetune_dataset+'_test')

train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_mse = 1000
best_ci = 0
best_epoch = -1
model_file_name = 'model_' + model_st + '_transfer.model'
result_file_name = 'result_' + model_st + '_transfer.csv'

for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch+1)
    G,P = predicting(model, device, test_loader)
    ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
    if ret[1]<best_mse:
        torch.save(model.state_dict(), model_file_name)
        with open(result_file_name,'w') as f:
            f.write(','.join(map(str,ret)))
        best_epoch = epoch+1
        best_mse = ret[1]
        best_ci = ret[-1]
        print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,'transfer')
    else:
        print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,'transfer')
