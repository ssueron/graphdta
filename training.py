import numpy as np
import pandas as pd
import sys, os
import argparse
import time
import re
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.pna import PNANet
from models.pna_deep import PNANet_Deep
from models.protein_cnn_simple import SimpleProteinCNN
from models.protein_cnn import DeepProteinCNN
from models.protein_cnn_blosum import DeepProteinCNN_BLOSUM
from utils import *
from utils_experiment import ExperimentManager
from utils_degree import get_or_compute_degree

def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return total_loss / len(train_loader)

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


parser = argparse.ArgumentParser(description='Train GraphDTA model')
parser.add_argument('dataset', type=int, help='Dataset index: 0=davis_klifs, 1=kiba_klifs, 2=chembl_pretraining, 3=pkis2_finetuning')
parser.add_argument('model', type=int, help='Model index: 0=GINConvNet, 1=GATNet, 2=GAT_GCN, 3=GCNNet, 4=PNANet, 5=PNANet_Deep')
parser.add_argument('protein_model', type=int, help='Protein encoder index: 0=SimpleProteinCNN, 1=DeepProteinCNN, 2=DeepProteinCNN_BLOSUM')
parser.add_argument('cuda', type=int, default=0, help='CUDA device index')
parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
parser.add_argument('--exp-name', type=str, default=None, help='Custom experiment name')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--save-freq', type=int, default=10, help='Save checkpoint every N epochs')
parser.add_argument('--num-workers', type=int, default=None, help='DataLoader workers for training (auto if omitted)')
parser.add_argument('--eval-num-workers', type=int, default=None, help='DataLoader workers for evaluation (auto if omitted)')

args = parser.parse_args()

dataset_options = ['davis_klifs', 'kiba_klifs', 'chembl_pretraining', 'pkis2_finetuning']
datasets = [dataset_options[args.dataset]]

modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet, PNANet, PNANet_Deep][args.model]
model_st = modeling.__name__

protein_model_classes = [SimpleProteinCNN, DeepProteinCNN, DeepProteinCNN_BLOSUM]
protein_model_factories = [
    lambda **kwargs: SimpleProteinCNN(**kwargs),
    lambda **kwargs: DeepProteinCNN(**kwargs),
    lambda **kwargs: DeepProteinCNN_BLOSUM(**kwargs),
]
protein_model_st = protein_model_classes[args.protein_model].__name__

cuda_name = f"cuda:{args.cuda}"
print('cuda_name:', cuda_name)
print('protein_encoder:', protein_model_st)

TRAIN_BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = args.batch_size
LR = args.lr
LOG_INTERVAL = 20
NUM_EPOCHS = args.epochs

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

def _parse_int_token(raw_value):
    if raw_value is None:
        return None
    match = re.search(r'\d+', str(raw_value))
    return int(match.group()) if match else None

def detect_available_cpus():
    env_candidates = [
        os.environ.get('SLURM_CPUS_PER_TASK'),
        os.environ.get('SLURM_JOB_CPUS_PER_NODE'),
        os.environ.get('SLURM_CPUS_ON_NODE')
    ]
    for raw in env_candidates:
        parsed = _parse_int_token(raw)
        if parsed:
            return parsed
    return os.cpu_count() or 1

def select_num_workers(requested, gpus):
    if requested is not None:
        return max(0, requested)
    cores = detect_available_cpus()
    gpus = max(1, gpus)
    baseline = max(2, cores // (2 * gpus))
    headroom = cores - 1 if cores > 1 else 0
    workers = min(baseline, 12, headroom)
    return max(0, workers)

visible_gpus = torch.cuda.device_count()
train_num_workers = select_num_workers(args.num_workers, visible_gpus)
if args.eval_num_workers is not None:
    eval_num_workers = max(0, args.eval_num_workers)
else:
    eval_num_workers = min(train_num_workers, 4) if train_num_workers > 0 else 0

pin_memory = torch.cuda.is_available()
print(f'DataLoader workers -> train: {train_num_workers}, eval: {eval_num_workers}, pin_memory: {pin_memory}')

for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print(f'please run: python create_data_unified.py {dataset}')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')

        if hasattr(train_data.data, 'target'):
            target_tensor = train_data.data.target
        else:
            target_tensor = torch.cat([sample.target for sample in train_data], dim=0)

        protein_seq_len = target_tensor.size(-1)
        protein_vocab_size = int(target_tensor.max().item()) if target_tensor.numel() > 0 else 0
        protein_kwargs = {
            'num_features_xt': protein_vocab_size,
            'seq_len': protein_seq_len
        }

        hyperparams = {
            'lr': LR,
            'batch_size': TRAIN_BATCH_SIZE,
            'epochs': NUM_EPOCHS,
            'model': model_st,
            'dataset': dataset,
            'protein_model': protein_model_st,
            'protein_model_index': args.protein_model,
            'protein_vocab_size': protein_vocab_size,
            'protein_seq_len': protein_seq_len
        }
        exp_manager = ExperimentManager(model_st, dataset, hyperparams, args.exp_name)
        print(f'Experiment directory: {exp_manager.exp_dir}')

        train_loader_kwargs = {
            'batch_size': TRAIN_BATCH_SIZE,
            'shuffle': True,
            'num_workers': train_num_workers,
            'pin_memory': pin_memory
        }
        test_loader_kwargs = {
            'batch_size': TEST_BATCH_SIZE,
            'shuffle': False,
            'num_workers': eval_num_workers,
            'pin_memory': pin_memory
        }
        if train_num_workers > 0:
            train_loader_kwargs['persistent_workers'] = True
        if eval_num_workers > 0:
            test_loader_kwargs['persistent_workers'] = True

        train_loader = DataLoader(train_data, **train_loader_kwargs)
        test_loader = DataLoader(test_data, **test_loader_kwargs)

        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

        if model_st in ['PNANet', 'PNANet_Deep']:
            deg = get_or_compute_degree(dataset)
            if model_st == 'PNANet':
                print(f'Info: {model_st} uses its built-in protein branch; selection {protein_model_st} is ignored.')
                model = modeling(deg=deg).to(device)
            else:
                protein_encoder = protein_model_factories[args.protein_model](**protein_kwargs)
                model = modeling(deg=deg, protein_encoder=protein_encoder).to(device)
        else:
            protein_encoder = protein_model_factories[args.protein_model](**protein_kwargs)
            model = modeling(protein_encoder=protein_encoder).to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        start_epoch = 0
        best_mse = 1000
        best_ci = 0
        best_epoch = -1

        if args.resume:
            checkpoint = exp_manager.load_checkpoint('latest')
            if checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_mse = checkpoint['metrics'][1]
                best_ci = checkpoint['metrics'][4]
                print(f'Resumed from epoch {start_epoch}, best_mse: {best_mse:.4f}, best_ci: {best_ci:.4f}')

        start_time = time.time()

        for epoch in range(start_epoch, NUM_EPOCHS):
            train_loss = train(model, device, train_loader, optimizer, epoch+1, loss_fn, LOG_INTERVAL)
            G, P = predicting(model, device, test_loader)
            ret = [rmse(G,P), mse(G,P), pearson(G,P), spearman(G,P), ci(G,P)]

            exp_manager.log_epoch(epoch+1, train_loss, ret)

            is_best = ret[1] < best_mse
            if is_best:
                best_epoch = epoch+1
                best_mse = ret[1]
                best_ci = ret[-1]
                print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
            else:
                print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)

            if (epoch+1) % args.save_freq == 0 or is_best:
                exp_manager.save_checkpoint(model, optimizer, epoch, ret, is_best=is_best)

        duration_hours = (time.time() - start_time) / 3600
        exp_manager.save_final_results(P, G, ret)
        exp_manager.update_summary(ret, best_epoch, NUM_EPOCHS, duration_hours)

        print(f'\nTraining completed! Results saved to {exp_manager.exp_dir}')
