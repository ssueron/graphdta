import pandas as pd
import numpy as np
import os
import sys
from rdkit import Chem
import networkx as nx
from utils import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)
    return x

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ-"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 85

if len(sys.argv) < 2:
    print("Usage: python create_data_unified.py <dataset_name>")
    print("Example: python create_data_unified.py davis_klifs")
    print("         python create_data_unified.py chembl_pretraining")
    sys.exit(1)

dataset_name = sys.argv[1]

splits = ['train', 'test'] if dataset_name not in ['chembl_pretraining', 'pkis2_finetuning'] else ['train', 'val', 'test']

print(f'Processing {dataset_name} dataset...')

for split in splits:
    csv_file = f'data/{dataset_name}_{split}.csv'

    if not os.path.isfile(csv_file):
        print(f'Error: {csv_file} not found')
        continue

    df = pd.read_csv(csv_file)
    print(f'{dataset_name}_{split}: {len(df)} samples')

    compound_iso_smiles = list(df['compound_iso_smiles'])
    target_sequences = list(df['target_sequence'])
    affinities = list(df['affinity'])

    smile_graph = {}
    for smile in set(compound_iso_smiles):
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    target_seqs_encoded = [seq_cat(seq) for seq in target_sequences]

    data = TestbedDataset(root='data', dataset=f'{dataset_name}_{split}',
                        xd=compound_iso_smiles, xt=target_seqs_encoded,
                        y=affinities, smile_graph=smile_graph)

    print(f'Saved to data/processed/{dataset_name}_{split}.pt')

print('Done!')
