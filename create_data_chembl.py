import pandas as pd
import numpy as np
import os
import json
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
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
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def seq_cat(prot):
    """
    Encode protein sequence to integer array.
    For KLIFS sequences (85 amino acids with gaps):
    - A-Z amino acids -> 1-26
    - '-' (gap/unknown) -> 27
    - Padding (0) for sequences < 85
    """
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict.get(ch, 0)  # Use 0 if character not in dict
    return x


# Vocabulary: A-Z amino acids + gap character
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ-"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}  # A=1, ..., Z=26, -=27
seq_dict_len = len(seq_dict)
max_seq_len = 85  # KLIFS active site sequences

print(f"Sequence vocabulary size: {seq_dict_len} (A-Z + gap)")
print(f"Max sequence length: {max_seq_len}")
print(f"Gap character '-' maps to: {seq_dict['-']}")

# Load KLIFS protein sequences
proteins_file = 'data/chembl_pkis2/proteins_klifs.txt'
print(f"\nLoading protein sequences from {proteins_file}...")
with open(proteins_file, 'r') as f:
    proteins_klifs = json.load(f)

print(f"Loaded {len(proteins_klifs)} protein sequences")

# Check for sequences with gaps
seqs_with_gaps = {k: v for k, v in proteins_klifs.items() if '-' in v}
print(f"Sequences with gaps: {len(seqs_with_gaps)}/{len(proteins_klifs)}")

# Process both chembl_pretraining and pkis2_finetuning datasets
datasets = [
    ('chembl_pretraining', 'chembl_pretraining'),
    ('pkis2_finetuning', 'pkis2_finetuning')
]

for dataset_name, file_prefix in datasets:
    print(f'\n{"="*60}')
    print(f'Processing dataset: {dataset_name}')
    print(f'{"="*60}')

    data_path = 'data/chembl_pkis2/'

    # Process train, val, test splits
    splits = ['train', 'val', 'test']

    # Collect all unique SMILES across splits for graph generation
    all_smiles = set()

    for split in splits:
        csv_file = f'{data_path}{file_prefix}_{split}.csv'
        print(f'\nReading {csv_file}...')

        df = pd.read_csv(csv_file)
        print(f'  Shape: {df.shape}')
        print(f'  Columns: smiles + {df.shape[1]-1} kinases')

        # Get SMILES column
        smiles_col = df.iloc[:, 0]  # First column is SMILES
        all_smiles.update(smiles_col)

        # Get kinase columns (all except first)
        kinase_columns = df.columns[1:]

        # Convert matrix format to pairwise format
        pairs = []
        for idx, row in df.iterrows():
            smiles = row.iloc[0]

            # Check each kinase
            for kinase_name in kinase_columns:
                affinity = row[kinase_name]

                # Skip if missing value (NaN means not tested)
                # Keep zeros - they represent no biological activity
                if pd.isna(affinity):
                    continue

                # Get protein sequence
                if kinase_name not in proteins_klifs:
                    print(f'Warning: Kinase {kinase_name} not found in proteins_klifs.txt')
                    continue

                protein_seq = proteins_klifs[kinase_name]

                pairs.append({
                    'compound_iso_smiles': smiles,
                    'target_sequence': protein_seq,
                    'affinity': affinity
                })

        # Create pairwise CSV
        output_csv = f'data/{dataset_name}_{split}.csv'
        pairs_df = pd.DataFrame(pairs)
        pairs_df.to_csv(output_csv, index=False)

        print(f'  Created {output_csv}')
        print(f'  Pairwise interactions: {len(pairs)}')
        print(f'  Unique compounds: {pairs_df["compound_iso_smiles"].nunique()}')
        print(f'  Unique proteins: {pairs_df["target_sequence"].nunique()}')

    print(f'\nTotal unique SMILES across all splits: {len(all_smiles)}')

    # Generate graphs for all unique SMILES
    print(f'\nGenerating molecular graphs for {len(all_smiles)} compounds...')
    smile_graph = {}
    failed_smiles = []

    for i, smile in enumerate(all_smiles):
        if (i + 1) % 100 == 0:
            print(f'  Progress: {i+1}/{len(all_smiles)}')

        try:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
        except Exception as e:
            print(f'  Warning: Failed to process SMILES: {smile[:50]}... Error: {e}')
            failed_smiles.append(smile)

    print(f'Successfully generated {len(smile_graph)} graphs')
    if failed_smiles:
        print(f'Failed to process {len(failed_smiles)} SMILES')

    # Convert to PyTorch data format
    print(f'\nConverting to PyTorch format...')

    for split in splits:
        processed_data_file = f'data/processed/{dataset_name}_{split}.pt'

        # Read the pairwise CSV we just created
        csv_file = f'data/{dataset_name}_{split}.csv'
        df = pd.read_csv(csv_file)

        drugs = list(df['compound_iso_smiles'])
        prots = list(df['target_sequence'])
        Y = list(df['affinity'])

        # Encode protein sequences
        print(f'  Encoding protein sequences for {split}...')
        XT = [seq_cat(t) for t in prots]

        drugs = np.asarray(drugs)
        XT = np.asarray(XT)
        Y = np.asarray(Y)

        # Create PyTorch Geometric dataset
        print(f'  Creating {processed_data_file}...')
        data = TestbedDataset(
            root='data',
            dataset=f'{dataset_name}_{split}',
            xd=drugs,
            xt=XT,
            y=Y,
            smile_graph=smile_graph
        )

        print(f'  Saved {processed_data_file}')

    print(f'\nDataset {dataset_name} processing complete!')

print(f'\n{"="*60}')
print('All datasets processed successfully!')
print(f'{"="*60}')
