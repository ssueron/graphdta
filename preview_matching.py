#!/usr/bin/env python3
"""Preview which proteins will be matched before full extraction."""

import json
import csv

# Load target proteins
with open('data/chembl_pkis2/proteins_klifs.txt', 'r') as f:
    proteins_dict = json.load(f)
target_names = set(proteins_dict.keys())

# Load KLIFS data
with open('klifs_pocket_sequences_human_wt_from_uniprot.csv', 'r') as f:
    reader = csv.DictReader(f)
    klifs_data = {}
    for row in reader:
        name = row['kinase_name']
        if name not in klifs_data:
            klifs_data[name] = {
                'uniprot_id': row['uniprot_id'],
                'klifs_id': row['kinase_klifs_id'],
                'count': 0
            }
        klifs_data[name]['count'] += 1

klifs_names = set(klifs_data.keys())

# Name mappings
NAME_MAPPINGS = {
    'AURKA': 'AurA', 'AURKB': 'AurB', 'AURKC': 'AurC',
    'p38-alpha': 'p38a', 'p38-beta': 'p38b', 'p38-gamma': 'p38g', 'p38-delta': 'p38d',
    'AMPK-alpha1': 'AMPKa1',
    'ERK1': 'Erk1', 'ERK2': 'Erk2',
    'CHEK1': 'CHK1', 'CHEK2': 'CHK2',
    'CSNK1A1': 'CK1a', 'CSNK1D': 'CK1d', 'CSNK1E': 'CK1e',
    'CSNK2A1': 'CK2a1', 'CSNK2A2': 'CK2a2',
    'IKK-alpha': 'IKKa', 'IKK-beta': 'IKKb', 'IKK-epsilon': 'IKKe',
    'MEK1': 'MAP2K1', 'MEK2': 'MAP2K2', 'MKK7': 'MAP2K7',
    'PDPK1': 'PDK1',
    'PKAC-alpha': 'PKA',
    'SGK': 'SGK1',
}

# Create mapping
mapping = {}

# Predefined mappings
for target, klifs in NAME_MAPPINGS.items():
    if target in target_names and klifs in klifs_names:
        mapping[target] = klifs

# Exact matches
for name in target_names:
    if name in klifs_names and name not in mapping:
        mapping[name] = name

# Case-insensitive
klifs_lower = {k.lower(): k for k in klifs_names}
for target in target_names:
    if target not in mapping and target.lower() in klifs_lower:
        mapping[target] = klifs_lower[target.lower()]

# Results
matched = set(mapping.keys())
unmatched = target_names - matched

print("="*80)
print("PROTEIN MATCHING PREVIEW")
print("="*80)
print(f"\nTarget proteins:       {len(target_names)}")
print(f"KLIFS proteins:        {len(klifs_names)}")
print(f"MATCHED proteins:      {len(matched)}")
print(f"UNMATCHED proteins:    {len(unmatched)}")

# Count total structures
total_structures = sum(klifs_data[mapping[t]]['count'] for t in matched)
print(f"Total structures:      {total_structures}")

print("\n" + "-"*80)
print("MATCHED PROTEINS (with structure counts):")
print("-"*80)
for target in sorted(matched):
    klifs = mapping[target]
    count = klifs_data[klifs]['count']
    uniprot = klifs_data[klifs]['uniprot_id']
    if target != klifs:
        print(f"  {target:20s} -> {klifs:15s} ({uniprot:8s}) [{count:3d} structures]")
    else:
        print(f"  {target:20s} ({uniprot:8s}) [{count:3d} structures]")

print("\n" + "-"*80)
print("UNMATCHED PROTEINS (not in KLIFS):")
print("-"*80)
for name in sorted(unmatched):
    print(f"  - {name}")

print("\n" + "="*80)
print(f"Ready to extract {len(matched)} proteins with ~{total_structures} structures")
print("="*80)
