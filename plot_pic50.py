import pandas as pd
import matplotlib.pyplot as plt

print("Loading datasets...")

davis_files = [
    'data/davis_train.csv',
    'data/davis_test.csv'
]
kiba_files = [
    'data/kiba_train.csv',
    'data/kiba_test.csv'
]
chembl_files = [
    'data/chembl_pretraining_train.csv',
    'data/chembl_pretraining_val.csv',
    'data/chembl_pretraining_test.csv'
]
pkis2_files = [
    'data/chembl_pkis2/pkis2_finetuning_train.csv',
    'data/chembl_pkis2/pkis2_finetuning_val.csv',
    'data/chembl_pkis2/pkis2_finetuning_test.csv'
]

davis_df = pd.concat([pd.read_csv(f) for f in davis_files], ignore_index=True)
kiba_df = pd.concat([pd.read_csv(f) for f in kiba_files], ignore_index=True)
chembl_df = pd.concat([pd.read_csv(f) for f in chembl_files], ignore_index=True)

# Load PKIS2 data and extract all pIC50 values (excluding 0.0 which means no data)
pkis2_dfs = [pd.read_csv(f) for f in pkis2_files]
pkis2_all_values = []
for df in pkis2_dfs:
    # Get all columns except 'smiles'
    kinase_cols = [col for col in df.columns if col != 'smiles']
    # Extract all non-zero values
    for col in kinase_cols:
        values = df[col].values
        pkis2_all_values.extend(values[values > 0])
pkis2_affinities = pd.Series(pkis2_all_values)

print(f"Davis: {len(davis_df)} samples")
print(f"KIBA: {len(kiba_df)} samples")
print(f"ChEMBL: {len(chembl_df)} samples")
print(f"PKIS2: {len(pkis2_affinities)} samples")

print("Plotting pIC50 distributions...")

fig, axes = plt.subplots(4, 2, figsize=(12, 16))

datasets = [
    ('Davis', davis_df['affinity']),
    ('KIBA', kiba_df['affinity']),
    ('ChEMBL', chembl_df['affinity']),
    ('PKIS2', pkis2_affinities)
]

for idx, (name, data) in enumerate(datasets):
    axes[idx, 0].hist(data, bins=50, edgecolor='black', alpha=0.7)
    axes[idx, 0].set_title(f'{name} - Linear Scale')
    axes[idx, 0].set_xlabel('pIC50')
    axes[idx, 0].set_ylabel('Frequency')

    axes[idx, 1].hist(data, bins=50, edgecolor='black', alpha=0.7)
    axes[idx, 1].set_title(f'{name} - Log Scale')
    axes[idx, 1].set_xlabel('pIC50')
    axes[idx, 1].set_ylabel('Frequency (log)')
    axes[idx, 1].set_yscale('log')

plt.tight_layout()
plt.savefig('pic50_distributions.png', dpi=300)
print("Saved pic50_distributions.png")
