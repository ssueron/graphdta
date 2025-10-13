import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE

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

davis_df = pd.concat([pd.read_csv(f) for f in davis_files], ignore_index=True)
kiba_df = pd.concat([pd.read_csv(f) for f in kiba_files], ignore_index=True)
chembl_df = pd.concat([pd.read_csv(f) for f in chembl_files], ignore_index=True)

print(f"Davis: {len(davis_df)} samples")
print(f"KIBA: {len(kiba_df)} samples")
print(f"ChEMBL: {len(chembl_df)} samples")

print("Plotting pIC50 distributions...")

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

for idx, (name, df) in enumerate([('Davis', davis_df), ('KIBA', kiba_df), ('ChEMBL', chembl_df)]):
    axes[idx, 0].hist(df['affinity'], bins=50, edgecolor='black', alpha=0.7)
    axes[idx, 0].set_title(f'{name} - Linear Scale')
    axes[idx, 0].set_xlabel('pIC50')
    axes[idx, 0].set_ylabel('Frequency')

    axes[idx, 1].hist(df['affinity'], bins=50, edgecolor='black', alpha=0.7)
    axes[idx, 1].set_title(f'{name} - Log Scale')
    axes[idx, 1].set_xlabel('pIC50')
    axes[idx, 1].set_ylabel('Frequency (log)')
    axes[idx, 1].set_yscale('log')

plt.tight_layout()
plt.savefig('pic50_distributions.png', dpi=300)
print("Saved pic50_distributions.png")

print("Computing ECFP6 fingerprints...")

def get_ecfp6(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)

davis_smiles = davis_df['compound_iso_smiles'].unique()
kiba_smiles = kiba_df['compound_iso_smiles'].unique()
chembl_smiles = chembl_df['compound_iso_smiles'].unique()

davis_fps = [(s, get_ecfp6(s)) for s in davis_smiles]
kiba_fps = [(s, get_ecfp6(s)) for s in kiba_smiles]
chembl_fps = [(s, get_ecfp6(s)) for s in chembl_smiles]

davis_fps = [(s, fp) for s, fp in davis_fps if fp is not None]
kiba_fps = [(s, fp) for s, fp in kiba_fps if fp is not None]
chembl_fps = [(s, fp) for s, fp in chembl_fps if fp is not None]

all_fps = (
    [(fp, 'Davis') for _, fp in davis_fps] +
    [(fp, 'KIBA') for _, fp in kiba_fps] +
    [(fp, 'ChEMBL') for _, fp in chembl_fps]
)

X = np.array([list(fp) for fp, _ in all_fps])
labels = [label for _, label in all_fps]

print(f"Running t-SNE on {len(X)} compounds...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

print("Plotting t-SNE...")

fig, ax = plt.subplots(figsize=(12, 10))

colors = {'Davis': 'blue', 'KIBA': 'red', 'ChEMBL': 'green'}
for dataset_name, color in colors.items():
    mask = np.array(labels) == dataset_name
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=dataset_name, alpha=0.6, s=10)

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('Chemical Space - t-SNE (ECFP6)')
ax.legend()
plt.tight_layout()
plt.savefig('tsne_chemical_space.png', dpi=300)
print("Saved tsne_chemical_space.png")
