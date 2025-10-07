# Davis and KIBA Datasets with KLIFS Pocket Sequences

## Summary

Successfully created new versions of Davis and KIBA datasets using 85 AA KLIFS pocket sequences instead of full protein sequences. This enables training with the same architecture used for ChEMBL/PKIS2 datasets.

## Created Files

### CSV Files (human-readable):
- `data/davis_train_klifs.csv` - 14,893 samples
- `data/davis_test_klifs.csv` - 2,991 samples
- `data/kiba_train_klifs.csv` - 97,684 samples
- `data/kiba_test_klifs.csv` - 19,522 samples

### PyTorch Processed Files:
- `data/processed/davis_train_klifs.pt` - 169 MB
- `data/processed/davis_test_klifs.pt` - 34 MB
- `data/processed/kiba_train_klifs.pt` - 952 MB
- `data/processed/kiba_test_klifs.pt` - 190 MB

## Matching Statistics

### Davis Dataset:
- **Matched**: ~60% of samples (14,893 train + 2,991 test)
- **Unmatched**: ~40% of samples (167 unique proteins couldn't be matched)
- **Reason**: Many Davis proteins are mutant variants (e.g., ABL1(E255K), EGFR(T790M)) or non-kinases not present in KLIFS database

### KIBA Dataset:
- **Matched**: ~99% of samples (97,684 train + 19,522 test)
- **Unmatched**: Only 10 proteins (mostly atypical or non-kinases)
- **High match rate** due to UniProt ID matching

## Usage

### Training with KLIFS Sequences

To train models on Davis/KIBA with KLIFS pocket sequences, add these options to `training.py`:

```python
# Add to dataset_options list in training.py (line 51):
dataset_options = ['davis', 'kiba', 'chembl_pretraining', 'pkis2_finetuning',
                   'davis_klifs', 'kiba_klifs']
```

Or create a new training script:

```bash
# Train GINConvNet on davis_klifs using same architecture as chembl/pkis2
python training.py 4 0 0  # 4=davis_klifs, 0=GINConvNet, 0=cuda:0
```

### Transfer Learning Example

Since davis_klifs and chembl use the same 85 AA sequences, transfer learning is now possible:

```bash
# 1. Pretrain on chembl
python training.py 2 0 0

# 2. Transfer to davis_klifs
python training_transfer.py 0 0  # Modify to use davis_klifs instead of pkis2
```

## Architecture Compatibility

All datasets now use:
- **Protein sequence length**: 85 amino acids
- **Vocabulary**: 27 characters (A-Z + gap '-')
- **Model**: `GINConvNet` from `models/ginconv.py` (NOT `ginconv_old.py`)
- **Embedding size**: 128
- **Conv1d input channels**: 128 (after transpose)

This enables:
- Transfer learning between chembl ↔ davis ↔ kiba ↔ pkis2
- Fair comparison of performance across datasets
- Unified model architecture

## Scripts

### `create_klifs_datasets.py`
Parses Excel file and matches Davis/KIBA proteins with KLIFS sequences by:
- Davis: Gene name matching (handles mutants by stripping suffixes)
- KIBA: UniProt ID matching

### `create_data_klifs.py`
Converts CSV files to PyTorch `.pt` format for training

## Unmatched Proteins

### Davis (167 proteins):
ACVR1, ACVR1B, ACVR2A, AMPK-alpha1, AURKA, AURKB, CAMK1, CAMK2A, CDC2L5, CDK2, etc.
- Many are mutant variants
- Some are non-KLIFS kinases

### KIBA (10 proteins):
O00418, O43741, P54619, P67870, P78527, Q15078, Q15118, Q9UGI9, Q9UGJ0, Q9Y478
- Mostly atypical or non-kinase proteins

## Next Steps

1. **Update training.py** to include davis_klifs/kiba_klifs dataset options
2. **Train baseline models** on these datasets for comparison with full sequences
3. **Test transfer learning** from chembl → davis_klifs or kiba_klifs
4. **Compare performance**:
   - Full sequence vs KLIFS pocket sequence
   - Same architecture across all datasets
