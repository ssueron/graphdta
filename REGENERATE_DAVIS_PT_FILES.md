# Regenerate Davis KLIFS Processed Files

## Background
The CSV files have been fixed (X → - replacement completed).
The processed .pt files need to be regenerated to reflect this change.

## Status
✅ **CSV files fixed**: X characters replaced with '-' in both train and test
❌ **Processed .pt files**: Deleted, need regeneration

## To Regenerate

The processed PyTorch Geometric files need to be created with the corrected sequences.

### Option 1: Using Conda Environment
```bash
# Activate the environment (adjust name as needed)
conda activate geometric

# Regenerate processed files
cd /Users/sebastien/Documents/chembl_kinases/GraphDTA
python create_data_unified.py davis_klifs
```

### Option 2: Using Python with PyTorch Geometric Installed
```bash
cd /Users/sebastien/Documents/chembl_kinases/GraphDTA
python3 create_data_unified.py davis_klifs
```

## Expected Output
```
Processing davis_klifs dataset...
davis_klifs_train: 14893 samples
davis_klifs_test: 2992 samples
Converting SMILES to graph: 14893/14893
Graph construction done. Saving to file.
Saved to data/processed/davis_klifs_train.pt
Converting SMILES to graph: 2992/2992
Graph construction done. Saving to file.
Saved to data/processed/davis_klifs_test.pt
Done!
```

## Files That Will Be Created
- `data/processed/davis_klifs_train.pt` (~169 MB)
- `data/processed/davis_klifs_test.pt` (~34 MB)

## Verification
After regeneration, verify the fix worked:
```bash
# Check that no X characters exist in CSV
grep -c "X" data/davis_klifs_train.csv  # Should output: 0
grep -c "X" data/davis_klifs_test.csv   # Should output: 0

# Verify processed files exist
ls -lh data/processed/davis_klifs*.pt
```

## What Changed
- **Before**: RIOK1 sequence had `...MLYIIXVSQS` (X encoded as 0 = unknown token)
- **After**: RIOK1 sequence has `...MLYII-VSQS` (- encoded as 21 = gap token)
- **Impact**: More consistent with ChEMBL/PKIS2 encoding, better semantic meaning

## Date
Fixed: 2025-10-14
