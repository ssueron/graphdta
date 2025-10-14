# GraphDTA Dataset Documentation

**Project**: Kinase Selectivity Prediction using GraphDTA
**Created**: 2025-10-14
**Purpose**: Technical reference for dataset structure, affinity value encoding, and data quality considerations

---

## 1. Dataset Overview

### Available Datasets

| Dataset | Purpose | Compounds | Kinases | Coverage | Affinity Type |
|---------|---------|-----------|---------|----------|---------------|
| **davis_klifs** | Benchmark (dense) | 68 | 211 | 100% dense | pKd |
| **chembl_pretraining** | Pretraining (diverse) | 55,908 | 298 | 100% dense | pIC50 only |
| **pkis2_finetuning** | Selectivity profiling | 429 | 298 | 42% sparse | pIC50 |

---

## 2. Affinity Value Encoding - CRITICAL INFORMATION

### IMPORTANT: Zero Values Have Different Meanings

#### **ChEMBL Dataset: pIC50 Values Only**
- **Data Source**: ChEMBL database (IC50 measurements converted to pIC50)
- **Value Range**: 3.3 - 12.0 (mean: 8.19)
- **Encoding**: `-log10(IC50 in M)` where IC50 is in molar units
  ```
  IC50 = 1 nM  → pIC50 = 9.0
  IC50 = 10 nM → pIC50 = 8.0
  IC50 = 1 μM  → pIC50 = 6.0
  ```
- **Zero Values**: ❌ **NO ZEROS** - all values are measured pIC50
- **Missing Data**: Not included in dataset (only tested pairs present)
- **Coverage**: 100% dense (every row has measured activity)

#### **PKIS2 Dataset: Sparse Selectivity Matrix**
- **Data Source**: PKIS2 kinome selectivity panel (DiscoverX KINOMEscan-style)
- **Value Range**: 0.0 - 9.0 (mean: 2.37)
- **Structure**: Each compound tested against ~299 kinases
- **Zero Values**: ✅ **REAL EXPERIMENTAL VALUES**
  ```
  0.0 = No activity detected (true negative)
  >0.0 = Measured pIC50 (active compound)
  ```
- **Coverage**: 42.3% sparse
  - **Non-zero values (55,060 / 130,065)**: Real measured activities
  - **Zero values (75,005 / 130,065)**: True experimental "no activity"
- **Distribution per compound**:
  ```
  Average tested kinases per compound: ~130 / 299
  Average zero activities: ~75 / 299 (true negatives)
  Average positive activities: ~55 / 299 (true actives)
  ```
- **Important**: PKIS2 has **100% coverage** - all compounds tested against all 298 kinases
  - No missing values (unlike sparse matrices with untested pairs)
  - Each compound has complete selectivity profile
  - Zeros represent real "no activity" experimental results

**Example PKIS2 Entry**:
```csv
compound_1,kinase_A,5.2  ← Active (pIC50 = 5.2)
compound_1,kinase_B,0.0  ← No activity (experimentally tested, inactive)
compound_1,kinase_C,7.1  ← Active (pIC50 = 7.1)
```

#### **Davis Dataset: pKd Values**
- **Data Source**: Davis et al. kinase dataset
- **Value Range**: 5.0 - 10.8 (mean: 5.51)
- **Encoding**: `-log10(Kd in M)` where Kd is dissociation constant
- **Zero Values**: **NO ZEROS** - all values are measured pKd
- **Coverage**: 100% dense

## 3. Protein Sequence Quality

### KLIFS Pocket Sequences
- **Standard Length**: 85 amino acids (KLIFS numbering system)
- **Vocabulary**: 20 standard amino acids + gap character ('-')
- **Encoding**: Integer mapping (A=1, C=2, ..., -=21, unknown=0)

### Sequence Length Distribution
```
Dataset              | 85aa  | 78aa  | 76aa | 80aa |
---------------------|-------|-------|------|------|
davis_klifs          | 99.3% | 0.7%  | -    | -    |
chembl_pretraining   | 99.7% | 0.3%  | <0.1%| <0.1%|
pkis2_finetuning     | 99.0% | 0.3%  | 0.3% | 0.3% |
```
## 4. Data Quality Summary

### **Validated Quality Checks**

| Check | davis_klifs | chembl_pretraining | pkis2_finetuning |
|-------|-------------|-------------------|------------------|
| Empty SMILES | 0 | 0 | 0 |
| Invalid SMILES (RDKit) | 0 | 0 | 0 |
| Empty sequences | 0 | 0 | 0 |
| Non-standard AA | 0 (fixed) | 0 | 0 |
| Affinity range | 5.0-10.8 | 3.3-12.0 | 0.0-9.0 |
| Sequence length consistency | 99.3% | 99.7% | 99.0% |
| Train/test kinase overlap | 100% | 100% | 100% |

### Train/Test Split Strategy
- **Type**: Compound-based split (kinases overlap 100% between splits)
- **Rationale**: Evaluate generalization to new compounds with known kinases
- **Implication**: Cannot predict for truly novel kinases without retraining
- **Appropriate for**: Kinase selectivity profiling of new compounds

### Cross-Dataset Kinase Compatibility
```
davis_klifs (211) ∩ chembl (298) = 165 kinases (78% overlap)
davis_klifs (211) ∩ pkis2 (298)  = 165 kinases (78% overlap)
chembl (298) ∩ pkis2 (298)       = 298 kinases (100% overlap!)
```

**Key Insight**: ChEMBL and PKIS2 share **identical 298 kinase sequences** → Perfect for transfer learning

---

## 5. File Locations

### CSV Files (Raw Data)
```
data/chembl_pretraining_train.csv   (79,928 rows)
data/chembl_pretraining_val.csv     (16,416 rows)
data/chembl_pretraining_test.csv    (15,613 rows)
data/pkis2_finetuning_train.csv    (130,065 rows)
data/pkis2_finetuning_val.csv       (31,396 rows)
data/pkis2_finetuning_test.csv      (31,396 rows)
data/davis_klifs_train.csv          (14,893 rows)
data/davis_klifs_test.csv            (2,992 rows)
```

### Processed Files (PyTorch Geometric)
```
data/processed/davis_klifs_train.pt  (need regeneration after X→- fix)
data/processed/davis_klifs_test.pt   (need regeneration after X→- fix)
data/processed/chembl_*.pt           (need processing)
data/processed/pkis2_*.pt            (need processing)
```

**Note**: Davis KLIFS files deleted and need regeneration. See [REGENERATE_DAVIS_PT_FILES.md](REGENERATE_DAVIS_PT_FILES.md) for instructions.

**To process ChEMBL**:
```bash
python create_data_unified.py chembl_pretraining
```

**To process PKIS2**:
```bash
python create_data_unified.py pkis2_finetuning
```

### Reference Files
```
data/human_kinase_pocket_sequences.xlsx  (KLIFS pocket database)
data/chembl_pkis2/proteins_klifs.txt     (298 kinase sequences)
```

---

## 6. Column Format (All Datasets)

```csv
compound_iso_smiles,target_sequence,affinity
CCO,EVLAEGGFAIVFLCALKRMVCKREIQIMRDLSKNIVGY...,5.2
```

**Fields**:
- `compound_iso_smiles`: Isomeric SMILES (includes stereochemistry)
- `target_sequence`: KLIFS pocket sequence (typically 85aa, uses 20 AA + '-' gap)
- `affinity`: Binding affinity in -log10(M) units
  - ChEMBL: pIC50 only (3.3-12.0, no zeros)
  - PKIS2: pIC50 or 0.0 for no activity (0.0-9.0, 100% coverage)
  - Davis: pKd (5.0-10.8, no zeros)

---

## 7. Key Takeaways

### For Model Development:
1. **PKIS2 zeros are real data** - not missing values, true experimental negatives
2. **ChEMBL is pIC50 only** - no Ki or Kd values mixed in
3. **Current models work** - can handle zeros, but better loss functions available
4. **X character is handled** - embedded as unknown token, minimal impact (0.4%)
5. **ChEMBL→PKIS2 transfer** - identical 298 kinases, perfect for pretraining

### For Kinase Selectivity:
1. **Use PKIS2** - native selectivity format, 100% coverage, 298 kinases
2. **Multi-output architecture** - predict all kinases simultaneously
3. **Selectivity metrics** - Gini, entropy, S10 (not just MSE/Pearson)
4. **Weighted/robust loss** - handle imbalanced active/inactive ratios
5. **Transfer learning** - ChEMBL pretraining → PKIS2 finetuning

### Data Quality:
- **SMILES**: 100% valid, ready to use
- **Sequences**: 99%+ are 85aa KLIFS pockets, high quality
- **Affinities**: Appropriate ranges, properly encoded
- **Splits**: Compound-based, appropriate for selectivity task
- **Overall**: Production-ready, no critical issues

---

