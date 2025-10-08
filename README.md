# Resources:

+ README.md: this file.
+ data/davis/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/davis/Y,ligands_can.txt,proteins.txt
  data/kiba/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/kiba/Y,ligands_can.txt,proteins.txt
  These file were downloaded from https://github.com/hkmztrk/DeepDTA/tree/master/data

###  Source codes:
+ create_data_unified.py: create data in pytorch format for any dataset
+ create_klifs_datasets.py: convert full sequences to KLIFS pocket sequences (85aa)
+ create_data_chembl.py: create ChEMBL/PKIS2 datasets
+ utils.py: include TestbedDataset and performance measures
+ training.py: train a GraphDTA model
+ training_validation.py: train with validation split
+ models/ginconv.py, gat.py, gat_gcn.py, gcn.py: models for KLIFS sequences (85aa)

# Step-by-step running:

## 0. Install Python libraries needed
+ Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
+ Install rdkit: conda install -y -c conda-forge rdkit
+ Or run the following commands to install both pytorch_geometric and rdkit:
```
conda create -n geometric python=3
conda activate geometric
conda install -y -c conda-forge rdkit
conda install pytorch torchvision cudatoolkit -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric

```

## 1. Create data in pytorch format
Running
```sh
conda activate geometric
python create_data_unified.py davis_klifs
python create_data_unified.py kiba_klifs
python create_data_unified.py chembl_pretraining
python create_data_unified.py pkis2_finetuning
```
This creates .pt files in data/processed/ from CSV files in data/. All datasets use KLIFS pocket sequences (85aa).

## 2. Train a prediction model
To train a model using training data. The model is chosen if it gains the best MSE for testing data.  
Running 

```sh
conda activate geometric
python training.py 0 0 0
```

where the first argument is for the index of the datasets, 0/1/2/3 for 'davis_klifs', 'kiba_klifs', 'chembl_pretraining', or 'pkis2_finetuning';
 the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet;
 and the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1'.

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.
For example, it returns two files model_GATNet_davis_klifs.model and result_GATNet_davis_klifs.csv when running GATNet on Davis KLIFS data.

## 3. Train a prediction model with validation

A model is trained on 80% of training data and chosen if it gains the best MSE for validation data (20% of training data).
Then the model is used to predict affinity for testing data.

Same arguments as in "2. Train a prediction model" are used. E.g., running

```sh
python training_validation.py 0 0 0
```

This returns the model achieving the best MSE for validation data throughout the training and performance results of the model on testing data.
For example, it returns two files model_GATNet_davis_klifs.model and result_GATNet_davis_klifs.csv when running GATNet on Davis KLIFS data.
