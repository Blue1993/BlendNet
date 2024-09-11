import os
import sys
import yaml
import copy
import math
import pickle
import random
import logging
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import _LRScheduler

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from modules.common.utils import load_cfg
from modules.interaction_modules.loaders import BADataset, pad_data
from modules.interaction_modules.models import BlendNetT
from modules.interaction_modules.trainers import BlendNetT_trainer

def main():
    
    #############
    # Load config
    #############
    
    config_path = "PDBbind.yml"
    config = load_cfg(config_path)
    
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu")

    ############
    # Load Data
    ############

    with open(f"{config['Path']['KFold']}", "rb") as f:
        Kfold_index_dict = pickle.load(f)

    Training_df = pd.read_csv(f"{config['Path']['Training_df']}", sep = "\t")
    PDB_IDs, Uniprot_IDs, Lig_codes, BA_labels = Training_df.iloc[:, 0].values, Training_df.iloc[:, 1].values, Training_df.iloc[:, 2].values, Training_df.iloc[:, 3].values
    Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(PDB_IDs, Uniprot_IDs, Lig_codes)])

    CASF2016_data = pd.read_csv(f"{config['Path']['CASF2016_df']}", sep = "\t")
    CASF2016_PDB_IDs, CASF2016_Uniprot_IDs, CASF2016_Lig_codes, CASF2016_BA_labels = CASF2016_data.iloc[:, 0].values, CASF2016_data.iloc[:, 1].values, CASF2016_data.iloc[:, 2].values, CASF2016_data.iloc[:, 3].values
    CASF2016_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CASF2016_PDB_IDs, CASF2016_Uniprot_IDs, CASF2016_Lig_codes)])

    CASF2013_data = pd.read_csv(f"{config['Path']['CASF2013_df']}", sep = "\t")
    CASF2013_PDB_IDs, CASF2013_Uniprot_IDs, CASF2013_Lig_codes, CASF2013_BA_labels = CASF2013_data.iloc[:, 0].values, CASF2013_data.iloc[:, 1].values, CASF2013_data.iloc[:, 2].values, CASF2013_data.iloc[:, 3].values
    CASF2013_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CASF2013_PDB_IDs, CASF2013_Uniprot_IDs, CASF2013_Lig_codes)])

    CSAR2014_data = pd.read_csv(f"{config['Path']['CSAR2014_df']}", sep = "\t")
    CSAR2014_PDB_IDs, CSAR2014_Uniprot_IDs, CSAR2014_Lig_codes, CSAR2014_BA_labels = CSAR2014_data.iloc[:, 0].values, CSAR2014_data.iloc[:, 1].values, CSAR2014_data.iloc[:, 2].values, CSAR2014_data.iloc[:, 3].values
    CSAR2014_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSAR2014_PDB_IDs, CSAR2014_Uniprot_IDs, CSAR2014_Lig_codes)])

    CSAR2012_data = pd.read_csv(f"{config['Path']['CSAR2012_df']}", sep = "\t")
    CSAR2012_PDB_IDs, CSAR2012_Uniprot_IDs, CSAR2012_Lig_codes, CSAR2012_BA_labels = CSAR2012_data.iloc[:, 0].values, CSAR2012_data.iloc[:, 1].values, CSAR2012_data.iloc[:, 2].values, CSAR2012_data.iloc[:, 3].values
    CSAR2012_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSAR2012_PDB_IDs, CSAR2012_Uniprot_IDs, CSAR2012_Lig_codes)])

    CSARset1_data = pd.read_csv(f"{config['Path']['CSARset1_df']}", sep = "\t")
    CSARset1_PDB_IDs, CSARset1_Uniprot_IDs, CSARset1_Lig_codes, CSARset1_BA_labels = CSARset1_data.iloc[:, 0].values, CSARset1_data.iloc[:, 1].values, CSARset1_data.iloc[:, 2].values, CSARset1_data.iloc[:, 3].values
    CSARset1_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSARset1_PDB_IDs, CSARset1_Uniprot_IDs, CSARset1_Lig_codes)])

    CSARset2_data = pd.read_csv(f"{config['Path']['CSARset2_df']}", sep = "\t")
    CSARset2_PDB_IDs, CSARset2_Uniprot_IDs, CSARset2_Lig_codes, CSARset2_BA_labels = CSARset2_data.iloc[:, 0].values, CSARset2_data.iloc[:, 1].values, CSARset2_data.iloc[:, 2].values, CSARset2_data.iloc[:, 3].values
    CSARset2_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSARset2_PDB_IDs, CSARset2_Uniprot_IDs, CSARset2_Lig_codes)])

    Astex_data = pd.read_csv(f"{config['Path']['Astex_df']}", sep = "\t")
    Astex_PDB_IDs, Astex_Uniprot_IDs, Astex_Lig_codes, Astex_BA_labels = Astex_data.iloc[:, 0].values, Astex_data.iloc[:, 1].values, Astex_data.iloc[:, 2].values, Astex_data.iloc[:, 3].values
    Astex_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(Astex_PDB_IDs, Astex_Uniprot_IDs, Astex_Lig_codes)])

    #################
    # Define datasets
    #################
    CASF2016_DTADataset = BADataset(interaction_IDs = CASF2016_Interactions_IDs, labels = CASF2016_BA_labels,
                                    protein_feature_path = config['Path']['CASF2016_protein_feat'],
                                    pocket_path = config['Path']['CASF2016_pocket'],
                                    compound_feature_path = config['Path']['Training_ligand_graph'],
                                    interaction_sites_path = config['Path']['CASF2016_interaction'],
                                    device = device)
    CASAF2016_Loader = DataLoader(CASF2016_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    CASF2013_DTADataset = BADataset(interaction_IDs = CASF2013_Interactions_IDs, labels = CASF2013_BA_labels, 
                                    protein_feature_path = config['Path']['CASF2013_protein_feat'],
                                    pocket_path = config['Path']['CASF2013_pocket'],  
                                    compound_feature_path = config['Path']['Training_ligand_graph'],
                                    interaction_sites_path = config['Path']['CASF2013_interaction'],
                                    device = device)
    CASF2013_Loader = DataLoader(CASF2013_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    CSAR2014_DTADataset = BADataset(interaction_IDs = CSAR2014_Interactions_IDs, labels = CSAR2014_BA_labels, 
                                    protein_feature_path = config['Path']['CSAR2014_protein_feat'],
                                    pocket_path = config['Path']['CSAR2014_pocket'],  
                                    compound_feature_path = config['Path']['Training_ligand_graph'],
                                    interaction_sites_path = config['Path']['CSAR2014_interaction'],
                                    device = device)
    CSAR2014_Loader = DataLoader(CSAR2014_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    CSAR2012_DTADataset = BADataset(interaction_IDs = CSAR2012_Interactions_IDs, labels = CSAR2012_BA_labels, 
                                    protein_feature_path = config['Path']['CSAR2012_protein_feat'],
                                    pocket_path = config['Path']['CSAR2012_pocket'],  
                                    compound_feature_path = config['Path']['Training_ligand_graph'],
                                    interaction_sites_path = config['Path']['CSAR2012_interaction'],
                                    device = device)
    CSAR2012_Loader = DataLoader(CSAR2012_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    CSARset1_DTADataset = BADataset(interaction_IDs = CSARset1_Interactions_IDs, labels = CSARset1_BA_labels, 
                                    protein_feature_path = config['Path']['CSARset1_protein_feat'],
                                    pocket_path = config['Path']['CSARset1_pocket'],  
                                    compound_feature_path = config['Path']['Training_ligand_graph'],
                                    interaction_sites_path = config['Path']['CSARset1_interaction'],
                                    device = device)
    CSARset1_Loader = DataLoader(CSARset1_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    CSARset2_DTADataset = BADataset(interaction_IDs = CSARset2_Interactions_IDs, labels = CSARset2_BA_labels, 
                                    protein_feature_path = config['Path']['CSARset2_protein_feat'],
                                    pocket_path = config['Path']['CSARset2_pocket'],  
                                    compound_feature_path = config['Path']['Training_ligand_graph'],
                                    interaction_sites_path = config['Path']['CSARset2_interaction'],
                                    device = device)
    CSARset2_Loader = DataLoader(CSARset2_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    Astex_DTADataset = BADataset(interaction_IDs = Astex_Interactions_IDs, labels = Astex_BA_labels, 
                                    protein_feature_path = config['Path']['Astex_protein_feat'],
                                    pocket_path = config['Path']['Astex_pocket'],
                                    compound_feature_path = config['Path']['Training_ligand_graph'],
                                    interaction_sites_path = config['Path']['Astex_interaction'],
                                    device = device)
    Astex_Loader = DataLoader(Astex_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    #################
    # Run predictions
    #################
    for idx in range(5):
        train_index, val_index, test_index = Kfold_index_dict[idx]["train"], Kfold_index_dict[idx]["val"], Kfold_index_dict[idx]["test"]

        TotalDataset = BADataset(interaction_IDs = Interactions_IDs, labels = BA_labels, 
                                protein_feature_path = config['Path']['Training_protein_feat'],
                                pocket_path = config['Path']['Training_pocket'],
                                compound_feature_path = config['Path']['Training_ligand_graph'],
                                interaction_sites_path = config['Path']['Training_interaction'],
                                device = device)
                                
        train_PDB_IDs, train_Lig_codes = PDB_IDs[train_index], Lig_codes[train_index]
        val_PDB_IDs, val_Lig_codes = PDB_IDs[val_index], Lig_codes[val_index]
        test_PDB_IDs, test_Lig_codes = PDB_IDs[test_index], Lig_codes[test_index]
        
        trainloader = DataLoader(Subset(TotalDataset, train_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        valloader = DataLoader(Subset(TotalDataset, val_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        testloader = DataLoader(Subset(TotalDataset, test_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        
        # Define model and load weights
        Model = BlendNetT(config, device).cuda()
        checkpoint = torch.load(f"{config['Path']['save_path']}/CV{idx}/BlendNet_T.pth")
        Model.load_state_dict(checkpoint)
        
        for parameter in Model.parameters():
            parameter.requires_grad = False
        Model.eval()
        
        # Define trainer
        trainer = BlendNetT_trainer(config, Model, None, device)

        train_predictions, train_labels = trainer.ba_prediction(trainloader)
        MSE, MAE, RMSE, PCC, SPEARMAN, CI, r2 = get_results(train_labels, train_predictions)
        print(f"[CV{idx}] Train - MSE: {MSE:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}, r2: {r2:.4f}")
        fwrite(train_PDB_IDs, train_Lig_codes, train_labels, train_predictions, open(f"../results/PDBbind/complex-free/BlendNet-T/CV{idx}/Train_results.tsv", "w"))
        
        val_predictions, val_labels = trainer.ba_prediction(valloader)
        MSE, MAE, RMSE, PCC, SPEARMAN, CI, r2 = get_results(val_labels, val_predictions)
        print(f"[CV{idx}] Validation - MSE: {MSE:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}, r2: {r2:.4f}")
        fwrite(val_PDB_IDs, val_Lig_codes, val_labels, val_predictions, open(f"../results/PDBbind/complex-free/BlendNet-T/CV{idx}/Validation_results.tsv", "w"))
        
        test_predictions, test_labels = trainer.ba_prediction(testloader)
        MSE, MAE, RMSE, PCC, SPEARMAN, CI, r2 = get_results(test_labels, test_predictions)
        print(f"[CV{idx}] Test - MSE: {MSE:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}, r2: {r2:.4f}")
        fwrite(test_PDB_IDs, test_Lig_codes, test_labels, test_predictions, open(f"../results/PDBbind/complex-free/BlendNet-T/CV{idx}/Test_results.tsv", "w"))
        
        CASF2016_predictions, CASF2016_labels = trainer.ba_prediction(CASAF2016_Loader)
        MSE, MAE, RMSE, PCC, SPEARMAN, CI, r2 = get_results(CASF2016_labels, CASF2016_predictions)
        print(f"[CV{idx}] CASF2016 - MSE: {MSE:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}, r2: {r2:.4f}")
        fwrite(CASF2016_PDB_IDs, CASF2016_Lig_codes, CASF2016_labels, CASF2016_predictions, open(f"../results/PDBbind/complex-free/BlendNet-T/CV{idx}/CASF2016_results.tsv", "w"))
        
        CASF2013_predictions, CASF2013_labels = trainer.ba_prediction(CASF2013_Loader)
        MSE, MAE, RMSE, PCC, SPEARMAN, CI, r2 = get_results(CASF2013_labels, CASF2013_predictions)
        print(f"[CV{idx}] CASF2013 - MSE: {MSE:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}, r2: {r2:.4f}")
        fwrite(CASF2013_PDB_IDs, CASF2013_Lig_codes, CASF2013_labels, CASF2013_predictions, open(f"../results/PDBbind/complex-free/BlendNet-T/CV{idx}/CASF2013_results.tsv", "w"))
        
        CSAR2014_predictions, CSAR2014_labels = trainer.ba_prediction(CSAR2014_Loader)
        MSE, MAE, RMSE, PCC, SPEARMAN, CI, r2 = get_results(CSAR2014_labels, CSAR2014_predictions)
        print(f"[CV{idx}] CSAR2014 - MSE: {MSE:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}, r2: {r2:.4f}")
        fwrite(CSAR2014_PDB_IDs, CSAR2014_Lig_codes, CSAR2014_labels, CSAR2014_predictions, open(f"../results/PDBbind/complex-free/BlendNet-T/CV{idx}/CSAR2014_results.tsv", "w"))
        
        CSAR2012_predictions, CSAR2012_labels = trainer.ba_prediction(CSAR2012_Loader)
        MSE, MAE, RMSE, PCC, SPEARMAN, CI, r2 = get_results(CSAR2012_labels, CSAR2012_predictions)
        print(f"[CV{idx}] CSAR2012 - MSE: {MSE:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}, r2: {r2:.4f}")
        fwrite(CSAR2012_PDB_IDs, CSAR2012_Lig_codes, CSAR2012_labels, CSAR2012_predictions, open(f"../results/PDBbind/complex-free/BlendNet-T/CV{idx}/CSAR2012_results.tsv", "w"))
        
        CSARset1_predictions, CSARset1_labels = trainer.ba_prediction(CSARset1_Loader)
        MSE, MAE, RMSE, PCC, SPEARMAN, CI, r2 = get_results(CSARset1_labels, CSARset1_predictions)
        print(f"[CV{idx}] CSARset1 - MSE: {MSE:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}, r2: {r2:.4f}")
        fwrite(CSARset1_PDB_IDs, CSARset1_Lig_codes, CSARset1_labels, CSARset1_predictions, open(f"../results/PDBbind/complex-free/BlendNet-T/CV{idx}/CSARset1_results.tsv", "w"))
        
        CSARset2_predictions, CSARset2_labels = trainer.ba_prediction(CSARset2_Loader)
        MSE, MAE, RMSE, PCC, SPEARMAN, CI, r2 = get_results(CSARset2_labels, CSARset2_predictions)
        print(f"[CV{idx}] CSARset2 - MSE: {MSE:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}, r2: {r2:.4f}")
        fwrite(CSARset2_PDB_IDs, CSARset2_Lig_codes, CSARset2_labels, CSARset2_predictions, open(f"../results/PDBbind/complex-free/BlendNet-T/CV{idx}/CSARset2_results.tsv", "w"))

        Astex_predictions, Astex_labels = trainer.ba_prediction(Astex_Loader)
        MSE, MAE, RMSE, PCC, SPEARMAN, CI, r2 = get_results(Astex_labels, Astex_predictions)
        print(f"[CV{idx}] Astex - MSE: {MSE:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}, r2: {r2:.4f}")
        fwrite(Astex_PDB_IDs, Astex_Lig_codes, Astex_labels, Astex_predictions, open(f"../results/PDBbind/complex-free/BlendNet-T/CV{idx}/Astex_results.tsv", "w"))
        print()

def fwrite(PDB_IDs, Ligand_codes, labels, predictions, fw): 
    
    fw.write(f"PDB\tLabels\tPredictions\n")
    for pdbid, ligandcode, label, prediction in zip(PDB_IDs, Ligand_codes, labels, predictions):
        fw.write(f"{pdbid}\t{ligandcode}\t{label:.2f}\t{prediction:.2f}\n")
    
    fw.close()

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))
    
def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)
    
### Get results
def get_results(labels, predictions):
    
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    MSE = mean_squared_error(labels, predictions)
    RMSE = mean_squared_error(labels, predictions)**0.5
    MAE = mean_absolute_error(labels, predictions)
    PCC = pearsonr(labels, predictions)
    CI = concordance_index(labels, predictions)
    r2 = r2_score(labels, predictions)
    SPEARMAN = spearmanr(labels, predictions)

    return MSE, MAE, RMSE, PCC[0], SPEARMAN[0], CI, r2 
    
if __name__ == "__main__":
    main()