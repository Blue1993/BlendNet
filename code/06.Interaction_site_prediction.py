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

from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

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
    
    COACH420_data = pd.read_csv(f"{config['Path']['COACH420_df']}", sep = "\t")
    COACH420_PDB_IDs, COACH420_Uniprot_IDs, COACH420_Lig_codes = COACH420_data.iloc[:, 0].values, COACH420_data.iloc[:, 1].values, COACH420_data.iloc[:, 2].values
    COACH420_BA_labels = np.array([-1 for i in range(len(COACH420_PDB_IDs))])
    COACH420_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(COACH420_PDB_IDs, COACH420_Uniprot_IDs, COACH420_Lig_codes)])
    
    HOLO4K_data = pd.read_csv(f"{config['Path']['HOLO4K_df']}", sep = "\t")
    HOLO4K_PDB_IDs, HOLO4K_Uniprot_IDs, HOLO4K_Lig_codes = HOLO4K_data.iloc[:, 0].values, HOLO4K_data.iloc[:, 1].values, HOLO4K_data.iloc[:, 2].values
    HOLO4K_BA_labels = np.array([-1 for i in range(len(HOLO4K_PDB_IDs))])
    HOLO4K_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(HOLO4K_PDB_IDs, HOLO4K_Uniprot_IDs, HOLO4K_Lig_codes)])
    
    #################
    # Define datasets
    #################
    CASF2016_DTADataset = BADataset(interaction_IDs = CASF2016_Interactions_IDs, labels = CASF2016_BA_labels,
                                    protein_feature_path = config['Path']['CASF2016_protein_feat'],
                                    pocket_path = config['Path']['CASF2016_pocket'],
                                    compound_feature_path = config['Path']['Training_ligand_graph'],
                                    interaction_sites_path = config['Path']['CASF2016_interaction'],
                                    device = device)
    CASF2016_Loader = DataLoader(CASF2016_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

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

    COACH420_DTADataset = BADataset(interaction_IDs = COACH420_Interactions_IDs, labels = COACH420_BA_labels, 
                                    protein_feature_path = config['Path']['COACH420_protein_feat'],
                                    pocket_path = config['Path']['COACH420_pocket'],
                                    compound_feature_path = config['Path']['Training_ligand_graph'],
                                    interaction_sites_path = config['Path']['COACH420_interaction'],
                                    device = device)
    COACH420_Loader = DataLoader(COACH420_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    HOLO4K_DTADataset = BADataset(interaction_IDs = HOLO4K_Interactions_IDs, labels = HOLO4K_BA_labels, 
                                    protein_feature_path = config['Path']['HOLO4K_protein_feat'],
                                    pocket_path = config['Path']['HOLO4K_pocket'],
                                    compound_feature_path = config['Path']['Training_ligand_graph'],
                                    interaction_sites_path = config['Path']['HOLO4K_interaction'],
                                    device = device)
    HOLO4K_Loader = DataLoader(HOLO4K_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

    #############################
    # Get predictions and results
    #############################
    for idx in range(5):
        print(f"CV{idx} results")
        # Define model and load weights
        Model = BlendNetT(config, device).cuda()
        checkpoint = torch.load(f"{config['Path']['save_path']}/CV{idx}/BlendNet_T.pth")
        Model.load_state_dict(checkpoint)
        
        for parameter in Model.parameters():
            parameter.requires_grad = False
        Model.eval()
        
        # Define trainer
        trainer = BlendNetT_trainer(config, Model, None, device)
        
        # CASF2016
        CASF2016_pairwise_preds, CASF2016_pairwise_mask, CASF2016_pairwise_labels, CASF2016_lengths = trainer.pairwise_map_prediction(CASF2016_Loader)
        atom_level_results_df = get_results("Atom", CASF2016_Interactions_IDs, config["Path"]["CASF2016_interaction"], CASF2016_pairwise_preds)
        print(f"[CASF2016] atom-level AUPRC: {atom_level_results_df['AUPRC'].mean():.4f}")
        residue_level_results_df = get_results("Residue", CASF2016_Interactions_IDs, config["Path"]["CASF2016_interaction"], CASF2016_pairwise_preds)
        print(f"[CASF2016] residue-level AUPRC: {residue_level_results_df['AUPRC'].mean():.4f}")
        pair_level_results_df = get_results("Pair", CASF2016_Interactions_IDs, config["Path"]["CASF2016_interaction"], CASF2016_pairwise_preds)
        print(f"[CASF2016] pair-level AUPRC: {pair_level_results_df['AUPRC'].mean():.4f}")
        print()
        
        # CASF2013
        CASF2013_pairwise_preds, CASF2013_pairwise_mask, CASF2013_pairwise_labels, CASF2013_lengths = trainer.pairwise_map_prediction(CASF2013_Loader)
        atom_level_results_df = get_results("Atom", CASF2013_Interactions_IDs, config["Path"]["CASF2013_interaction"], CASF2013_pairwise_preds)
        print(f"[CASF2013] atom-level AUPRC: {atom_level_results_df['AUPRC'].mean():.4f}")
        residue_level_results_df = get_results("Residue", CASF2013_Interactions_IDs, config["Path"]["CASF2013_interaction"], CASF2013_pairwise_preds)
        print(f"[CASF2013] residue-level AUPRC: {residue_level_results_df['AUPRC'].mean():.4f}")
        pair_level_results_df = get_results("Pair", CASF2013_Interactions_IDs, config["Path"]["CASF2013_interaction"], CASF2013_pairwise_preds)
        print(f"[CASF2013] pair-level AUPRC: {pair_level_results_df['AUPRC'].mean():.4f}")
        print()
        
        # CSAR2014
        CSAR2014_pairwise_preds, CSAR2014_pairwise_mask, CSAR2014_pairwise_labels, CSAR2014_lengths = trainer.pairwise_map_prediction(CSAR2014_Loader)
        atom_level_results_df = get_results("Atom", CSAR2014_Interactions_IDs, config["Path"]["CSAR2014_interaction"], CSAR2014_pairwise_preds)
        print(f"[CSAR2014] atom-level AUPRC: {atom_level_results_df['AUPRC'].mean():.4f}")
        residue_level_results_df = get_results("Residue", CSAR2014_Interactions_IDs, config["Path"]["CSAR2014_interaction"], CSAR2014_pairwise_preds)
        print(f"[CSAR2014] residue-level AUPRC: {residue_level_results_df['AUPRC'].mean():.4f}")
        pair_level_results_df = get_results("Pair", CSAR2014_Interactions_IDs, config["Path"]["CSAR2014_interaction"], CSAR2014_pairwise_preds)
        print(f"[CSAR2014] pair-level AUPRC: {pair_level_results_df['AUPRC'].mean():.4f}")
        print()
        
        # CSAR2012
        CSAR2012_pairwise_preds, CSAR2012_pairwise_mask, CSAR2012_pairwise_labels, CSAR2012_lengths = trainer.pairwise_map_prediction(CSAR2012_Loader)
        atom_level_results_df = get_results("Atom", CSAR2012_Interactions_IDs, config["Path"]["CSAR2012_interaction"], CSAR2012_pairwise_preds)
        print(f"[CSAR2012] atom-level AUPRC: {atom_level_results_df['AUPRC'].mean():.4f}")
        residue_level_results_df = get_results("Residue", CSAR2012_Interactions_IDs, config["Path"]["CSAR2012_interaction"], CSAR2012_pairwise_preds)
        print(f"[CSAR2012] residue-level AUPRC: {residue_level_results_df['AUPRC'].mean():.4f}")
        pair_level_results_df = get_results("Pair", CSAR2012_Interactions_IDs, config["Path"]["CSAR2012_interaction"], CSAR2012_pairwise_preds)
        print(f"[CSAR2012] pair-level AUPRC: {pair_level_results_df['AUPRC'].mean():.4f}")
        print()
        
        # CSARset1
        CSARset1_pairwise_preds, CSARset1_pairwise_mask, CSARset1_pairwise_labels, CSARset1_lengths = trainer.pairwise_map_prediction(CSARset1_Loader)
        atom_level_results_df = get_results("Atom", CSARset1_Interactions_IDs, config["Path"]["CSARset1_interaction"], CSARset1_pairwise_preds)
        print(f"[CSARset1] atom-level AUPRC: {atom_level_results_df['AUPRC'].mean():.4f}")
        residue_level_results_df = get_results("Residue", CSARset1_Interactions_IDs, config["Path"]["CSARset1_interaction"], CSARset1_pairwise_preds)
        print(f"[CSARset1] residue-level AUPRC: {residue_level_results_df['AUPRC'].mean():.4f}")
        pair_level_results_df = get_results("Pair", CSARset1_Interactions_IDs, config["Path"]["CSARset1_interaction"], CSARset1_pairwise_preds)
        print(f"[CSARset1] pair-level AUPRC: {pair_level_results_df['AUPRC'].mean():.4f}")
        print()
        
        # CSARset2
        CSARset2_pairwise_preds, CSARset2_pairwise_mask, CSARset2_pairwise_labels, CSARset2_lengths = trainer.pairwise_map_prediction(CSARset2_Loader)
        atom_level_results_df = get_results("Atom", CSARset2_Interactions_IDs, config["Path"]["CSARset2_interaction"], CSARset2_pairwise_preds)
        print(f"[CSARset2] atom-level AUPRC: {atom_level_results_df['AUPRC'].mean():.4f}")
        residue_level_results_df = get_results("Residue", CSARset2_Interactions_IDs, config["Path"]["CSARset2_interaction"], CSARset2_pairwise_preds)
        print(f"[CSARset2] residue-level AUPRC: {residue_level_results_df['AUPRC'].mean():.4f}")
        pair_level_results_df = get_results("Pair", CSARset2_Interactions_IDs, config["Path"]["CSARset2_interaction"], CSARset2_pairwise_preds)
        print(f"[CSARset2] pair-level AUPRC: {pair_level_results_df['AUPRC'].mean():.4f}")
        print()
        
        # Astex
        Astex_pairwise_preds, Astex_pairwise_mask, Astex_pairwise_labels, Astex_lengths = trainer.pairwise_map_prediction(Astex_Loader)
        atom_level_results_df = get_results("Atom", Astex_Interactions_IDs, config["Path"]["Astex_interaction"], Astex_pairwise_preds)
        print(f"[Astex] atom-level AUPRC: {atom_level_results_df['AUPRC'].mean():.4f}")
        residue_level_results_df = get_results("Residue", Astex_Interactions_IDs, config["Path"]["Astex_interaction"], Astex_pairwise_preds)
        print(f"[Astex] residue-level AUPRC: {residue_level_results_df['AUPRC'].mean():.4f}")
        pair_level_results_df = get_results("Pair", Astex_Interactions_IDs, config["Path"]["Astex_interaction"], Astex_pairwise_preds)
        print(f"[Astex] pair-level AUPRC: {pair_level_results_df['AUPRC'].mean():.4f}")
        print()
        
        # COACH420
        COACH420_pairwise_preds, COACH420_pairwise_mask, COACH420_pairwise_labels, COACH420_lengths = trainer.pairwise_map_prediction(COACH420_Loader)
        atom_level_results_df = get_results("Atom", COACH420_Interactions_IDs, config["Path"]["COACH420_interaction"], COACH420_pairwise_preds)
        print(f"[COACH420] atom-level AUPRC: {atom_level_results_df['AUPRC'].mean():.4f}")
        residue_level_results_df = get_results("Residue", COACH420_Interactions_IDs, config["Path"]["COACH420_interaction"], COACH420_pairwise_preds)
        print(f"[COACH420] residue-level AUPRC: {residue_level_results_df['AUPRC'].mean():.4f}")
        pair_level_results_df = get_results("Pair", COACH420_Interactions_IDs, config["Path"]["COACH420_interaction"], COACH420_pairwise_preds)
        print(f"[COACH420] pair-level AUPRC: {pair_level_results_df['AUPRC'].mean():.4f}")
        print()
        
        # HOLO4K
        HOLO4K_pairwise_preds, HOLO4K_pairwise_mask, HOLO4K_pairwise_labels, HOLO4K_lengths = trainer.pairwise_map_prediction(HOLO4K_Loader)
        atom_level_results_df = get_results("Atom", HOLO4K_Interactions_IDs, config["Path"]["HOLO4K_interaction"], HOLO4K_pairwise_preds)
        print(f"[HOLO4K] atom-level AUPRC: {atom_level_results_df['AUPRC'].mean():.4f}")
        residue_level_results_df = get_results("Residue", HOLO4K_Interactions_IDs, config["Path"]["HOLO4K_interaction"], HOLO4K_pairwise_preds)
        print(f"[HOLO4K] residue-level AUPRC: {residue_level_results_df['AUPRC'].mean():.4f}")
        pair_level_results_df = get_results("Pair", HOLO4K_Interactions_IDs, config["Path"]["HOLO4K_interaction"], HOLO4K_pairwise_preds)
        print(f"[HOLO4K] pair-level AUPRC: {pair_level_results_df['AUPRC'].mean():.4f}")
        print()

def get_pair_level_metric(pairwise_pred, labels):

    num_vertex = labels.shape[0]
    num_residue = labels.shape[1]
    
    pairwise_pred = pairwise_pred[:num_vertex, :num_residue].reshape(-1)
    pairwise_label = labels.reshape(-1)

    precision, recall, thresholds = precision_recall_curve(pairwise_label, pairwise_pred)
    AUPRC = auc(recall, precision)

    return AUPRC
    
def get_residue_level_metric(pairwise_pred, labels):

    num_vertex = labels.shape[0]
    num_residue = labels.shape[1]
    
    pairwise_pred = np.max(pairwise_pred[:num_vertex, :num_residue], axis = 0)
    pairwise_label = np.clip(np.sum(labels, axis = 0), 0, 1)

    precision, recall, thresholds = precision_recall_curve(pairwise_label, pairwise_pred)
    AUPRC = auc(recall, precision)

    return AUPRC
    
def get_atom_level_metric(pairwise_pred, labels):

    num_vertex = labels.shape[0]
    num_residue = labels.shape[1]
    
    pairwise_pred = np.max(pairwise_pred[:num_vertex, :num_residue], axis = 1)
    pairwise_label = np.clip(np.sum(labels, axis = 1), 0, 1)

    try:
        AUROC = roc_auc_score(pairwise_label, pairwise_pred)
    except:
        return None

    precision, recall, thresholds = precision_recall_curve(pairwise_label, pairwise_pred)
    AUPRC = auc(recall, precision)

    return AUPRC
    
def get_results(method, interaction_IDs, interaction_site_labels_path, interaction_site_predictions):
    
    data = {"PDBID":[], "LigID":[], "AUPRC":[]}

    with open(f"{interaction_site_labels_path}", "rb") as f:
        interaction_site_labels = pickle.load(f)

    for idx, sample_key in enumerate(interaction_IDs):

        sample_key = f'{sample_key.split("_")[0]}_{sample_key.split("_")[2]}'
        sample_pred, sample_labels = interaction_site_predictions[idx], interaction_site_labels[sample_key]

        if method == "Atom":
            AUPRC = get_atom_level_metric(sample_pred, sample_labels)

        elif method == "Residue":
            AUPRC = get_residue_level_metric(sample_pred, sample_labels)

        elif method == "Pair":
            AUPRC = get_pair_level_metric(sample_pred, sample_labels)

        data["PDBID"].append(sample_key.split("_")[0])
        data["LigID"].append(sample_key.split("_")[1])
        data["AUPRC"].append(AUPRC)

    return pd.DataFrame(data)

if __name__ == "__main__":
    main()
    