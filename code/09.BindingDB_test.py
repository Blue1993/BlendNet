import os
import sys
import yaml
import pickle
import random
import logging
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error

from modules.common.utils import load_cfg
from modules.interaction_modules.BDB_loaders import BADataset, pad_data
from modules.interaction_modules.BDB_models import BlendNetS
from modules.interaction_modules.BDB_trainers import BlendNetS_trainer

def main():

    ##############
    # Load config
    ##############
    
    config_path = "BindingDB.yml"
    config = load_cfg(config_path)

    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu")
    
    ###############
    # Load Ki Data
    ###############
    Ki_data = pd.read_csv(f"{config['Path']['Ki_df']}", sep = "\t")
    UniProt_IDs, PubChem_CIDs, labels = Ki_data.iloc[:, 0].values, Ki_data.iloc[:, 1].values, np.round(Ki_data.iloc[:, 2].values, 4)
    Interactions_IDs = np.array([f"{u}_{c}" for u, c in zip(UniProt_IDs, PubChem_CIDs)])
    
    Ki_Dataset = BADataset(interaction_IDs = Interactions_IDs, labels = labels,
                                    protein_feature_path = config['Path']['Ki_protein_feat'],
                                    pocket_path = config['Path']['Ki_pockets'],
                                    compound_feature_path = config['Path']['Ligand_graph'],
                                    device = device)

    ##################################
    # Load Ki random split CV settings
    ##################################
    with open(f"{config['Path']['Ki_random_split']}", "rb") as f:
        Ki_Kfold_index_dict = pickle.load(f)
    
    for idx in range(3):
        print(f"[Ki] random split CV {idx} predictions...")
        train_index, val_index, test_index = Ki_Kfold_index_dict[idx]["train"], Ki_Kfold_index_dict[idx]["val"], Ki_Kfold_index_dict[idx]["test"]
        
        # Define loaders
        Train_loader = DataLoader(Subset(Ki_Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Validation_loader = DataLoader(Subset(Ki_Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Test_loader = DataLoader(Subset(Ki_Dataset, test_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        # Define model and load weights
        Model = BlendNetS(config["Path"]["Ki_interaction_site_predictor"], config, device).cuda()

        checkpoint = torch.load(f"{config['Path']['Ki_save_path']}/random_split/CV{idx}/BlendNet_S.pth")
        Model.load_state_dict(checkpoint)
        
        for parameter in Model.parameters():
            parameter.requires_grad = False
        Model.eval()
        
        # Define trainer
        trainer = BlendNetS_trainer(config, Model, [None, None], device)

        # Run predictions
        train_predictions, train_labels = trainer.test(Train_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(train_labels, train_predictions)
        print(f"[Train] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[train_index], train_predictions, train_labels, open(f"{config['Path']['Ki_results']}/random_split/CV{idx}/Train_results.txt", "w"))
        
        validation_predictions, validation_labels = trainer.test(Validation_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(validation_labels, validation_predictions)
        print(f"[Validation] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[val_index], validation_predictions, validation_labels, open(f"{config['Path']['Ki_results']}/random_split/CV{idx}/Validation_results.txt", "w"))

        test_predictions, test_labels = trainer.test(Test_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(test_labels, test_predictions)
        print(f"[Test] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[test_index], test_predictions, test_labels, open(f"{config['Path']['Ki_results']}/random_split/CV{idx}/Test_results.txt", "w"))
        print()

    ##################################
    # Load Ki new protein CV settings
    ##################################
    with open(f"{config['Path']['Ki_new_protein']}", "rb") as f:
        Ki_Kfold_index_dict = pickle.load(f)
    
    for idx in range(3):
        print(f"[Ki] new protein CV {idx} predictions...")
        train_index, val_index, test_index = Ki_Kfold_index_dict[idx]["train"], Ki_Kfold_index_dict[idx]["val"], Ki_Kfold_index_dict[idx]["test"]
        
        # Define loaders
        Train_loader = DataLoader(Subset(Ki_Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Validation_loader = DataLoader(Subset(Ki_Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Test_loader = DataLoader(Subset(Ki_Dataset, test_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        # Define model and load weights
        Model = BlendNetS(config["Path"]["Ki_interaction_site_predictor"], config, device).cuda()

        checkpoint = torch.load(f"{config['Path']['Ki_save_path']}/new_protein/CV{idx}/BlendNet_S.pth")
        Model.load_state_dict(checkpoint)
        
        for parameter in Model.parameters():
            parameter.requires_grad = False
        Model.eval()
        
        # Define trainer
        trainer = BlendNetS_trainer(config, Model, [None, None], device)

        # Run predictions
        train_predictions, train_labels = trainer.test(Train_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(train_labels, train_predictions)
        print(f"[Train] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[train_index], train_predictions, train_labels, open(f"{config['Path']['Ki_results']}/new_protein/CV{idx}/Train_results.txt", "w"))
        
        validation_predictions, validation_labels = trainer.test(Validation_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(validation_labels, validation_predictions)
        print(f"[Validation] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[val_index], validation_predictions, validation_labels, open(f"{config['Path']['Ki_results']}/new_protein/CV{idx}/Validation_results.txt", "w"))

        test_predictions, test_labels = trainer.test(Test_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(test_labels, test_predictions)
        print(f"[Test] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[test_index], test_predictions, test_labels, open(f"{config['Path']['Ki_results']}/new_protein/CV{idx}/Test_results.txt", "w"))
        print()
        
    ##################################
    # Load Ki new compound CV settings
    ##################################
    with open(f"{config['Path']['Ki_new_compound']}", "rb") as f:
        Ki_Kfold_index_dict = pickle.load(f)
    
    for idx in range(3):
        print(f"[Ki] new compound CV {idx} predictions...")
        train_index, val_index, test_index = Ki_Kfold_index_dict[idx]["train"], Ki_Kfold_index_dict[idx]["val"], Ki_Kfold_index_dict[idx]["test"]
        
        # Define loaders
        Train_loader = DataLoader(Subset(Ki_Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Validation_loader = DataLoader(Subset(Ki_Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Test_loader = DataLoader(Subset(Ki_Dataset, test_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        # Define model and load weights
        Model = BlendNetS(config["Path"]["Ki_interaction_site_predictor"], config, device).cuda()

        checkpoint = torch.load(f"{config['Path']['Ki_save_path']}/new_compound/CV{idx}/BlendNet_S.pth")
        Model.load_state_dict(checkpoint)
        
        for parameter in Model.parameters():
            parameter.requires_grad = False
        Model.eval()
        
        # Define trainer
        trainer = BlendNetS_trainer(config, Model, [None, None], device)

        # Run predictions
        train_predictions, train_labels = trainer.test(Train_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(train_labels, train_predictions)
        print(f"[Train] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[train_index], train_predictions, train_labels, open(f"{config['Path']['Ki_results']}/new_compound/CV{idx}/Train_results.txt", "w"))
        
        validation_predictions, validation_labels = trainer.test(Validation_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(validation_labels, validation_predictions)
        print(f"[Validation] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[val_index], validation_predictions, validation_labels, open(f"{config['Path']['Ki_results']}/new_compound/CV{idx}/Validation_results.txt", "w"))

        test_predictions, test_labels = trainer.test(Test_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(test_labels, test_predictions)
        print(f"[Test] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[test_index], test_predictions, test_labels, open(f"{config['Path']['Ki_results']}/new_compound/CV{idx}/Test_results.txt", "w"))
        print()

    ##################################
    # Load Ki blind split CV settings
    ##################################
    with open(f"{config['Path']['Ki_blind_split']}", "rb") as f:
        Ki_Kfold_index_dict = pickle.load(f)
    
    for idx in range(3):
        print(f"[Ki] blind split CV {idx} predictions...")
        train_index, val_index, test_index = Ki_Kfold_index_dict[idx]["train"], Ki_Kfold_index_dict[idx]["val"], Ki_Kfold_index_dict[idx]["test"]
        
        # Define loaders
        Train_loader = DataLoader(Subset(Ki_Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Validation_loader = DataLoader(Subset(Ki_Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Test_loader = DataLoader(Subset(Ki_Dataset, test_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        # Define model and load weights
        Model = BlendNetS(config["Path"]["Ki_interaction_site_predictor"], config, device).cuda()

        checkpoint = torch.load(f"{config['Path']['Ki_save_path']}/blind_split/CV{idx}/BlendNet_S.pth")
        Model.load_state_dict(checkpoint)
        
        for parameter in Model.parameters():
            parameter.requires_grad = False
        Model.eval()
        
        # Define trainer
        trainer = BlendNetS_trainer(config, Model, [None, None], device)

        # Run predictions
        train_predictions, train_labels = trainer.test(Train_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(train_labels, train_predictions)
        print(f"[Train] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[train_index], train_predictions, train_labels, open(f"{config['Path']['Ki_results']}/blind_split/CV{idx}/Train_results.txt", "w"))
        
        validation_predictions, validation_labels = trainer.test(Validation_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(validation_labels, validation_predictions)
        print(f"[Validation] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[val_index], validation_predictions, validation_labels, open(f"{config['Path']['Ki_results']}/blind_split/CV{idx}/Validation_results.txt", "w"))

        test_predictions, test_labels = trainer.test(Test_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(test_labels, test_predictions)
        print(f"[Test] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[test_index], test_predictions, test_labels, open(f"{config['Path']['Ki_results']}/blind_split/CV{idx}/Test_results.txt", "w"))
        print()

    ###############
    # Load IC50 Data
    ###############
    IC50_data = pd.read_csv(f"{config['Path']['IC50_df']}", sep = "\t")
    UniProt_IDs, PubChem_CIDs, labels = IC50_data.iloc[:, 0].values, IC50_data.iloc[:, 1].values, np.round(IC50_data.iloc[:, 2].values, 4)
    Interactions_IDs = np.array([f"{u}_{c}" for u, c in zip(UniProt_IDs, PubChem_CIDs)])
    
    IC50_Dataset = BADataset(interaction_IDs = Interactions_IDs, labels = labels,
                                    protein_feature_path = config['Path']['IC50_protein_feat'],
                                    pocket_path = config['Path']['IC50_pockets'],
                                    compound_feature_path = config['Path']['Ligand_graph'],
                                    device = device)

    ####################################
    # Load IC50 random split CV settings
    ####################################
    with open(f"{config['Path']['IC50_random_split']}", "rb") as f:
        IC50_Kfold_index_dict = pickle.load(f)
    
    for idx in range(3):
        print(f"[IC50] random split CV {idx} predictions...")
        train_index, val_index, test_index = IC50_Kfold_index_dict[idx]["train"], IC50_Kfold_index_dict[idx]["val"], IC50_Kfold_index_dict[idx]["test"]
        
        # Define loaders
        Train_loader = DataLoader(Subset(IC50_Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Validation_loader = DataLoader(Subset(IC50_Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Test_loader = DataLoader(Subset(IC50_Dataset, test_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        # Define model and load weights
        Model = BlendNetS(config["Path"]["IC50_interaction_site_predictor"], config, device).cuda()

        checkpoint = torch.load(f"{config['Path']['IC50_save_path']}/random_split/CV{idx}/BlendNet_S.pth")
        Model.load_state_dict(checkpoint)
        
        for parameter in Model.parameters():
            parameter.requires_grad = False
        Model.eval()
        
        # Define trainer
        trainer = BlendNetS_trainer(config, Model, [None, None], device)

        # Run predictions
        train_predictions, train_labels = trainer.test(Train_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(train_labels, train_predictions)
        print(f"[Train] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[train_index], train_predictions, train_labels, open(f"{config['Path']['IC50_results']}/random_split/CV{idx}/Train_results.txt", "w"))
        
        validation_predictions, validation_labels = trainer.test(Validation_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(validation_labels, validation_predictions)
        print(f"[Validation] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[val_index], validation_predictions, validation_labels, open(f"{config['Path']['IC50_results']}/random_split/CV{idx}/Validation_results.txt", "w"))

        test_predictions, test_labels = trainer.test(Test_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(test_labels, test_predictions)
        print(f"[Test] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[test_index], test_predictions, test_labels, open(f"{config['Path']['IC50_results']}/random_split/CV{idx}/Test_results.txt", "w"))
        print()

    ####################################
    # Load IC50 new protein CV settings
    ####################################
    with open(f"{config['Path']['IC50_new_protein']}", "rb") as f:
        IC50_Kfold_index_dict = pickle.load(f)
    
    for idx in range(3):
        print(f"[IC50] new protein CV {idx} predictions...")
        train_index, val_index, test_index = IC50_Kfold_index_dict[idx]["train"], IC50_Kfold_index_dict[idx]["val"], IC50_Kfold_index_dict[idx]["test"]
        
        # Define loaders
        Train_loader = DataLoader(Subset(IC50_Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Validation_loader = DataLoader(Subset(IC50_Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Test_loader = DataLoader(Subset(IC50_Dataset, test_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        # Define model and load weights
        Model = BlendNetS(config["Path"]["IC50_interaction_site_predictor"], config, device).cuda()

        checkpoint = torch.load(f"{config['Path']['IC50_save_path']}/new_protein/CV{idx}/BlendNet_S.pth")
        Model.load_state_dict(checkpoint)
        
        for parameter in Model.parameters():
            parameter.requires_grad = False
        Model.eval()
        
        # Define trainer
        trainer = BlendNetS_trainer(config, Model, [None, None], device)

        # Run predictions
        train_predictions, train_labels = trainer.test(Train_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(train_labels, train_predictions)
        print(f"[Train] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[train_index], train_predictions, train_labels, open(f"{config['Path']['IC50_results']}/new_protein/CV{idx}/Train_results.txt", "w"))
        
        validation_predictions, validation_labels = trainer.test(Validation_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(validation_labels, validation_predictions)
        print(f"[Validation] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[val_index], validation_predictions, validation_labels, open(f"{config['Path']['IC50_results']}/new_protein/CV{idx}/Validation_results.txt", "w"))

        test_predictions, test_labels = trainer.test(Test_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(test_labels, test_predictions)
        print(f"[Test] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[test_index], test_predictions, test_labels, open(f"{config['Path']['IC50_results']}/new_protein/CV{idx}/Test_results.txt", "w"))
        print()
    
    ####################################
    # Load IC50 new compound CV settings
    ####################################
    with open(f"{config['Path']['IC50_new_compound']}", "rb") as f:
        IC50_Kfold_index_dict = pickle.load(f)
        
    for idx in range(3):
        print(f"[IC50] new compound CV {idx} test...")
        train_index, val_index, test_index = IC50_Kfold_index_dict[idx]["train"], IC50_Kfold_index_dict[idx]["val"], IC50_Kfold_index_dict[idx]["test"]
        
        # Define loaders
        Train_loader = DataLoader(Subset(IC50_Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Validation_loader = DataLoader(Subset(IC50_Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Test_loader = DataLoader(Subset(IC50_Dataset, test_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        # Define model and load weights
        Model = BlendNetS(config["Path"]["IC50_interaction_site_predictor"], config, device).cuda()

        checkpoint = torch.load(f"{config['Path']['IC50_save_path']}/new_compound/CV{idx}/BlendNet_S.pth")
        Model.load_state_dict(checkpoint)
        
        for parameter in Model.parameters():
            parameter.requires_grad = False
        Model.eval()
        
        # Define trainer
        trainer = BlendNetS_trainer(config, Model, [None, None], device)

        # Run predictions
        train_predictions, train_labels = trainer.test(Train_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(train_labels, train_predictions)
        print(f"[Train] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[train_index], train_predictions, train_labels, open(f"{config['Path']['IC50_results']}/new_compound/CV{idx}/Train_results.txt", "w"))
        
        validation_predictions, validation_labels = trainer.test(Validation_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(validation_labels, validation_predictions)
        print(f"[Validation] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[val_index], validation_predictions, validation_labels, open(f"{config['Path']['IC50_results']}/new_compound/CV{idx}/Validation_results.txt", "w"))

        test_predictions, test_labels = trainer.test(Test_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(test_labels, test_predictions)
        print(f"[Test] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[test_index], test_predictions, test_labels, open(f"{config['Path']['IC50_results']}/new_compound/CV{idx}/Test_results.txt", "w"))
        print() 
        
    ####################################
    # Load IC50 blind split CV settings
    ####################################
    with open(f"{config['Path']['IC50_blind_split']}", "rb") as f:
        IC50_Kfold_index_dict = pickle.load(f) 

    for idx in range(3):
        print(f"[IC50] blind split CV {idx} test...")
        train_index, val_index, test_index = IC50_Kfold_index_dict[idx]["train"], IC50_Kfold_index_dict[idx]["val"], IC50_Kfold_index_dict[idx]["test"]
        
        # Define loaders
        Train_loader = DataLoader(Subset(IC50_Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Validation_loader = DataLoader(Subset(IC50_Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
        Test_loader = DataLoader(Subset(IC50_Dataset, test_index), batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        # Define model and load weights
        Model = BlendNetS(config["Path"]["IC50_interaction_site_predictor"], config, device).cuda()

        checkpoint = torch.load(f"{config['Path']['IC50_save_path']}/blind_split/CV{idx}/BlendNet_S.pth")
        Model.load_state_dict(checkpoint)
        
        for parameter in Model.parameters():
            parameter.requires_grad = False
        Model.eval()
        
        # Define trainer
        trainer = BlendNetS_trainer(config, Model, [None, None], device)

        # Run predictions
        train_predictions, train_labels = trainer.test(Train_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(train_labels, train_predictions)
        print(f"[Train] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[train_index], train_predictions, train_labels, open(f"{config['Path']['IC50_results']}/blind_split/CV{idx}/Train_results.txt", "w"))
        
        validation_predictions, validation_labels = trainer.test(Validation_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(validation_labels, validation_predictions)
        print(f"[Validation] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[val_index], validation_predictions, validation_labels, open(f"{config['Path']['IC50_results']}/blind_split/CV{idx}/Validation_results.txt", "w"))

        test_predictions, test_labels = trainer.test(Test_loader) 
        MSE, RMSE, PCC, SPEARMAN, CI = get_results(test_labels, test_predictions)
        print(f"[Test] MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SPEARMAN: {SPEARMAN:.4f}, CI: {CI:.4f}")
        fwrite(Interactions_IDs[test_index], test_predictions, test_labels, open(f"{config['Path']['IC50_results']}/blind_split/CV{idx}/Test_results.txt", "w"))
        print()

def fwrite(Interactions_IDs, predictions, labels, fw):
    
    fw.write("UniProtIDs\tPubChemCIDs\tLabels\tPredictions\n")
    
    for inter_id, lab, pre in zip(Interactions_IDs, labels, predictions):
        prot, comp = inter_id.split("_")[0], inter_id.split("_")[1]
        fw.write(f"{prot}\t{comp}\t{lab:.4f}\t{pre:.4f}\n")
    fw.close()

### Get results
def get_results(labels, predictions):
    
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    MSE = mean_squared_error(labels, predictions)
    RMSE = mean_squared_error(labels, predictions)**0.5
    PCC = pearsonr(labels, predictions)
    CI = concordance_index(labels, predictions)
    SPEARMAN = spearmanr(labels, predictions)

    return MSE, RMSE, PCC[0], SPEARMAN[0], CI
    
if __name__ == "__main__":
    main()