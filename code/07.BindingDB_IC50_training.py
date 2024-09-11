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
    
    ##############
    # Load IC50 data
    ##############
    IC50_training_data = pd.read_csv(f"{config['Path']['IC50_df']}", sep = '\t')
    Uniprot_IDs, Lig_codes, BA_labels = IC50_training_data.iloc[:, 0].values, IC50_training_data.iloc[:, 1].values, np.round(IC50_training_data.iloc[:, 2].values, 4)
    Interactions_IDs = np.array([f"{u}_{l}" for u, l in zip(Uniprot_IDs, Lig_codes)])

    ##################
    # Random split CV 
    ##################
    with open(f"{config['Path']['IC50_random_split']}", "rb") as f:
        IC50_Kfold_index_dict = pickle.load(f)

    Dataset = BADataset(interaction_IDs = Interactions_IDs, labels = BA_labels,
                                    protein_feature_path = config['Path']['IC50_protein_feat'],
                                    pocket_path = config['Path']['IC50_pockets'],
                                    compound_feature_path = config['Path']['Ligand_graph'], device = device)

    print(f"[IC50] - random split case training...")
    for idx in range(3):
        reset_seed(config['Train']['seed'])
        
        train_index, val_index, test_index = IC50_Kfold_index_dict[idx]["train"], IC50_Kfold_index_dict[idx]["val"], IC50_Kfold_index_dict[idx]["test"]
        
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        np.random.shuffle(test_index)
        
        ##############
        # Define model
        ##############
        Model = BlendNetS(config["Path"]["IC50_interaction_site_predictor"], config, device).cuda()
        
        ###################
        # Define optimizer 
        ###################
        KD_parameters = [v for k, v in Model.named_parameters() if "resi_encoder" in k or "cross_encoder" in k or "intersites_predictor.pairwise_compound" in k or "intersites_predictor.pairwise_protein" in k]
        BA_parameters = [v for k, v in Model.named_parameters() if v.requires_grad == True]

        KDOptimizer = optim.Adam([{'params':KD_parameters}], lr=0.0001, amsgrad=False) 
        BAOptimizer = optim.Adam([{'params':BA_parameters}], lr=0.0001, amsgrad=False) 

        scheduler_dta = optim.lr_scheduler.ReduceLROnPlateau(BAOptimizer, mode='min', factor=0.9, patience=5, threshold=1e-3, verbose = True)
        scheduler_kd = optim.lr_scheduler.ReduceLROnPlateau(KDOptimizer, mode='min', factor=0.9, patience=5, threshold=1e-3, verbose = True)

        TrainLoader = DataLoader(Subset(Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        ValLoader = DataLoader(Subset(Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        print(f"> Training: {len(train_index)}, Validation: {len(val_index)}")

        ###################
        # Define trainer 
        ###################
        trainer = BlendNetS_trainer(config, Model, [KDOptimizer, BAOptimizer], device)

        #########################
        # Get teacher predictions 
        #########################
        train_teacher_pred, train_teacher_label = trainer.extract_total_teacher_predictions(TrainLoader)
        validation_teacher_pred, validation_teacher_label = trainer.extract_total_teacher_predictions(ValLoader)
       
        train_error = (train_teacher_pred - train_teacher_label)** 2
        validation_error = (validation_teacher_pred - validation_teacher_label)** 2
        
        max_train_teacher_error, min_train_teacher_error = torch.max(train_error), torch.min(train_error)
        max_validation_teacher_error, min_validation_teacher_error = torch.max(validation_error), torch.min(validation_error)

        ###############
        # Run training 
        ###############
        for epoch in range(1, config["Train"]["epochs"] + 1):
            TrainLoss = trainer.train(TrainLoader, max_teacher_error = max_train_teacher_error, min_teacher_error = min_train_teacher_error)
            print(f"[Train ({epoch})] ToTal BA loss: {TrainLoss['Total_BALoss']:.4f}, BA loss: {TrainLoss['BALoss']:.4f}, Imitation loss: {TrainLoss['ImitationLoss']:.4f}")

            save_path = f"{config['Path']['IC50_save_path']}/random_split/CV{idx}/BlendNet_S.pth"
            ValLoss, patience = trainer.eval(ValLoader, idx, save_path = save_path, max_teacher_error = max_validation_teacher_error, min_teacher_error = min_validation_teacher_error)
            print(f"[Val ({epoch})] ToTal BA loss: {ValLoss['Total_BALoss']:.4f}, BA loss: {ValLoss['BALoss']:.4f}, Imitation loss: {ValLoss['ImitationLoss']:.4f}")
            
            if patience > 15:
                print(f"Validation loss do not improves, stop training")
                break
                
            if scheduler_dta is not None:
                scheduler_dta.step(ValLoss['Total_BALoss']) 
                scheduler_kd.step(ValLoss['Total_BALoss'])
            print()


    ##################
    # New protein CV 
    ##################
    with open(f"{config['Path']['IC50_new_protein']}", "rb") as f:
        IC50_Kfold_index_dict = pickle.load(f)

    print(f"[IC50] - new protein case training...")
    for idx in range(3):
        reset_seed(config['Train']['seed'])
        
        train_index, val_index, test_index = IC50_Kfold_index_dict[idx]["train"], IC50_Kfold_index_dict[idx]["val"], IC50_Kfold_index_dict[idx]["test"]
        
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        np.random.shuffle(test_index)
        
        ##############
        # Define model
        ##############
        Model = BlendNetS(config["Path"]["IC50_interaction_site_predictor"], config, device).cuda()
        
        ###################
        # Define optimizer 
        ###################
        KD_parameters = [v for k, v in Model.named_parameters() if "resi_encoder" in k or "cross_encoder" in k or "intersites_predictor.pairwise_compound" in k or "intersites_predictor.pairwise_protein" in k]
        BA_parameters = [v for k, v in Model.named_parameters() if v.requires_grad == True]

        KDOptimizer = optim.Adam([{'params':KD_parameters}], lr=0.0001, amsgrad=False) 
        BAOptimizer = optim.Adam([{'params':BA_parameters}], lr=0.0001, amsgrad=False) 

        scheduler_dta = optim.lr_scheduler.ReduceLROnPlateau(BAOptimizer, mode='min', factor=0.9, patience=5, threshold=1e-3, verbose = True)
        scheduler_kd = optim.lr_scheduler.ReduceLROnPlateau(KDOptimizer, mode='min', factor=0.9, patience=5, threshold=1e-3, verbose = True)

        TrainLoader = DataLoader(Subset(Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        ValLoader = DataLoader(Subset(Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        print(f"> Training: {len(train_index)}, Validation: {len(val_index)}")

        ###################
        # Define trainer 
        ###################
        trainer = BlendNetS_trainer(config, Model, [KDOptimizer, BAOptimizer], device)

        #########################
        # Get teacher predictions 
        #########################
        train_teacher_pred, train_teacher_label = trainer.extract_total_teacher_predictions(TrainLoader)
        validation_teacher_pred, validation_teacher_label = trainer.extract_total_teacher_predictions(ValLoader)
       
        train_error = (train_teacher_pred - train_teacher_label)** 2
        validation_error = (validation_teacher_pred - validation_teacher_label)** 2
        
        max_train_teacher_error, min_train_teacher_error = torch.max(train_error), torch.min(train_error)
        max_validation_teacher_error, min_validation_teacher_error = torch.max(validation_error), torch.min(validation_error)

        ###############
        # Run training 
        ###############
        for epoch in range(1, config["Train"]["epochs"] + 1):
            TrainLoss = trainer.train(TrainLoader, max_teacher_error = max_train_teacher_error, min_teacher_error = min_train_teacher_error)
            print(f"[Train ({epoch})] ToTal BA loss: {TrainLoss['Total_BALoss']:.4f}, BA loss: {TrainLoss['BALoss']:.4f}, Imitation loss: {TrainLoss['ImitationLoss']:.4f}")

            save_path = f"{config['Path']['IC50_save_path']}/new_protein/CV{idx}/BlendNet_S.pth"
            ValLoss, patience = trainer.eval(ValLoader, idx, save_path = save_path, max_teacher_error = max_validation_teacher_error, min_teacher_error = min_validation_teacher_error)
            print(f"[Val ({epoch})] ToTal BA loss: {ValLoss['Total_BALoss']:.4f}, BA loss: {ValLoss['BALoss']:.4f}, Imitation loss: {ValLoss['ImitationLoss']:.4f}")
            
            if patience > 15:
                print(f"Validation loss do not improves, stop training")
                break
                
            if scheduler_dta is not None:
                scheduler_dta.step(ValLoss['Total_BALoss']) 
                scheduler_kd.step(ValLoss['Total_BALoss'])
            print()


    ##################
    # New compound CV 
    ##################
    with open(f"{config['Path']['IC50_new_compound']}", "rb") as f:
        IC50_Kfold_index_dict = pickle.load(f)

    print(f"[IC50] - new compound case training...")
    for idx in range(3):
        reset_seed(config['Train']['seed'])
        
        train_index, val_index, test_index = IC50_Kfold_index_dict[idx]["train"], IC50_Kfold_index_dict[idx]["val"], IC50_Kfold_index_dict[idx]["test"]
        
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        np.random.shuffle(test_index)
        
        ##############
        # Define model
        ##############
        Model = BlendNetS(config["Path"]["IC50_interaction_site_predictor"], config, device).cuda()
        
        ###################
        # Define optimizer 
        ###################
        KD_parameters = [v for k, v in Model.named_parameters() if "resi_encoder" in k or "cross_encoder" in k or "intersites_predictor.pairwise_compound" in k or "intersites_predictor.pairwise_protein" in k]
        BA_parameters = [v for k, v in Model.named_parameters() if v.requires_grad == True]

        KDOptimizer = optim.Adam([{'params':KD_parameters}], lr=0.0001, amsgrad=False) 
        BAOptimizer = optim.Adam([{'params':BA_parameters}], lr=0.0001, amsgrad=False) 

        scheduler_dta = optim.lr_scheduler.ReduceLROnPlateau(BAOptimizer, mode='min', factor=0.9, patience=5, threshold=1e-3, verbose = True)
        scheduler_kd = optim.lr_scheduler.ReduceLROnPlateau(KDOptimizer, mode='min', factor=0.9, patience=5, threshold=1e-3, verbose = True)

        TrainLoader = DataLoader(Subset(Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        ValLoader = DataLoader(Subset(Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        print(f"> Training: {len(train_index)}, Validation: {len(val_index)}")

        ###################
        # Define trainer 
        ###################
        trainer = BlendNetS_trainer(config, Model, [KDOptimizer, BAOptimizer], device)

        #########################
        # Get teacher predictions 
        #########################
        train_teacher_pred, train_teacher_label = trainer.extract_total_teacher_predictions(TrainLoader)
        validation_teacher_pred, validation_teacher_label = trainer.extract_total_teacher_predictions(ValLoader)
       
        train_error = (train_teacher_pred - train_teacher_label)** 2
        validation_error = (validation_teacher_pred - validation_teacher_label)** 2
        
        max_train_teacher_error, min_train_teacher_error = torch.max(train_error), torch.min(train_error)
        max_validation_teacher_error, min_validation_teacher_error = torch.max(validation_error), torch.min(validation_error)

        ###############
        # Run training 
        ###############
        for epoch in range(1, config["Train"]["epochs"] + 1):
            TrainLoss = trainer.train(TrainLoader, max_teacher_error = max_train_teacher_error, min_teacher_error = min_train_teacher_error)
            print(f"[Train ({epoch})] ToTal BA loss: {TrainLoss['Total_BALoss']:.4f}, BA loss: {TrainLoss['BALoss']:.4f}, Imitation loss: {TrainLoss['ImitationLoss']:.4f}")

            save_path = f"{config['Path']['IC50_save_path']}/new_compound/CV{idx}/BlendNet_S.pth"
            ValLoss, patience = trainer.eval(ValLoader, idx, save_path = save_path, max_teacher_error = max_validation_teacher_error, min_teacher_error = min_validation_teacher_error)
            print(f"[Val ({epoch})] ToTal BA loss: {ValLoss['Total_BALoss']:.4f}, BA loss: {ValLoss['BALoss']:.4f}, Imitation loss: {ValLoss['ImitationLoss']:.4f}")
            
            if patience > 15:
                print(f"Validation loss do not improves, stop training")
                break
                
            if scheduler_dta is not None:
                scheduler_dta.step(ValLoss['Total_BALoss']) 
                scheduler_kd.step(ValLoss['Total_BALoss'])
            print()


    ##################
    # Blind split CV 
    ##################
    with open(f"{config['Path']['IC50_blind_split']}", "rb") as f:
        IC50_Kfold_index_dict = pickle.load(f)

    print(f"[IC50] - blind split case training...")
    for idx in range(3):
        reset_seed(config['Train']['seed'])
        
        train_index, val_index, test_index = IC50_Kfold_index_dict[idx]["train"], IC50_Kfold_index_dict[idx]["val"], IC50_Kfold_index_dict[idx]["test"]
        
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        np.random.shuffle(test_index)
        
        ##############
        # Define model
        ##############
        Model = BlendNetS(config["Path"]["IC50_interaction_site_predictor"], config, device).cuda()
        
        ###################
        # Define optimizer 
        ###################
        KD_parameters = [v for k, v in Model.named_parameters() if "resi_encoder" in k or "cross_encoder" in k or "intersites_predictor.pairwise_compound" in k or "intersites_predictor.pairwise_protein" in k]
        BA_parameters = [v for k, v in Model.named_parameters() if v.requires_grad == True]

        KDOptimizer = optim.Adam([{'params':KD_parameters}], lr=0.0001, amsgrad=False) 
        BAOptimizer = optim.Adam([{'params':BA_parameters}], lr=0.0001, amsgrad=False) 

        scheduler_dta = optim.lr_scheduler.ReduceLROnPlateau(BAOptimizer, mode='min', factor=0.9, patience=5, threshold=1e-3, verbose = True)
        scheduler_kd = optim.lr_scheduler.ReduceLROnPlateau(KDOptimizer, mode='min', factor=0.9, patience=5, threshold=1e-3, verbose = True)

        TrainLoader = DataLoader(Subset(Dataset, train_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        ValLoader = DataLoader(Subset(Dataset, val_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        print(f"> Training: {len(train_index)}, Validation: {len(val_index)}")

        ###################
        # Define trainer 
        ###################
        trainer = BlendNetS_trainer(config, Model, [KDOptimizer, BAOptimizer], device)

        #########################
        # Get teacher predictions 
        #########################
        train_teacher_pred, train_teacher_label = trainer.extract_total_teacher_predictions(TrainLoader)
        validation_teacher_pred, validation_teacher_label = trainer.extract_total_teacher_predictions(ValLoader)
       
        train_error = (train_teacher_pred - train_teacher_label)** 2
        validation_error = (validation_teacher_pred - validation_teacher_label)** 2
        
        max_train_teacher_error, min_train_teacher_error = torch.max(train_error), torch.min(train_error)
        max_validation_teacher_error, min_validation_teacher_error = torch.max(validation_error), torch.min(validation_error)

        ###############
        # Run training 
        ###############
        for epoch in range(1, config["Train"]["epochs"] + 1):
            TrainLoss = trainer.train(TrainLoader, max_teacher_error = max_train_teacher_error, min_teacher_error = min_train_teacher_error)
            print(f"[Train ({epoch})] ToTal BA loss: {TrainLoss['Total_BALoss']:.4f}, BA loss: {TrainLoss['BALoss']:.4f}, Imitation loss: {TrainLoss['ImitationLoss']:.4f}")

            save_path = f"{config['Path']['IC50_save_path']}/blind_split/CV{idx}/BlendNet_S.pth"
            ValLoss, patience = trainer.eval(ValLoader, idx, save_path = save_path, max_teacher_error = max_validation_teacher_error, min_teacher_error = min_validation_teacher_error)
            print(f"[Val ({epoch})] ToTal BA loss: {ValLoss['Total_BALoss']:.4f}, BA loss: {ValLoss['BALoss']:.4f}, Imitation loss: {ValLoss['ImitationLoss']:.4f}")
            
            if patience > 15:
                print(f"Validation loss do not improves, stop training")
                break
                
            if scheduler_dta is not None:
                scheduler_dta.step(ValLoss['Total_BALoss']) 
                scheduler_kd.step(ValLoss['Total_BALoss'])
            print()
        
def reset_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
if __name__ == "__main__":
    main()