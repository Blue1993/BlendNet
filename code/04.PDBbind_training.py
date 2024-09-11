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
    
    ###########
    # Load data
    ###########
    print(f"Load data...")
    with open(f"{config['Path']['KFold']}", "rb") as f:
        Kfold_index_dict = pickle.load(f)
        
    Training_df = pd.read_csv(f"{config['Path']['Training_df']}", sep = "\t")
    PDB_IDs, Uniprot_IDs, Lig_codes, BA_labels = Training_df.iloc[:, 0].values, Training_df.iloc[:, 1].values, Training_df.iloc[:, 2].values, Training_df.iloc[:, 3].values
    Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(PDB_IDs, Uniprot_IDs, Lig_codes)])
    
    ##############
    # Run training
    ##############
    for idx in range(5):
        print(f"CV {idx} is running...")
        reset_seed(config['Train']['seed'])
        
        train_index, val_index, test_index = Kfold_index_dict[idx]["train"], Kfold_index_dict[idx]["val"], Kfold_index_dict[idx]["test"]
        print(f"\tTraining: {len(train_index)}, Validation: {len(val_index)}, Test: {len(test_index)}")
        
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        np.random.shuffle(test_index) 
        
        TotalDataset = BADataset(interaction_IDs = Interactions_IDs, labels = BA_labels, 
                                protein_feature_path = config['Path']['Training_protein_feat'],
                                pocket_path = config['Path']['Training_pocket'],
                                compound_feature_path = config['Path']['Training_ligand_graph'],
                                interaction_sites_path = config['Path']['Training_interaction'],
                                device = device)
                                
        ################################
        # Define Binding Affinity model
        ################################
        Model = BlendNetT(config, device).cuda()
        
        ###################
        # Define optimizer 
        ###################
        parameters = [v for k, v in Model.named_parameters() if v.requires_grad == True]
        optimizer = optim.Adam([{'params':parameters}], lr=1e-10, weight_decay=config['Train']['decay'], amsgrad=False)        
        print('model trainable params: ', sum(p.numel() for p in Model.parameters() if p.requires_grad))
        print('model Total params: ', sum(p.numel() for p in Model.parameters()))

        scheduler_dta = CosineAnnealingWarmUpRestarts(optimizer, T_0=15, T_mult=1, eta_max=0.001, T_up=1, gamma=0.96)
        
        TrainLoader = DataLoader(Subset(TotalDataset, train_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        ValLoader = DataLoader(Subset(TotalDataset, val_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)

        trainer = BlendNetT_trainer(config, Model, optimizer, device)

        ###########
        # BA Train
        ###########
        for epoch in range(1, config["Train"]["epochs"] + 1):
            TrainLoss = trainer.train(TrainLoader)
            print(f"[Train ({epoch})] BA MSE: {TrainLoss['MSE']:.4f}, PairCE loss: {TrainLoss['InterSitesLoss']:.4f}")
        
            ValLoss, patience = trainer.eval(ValLoader, idx)
            print(f"[Val ({epoch})] BA MSE: {ValLoss['MSE']:.4f}, PairCE loss: {ValLoss['InterSitesLoss']:.4f}")

            if patience > config["Train"]["patience"]:
                print(f"Validation loss do not improves, stop training")
                break
                
            if scheduler_dta is not None:
                scheduler_dta.step()   
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
            
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):

        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
if __name__ == "__main__":
    main()