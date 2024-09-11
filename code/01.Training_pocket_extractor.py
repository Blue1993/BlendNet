import os
import sys
import yaml
import pickle
import random
import logging
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Subset, DataLoader

from modules.common.utils import load_cfg
from modules.pocket_modules.loaders import PocketDataset
from modules.pocket_modules.trainers import Pseq2SitesTrainer

def main():
    
    ##############
    # Load config
    ##############
    config_path = "./pocket_extractor_config.yml"
    config = load_cfg(config_path)
    
    ############
    # Load data
    ############
    print("Load data...")
    with open(config["Path"]["prot_feats"], "rb") as f:
        protein_feats = pickle.load(f)
        
    training_df = pd.read_csv(config["Path"]["training"], sep = "\t")
    IDs, seqs, prot_BS = training_df.iloc[:,0].values, training_df.iloc[:,1].values, training_df.iloc[:,3].values
    print(f"\t> Training seqs: {len(IDs)}")
    
    ###############################
    # training of pocket extractor 
    ###############################
    print(f"Start training")
    for idx in range(5):
        print(f"[CV{idx}] Split dataset to train and validation")
        all_idx = [i for i in range(len(IDs))]
        PSeqTrainIdx = random.sample(all_idx, int(len(all_idx) * 0.9))
        PSeqValIdx = list(set(all_idx) - set(PSeqTrainIdx))  
        print(f"\t[Training]: {len(PSeqTrainIdx)} samples, [Validation]: {len(PSeqValIdx)} samples")
        print()

        torch.manual_seed(config['Train']['seed'])
        np.random.seed(config['Train']['seed'])
        
        device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['Train']['seed'])
        
        PSeqDataset = PocketDataset(PID = IDs, Pseqs = seqs, Pfeatures = protein_feats, Labels = prot_BS)

        PseqTrainLoader = DataLoader(Subset(PSeqDataset, PSeqTrainIdx), batch_size=config['Train']['batch_size'], shuffle=True)
        PseqValLoader = DataLoader(Subset(PSeqDataset, PSeqValIdx), batch_size=config['Train']['batch_size'], shuffle=True)
        
        trainer = Pseq2SitesTrainer(config, device)
        for epoch in range(1, config['Train']['epochs'] + 1):
            print(f"====Epoch: {epoch}====")
            train_loss = trainer.train(PseqTrainLoader)
            print(f"[Train ({epoch})] Loss: {train_loss:.4f}")
            
            val_loss, patience = trainer.eval(PseqValLoader, idx)
            print(f"[Val ({epoch})] Loss: {val_loss:.4f}")
            
            if patience > config["Train"]["patience"]:
                print(f"Validation loss do not improves, stop training")
                break
        print()
        
if __name__ == "__main__":
    main()          