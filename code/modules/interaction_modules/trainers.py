import os
import random
import numpy as np
from tqdm import tqdm

import torch 
import torch.nn as nn

from sklearn import linear_model
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class BlendNetT_trainer():
     def __init__(self, config, model, optimizer, device):
        self.config = config
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optimizer
        self.best_eval_loss = np.inf
        self.patience = 0

     def train(self, loader):        
        results = {"MSE": 0, "InterSitesLoss":0}   
            
        pred_list, label_list = list(), list() 
        
        self.model.residue_encoder.train()
        self.model.cross_encoder.train()
        self.model.intersites_predictor.train()
        self.model.ba_predictor.train()

        for idx, batch in enumerate(tqdm(loader)):

            prot_feat_set, compound_graph, labels, interaction_labels, ori_interaction_labels, ori_pairwise_mask, pocket_mask, compound_mask = batch

            ba_predictions, pairwise_map, pairwise_mask = self.model(prot_feat_set[0], compound_graph, pocket_mask, compound_mask)

            ba_loss = self.criterion(ba_predictions, labels)
            pairwise_loss = self.pairwise_criterion(pairwise_map, interaction_labels, pairwise_mask)
            
            loss = ba_loss + pairwise_loss * 0.5

            results['InterSitesLoss'] += float(pairwise_loss.cpu().item())
            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            
            self.optimizer.step()

            pred_list.extend(ba_predictions.detach().cpu().tolist())
            label_list.extend(labels.detach().cpu().tolist())
            
        results = {k: v / len(loader) for k, v in results.items()}
        MSE = testmse = mean_squared_error(np.array(label_list), np.array(pred_list))
        results['MSE'] = MSE
        
        return results

     @torch.no_grad()
     def eval(self, loader, fold):
        results = {"MSE": 0, "InterSitesLoss":0}  
        
        pred_list, label_list = list(), list()
        self.model.eval()

        for idx, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                prot_feat_set, compound_graph, labels, interaction_labels, ori_interaction_labels, ori_pairwise_mask, pocket_mask, compound_mask = batch
                ba_predictions, pairwise_map, pairwise_mask = self.model(prot_feat_set[0], compound_graph, pocket_mask, compound_mask)

                ba_loss = self.criterion(ba_predictions, labels)
                pairwise_loss = self.pairwise_criterion(pairwise_map, interaction_labels, pairwise_mask)

                loss = ba_loss + pairwise_loss * 0.5

                results['InterSitesLoss'] += float(pairwise_loss.cpu().item())

                pred_list.extend(ba_predictions.detach().cpu().tolist())
                label_list.extend(labels.detach().cpu().tolist())
                
        results = {k: v / len(loader) for k, v in results.items()}
        MSE = mean_squared_error(np.array(label_list), np.array(pred_list))
        results['MSE'] = MSE

        if results["MSE"] < self.best_eval_loss:
            self.patience = 0
            torch.save(self.model.state_dict(), f"{self.config['Path']['save_path']}/CV{fold}/BlendNet_T.pth")
            print(f"Save model improvements: {(self.best_eval_loss - results['MSE']):.4f}")
            self.best_eval_loss = results['MSE']
        else:
            self.patience += 1

        return results, self.patience

     def pairwise_criterion(self, pred, labels, mask, dim = None):

        loss_ft = nn.BCELoss(reduction = 'none')
        loss_all = loss_ft(pred, labels)

        loss = torch.sum(loss_all*mask) / pred.size()[0]

        return loss

     def return_matrix(self, pairwise_map, pocket_ind_list, protein_seq_list, ori_interaction_labels):

        final_pairwise_map = np.transpose(np.zeros(ori_interaction_labels.shape), (0, 2, 1))
        pairwise_map = np.transpose(pairwise_map, (0, 2, 1))

        for idx, (pmap, pind) in enumerate(zip(pairwise_map, pocket_ind_list)):
            for jdx, ind in enumerate(pind):
                if ind != -1:
                    final_pairwise_map[idx, ind, :] = pmap[jdx]

        return np.transpose(final_pairwise_map, (0, 2, 1))

     @torch.no_grad()
     def ba_prediction(self, loader):
        pred_list, label_list = list(), list()
        
        self.model.eval()
        
        for idx, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                prot_feat_set, compound_graph, labels, interaction_labels, ori_interaction_labels, ori_pairwise_mask, pocket_mask, compound_mask = batch
                
                ba_predictions, _, _ = self.model(prot_feat_set[0], compound_graph, pocket_mask, compound_mask)
                
                pred_list.extend(ba_predictions.detach().cpu().tolist())
                label_list.extend(labels.detach().cpu().tolist())

        return np.array(pred_list), np.array(label_list)

     @torch.no_grad()
     def pairwise_map_prediction(self, loader):
        pairwise_pred_list, pairwise_mask_list, pairwise_label_list, protein_lengths = list(), list(), list(), list()
        
        for idx, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                prot_feat_set, compound_graph, labels, interaction_labels, ori_interaction_labels, ori_pairwise_mask, pocket_mask, compound_mask = batch
                ba_predictions, pairwise_map, pairwise_mask = self.model(prot_feat_set[0], compound_graph, pocket_mask, compound_mask)
                
                protein_lengths.extend(prot_feat_set[2].detach().cpu().tolist())
                pairwise_map = self.return_matrix(pairwise_map.detach().cpu().numpy(), prot_feat_set[1].detach().cpu().numpy(), prot_feat_set[2].detach().cpu().numpy(), ori_interaction_labels.detach().cpu().numpy())
                
                for i in pairwise_map:
                    pairwise_pred_list.append(i)

                for i in ori_pairwise_mask.detach().cpu().tolist():
                    pairwise_mask_list.append(i)
 
                for i in ori_interaction_labels.detach().cpu().tolist():
                    pairwise_label_list.append(i)

        return pairwise_pred_list, pairwise_mask_list, pairwise_label_list, protein_lengths
