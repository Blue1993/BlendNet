import os
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy

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

class BlendNetS_trainer():
    def __init__(self, config, model, optimizer_list, device):
        self.config = config
        self.model = model
        self.ba_criterion = nn.MSELoss()
        self.hint_optimizer = optimizer_list[0]
        self.ba_optimizer = optimizer_list[1]
        self.best_eval_loss = np.inf
        self.patience = 0
    
    def train(self, loader, max_teacher_error, min_teacher_error):
        results = {"Total_BALoss": 0, "BALoss": 0, "ImitationLoss": 0}

        self.model.residue_encoder.train()
        self.model.cross_encoder.train()
        self.model.intersites_predictor.train()
        self.model.ba_predictor.train()
        
        for idx, batch in enumerate(tqdm(loader)):
            
            prot_feat_set, compound_graph, labels, pocket_mask, compound_mask = batch
            compound_graph_for_teacher = deepcopy(compound_graph)
            compound_graph_for_ba = deepcopy(compound_graph)
            
            """
            Get teacher predictions for batch
            """
            with torch.no_grad():
                ba_teacher_labels, teacher_pairwise_maps, teacher_compound_representations, teacher_protein_representations = self.model.get_teacher_labels(prot_feat_set[0], compound_graph_for_teacher, pocket_mask, compound_mask)
            
            """
            Update hint loss
            """
            _, pairwise_maps, pairwise_masks, compound_representations, protein_representations = self.model(prot_feat_set[0], compound_graph, pocket_mask, compound_mask)
            
            representation_loss = self.pairwise_criterion(teacher_compound_representations, teacher_protein_representations,
                                                compound_representations, protein_representations,
                                                    ba_teacher_labels, labels, max_teacher_error, min_teacher_error, pocket_mask, compound_mask)
            
            inter_site_loss = self.inter_site_loss(teacher_pairwise_maps, pairwise_maps, pairwise_masks,
                                    ba_teacher_labels, labels, max_teacher_error, min_teacher_error)
            
            hint_loss = representation_loss + inter_site_loss
            hint_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            
            self.hint_optimizer.step()
            self.hint_optimizer.zero_grad()
            
            """
            Update imitation and ba losses
            """
            ba_predictions, _, _, _, _ = self.model(prot_feat_set[0], compound_graph_for_ba, pocket_mask, compound_mask)
            
            imitation_loss = self.imitation_loss(ba_predictions, ba_teacher_labels, labels, max_teacher_error, min_teacher_error)
            ba_loss = self.ba_criterion(ba_predictions, labels)
            
            loss = imitation_loss + ba_loss * 0.4            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            
            self.ba_optimizer.step()
            self.ba_optimizer.zero_grad()
            
            results['Total_BALoss'] += float(loss.cpu().item())
            results['BALoss'] += float(ba_loss.cpu().item())
            results['ImitationLoss'] += float(imitation_loss.cpu().item())

        results = {k: v / len(loader) for k, v in results.items()}
        
        return results
    
    @torch.no_grad()
    def eval(self, loader, fold, save_path, max_teacher_error, min_teacher_error):
        results = {"Total_BALoss": 0, "BALoss": 0, "ImitationLoss": 0}
        self.model.eval()
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(loader)):
                prot_feat_set, compound_graph, labels, pocket_mask, compound_mask = batch
                compound_graph_for_teacher = deepcopy(compound_graph)
    
                ba_teacher_labels, teacher_pairwise_maps, teacher_compound_representations, teacher_protein_representations = self.model.get_teacher_labels(prot_feat_set[0], compound_graph_for_teacher, pocket_mask, compound_mask)

                ba_predictions, pairwise_maps, pairwise_masks, compound_representations, protein_representations = self.model(prot_feat_set[0], compound_graph, pocket_mask, compound_mask)
                
                imitation_loss = self.imitation_loss(ba_predictions, ba_teacher_labels, labels, max_teacher_error, min_teacher_error)
                ba_loss = self.ba_criterion(ba_predictions, labels)

                loss = ba_loss + imitation_loss * 0.4

                results['Total_BALoss'] += float(loss.cpu().item())
                results['BALoss'] += float(ba_loss.cpu().item())
                results['ImitationLoss'] += float(imitation_loss.cpu().item())

        results = {k: v / len(loader) for k, v in results.items()}

        if results["Total_BALoss"] < self.best_eval_loss:
            self.patience = 0
            
            torch.save(self.model.state_dict(), f"{save_path}")
            print(f"Save model improvements: {(self.best_eval_loss - results['Total_BALoss']):.4f}")
            self.best_eval_loss = results['Total_BALoss']
        else:
            self.patience += 1

        return results, self.patience
    
    @torch.no_grad()
    def extract_total_teacher_predictions(self, loader):
        pred_list, label_list = list(), list()
        self.model.eval()
        
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                prot_feat_set, compound_graph, labels, pocket_masks, compound_masks = batch
                ba_teacher_labels, _, _, _ = self.model.get_teacher_labels(prot_feat_set[0], compound_graph, pocket_masks, compound_masks)
                pred_list.extend(ba_teacher_labels.detach().cpu().tolist())
                label_list.extend(labels.detach().cpu().tolist())

        return torch.tensor(pred_list).cuda(), torch.tensor(label_list).cuda()
      
    @torch.no_grad()
    def test(self, loader):
        pred_list, label_list = list(), list()
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(loader)):
                prot_feat_set, compound_graph, labels, pocket_mask, compoun_mask = batch
                ba_predictions, pairwise_maps, pairwise_masks, _, _ = self.model(prot_feat_set[0], compound_graph, pocket_mask, compoun_mask)
                
                pred_list.extend(ba_predictions.detach().cpu().tolist())
                label_list.extend(labels.detach().cpu().tolist())
        
        return np.array(pred_list), np.array(label_list)
        
        
    def imitation_loss(self, student_predictions, teacher_predictions, labels, max_teacher_error, min_teacher_error):
        loss_ft = nn.MSELoss(reduction='none')
        
        weight = max_teacher_error - min_teacher_error
        
        stu_tea_loss = loss_ft(student_predictions, student_predictions)
        tea_gt_loss = (student_predictions - labels)** 2
        tea_gt_loss = torch.clamp(tea_gt_loss, max = max_teacher_error.item())

        weigths = 1 - torch.clamp(tea_gt_loss/weight, max = 1)

        loss = torch.mean(weigths * stu_tea_loss)
        
        return loss
        
    def inter_site_loss(self, teacher_pairwise_maps, student_pairwise_maps, pairwise_masks,
                    teacher_predictions, labels, max_teacher_error, min_teacher_error): 
                    
        weight = max_teacher_error - min_teacher_error
        
        tea_gt_loss = (teacher_predictions - labels)** 2
        tea_gt_loss = torch.clamp(tea_gt_loss, max = max_teacher_error.item())
        
        weigths = 1 - torch.clamp(tea_gt_loss/weight, max = 1)
        weigths = weigths.unsqueeze(-1)

        loss_ft = nn.BCELoss(reduction = 'none')
        
        teacher_pairwise_maps = torch.sigmoid(teacher_pairwise_maps)
        teacher_pairwise_maps = torch.multiply(teacher_pairwise_maps, pairwise_masks)

        teacher_pairwise_maps[teacher_pairwise_maps >= 0.1] = 1
        teacher_pairwise_maps[teacher_pairwise_maps < 0.1] = 0
        
        student_pairwise_maps = torch.sigmoid(student_pairwise_maps)
        student_pairwise_maps = torch.multiply(student_pairwise_maps, pairwise_masks)

        loss = loss_ft(student_pairwise_maps, teacher_pairwise_maps)
        loss = torch.sum(loss) / student_pairwise_maps.size()[0]

        return loss
        
    def pairwise_criterion(self, teacher_compound_representations, teacher_protein_representations, 
                            student_compound_representations, student_protein_representations, 
                                    teacher_predictions, labels, max_teacher_error, min_teacher_error, pocket_masks, compound_masks): 

        weight = max_teacher_error - min_teacher_error
        
        tea_gt_loss = (teacher_predictions - labels)** 2
        tea_gt_loss = torch.clamp(tea_gt_loss, max = max_teacher_error.item())
        
        weigths = 1 - torch.clamp(tea_gt_loss/weight, max = 1)
        weigths = weigths.unsqueeze(-1)

        loss_ft = nn.MSELoss(reduction='none')
        
        # Compound loss
        compound_loss = loss_ft(student_compound_representations, teacher_compound_representations)
        compound_loss = torch.sum(compound_loss, dim = 1)
        compound_loss = torch.sum(compound_loss * weigths) / teacher_compound_representations.size()[0]

        # Protein loss
        protein_loss = loss_ft(student_protein_representations, teacher_protein_representations)
        protein_loss = torch.sum(protein_loss, dim = 1)
        protein_loss = torch.sum(protein_loss * weigths) / teacher_protein_representations.size()[0]

        return torch.mean(compound_loss + protein_loss)