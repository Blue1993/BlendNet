import os

import dgl
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset

class BADataset(Dataset):
    def __init__(self, interaction_IDs, labels, 
                protein_feature_path, pocket_path,
                compound_feature_path, interaction_sites_path, device):

        self.interaction_IDs = interaction_IDs
        self.labels = labels
        self.protein_feature_path = protein_feature_path
        self.compound_feature_path = compound_feature_path
        self.interaction_sites_path = interaction_sites_path
        self.pocket_path = pocket_path
        self.device = device
        
        print("Preparing protein data ...")
        with open(f"{self.protein_feature_path}", "rb") as f:
            self.protein_data_dict = pickle.load(f)
        print(f"\tLoad No. of Protein: {len(self.protein_data_dict)}")
        
        with open(f"{pocket_path}", "rb") as f:
            self.pocket_ind_dict = pickle.load(f)
        print(f"\tLoad No. of Pockets: {len(self.pocket_ind_dict)}")   
        
        print("Preparing compound data ...")
        compound_data_dict = torch.load(f"{self.compound_feature_path}")
        print(f"\tLoad No. of Compound: {len(compound_data_dict['mol_ids'])}")
        
        print("Preparing interaction data ...")
        with open(f"{self.interaction_sites_path}", "rb") as f:
            self.interaction_sites_data_dict = pickle.load(f)
        print(f"\tLoad No. of Interactions: {len(self.interaction_sites_data_dict)}")

        self.compound_ids = compound_data_dict['mol_ids']
        self.compound_id_dict = {molid:idx for idx, molid in enumerate(self.compound_ids)}
        self.compound_feature_tensor = compound_data_dict['atom_features']
        self.compound_e_features_tensor = compound_data_dict['edge_features']
        self.edge_indices = compound_data_dict['edge_indices']
        
        self.compound_meta_dict = {k: compound_data_dict[k] for k in ('mol_ids', 'edge_slices', 'atom_slices', 'n_atoms')} 
        self.avg_degree = compound_data_dict['avg_degree']
        self.dgl_graphs = {}
        
    def __len__(self):
        return len(self.interaction_IDs)
        
    def __getitem__(self, idx):
        pdbid, pid, cid = self.interaction_IDs[idx].split("_")[0], self.interaction_IDs[idx].split("_")[1], self.interaction_IDs[idx].split("_")[2]
        label = self.labels[idx]

        ####################
        # Get Protein data
        ####################
        pfeat = self.protein_data_dict[pid]
        seqlength = pfeat.shape[0]
        pocket = self.pocket_ind_dict[pid]
        
        ####################
        # Get Compound data
        ####################
        comp_idx = self.compound_id_dict[cid]
        e_start = self.compound_meta_dict['edge_slices'][comp_idx].item()
        e_end = self.compound_meta_dict['edge_slices'][comp_idx + 1].item()
        start = self.compound_meta_dict['atom_slices'][comp_idx].item()
        n_atoms = self.compound_meta_dict['n_atoms'][comp_idx].item()
        
        compound_graph = self.data_by_type(comp_idx, e_start, e_end, start, n_atoms)
        num_node = compound_graph.number_of_nodes()
        
        #######################
        # Get Interaction data
        #######################
        interaction_site = self.interaction_sites_data_dict[f"{pdbid}_{cid}"]
        
        return {"pfeat":pfeat, "seqlength": seqlength, "pocket":pocket,
                    "compound_graph":compound_graph, "num_node":num_node,
                        "interaction_site":interaction_site, "label":label}
        
    def data_by_type(self, idx, e_start, e_end, start, n_atoms):
        g = self.get_graph(idx, e_start, e_end, n_atoms, start)
        return g
        
    def get_graph(self, idx, e_start, e_end, n_atoms, start):

        edge_indices = self.edge_indices[:, e_start: e_end]
        g = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms, device=self.device)
        g.ndata['feat'] = self.compound_feature_tensor[start: start + n_atoms].to(self.device)
        g.edata['feat'] = self.compound_e_features_tensor[e_start: e_end].to(self.device)
        return g
       
def pad_data(samples):

    protein_feats = [sample['pfeat'] for sample in samples]
    pocket_ind = [sample['pocket'] for sample in samples]
    seq_lengths = [sample['seqlength'] for sample in samples]
    
    pocket_seq_lengths = [len(sample) for sample in pocket_ind]
    compound_graphs = [sample['compound_graph'] for sample in samples]
    num_nodes = [sample['num_node'] for sample in samples]
    interaction_sites = [sample['interaction_site'] for sample in samples]
    labels = [sample['label'] for sample in samples]

    batch_size, protein_seq_max, pocket_seq_max, compound_atom_max = len(protein_feats), np.max(seq_lengths), np.max(pocket_seq_lengths), np.max(num_nodes)
    
    protein_mask = np.zeros((batch_size, protein_seq_max))
    pocket_mask, compound_mask, interaction_labels = np.zeros((batch_size, pocket_seq_max)), np.zeros((batch_size, compound_atom_max)), np.zeros((batch_size, compound_atom_max, protein_seq_max))
    extract_interaction_labels = np.zeros((batch_size, compound_atom_max, pocket_seq_max))

    resi_feat = np.zeros((batch_size, pocket_seq_max, 1024))
    pocket_ids = np.zeros((batch_size, pocket_seq_max)) - 1
    
    ## extract pocket
    for idx, arr in enumerate(protein_feats):
        protein_mask[idx, :arr.shape[0]] = 1
        extract_arr = arr[pocket_ind[idx],:]
        resi_feat[idx, :extract_arr.shape[0], :] = extract_arr
        pocket_mask[idx, :extract_arr.shape[0]] = 1
        pocket_ids[idx, :pocket_seq_lengths[idx]] = pocket_ind[idx]
        
    for idx, n_node in enumerate(num_nodes):
        compound_mask[idx, :n_node] = 1

    ## extract pocket labels
    for idx, arr in enumerate(interaction_sites):
        interaction_labels[idx, :arr.shape[0], :arr.shape[1]] = arr
        
        extract_isarr = arr[:,pocket_ind[idx]]
        extract_interaction_labels[idx, :extract_isarr.shape[0], :extract_isarr.shape[1]] = extract_isarr
      
    pairwise_mask = np.matmul(np.expand_dims(compound_mask, axis = 2), np.expand_dims(protein_mask, axis = 1))

    resi_feat = torch.tensor(resi_feat, dtype = torch.float32).cuda()
    pocket_mask = torch.tensor(pocket_mask, dtype = torch.long).cuda()
    pocket_ids = torch.tensor(pocket_ids, dtype = torch.long).cuda()
    seq_lengths = torch.tensor(seq_lengths, dtype = torch.long).cuda()
    pairwise_mask = torch.tensor(pairwise_mask, dtype = torch.long).cuda()
    
    compound_mask = torch.tensor(compound_mask, dtype = torch.long).cuda()
    extract_interaction_labels = torch.tensor(extract_interaction_labels, dtype = torch.float32).cuda()
    interaction_labels = torch.tensor(interaction_labels, dtype = torch.float32).cuda()
    labels = torch.tensor(labels, dtype = torch.float32).cuda()
    
    return (resi_feat, pocket_ids, seq_lengths), dgl.batch(compound_graphs).to("cuda:0"), labels, extract_interaction_labels, interaction_labels, pairwise_mask, pocket_mask, compound_mask    
        