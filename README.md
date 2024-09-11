# BlendNet

BlendNet is a complex-free binding affinity prediction model. \
To improve binding affinity prediction accuracy, BlendNet aims to learn interdependent information between proteins and compounds.

BlendNet consists of two networks: BlendNet (T) and BlendNet (S). 

BlendNet (T) trains on data related to non-covalent interaction sites extracted from the PDB database. The trained BlendNet (T) transfers interdependent information to BlendNet (S), which is trained with limited complex structure data, thereby enhancing the accuracy of BlendNet (S)’s binding affinity prediction. This training strategy can provide the predictive model with information about the interaction that is useful for binding affinity prediction in the context of limited compound-protein complex structure.

BlendNet (S) outperforms existing complex-free models in a various cold-start cases based on the information received from the teacher network. In addition, BlendNet (S) shows the interpretability of the interactions even without utilizing complex structure data.

### Experiment results
The BlendNet framework was compared to 9 complex-based and 20 complex-free models on eight test datasets, including internal test dataset. \
The results of all baseline experiments, including BlendNet, are available in '/results/PDBbind' dir for public use by other researchers.
- Experiment setting: 5-fold cross-valiation
- External test datasets: CASF2016, CASF2013, CSAR2014, CSAR2012, CSARset1, CSARset2, and Astex
- Complex-based models: RF-Score, Pafnucy, OnionNet, PLECC-NN, PSH-ML, SMLPLIP-Score, EISA-Score, PointCloud-ML, and KIDA (T)
- Complex-free models: DeepDTA, GraphDTA, DeepGLSTM, MGraphDTA, BiCompDTA, DeepAffinity, ML-DTI, MolTrans, AttentionDTA, BACPI, FusionDTA, PSG-BAR, DeepDTAF, AttentionSiteDTI, HoTS, CAPLA, MONN, and KIDA (S)

We also provide results from curated external test datasets based on protein or compound structural similarity to evaluate generalization performance.
- low compound similarity, low protein similarity

In addition to results for the compound-protein complex structure dataset (PDBbind), we also provide results for the BindingDB dataset (IC50 and Ki-labeled datasets).
The BindingDB dataset was divided into four cases, and experiments were perfomred using 3-fold cross-validation for each case. The experimental results are provided in the [Dropbox](https://www.dropbox.com/scl/fo/9u9aw7xxjfmk3mnui0rwq/AI7vF-DoqDUM8s6dJR6JOwg?rlkey=5ig47zrxvobz3e3aso7mmww0o&st=w0rgyalx&dl=0)
- Random split: No overlapping interactions between the training and test splits are present; however, compound overlapping (or proteins) could be found. This case simulates a virtual screening scenario in which the model predicts proteins or compounds observed during training.
- New-protein case: No proteins overlapped between the training and test splits. No restrictions were imposed for the compounds. This case simulates a drug-repositioning scenario in which the model predicts unseen proteins.
- New-compound case: No compounds overlapped between the training and test splits. No restrictions were imposed for the proteins. A virtual screening scenario was simulated, in which the model predicts unseen compounds.
- Blind splitting case: No proteins or compounds overlapped between the training and test splits. A virtual screening scenario was simulated, in which the model predicts unseen proteins and compounds. 
- Complex-free models: DeepDTA, GraphDTA, DeepGLSTM, MGraphDTA, BiCompDTA, DeepAffinity, ML-DTI, MolTrans, AttentionDTA, BACPI, FusionDTA, PSG-BAR, DeepDTAF, AttentionSiteDTI, HoTS, CAPLA, and MONN

To evaluate newly proposed models against the baseline models provided on GitHub, we offer not only the experimental results but also all datasets used in the experiments. These datasets can be found in the '/input_data/' directory. However, due to storage limitations, the data is provided via [Dropbox](https://www.dropbox.com/scl/fo/9u9aw7xxjfmk3mnui0rwq/AI7vF-DoqDUM8s6dJR6JOwg?rlkey=5ig47zrxvobz3e3aso7mmww0o&st=w0rgyalx&dl=0) instead of GitHub. Therefore, you will need to replace the '/input_data' directory with the data downloaded from Dropbox. 
- PDBbind (Path: /input_data/PDB/BA/): kfold_indices.pkl, Training_BA_data.tsv, CASF2016_BA_data.tsv, CASF2013_BA_data.tsv, CSAR2014_BA_data.tsv, CSAR2012_BA_data.tsv, CSARset1_BA_data.tsv, CASRset2_BA_data.tsv, Astex_BA_data.tsv, COACH420_IS_data.tsv, and HOLO4K_IS_data.tsv
- BindingDB (Path: /input_data/BindingDB): IC50_data.tsv, Ki_data.tsv, IC50_randdom_split_indices.pkl, IC50_new_protein_indices.pkl, IC50_new_compound_indices.pkl, IC50_blind_split_indices.pkl, Ki_random_split_indices.pkl, Ki_new_protein_indices.pkl, Ki_new_compound_indices.pkl, Ki_blind_split_indices.pkl
  
Results from more recent complex-free models will be continuously updated on our GitHub. We hope to contribute to the advancement of the field of complex-free modeling by providing these results.

## Requirements
python==3.7 \
Pytorch==1.7.1 \
RDKit==2021.03.01 \
Openbabel==2.4.1 \
[ProtTrans](https://github.com/agemagician/ProtTrans)==3.20.1

## (Optional) Preprocessing datasets for BlendNet (T)
BlendNet (T) utilizes compound-protein complex structures collected from PDB databases for training. In order to collect high-quality data, we perform the following preprocessing steps
We provide the preprocessed data via [Dropbox](https://www.dropbox.com/scl/fo/9u9aw7xxjfmk3mnui0rwq/AI7vF-DoqDUM8s6dJR6JOwg?rlkey=5ig47zrxvobz3e3aso7mmww0o&st=w0rgyalx&dl=0), so if preprocessing for new data is not required, you may skip the seven preprocessing steps.
The code for preprocessing is provided in the '/preprocessing_PDB/' directory. The preprocessing results for each step are provided via [Dropbox](https://www.dropbox.com/scl/fo/9u9aw7xxjfmk3mnui0rwq/AI7vF-DoqDUM8s6dJR6JOwg?rlkey=5ig47zrxvobz3e3aso7mmww0o&st=w0rgyalx&dl=0).

#### 1. Download protein and ligand structures
BlendNet (T) utilizes PDBbind, CASF2016, CASF2013, CSAR2014, CSAR2012, CSARset1, CSARset2, Astex, COACH420, and HOLO4K datasets for training and evaluation. Complex structures are collected from the PDB database based on the (PDB ID - Ligand code) of each dataset. Additionally, complexes containing compounds that cannot be analyzed by RDKit or for which IDEAL compound structures cannot be retrieved from PDB (https://files.rcsb.org/ligands) are excluded (see __01.Download protein and ligand structures.ipynb__). 

#### 2. Preprocessing PDB structures
To ensure the collection of high-quality complex structures, those containing compounds that significantly differ from their IDEAL structures are excluded. Complexes containing proteins that cannot be found in the UniProt database or whose pockets are not composed of a unique chain are also excluded to ensure mapping to a single UniProt sequence (see __02. Preprocessing download PDB files.ipynb__). 

#### 3. Run Protein-Ligand Interaction Profiler
BlendNet (T) is trained through supervised learning using interaction site labels. To extract these labels, we utilize the Protein-Ligand Interaction Profiler (PLIP) tool [1]. Complex structures are analyzed using PLIP, and any complexes for which analysis results cannot be obtained are excluded (see __03. Run PLIP.ipynb__). 

#### 4. Mappling PLIP results
The PLIP results collected in the previous step are mapped onto the compound-protein complex structures. The compound atoms involved in non-covalent interactions were mapped to IDEAL compound structures collected from the RCSB PDB (https://files.rcsb.org/ligands) based on their unique names and indices (see __04. Mapping PLIP results.ipynb__). 

#### 5. Mapping PLIP results for UniProt sequences
The protein sequences in the PDB structure were mapped to UniProt sequences using a sequence alignment tool [2], and complexes with less than 90% identity were discarded. The residues involved in the non-covalent interactions extracted by PLIP were then mapped to the UniProt sequences based on the alignment results (see __05. Mapping PLIP results for Uniprot sequences.ipynb__). 

#### 6. Extract binding sites
As the final step in collecting high-quality data, only complexes where at least one amino acid residue is present within 4 Å of each atom of the compound are selected. As with the PLIP results, binding sites extracted from the PDB structures are mapped to UniProt sequences. Complexes containing proteins that cannot be retrieved in the UniProt database or fail to map are excluded (see __06.1 Extract binding sites I.ipynb__ and __06.1 Extract binding sites II.ipynb__). 

#### 7. Get final datasets
All data collected through step 6 are integrated to construct the training dataset. To prevent data duplication, complexes from all external test datasets are excluded from the training dataset based on their (PDB ID - Ligand code) (see __07. Get final data.ipynb__). 

## (Optional) Preprocessing datasets for BlendNet (S)
BlendNet (S) was trained on the BindingDB database, which provides compound-protein interactions collected from the literature. The related data are also available via [Dropbox](https://www.dropbox.com/scl/fo/9u9aw7xxjfmk3mnui0rwq/AI7vF-DoqDUM8s6dJR6JOwg?rlkey=5ig47zrxvobz3e3aso7mmww0o&st=w0rgyalx&dl=0), so if new data are not required, you may skip the corresponding preprocessing steps. The code for preprocessing is provided in the '/preprocessing_BindingDB/' directory. The preprocessing results for each step are provided via [Dropbox](https://www.dropbox.com/scl/fo/9u9aw7xxjfmk3mnui0rwq/AI7vF-DoqDUM8s6dJR6JOwg?rlkey=5ig47zrxvobz3e3aso7mmww0o&st=w0rgyalx&dl=0).

#### 1. Preprocessing BindingDB dataset
Interactions that violate the following seven rules are excluded from the BindingDB v.2023 data. \
1. the protein must consist of only a single chain
2. the UniProt ID and PubChem CID must be searchable
3. the length of the protein sequence must not exceed 1,500 based on the UniProt sequence
4. the protein sequence must be a complete sample (e.g., exclude if the amino acid sequence contains an ‘X’)
5. each affinity value must be an exact number, not a range or an approximation
6. if there are multiple affinity values, the minimum and maximum values must not differ by more than a factor of 100.

The interactions that pass the above rules are divided into IC50 and Ki-labeled datasets (see __01. Preprocessing raw BindingDB dataset.ipynb__). 

#### 2. Preprocessing compound data
From the IC50 and Ki-labeled datasets collected in the previous step, interactions that violate the following three rules are excluded.
1. the compound must be analyzable with RDKit
2. the compound must not be an ion or a single atom
3. the SMILES of the compound must not exceed 150 in length

As a result of the preprocessing, the IC50-labeled and Ki-labeled datasets contain 837,155 and 328,122 interactions, respectively. 
Compound structure data for graph generation are collected from the PubChem database (see __02. Preprocessing compound data.ipynb__). 

## Training BlendNet
The BlendNet framework predicts binding affinity by utilizing compound graphs and predicted protein binding pockets. The predicted binding pockets are extracted using the state-of-the-art sequence-based binding pocket prediction model, Pseq2Sites [3]. Since the original model was trained to work on PDB protein structures, we have retrained it to enable predictions from UniProt sequences. As with the experimental data and results, we provide the trained weights for all modules via [Dropbox](https://www.dropbox.com/scl/fo/9u9aw7xxjfmk3mnui0rwq/AI7vF-DoqDUM8s6dJR6JOwg?rlkey=5ig47zrxvobz3e3aso7mmww0o&st=w0rgyalx&dl=0). Therefore, if you only wish to obtain prediction results for new compound-protein interactions, you can skip the training process and refer to the Test section.

### 1. Training pocket extractor
To train Pseq2Sites, input data is first extracted from the collected protein sequences (see __01. Get protein features for pocket prediction training.ipynb__). \
Pseq2Sites was trained using 5 repeated experiments, and pocket extraction was performed using the model that exhibited the best performance based on the validation dataset (see __01.Training_pocket_extractor.py__).

### 2. Pretraining compound encoder
BlendNet employs Principal Neighborhood Aggregation for Graph Nets (PNA) [4], a type of MPNN, as the compound encoder. To extract more informative compound feature vectors, we have adopted the atom-level Masked Atom Modeling (MAM) and the graph-level triplet-masked contrastive learning (TMCL) techniques proposed by Jun et al [5] as pre-training tasks (see __02.Compound_VQVAE.py__ and __03.Pretraining_compound_encoder.py__).

The molecular graphs for training the compound encoder are generated using the script __01. Get compound features for pretraining.ipynb__.

### 3. PDBbind training 
The training data from the PDBbind dataset is generated using the following three scripts.
- Protein: __02. Get protein features for PDB.ipynb__
- Compound: __02. Get compound features for PDB.ipynb__
- Pocket: __01. Pocket prediction for PDB.ipynb__

The training is conducted in a 5-fold cross-validation setting (see __04.PDBbind_training.py__).

### 4. BindingDB training
The training data from the BindingDB dataset is generated using the following three scripts.
- Protein: __03. Get protein features for BindingDB.ipynb__
- Compound: __03. Get compound features of BindingDB.ipynb__
- Pocket: __02. Pocket preidiction for BindingDB.ipynb__

The training is conducted in a 3-fold cross-validation setting (see __07.BindingDB_IC50_training.py__ and __08.BindingDB_Ki_training.py__).

## Test
To perform testing using the trained model weights, you need to download the data provided via [Dropbox](https://www.dropbox.com/scl/fo/9u9aw7xxjfmk3mnui0rwq/AI7vF-DoqDUM8s6dJR6JOwg?rlkey=5ig47zrxvobz3e3aso7mmww0o&st=w0rgyalx&dl=0) into the 'model_checkpoint' directory.

### 1. Complex structure datasets test
Binding affinity predictions are performed on all external test datasets, including the cross-validation datasets. For binding affinity prediction, protein sequence features, predicted binding pocket sequences, and compound molecular graphs are required (see __05.PDBbind_test.py__).

### 2. Interaction sites test
Interaction sites are predicted using the model trained on the complex structure dataset. For the prediction, protein sequence features, predicted binding pocket sequences, and compound molecular graphs are required (see __06.Interaction_site_prediction.py__).

### 3. Complex-free datasets test
Binding affinity is predicted using the model trained on the complex-free datasets. For the prediction, protein sequence features, predicted binding pocket sequences, and compound molecular graphs are required (see __09.BindingDB_test.py__).


[1] Salentin, S., Schreiber, S., Haupt, V. J., Adasme, M. F., & Schroeder, M. (2015). PLIP: fully automated protein–ligand interaction profiler. Nucleic acids research, 43(W1), W443-W447. \
[2] Zhao, M., Lee, W. P., Garrison, E. P., & Marth, G. T. (2013). SSW library: an SIMD Smith-Waterman C/C++ library for use in genomic applications. PloS one, 8(12), e82138. \
[3] Chen, T., Zhang, Y., & Chatterjee, P. (2024). moPPIt: De Novo Generation of Motif-Specific Binders with Protein Language Models. bioRxiv, 2024-07. \
[4] Corso, G., Cavalleri, L., Beaini, D., Liò, P., & Veličković, P. (2020). Principal neighbourhood aggregation for graph nets. Advances in Neural Information Processing Systems, 33, 13260-13271. \
[5] Xia, J., Zhao, C., Hu, B., Gao, Z., Tan, C., Liu, Y., ... & Li, S. Z. (2023). Mole-bert: Rethinking pre-training graph neural networks for molecules.
