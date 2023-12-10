import json
import os
import random
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import pandas as pd
import torch

from metric import silhouette_coeff_ASW
from train_sda import cdca_alignment

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from expirements.load import assign_labels_to_numbers
from expirements.plot_benchmark import plotTSNE, plotUMAP, save_output_csv
from expirements.utils import make_combinations_from_config, sample_from_space, plot_adata
from pre_procesing.train_reduce_dim import pre_processing
# from scDML.scDML.metrics import evaluate_dataset, silhouette_coeff_ASW


data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\batch_effect\dataset5"

# myData1 = pd.read_csv(os.path.join(parent_dir,'b1_exprs.txt/b1_exprs.txt'),header=0, index_col=0, sep='\t')
# myData2 = pd.read_csv(os.path.join(parent_dir,'b2_exprs.txt/b2_exprs.txt'),header=0, index_col=0, sep='\t')
# mySample1 = pd.read_csv(os.path.join(parent_dir,'b1_celltype.txt/b1_celltype.txt'),header=0, index_col=0, sep='\t')
# mySample2 = pd.read_csv(os.path.join(parent_dir,'b2_celltype.txt/b2_celltype.txt'),header=0, index_col=0, sep='\t')
#
# adata1 = sc.AnnData(myData1.values.T)
# adata1.obs_names = myData1.keys()
# adata1.var_names = myData1.index
# adata1.obs['celltype'] = mySample1.loc[adata1.obs_names,['CellType']]
# adata1.obs['batch'] = 1
# adata1.obs['batch'] = adata1.obs['batch'].astype('category')
# adata1.obs['blb'] = 'batch1'
#
# adata2 = sc.AnnData(myData2.values.T)
# adata2.obs_names = myData2.keys()
# adata2.var_names = myData2.index
# adata2.obs['celltype'] = mySample2.loc[adata2.obs_names,['CellType']]
# adata2.obs['batch'] = 2
# adata2.obs['batch'] = adata2.obs['batch'].astype('category')
# adata2.obs['blb'] = 'batch2'
#
# # Combine 2 dataframe to run PCA
# # adata = sc.AnnData(adata1, adata2, batch_key='batch')
# # adata
# adata = sc.AnnData(np.concatenate([adata1.X, adata2.X]))
# adata.obs_names = adata1.obs_names.tolist() + adata2.obs_names.tolist()
# adata.var_names = adata1.var_names.tolist()
# adata.obs['celltype'] = adata1.obs['celltype'].tolist() + adata2.obs['celltype'].tolist()
# adata.obs['batch'] = adata1.obs['batch'].tolist() + adata2.obs['batch'].tolist()
# adata.obs['blb'] = adata1.obs['blb'].tolist() + adata2.obs['blb'].tolist()
#
# start = time.time()
# adata = sc.read_h5ad(os.path.join(parent_dir, 'myTotalData.h5ad'))
# sc.pp.filter_genes(adata, min_cells=10)
# sc.pp.normalize_per_cell(adata)
# sc.pp.log1p(adata)
# zero_columns = np.all(adata.X == 0, axis=0)
# adata.X = adata.X[:, ~zero_columns]
#
# npcs_train = 50
# sc.tl.pca(adata, svd_solver='arpack', n_comps=npcs_train)  # output save to adata.obsm['X_pca']
# adata.write_h5ad(os.path.join(parent_dir, 'myTotalData_scale_with_pca.h5ad'))
parent_dir = Path(r'C:/Users/avrah/PycharmProjects/cdca/expirements')

config = {
    "loss_type":["s&t&u&c"],
    "experiment_name":["mmd+s+t+cdca1"],
    "input_dim": [25],
    "hidden_dim": [200],
    "drop_prob": [0.2],
    "hidden_layers": [10],
    "lr": [0.001,0.0001],  # np.random.uniform(1e-2, 1e-1, size=5).tolist(),  # Nuber of covariates in the data
    "test_size":[0.3],
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2],  # or tune.choice([<list values>])
    "weight_decay": [0.15, 0.25, 0.2],  # or tune.choice([<list values>])
    "batch_size": [600],  # or tune.choice([<list values>])
    "epochs": [150],
    "coef_1": [50, 100, 150, 400, 800, 1000],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset5-benchmark/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset5-benchmark/")]
}

dim_reduce_weights_path = os.path.join(config["save_weights"][0], "dim_reduce")
os.makedirs(dim_reduce_weights_path, exist_ok=True)
configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=5)
# configurations = [configurations[1]]
if __name__ == "__main__":
    adata = sc.read_h5ad(os.path.join(data_dir, 'myTotalData_scale_with_pca.h5ad'))
    adata.obs['celltype'] = np.array(assign_labels_to_numbers(adata.obs['cell_type']))

    adata1 = adata[adata.obs['batch'] == 1, :].copy()
    adata2 = adata[adata.obs['batch'] == 2, :].copy()
    source, target, model_shrinking = pre_processing(adata1.X, adata2.X, num_epochs=20,
                                                              save_weights_path=dim_reduce_weights_path)
    adata1.obsm["dim_reduce"], adata2.obsm["dim_reduce"] = source, target

    for config in configurations:
        os.makedirs(config["save_weights"], exist_ok=True)
        os.makedirs(config["plots_dir"], exist_ok=True)
        # plot_adata(adata, plot_dir=config["plots_dir"],embed='X_pca',label='celltype', title='before-calibrationp')
        #
        # sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)

        with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
            file.write(json.dumps(config))

        silhouette_coeff_ASW(adata).to_csv(
            os.path.join(config["plots_dir"], "ASW_orignal_adata.csv"))
        evaluate_dataset(adata).to_csv(
            os.path.join(config["plots_dir"], "orignal_adata.csv"))  # Set the batch key for each cell

        adata_target_calibrated_src, adata_src_calibrated_target = ber_for_notebook(config, adata1=adata1,
                                                                                    adata2=adata2,
                                                                                    model_shrinking=model_shrinking,
                                                                                    embed='dim_reduce')

        plot_adata(adata_target_calibrated_src,embed='X_pca', plot_dir=config["plots_dir"],
                   title='after-calibration-target_calibrated_src')
        plot_adata(adata_src_calibrated_target,embed='X_pca', plot_dir=config["plots_dir"],
                   title='after-calibration-src_calibrated_target')

        silhouette_coeff_ASW(adata_src_calibrated_target).to_csv(os.path.join(config["plots_dir"],
                                                                              "ASW_adata_src_calibrated_target.csv"))

        silhouette_coeff_ASW(adata_target_calibrated_src).to_csv(os.path.join(config["plots_dir"],
                                                                              "ASW_adata_target_calibrated_src.csv"))

        evaluate_dataset(adata_src_calibrated_target, embed="X_pca").to_csv(
            os.path.join(config["plots_dir"], "adata_src_calibrated_target.csv")) # Set the batch key for each cell

        evaluate_dataset(adata_target_calibrated_src, embed="X_pca").to_csv(
            os.path.join(config["plots_dir"], "adata_target_calibrated_src.csv"))  # Set the batch key for each cell
