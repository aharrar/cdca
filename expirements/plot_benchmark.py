import random

import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as pl

def save_images(base_name, dpi=300, fig_type=".png"):
    output_dir = os.path.dirname(base_name)
    # os.makedirs(base_name,exist_ok=True)
    fn, fe = os.path.splitext(base_name)
    if (fe == ""):
        base_name = base_name + fig_type
    pl.savefig(base_name, dpi=dpi)
    pl.close()


def plotTSNE(adata, color_group, n_pcs=20, perplexity=30, save_filename='tsne', use_repx=False):
    # adata.var_names_make_unique()
    random.seed(42)
    if use_repx:
        sc.tl.tsne(adata, random_state=0, n_pcs=n_pcs, perplexity=perplexity, use_rep='X')
    else:
        sc.tl.tsne(adata, random_state=0, n_pcs=n_pcs, perplexity=perplexity, n_jobs=20)
    sc.pl.tsne(adata, color=color_group, show=False, wspace=.4)
    save_images(save_filename)


def plotUMAP(adata, color_group, save_filename, use_repx=False):
    if use_repx:
        sc.pp.neighbors(adata, use_rep='X')
    else:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)

    sc.tl.umap(adata)
    sc.pl.umap(adata, color=color_group, show=False, wspace=.4)
    save_images(save_filename)


def save_output_csv(adata, save_dir, usecase_name='ours'):
    colnu = []
    for i in range(adata.obsm['X_umap'].shape[1]):
        colnu.append("UMAP" + str(i + 1))
    df = pd.DataFrame(adata.obsm['X_umap'], columns=colnu, index=adata.obs_names)
    df['batch'] = pd.Series(adata.obs['batch'], index=adata.obs_names)
    df['celltype'] = pd.Series(adata.obs['cell_type'], index=adata.obs_names)
    df.to_csv(os.path.join(save_dir, usecase_name + '_umap.csv'))

    # Save output of tsne for visualization
    colnt = []

    for i in range(adata.obsm['X_tsne'].shape[1]):
        colnt.append("tSNE_" + str(i + 1))

    df = pd.DataFrame(adata.obsm['X_tsne'], columns=colnt, index=adata.obs_names)
    df['batch'] = pd.Series(adata.obs['batch'], index=adata.obs_names)
    df['celltype'] = pd.Series(adata.obs['cell_type'], index=adata.obs_names)
    df.to_csv(os.path.join(save_dir, usecase_name + '_tsne.csv'))

    # Save output of pca for evaluation ASW
    colnpc = []
    for i in range(20):
        colnpc.append("X_pca" + str(i + 1))

    df = pd.DataFrame(adata.obsm['X_pca'][:, :20], columns=colnpc, index=adata.obs_names)
    df['batch'] = pd.Series(adata.obs['batch'], index=adata.obs_names)
    df['celltype'] = pd.Series(adata.obs['cell_type'], index=adata.obs_names)
    df.to_csv(os.path.join(save_dir, usecase_name + '_pca.csv'))
