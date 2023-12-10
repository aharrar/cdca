import numpy as np
import scanpy as sc
from sklearn.impute import SimpleImputer


def make_adata_from_batches(batch1, batch2, batch1_labels, batch2_labels):
    adata = sc.AnnData(X=np.concatenate((batch1, batch2), axis=0))
    adata.obs['batch'] = [1 if i < len(batch1) else 2 for i in range(len(batch1) + len(batch2))]
    labels = np.concatenate((batch1_labels, batch2_labels), axis=0)
    adata.obs['celltype'] = labels
    sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
    adata.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat

    return adata

def assign_labels_to_numbers(labels):
    """
    Assigns a unique number to each distinct string label.

    Args:
        labels (list): List of string labels.

    Returns:
        dict: A dictionary mapping each label to its corresponding number.
    """
    unique_labels = set(labels)
    label_to_number = {label: number for number, label in enumerate(unique_labels)}
    encoding_labels = [label_to_number[label] for label in labels]
    return encoding_labels

def get_batch_from_adata(adata):
    adata1 = adata[adata.obs['batch'] == 1]
    adata2 = adata[adata.obs['batch'] == 2]

    return adata1, adata2


def load_to_adata_shaham_dataset(path_src, path_target, path_src_labels, path_target_labels):
    batch1 = np.loadtxt(path_src, delimiter=',')
    batch2 = np.loadtxt(path_target, delimiter=',')
    batch1_labels = np.loadtxt(path_src_labels, delimiter=',')
    batch2_labels = np.loadtxt(path_target_labels, delimiter=',')
    adata = make_adata_from_batches(batch1, batch2, batch1_labels, batch2_labels)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
    adata.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat

    # sc.pp.scale(adata)

    return adata


def remove_nan(data):
    if np.any(np.isnan(data)):
        my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data = my_imputer.fit_transform(data)

    return data


def pre_process_cytof_data(data):
    return np.log2(1 + data)


def load_and_pre_process_data(src_path, target_path):
    src_data = np.loadtxt(src_path, delimiter=',')
    target_data = np.loadtxt(target_path, delimiter=',')

    src_data = remove_nan(src_data)
    target_data = remove_nan(target_data)

    src_data = pre_process_cytof_data(src_data)
    target_data = pre_process_cytof_data(target_data)

    return src_data, target_data
