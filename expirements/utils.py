import itertools
import os
import random

from expirements.load import get_batch_from_adata
from plot_data import get_pca_data, plot_scatter


def plot_adata(adata, plot_dir='', embed='', label='celltype', title='before-calibrationp'):
    adata1, adata2 = get_batch_from_adata(adata)
    labels_b1 = adata1.obs[label]
    labels_b2 = adata2.obs[label]
    if embed != '':
        src_pca = get_pca_data(adata1.obsm[embed])
        target_pca = get_pca_data(adata2.obsm[embed])
    else:
        src_pca = get_pca_data(adata1.X)
        target_pca = get_pca_data(adata2.X)

    plot_scatter(src_pca, target_pca, labels_b1, labels_b2, plot_dir=plot_dir, title=title)


def make_combinations_from_config(config):
    param_combinations = list(itertools.product(*config.values()))
    configurations = []
    for params in param_combinations:
        new_config = {key: value for key, value in zip(config.keys(), params)}
        configurations.append(new_config)

    return configurations


def sample_from_space(configurations, num_of_samples):
    list_configurations = random.sample(configurations, num_of_samples)

    for index, config in enumerate(list_configurations):
        experiment_name = f"{config['experiment_name']}_{index}"
        config["experiment_name"] = experiment_name
        config["plots_dir"] = os.path.join(config["plots_dir"], experiment_name)
        config["save_weights"] = os.path.join(config["save_weights"], experiment_name)

    return list_configurations
