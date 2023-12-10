import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from torch import nn

from tqdm import tqdm
class MMD(nn.Module):
    def __init__(self,
                 src,
                 target,
                 target_sample_size=1000,
                 n_neighbors=25,
                 scales=None,
                 weights=None):
        super(MMD, self).__init__()
        if scales is None:
            med_list = torch.zeros(25)
            for i in range(25):
                sample = target[torch.randint(0, target.shape[0] - 1, (target_sample_size,))]
                distance_matrix = torch.cdist(sample, sample)
                sorted, indices = torch.sort(distance_matrix, dim=0)

                # nearest neighbor is the point so we need to exclude it
                med_list[i] = torch.median(sorted[:, 1:n_neighbors])
            med = torch.mean(med_list)

        scales = [med / 2, med, med * 2]  # CyTOF

        # print(scales)
        scales = torch.tensor(scales)
        weights = torch.ones(len(scales))
        self.src = src
        self.target = target
        self.target_sample_size = target_sample_size
        self.kernel = self.RaphyKernel
        self.scales = scales
        self.weights = weights

    def RaphyKernel(self, X, Y):
        # expand dist to a 1xnxm tensor where the 1 is broadcastable
        sQdist = (torch.cdist(X, Y) ** 2).unsqueeze(0)
        scales = self.scales.unsqueeze(-1).unsqueeze(-1)
        weights = self.weights.unsqueeze(-1).unsqueeze(-1)

        return torch.sum(weights * torch.exp(-sQdist / (torch.pow(scales, 2))), 0)

    # Calculate the MMD cost
    def cost(self):
        mmd_list = torch.zeros(25)
        for i in range(25):
            src = self.src[torch.randint(0, self.src.shape[0] - 1, (self.target_sample_size,))]
            target = self.target[torch.randint(0, self.target.shape[0] - 1, (self.target_sample_size,))]
            xx = self.kernel(src, src)
            xy = self.kernel(src, target)
            yy = self.kernel(target, target)
            # calculate the bias MMD estimater (cannot be less than 0)
            MMD = torch.mean(xx) - 2 * torch.mean(xy) + torch.mean(yy)
            mmd_list[i] = torch.sqrt(MMD)

            # return the square root of the MMD because it optimizes better
        return torch.mean(mmd_list)

def silhouette_coeff_ASW(adata, method_use='raw', save_dir='', save_fn='', percent_extract=0.8):
    random.seed(0)
    asw_fscore = []
    asw_bn = []
    asw_bn_sub = []
    asw_ctn = []
    iters = []

    for i in tqdm(range(20)):
        iters.append('iteration_' + str(i + 1))
        rand_cidx = np.random.choice(adata.obs_names, size=int(len(adata.obs_names) * percent_extract), replace=False)
        #         print('nb extracted cells: ',len(rand_cidx))
        adata_ext = adata[rand_cidx, :]
        asw_batch = silhouette_score(adata_ext.obsm["X_pca"], adata_ext.obs['batch'])
        asw_celltype = silhouette_score(adata_ext.obsm["X_pca"], adata_ext.obs['celltype'])
        min_val = -1
        max_val = 1
        asw_batch_norm = (asw_batch - min_val) / (max_val - min_val)
        asw_celltype_norm = (asw_celltype - min_val) / (max_val - min_val)

        fscoreASW = (2 * (1 - asw_batch_norm) * (asw_celltype_norm)) / (1 - asw_batch_norm + asw_celltype_norm)
        asw_fscore.append(fscoreASW)
        asw_bn.append(asw_batch_norm)
        asw_bn_sub.append(1 - asw_batch_norm)
        asw_ctn.append(asw_celltype_norm)

        iters.append('median_value')
        # asw_fscore.append(np.round(np.median(fscoreASW),3))
        # asw_bn.append(np.round(np.median(asw_batch_norm),3))
        # asw_bn_sub.append(np.round(1 - np.median(asw_batch_norm),3))
        # asw_ctn.append(np.round(np.median(asw_celltype_norm),3))
    df = pd.DataFrame({'asw_batch_norm': asw_bn, 'asw_batch_norm_sub': asw_bn_sub,
                       'asw_celltype_norm': asw_ctn, 'fscore': asw_fscore,
                       'method_use': np.repeat(method_use, len(asw_fscore))})
    print(df)
    print('Save output of pca in: ', save_dir)
    print(df.values.shape)
    print(df.keys())
    return df


def eval_mmd(source, target):
    mmd_value = MMD(source, target).cost()

    return mmd_value