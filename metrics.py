from sklearn.metrics import *


def compute_ARI(adata, gt_key, pred_key):
    return adjusted_rand_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_NMI(adata, gt_key, pred_key):
    return normalized_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key])

