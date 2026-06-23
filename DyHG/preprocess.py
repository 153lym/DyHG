import os
import ot
import pandas as pd
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import scipy.sparse as sp

def permutation(feature):
    # fix_seed(FLAGS.random_seed)
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]

    return feature_permutated 

def construct_hypergraph(adata, n_neighbors=5):
    """
    Constructing spot-to-spot hypergraph.
    Each spot and its k-nearest neighbors form a hyperedge.
    """
    print("Constructing Spatial Hypergraph...")
    position = adata.obsm['spatial']

    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    H = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(n_neighbors + 1):
            y = distance[t]
            H[y, i] = 1.0

    Dv = np.sum(H, axis=1)
    De = np.sum(H, axis=0)

    Dv_inv_sqrt = np.power(Dv, -0.5)
    Dv_inv_sqrt[np.isinf(Dv_inv_sqrt)] = 0.
    Dv_mat_inv_sqrt = sp.diags(Dv_inv_sqrt)

    De_inv = np.power(De, -1.0)
    De_inv[np.isinf(De_inv)] = 0.
    De_mat_inv = sp.diags(De_inv)

    H_sp = sp.coo_matrix(H)
    H_T_sp = H_sp.transpose()

    G = Dv_mat_inv_sqrt.dot(H_sp).dot(De_mat_inv).dot(H_T_sp).dot(Dv_mat_inv_sqrt)

    adata.obsm['XY_adj'] = G.toarray()


def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ]
    adata.obsm['feat'] = feat

def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    
