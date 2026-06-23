import torch
import os
import time
from DyHG import DyHG
from metrics import *
from DyHG.utils import clustering

def do_DyHG(datafile, fname, cluster_num, alpha, gamma):
    adata = sc.read_h5ad(datafile)
    model = DyHG.DyHG(adata, device=device, alpha=alpha, gamma=gamma)

    adata = model.train()
    return adata


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training model device on ..., ', device)

    datadir = 'DLPFC/' # dataset path
    filename_list = ['151507', '151508', '151509', '151510', '151669',
                     '151670', '151671', '151672', '151673', '151674',
                     '151675', '151676']
    clusters_num_list = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]
    alpha_list = [float(i) for i in range(11)]
    gamma_list = [float(i) for i in range(11)]
    tools_list = ['mclust']
    out_path = "results_DLPFC/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    local_time = time.localtime(time.time())
    formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", local_time)
    for fname, cluster_num, in zip(filename_list, clusters_num_list):
        run_all_re = []
        for  alpha in alpha_list:
            for  gamma in gamma_list:
                print(fname, alpha, gamma)
                # tracemalloc.start()
                adata = do_DyHG(f'{datadir}/{fname}.h5ad',
                                   fname,
                                   cluster_num, alpha, gamma)
                for tool in tools_list:
                    try:
                        if tool == 'mclust':
                            for radius in [l for l in range(51)]:
                                clustering(adata, cluster_num, radius=radius, method=tool,
                                           refinement=True)
                                # 保存结果
                                adata_not_nan = adata[
                                    np.logical_not(adata.obs['ground_truth'].isna())]  # remove NAN
                                # compute ari
                                ARI = compute_ARI(adata_not_nan, 'ground_truth', f'domain')
                                # compute nmi
                                NMI = compute_NMI(adata_not_nan, 'ground_truth', f'domain')

                                print(fname, tool, alpha, gamma, radius,
                                      ARI, NMI)
                                run_all_re.append([fname, tool, alpha, gamma, radius,
                                                   ARI, NMI])
                        elif tool in ['leiden', 'louvain']:
                            clustering(adata, cluster_num, radius=15, method=tool,
                                       start=0.1, end=2.0,
                                       increment=0.01, refinement=False)
                            adata_not_nan = adata[
                                np.logical_not(adata.obs['ground_truth'].isna())]  # remove NAN
                            # compute ari
                            ARI = compute_ARI(adata_not_nan, 'ground_truth', f'domain')
                            # compute nmi
                            NMI = compute_NMI(adata_not_nan, 'ground_truth', f'domain')
                            print(fname, tool, alpha, gamma, None,
                                  ARI, NMI)
                            run_all_re.append([fname, tool, alpha, gamma, None,
                                               ARI, NMI])
                    except:
                        print(fname, alpha, gamma, tool,
                              'error')
                    pd.DataFrame(run_all_re).to_csv(f'{fname}_{str(formatted_time)}.csv', index=False)

