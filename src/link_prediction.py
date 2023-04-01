import numpy as np
import pandas as pd
import networkx as nx
from utils import load_features, modify_nodes, get_test_edge, link_predict_without_lr, link_predict_with_lr
from utils import normalize_by_row, normalize_by_column, create_D_inverse, create_A_tilde, softmax_inv
from utils import load_graph, reduce_dimension_svd
from scipy import sparse
from index import CN, JC, RWR, KI, RPR, MTP
from karateclub import FSCNMF, TADW, TENE, BANE, MUSAE, FeatherNode, DeepWalk, Node2Vec, GLEE, NodeSketch, NetMF
import csv

#dataset = 'facebook'
#dataset = 'cora'
#dataset = 'lastfm'
dataset = 'blogcatalog'

graph_type = 'undirect'

method = 'RWR'
#method = 'MTP'
#method = 'KI'
#method = 'JC'
#method = 'CN'
#method = 'RPR'
#method = 'DeepWalk'
#method = 'NetMF'
#method = 'Node2Vec'
#method = 'NodeSketch'

print('method', method)

test_percent = 0.1

cnt = 10

m = 100

path_feature = '../dataset/node_level/{}/features.csv'.format(dataset)
path_graph = '../dataset/node_level/{}/edges.csv'.format(dataset)
path_target = '../dataset/node_level/{}/target.csv'.format(dataset)

# path_embedding = '../output/blogcatalog_RWR_embedding.csv'
# path_embedding = '../output/cora_DeepWalk_embedding.csv'
path_embedding = '../output/{}_{}_embedding.csv'.format(dataset, method)
#path_embedding = '../output/{}_{}_m_{}_embedding.csv'.format(dataset, method, m)

#path_result = '../result/{}_{}_{}_link.csv'.format(method, dataset, test_percent)
#path_result = '../result/{}_{}_{}_nv_2_2_link_norm.csv'.format(method, dataset, test_percent)
#path_result = '../result/{}_{}_{}_m_{}_link.csv'.format(method, dataset, test_percent, m)
#path_result = '../result/{}_{}_{}_m_{}_link_norm.csv'.format(method, dataset, test_percent, m)
#path_result = '../result/{}_{}_{}_m_{}_link_sumnorm.csv'.format(method, dataset, test_percent, m)
#path_result = '../result/{}_{}_{}_m_{}_link_emb.csv'.format(method, dataset, test_percent, m)
path_result = '../result/{}_{}_{}_m_{}_link_emb_norm_dot.csv'.format(method, dataset, test_percent, m)

result_file= open(path_result, 'a', encoding='utf-8', newline='')
writer = csv.writer(result_file)
writer.writerow(['method', 'dataset', 'cnt', 'test_perc', 'auc', 'ap', 'precision', 'recall', 'f1'])
result_file.close()

for c in range(cnt):
    graph, map_idx_nodes = load_graph(path_graph, graph_type)

    graph, positive_edge, negative_edge = get_test_edge(graph, test_percent)
    for j in range(graph.number_of_nodes()):
        graph.add_edge(j,j)

    # similarity score
    if method =='NetMF':
        model = NetMF(dimensions=128)
        model.fit(graph)
    elif method =='DeepWalk':
        model = DeepWalk(dimensions=128)
        model.fit(graph)
    elif method =='Node2Vec':
        model = Node2Vec(dimensions=128,p=2,q=2)
        model.fit(graph)
    elif method =='GLEE':
        model = GLEE(dimensions=128)
        model.fit(graph)
    elif method =='NodeSketch':
        model = NodeSketch(dimensions=128)
        model.fit(graph)

    elif method == 'RWR':
        alpha = 0.5
        model = RWR(graph, alpha=alpha)
    elif method == 'CN':
        model = CN(graph)
    elif method == 'JC':
        model = JC(graph)
    elif method == 'KI':
        model = KI(graph)
    elif method == 'MTP':
        model = MTP(graph)
    elif method == 'RPR':
        alpha = 0.5
        model = RPR(graph, alpha=alpha)

    X = model.get_scores_matrix()
    #X = model.get_embedding()

    X = softmax_inv(graph.number_of_nodes(), X, m)
    #X = reduce_dimension_svd(X, dimension=128)
    
    # X = np.array(pd.read_csv(path_embedding))[:, 1:]

    denum = np.sum(X, axis=-1)[:,None]
    denum[np.where(denum == 0)] = 1.0
    X = X / denum

    #denum = np.linalg.norm(X, axis=-1)[:,None]
    #denum[np.where(denum == 0)] = 1.0
    #X = X / denum

    X = X.dot(X.T)
    # scores = model.get_scores_matrix()

    # denum = np.sum(X, axis=-1)[:,None]
    # denum[np.where(denum == 0)] = 1.0
    # X = X / denum

    X = (X + X.T)

    # denum = np.sum(X, axis=-1)[:,None]
    # denum[np.where(denum == 0)] = 1.0
    # scores = X / denum

    # graph, positive_edge, negative_edge = get_test_edge(graph, test_percent)
    # for j in range(graph.number_of_nodes()):
    #     graph.add_edge(j,j)

    precision, recall, f1, auc, ap = link_predict_without_lr(graph, X, positive_edge, negative_edge, test_percent)
    print('precision', precision)
    print('recall', recall)
    print('f1', f1)
    print('auc', auc)
    print('ap', ap)
        
    result_file= open(path_result, 'a', encoding='utf-8', newline='')
    writer = csv.writer(result_file)
    writer.writerow([method, dataset, str(c), str(test_percent), str(auc), str(ap), str(precision), str(recall), str(f1)])
    result_file.close()


