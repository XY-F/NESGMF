import numpy as np
import pandas as pd
import networkx as nx
from utils import load_features, modify_nodes, get_test_edge, link_predict_without_lr, link_predict_with_lr
from utils import normalize_by_row, normalize_by_column, create_D_inverse, create_A_tilde
from utils import load_graph, reduce_dimension_svd, softmax_inv, save_embedding, evaluate_embeddings_nn, evaluate_embeddings
from scipy import sparse
from index import CN, JC, RWR, KI, RPR, MTP
from karateclub import FSCNMF, TADW, TENE, BANE, MUSAE, FeatherNode, DeepWalk, Node2Vec, GLEE, NodeSketch, NetMF
import csv

# dataset = 'citeseer'
# dataset = 'pubmed'
dataset = 'cora'
# dataset = 'lastfm'
# dataset = 'blogcatalog'

graph_type = 'undirect'

method = 'RWR'
# method = 'MTP'
# method = 'KI'
# method = 'JC'
# method = 'CN'
# method = 'RPR'
# method = 'DeepWalk'
# method = 'NetMF'
# method = 'Node2Vec'
# method = 'NodeSketch'

print('method', method)

test_percent = 0.9

clf = 'LR'
#
# train_percent = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# train_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
train_percent = [0.5, 0.6, 0.7, 0.8, 0.9]

cnt = 5
d = 128
m = 100


path_feature = '../dataset/node_level/{}/features.csv'.format(dataset)
path_graph = '../dataset/node_level/{}/edges.csv'.format(dataset)
path_target = '../dataset/node_level/{}/target.csv'.format(dataset)

# path_embedding = '../output/blogcatalog_RWR_embedding.csv'
# path_embedding = '../output/cora_DeepWalk_embedding.csv'
path_embedding = '../output/{}_{}_embedding.csv'.format(dataset, method)

path_result = '../result_a/{}_{}_{}_node_classify.csv'.format(method, dataset, d)
# path_result = '../result/{}_{}_{}_m_{}_link.csv'.format(method, dataset, test_percent, m)

X = load_features(path_feature)
denum = np.sum(X, axis=-1)[:,None]
denum[np.where(denum==0)] = 1
X = X / denum
denum = np.sum(X.T, axis=-1)[:,None]
denum[np.where(denum==0)] = 1
Xt = X.T / denum


graph, map_idx_nodes = load_graph(path_graph, graph_type)


A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes())).toarray()
denum = np.sum(A, axis=-1)[:,None]
denum[np.where(denum==0)] = 1.0
A = A / denum
A1 = np.hstack([A, X])
A2 = np.hstack([Xt, np.eye(X.shape[-1])])
A = np.vstack([A1, A2])
A_tilde = A / np.sum(A, axis=-1)[:,None]

alpha = 0.99
S = (1-alpha) * np.linalg.inv((np.eye(A_tilde.shape[0]) - alpha * A_tilde.T))
print('S.shape', S.shape)
#S = S[:graph.number_of_nodes(),:]
#print('S.shape', S.shape)

X = softmax_inv(graph.number_of_nodes(), S, 100)
X = reduce_dimension_svd(X, dimension=d)

nodes = list(graph.nodes())
embedding = {map_idx_nodes[i]: X[i, :] for i in range(len(nodes))}

save_embedding(X, path_embedding)
print('saved embedding')

result_file = open(path_result, 'a', encoding='utf-8', newline='')
writer = csv.writer(result_file)
writer.writerow(['method', 'dataset', 'cnt', 'train_percent', 'micro-f1', 'macro-f1', 'sampled-f1', 'micro-auc', 'macro-auc', 'weighted-auc', 'sampled-auc', 'acc'])
result_file.close()

for i in range(cnt):
    print('i', i)
    print('dataset', dataset)
    print('method',  method)
    for p in train_percent:
        if clf == 'NN':
            results, Y = evaluate_embeddings_nn(embedding, hidden, epochs, learning_rate, path_target, p, dataset, method)    
        else:
            results = evaluate_embeddings(embedding, path_target, p, clf)
        result_file= open(path_result, 'a', encoding='utf-8', newline='')
        writer = csv.writer(result_file)
        writer.writerow([method, dataset, str(i), str(p), str(results['micro_f1']), str(results['macro_f1']), str(results['weighted_f1']), str(results['samples_f1']), str(results['micro_auc']), 
            str(results['macro_auc']), str(results['weighted_auc']), str(results['samples_auc']), str(results['acc'])])
        result_file.close()


