import scipy.sparse as sps
import numpy as np
import pandas as pd
import networkx as nx
from utils import load_features, modify_nodes, get_test_edge, link_predict_without_lr, link_predict_with_lr
from utils import normalize_by_row, normalize_by_column, create_D_inverse, create_A_tilde
from utils import load_graph
from scipy import sparse
from index import CN, JC, RWR, KI, RPR, MTP
from karateclub import FSCNMF, TADW, TENE, BANE, MUSAE, FeatherNode, DeepWalk, Node2Vec, GLEE, NodeSketch, NetMF
import csv

#dataset = 'citeseer'
#dataset = 'pubmed'
#dataset = 'cora'
#dataset = 'lastfm'
#dataset = 'facebook'
dataset = 'wikipedia'
# dataset = 'blogcatalog'

graph_type = 'undirect'

#method = 'RWR'
#method = 'MTP'
#method = 'KI'
# method = 'JC'
method = 'CN'
# method = 'RPR'
# method = 'DeepWalk'
# method = 'NetMF'
# method = 'Node2Vec'
# method = 'NodeSketch'

print('method', method)

test_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

cnt = 10

m = 100

path_feature = '../dataset/node_level/{}/features.csv'.format(dataset)
path_graph = '../dataset/node_level/{}/edges.csv'.format(dataset)
path_target = '../dataset/node_level/{}/target.csv'.format(dataset)

# path_embedding = '../output/blogcatalog_RWR_embedding.csv'
# path_embedding = '../output/cora_DeepWalk_embedding.csv'
path_embedding = '../output/{}_{}_embedding.csv'.format(dataset, method)

path_result = '../result_a/{}_{}_link.csv'.format(method, dataset)
# path_result = '../result/{}_{}_{}_m_{}_link.csv'.format(method, dataset, test_percent, m)

f = load_features(path_feature)
denum = np.max(f,axis=-1)[:,None] - np.min(f,axis=-1)[:,None]
denum[np.where(denum==0)] = 1.0
f = f / denum
print('min-max')
X = f.copy()
denum = np.sum(X, axis=-1)[:,None]
denum[np.where(denum==0)] = 1
X1 = X / denum
denum = np.sum(X.T, axis=-1)[:,None]
denum[np.where(denum==0)] = 1
Xt = X.T / denum
X = X1.dot(Xt)
X += X.T
denum = np.sum(X, axis=-1)[:,None]
denum[np.where(denum==0)] = 1
X = X / denum


result_file= open(path_result, 'a', encoding='utf-8', newline='')
writer = csv.writer(result_file)
writer.writerow(['method', 'dataset', 'cnt', 'test_perc', 'auc', 'ap', 'precision', 'recall', 'f1'])
result_file.close()

for c in range(cnt):
    for t in test_percent:
        graph, map_idx_nodes = load_graph(path_graph, graph_type)

        graph, positive_edge, negative_edge = get_test_edge(graph, t)

        for j in range(graph.number_of_nodes()):
            graph.add_edge(j,j)


        # rwr
        if method == 'RWR':

            A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes())).toarray()
            denum = np.sum(A, axis=-1)[:,None]
            denum[np.where(denum==0)] = 1.0
            A = A / denum
            A1 = np.hstack([A, X1])
            A2 = np.hstack([Xt, np.eye(X1.shape[-1])])
            A = np.vstack([A1, A2])
            A_tilde = A / np.sum(A, axis=-1)[:,None]

            alpha = 0.5
            S = (1-alpha) * np.linalg.inv((np.eye(A_tilde.shape[0]) - alpha * A_tilde.T))
            print('S.shape', S.shape)
            S = S[:graph.number_of_nodes(), :graph.number_of_nodes()]
            print('S.shape', S.shape)

        # mtp
        if method == 'MTP':
            A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes())).toarray()
            denum = np.sum(A, axis=-1)[:,None]
            denum[np.where(denum==0)] = 1.0
            A_tilde = A / denum
            A_hat = A_tilde.copy()

            order = 5
            print('order', order)
            P = A_tilde + X
            for i in range(order-1):
                tmp = A_tilde.dot(X)
                P += tmp
                for j in range(i-1):
                    P += tmp.dot(A_hat)
                A_tilde = A_tilde.dot(A_hat)
                P += A_tilde
            S = P

        # ki
        if method == 'KI':
            A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes())).toarray()
            denum = np.sum(A, axis=-1)[:,None]
            denum[np.where(denum==0)] = 1.0
            A = A / denum
            A1 = np.hstack([A, X1])
            A2 = np.hstack([Xt, np.eye(X1.shape[-1])])
            A = np.vstack([A1, A2])
            A_tilde = A / np.sum(A, axis=-1)[:,None]

            value, _ = sps.linalg.eigs(sps.coo_matrix(A), k=1, which='LM')
            try:
                beta = 1 / (value[0].real + 1)
            except:
                beta = 1 / (value[0] + 1)
            I = np.eye(A.shape[0])
            S = np.linalg.inv(I - beta * A) - I

        # cn
        if method == 'CN':
            A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes())).toarray()
            denum = np.sum(A, axis=-1)[:,None]
            denum[np.where(denum==0)] = 1.0
            A = A / denum
            A1 = np.hstack([A, X1])
            A2 = np.hstack([Xt, np.eye(X1.shape[-1])])
            
            A = sps.coo_matrix(np.vstack([A1, A2]))
            A_tilde = A / np.sum(A, axis=-1)[:,None]

            S = A.dot(A.T).toarray()


        precision, recall, f1, auc, ap = link_predict_without_lr(graph, S, positive_edge, negative_edge, t)
        print('precision', precision)
        print('recall', recall)
        print('f1', f1)
        print('auc', auc)
        print('ap', ap)
            
        result_file= open(path_result, 'a', encoding='utf-8', newline='')
        writer = csv.writer(result_file)
        writer.writerow([method, dataset, str(c), str(test_percent), str(auc), str(ap), str(precision), str(recall), str(f1)])
        result_file.close()


