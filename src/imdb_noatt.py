# 异质节点 转换成 多层连边
# 属性转换成 异质节点
import csv
import networkx as nx
import scipy.sparse as sps
from utils import load_graph, load_features, save_embedding, evaluate_embeddings
from utils import evaluate_embeddings_nn,  reduce_dimension_svd, normalize_by_row_np
from utils import softmax_inv
from karateclub import FSCNMF, TADW, TENE, BANE, MUSAE, FeatherNode
from netmf import NetMF
from index import CN, JC, RWR, KI, CNA, ATT, EMB, KIA, RWRA, JCA, TP, ACT, RPR
from mine import MINE
from une import UNE
import numpy as np



dataset = 'imdb'

graph_type = 'undirect'

print('dataset', dataset)

method = 'RWR'
# method = 'RWRA'
print('method', method)


clf = 'LR'
train_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# train_percent = [0.5, 0.6, 0.7, 0.8, 0.9]

cnt = 10


# path_graph1 = '../dataset/heterogeneous/{}/PAP_edge.csv'.format(dataset)
# path_graph2 = '../dataset/heterogeneous/{}/PSP_edge.csv'.format(dataset)
# path_graph1 = '../dataset/heterogeneous/{}/AP.csv'.format(dataset)
# path_graph2 = '../dataset/heterogeneous/{}/PC.csv'.format(dataset)
# path_graph3 = '../dataset/heterogeneous/{}/PT.csv'.format(dataset)
path_graph1 = '../dataset/heterogeneous/{}/MAM_edge.csv'.format(dataset)
path_graph2 = '../dataset/heterogeneous/{}/MDM_edge.csv'.format(dataset)

path_feature = '../dataset/heterogeneous/{}/features.csv'.format(dataset)
path_embedding = '../output/{}_{}.csv'.format(dataset, method)
path_target = '../dataset/heterogeneous/{}/target.csv'.format(dataset)

path_result = '../result/{}_{}_noatt_node_classify.csv'.format(dataset, method)

g1,_ = load_graph(path_graph1)
g2,_ = load_graph(path_graph2)

num_nodes = g1.number_of_nodes()

# feature = None
# num_attribute = 0
feature = load_features(path_feature)
num_features = feature.shape[-1]

f = feature
ft = feature.T

denum = np.sum(f, axis=-1)[:,None]
denum[np.where(denum==0)] = 1.0
f = f / denum

denum = np.sum(ft, axis=-1)[:,None]
denum[np.where(denum==0)] = 1.0
ft = ft / denum


A1 = nx.adjacency_matrix(g1, nodelist = range(g1.number_of_nodes())).toarray()
A2 = nx.adjacency_matrix(g2, nodelist = range(g2.number_of_nodes())).toarray()
A1 = normalize_by_row_np(A1)
A2 = normalize_by_row_np(A2)


E1 = np.eye(num_nodes) / num_nodes
E2 = np.eye(num_features) / num_features

alpha = 0.5

# A1 = np.hstack([A_apa, f])
# A1 = np.vstack([A1, np.hstack([ft, E2])])
A1_tilde = normalize_by_row_np(A1)

S1 = (1-alpha) * np.linalg.inv((np.eye(A1_tilde.shape[0]) - alpha * A1_tilde.T))
S1 = np.asarray(S1)
S1 = S1 + S1.T

X1 = softmax_inv(S1,0)

# A2 = np.hstack([A_apcpa, f])
# A2 = np.vstack([A2, np.hstack([ft, E2])])
A2_tilde = normalize_by_row_np(A2)

S2 = (1-alpha) * np.linalg.inv((np.eye(A2_tilde.shape[0]) - alpha * A2_tilde.T))
S2 = np.asarray(S1)
S2 = S2 + S2.T

X2 = softmax_inv(S2,0)

X = np.hstack([X1, X2])

# A1 = np.hstack([A1, E1, f])
# A2 = np.hstack([E1, A2, f])
# A3 = np.hstack([ft, ft, E2])

# A = np.vstack([A1, A2, A3])

# denum = np.sum(A, axis=-1)[:,None]
# denum[np.where(denum==0)] = 1.0
# A_tilde = A / denum

# rwr
# alpha = 0.5
# print('A_tilde.shape', A_tilde.shape)

# S = (1-alpha) * np.linalg.inv((np.eye(A_tilde.shape[0]) - alpha * A_tilde.T))
# S = np.asarray(S)
# S = S + S.T

# ki
# value, _ = sps.linalg.eigs(A_tilde, k=1, which='LM')
# try:
#     value = value.real
# except:
#     pass

# beta = 1 / (value[0] + 1)
# I = np.eye(A_tilde.shape[0])
# S = np.linalg.inv(I - beta * A_tilde) - I
# S = S + S.T

# X = softmax_inv(S)

# X = np.hstack([X[:num_nodes, :], X[num_nodes:2*num_nodes,: ]])

X = reduce_dimension_svd(X, dimension=128)

# print('X.shape', X.shape)
# print('X', X)


embedding = {i: X[i, :] for i in range(X.shape[0])}

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
