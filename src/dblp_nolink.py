# 异质节点 转换成 多层连边
# 属性转换成 异质节点
import csv
import networkx as nx
import scipy.sparse as sps
from utils import load_graph, load_features, save_embedding, evaluate_embeddings
from utils import evaluate_embeddings_nn,  reduce_dimension_svd, normalize_by_row_np
from utils import reduce_dimension_KPCA, spearman, self_cosine_similarity,softmax_inv
from karateclub import FSCNMF, TADW, TENE, BANE, MUSAE, FeatherNode
from netmf import NetMF
from index import CN, JC, RWR, KI, CNA, ATT, EMB, KIA, RWRA, JCA, TP, ACT, RPR
from mine import MINE
from une import UNE
import numpy as np



dataset = 'dblp'

graph_type = 'undirect'

print('dataset', dataset)

method = 'RWR'
# method = 'RWRA'
print('method', method)

is_emb = False
# is_emb = True
clf = 'LR'
train_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# train_percent = [0.5, 0.6, 0.7, 0.8, 0.9]

cnt = 10


# path_graph1 = '../dataset/heterogeneous/{}/PAP_edge.csv'.format(dataset)
# path_graph2 = '../dataset/heterogeneous/{}/PSP_edge.csv'.format(dataset)
path_graph1 = '../dataset/heterogeneous/{}/AP.csv'.format(dataset)
path_graph2 = '../dataset/heterogeneous/{}/PC.csv'.format(dataset)
path_graph3 = '../dataset/heterogeneous/{}/PT.csv'.format(dataset)
# path_graph1 = '../dataset/heterogeneous/{}/MAM_edge.csv'.format(dataset)
# path_graph2 = '../dataset/heterogeneous/{}/MDM_edge.csv'.format(dataset)

path_feature = '../dataset/heterogeneous/{}/features.csv'.format(dataset)
path_embedding = '../output/{}_{}.csv'.format(dataset, method)
path_target = '../dataset/heterogeneous/{}/target.csv'.format(dataset)

path_result = '../result/{}_{}_nolink_node_classify.csv'.format(dataset, method)

A_ap = load_features(path_graph1)
A_pc = load_features(path_graph2)
A_pt = load_features(path_graph3)

num_nodes = A_ap.shape[0]

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


# A_ap = np.array(nx.adjacency_matrix(ap).todense())
# A_pc = np.array(nx.adjacency_matrix(pc).todense())
# A_pt = np.array(nx.adjacency_matrix(pt).todense())

A_pa = normalize_by_row_np(A_ap.T)
A_ap = normalize_by_row_np(A_ap)
A_apa = np.array(A_ap.dot(A_pa))
A_apa = A_apa + A_apa.T
A_apa = normalize_by_row_np(A_apa)
# zeros = np.zeros(A_apa.shape)
# zeros[np.where(A_apa!=0)] = 1.0
# A_apa = zeros
print('A_apa.shape', A_apa.shape)

A_cp = normalize_by_row_np(A_pc.T)
A_pc = normalize_by_row_np(A_pc)
A_apc = np.array((A_ap.dot(A_pc)))
A_cpa = np.array(A_cp.dot(A_pa))
A_apcpa = np.array(A_apc.dot(A_cpa))
A_apcpa = A_apcpa + A_apcpa.T
A_apcpa = normalize_by_row_np(A_apcpa)
# zeros = np.zeros(A_apcpa.shape)
# zeros[np.where(A_apcpa!=0)] = 1.0
# A_apcpa = zeros
print('A_apcpa.shape', A_apcpa.shape)

A_tp = normalize_by_row_np(A_pt.T)
A_pt = normalize_by_row_np(A_pt)
A_apt = np.array(A_ap.dot(A_pt))
A_tpa = np.array(A_tp.dot(A_pa))
A_aptpa = np.array(A_apt.dot(A_tpa))
A_aptpa = A_aptpa + A_aptpa.T
A_aptpa = normalize_by_row_np(A_aptpa)
# zeros = np.zeros(A_aptpa.shape)
# zeros[np.where(A_aptpa!=0)] = 1.0
# A_aptpa = zeros
print('A_aptpa.shape', A_aptpa.shape)

E1 = np.eye(num_nodes) / num_nodes
E2 = np.eye(num_features) / num_features

alpha = 0.5

A1 = np.hstack([A_apa, f])
A1 = np.vstack([A1, np.hstack([ft, E2])])
A1_tilde = normalize_by_row_np(A1)

S1 = (1-alpha) * np.linalg.inv((np.eye(A1_tilde.shape[0]) - alpha * A1_tilde.T))
S1 = np.asarray(S1)
S1 = S1 + S1.T

X1 = softmax_inv(S1,0)

A2 = np.hstack([A_apcpa, f])
A2 = np.vstack([A2, np.hstack([ft, E2])])
A2_tilde = normalize_by_row_np(A2)

S2 = (1-alpha) * np.linalg.inv((np.eye(A2_tilde.shape[0]) - alpha * A2_tilde.T))
S2 = np.asarray(S1)
S2 = S2 + S2.T

X2 = softmax_inv(S2,0)

A3 = np.hstack([A_aptpa, f])
A3 = np.vstack([A3, np.hstack([ft, E2])])
A3_tilde = normalize_by_row_np(A3)

S3 = (1-alpha) * np.linalg.inv((np.eye(A3_tilde.shape[0]) - alpha * A3_tilde.T))
S3 = np.asarray(S3)
S3 = S3 + S3.T

X3 = softmax_inv(S3,0)

X = np.hstack([X1, X2, X3])



# A1 = np.hstack([A_apa, E1, E1, f])
# A2 = np.hstack([E1, A_apcpa, E1, f])
# A3 = np.hstack([E1, E1, A_aptpa, f])
# A4 = np.hstack([ft, ft, ft, E2])
# A = np.vstack([A1, A2, A3, A4])

# denum = np.sum(A, axis=-1)[:,None]
# denum[np.where(denum==0)] = 1.0
# A_tilde = A / denum

# # rwr
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

# X = np.hstack([X[:num_nodes, :], X[num_nodes:2*num_nodes,: ], X[2*num_nodes: 3*num_nodes, :]])

X = reduce_dimension_svd(X, dimension=128)

print('X.shape', X.shape)
print('X', X)


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
