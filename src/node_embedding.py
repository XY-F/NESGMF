import networkx as nx
import scipy.sparse as sps
from utils import load_graph, load_features, save_embedding, evaluate_embeddings, KL, cross_entropy
from utils import evaluate_embeddings_nn, TF_IDF, reduce_dimension_svd, softmax_inv,reduce_dimension_nmf
from utils import reduce_dimension_KPCA, spearman, self_cosine_similarity
from karateclub import FSCNMF, TADW, TENE, BANE, MUSAE, FeatherNode, DeepWalk, Node2Vec, GLEE, NodeSketch, NetMF
# from netmf import NetMF
import pandas as pd
import csv
from sklearn.manifold import TSNE
from index import CN, JC, RWR, KI, CNA, ATT, EMB, KIA, RWRA, JCA, TP, ACT, RPR, MTP, CNK
from mine import MINE
from une import UNE
import numpy as np


# dataset = 'citeseer'
# dataset = 'cora' 
dataset = 'lastfm'
# dataset = 'wikipedia'
# dataset = 'pubmed'
# dataset = 'facebook'
#dataset = 'blogcatalog'
#dataset = 'emailEu'
#dataset = 'flickr'
#dataset = 'youtube'
# dataset = 'cornell'
# graph_type = 'direct'
graph_type = 'undirect'
# print('dataset', dataset)

#method = 'FeatherNode'
# method = 'MUSAE'
# method = ''
#method = 'TADW'
# method = 'NetMF'
# method = 'RWR'
# method = 'CNK'
# method = 'DeepWalk'
# method = 'NodeSketch'
# method = 'Node2Vec'
# method = 'MTP'
method = 'KI'
# method = 'JC'
# method = 'CN'
# method = 'MINE'
#method = 'TP'
# method = 'ACT'
# method = 'RPR'
# method = 'RWRA'
	

is_emb = False
# is_emb = True
clf = 'LR'
#clf = 'SVC'
#train_percent = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#train_percent = [0.1]
train_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#train_percent = [0.5, 0.6, 0.7, 0.8, 0.9]
test_perc = 0
cnt = 0
d = 128
m = 0
t = 100
alpha = 0.5
order = 5

path_graph = '../dataset/node_level/{}/edges.csv'.format(dataset)
path_feature = '../dataset/node_level/{}/features.csv'.format(dataset)
#path_embedding = '../output/{}_{}_embedding_d_{}.csv'.format(dataset, method, d)
# path_embedding = '../output/{}_{}_alpha_{}_embedding.csv'.format(dataset, method, alpha)
# path_embedding = '../output/{}_{}_order_{}_embedding.csv'.format(dataset, method, order)
path_embedding = '../output/{}_{}_full_embedding.csv'.format(dataset, method)

path_target = '../dataset/node_level/{}/target.csv'.format(dataset)

#path_result = '../result/{}_{}_node_classify_d_{}.csv'.format(dataset, method, d)
#path_result = '../result/{}_{}_node_classify_d_{}_test.csv'.format(dataset, method, d)
#path_result = '../result/{}_{}_node_classify_d_{}_m_{}_svm.csv'.format(dataset, method, d,m)
#path_result = '../result_0316/{}_{}_node_classify_d_{}_t_{}_svm.csv'.format(dataset, method, d,t)
path_result = '../result/{}_{}_node_classify_d_{}_t_{}_lr.csv'.format(dataset, method, d, t)
#path_result = '../result_0316/{}_{}_node_classify_m_{}_d_{}_t_{}_svm.csv'.format(dataset, method, m, d,t)
#path_result = '../result_0316/{}_{}_node_classify_d_{}_svm.csv'.format(dataset, method, d)
#path_result = '../result_0316/{}_{}_eigen_node_classify_d_{}_svm.csv'.format(dataset, method, d)
#path_result = '../result_0316/{}_{}_eigen_node_classify_d_{}_lr.csv'.format(dataset, method, d)
#path_result = '../result_lr_svc/{}_{}_origin_t_{}_dim_{}_node_classify_{}.csv'.format(dataset, method, t, d, clf)

# node_id = np.array(pd.read_csv(path_target))[:, 0]

graph, map_idx_nodes = load_graph(path_graph, graph_type)



# feature = load_features(path_feature)

if is_emb:
	if method == 'FeatherNode':
		model = FeatherNode(reduction_dimensions=32, eval_points=16, order=2)
	elif method == 'MUSAE':
		model = MUSAE(dimensions=256)
		model.fit(graph, feature)
	elif method == 'BANE':
		model = BANE(dimensions=256)
		model.fit(graph, feature)
	elif method == 'TENE':
		model = TENE(dimensions=128)
		model.fit(graph, feature)
	elif method == 'TADW':
		model = TADW(dimensions=128)
		model.fit(graph, feature)
	elif method =='FSCNMF':
		model = FSCNMF(dimensions=128)
		model.fit(graph, feature)
	elif method =='NetMF':
		model = NetMF(dimensions=128)
		model.fit(graph)
	elif method =='DeepWalk':
		model = DeepWalk(dimensions=128)
		model.fit(graph)
	elif method =='Node2Vec':
		model = Node2Vec(dimensions=128, p=2, q=2)
		model.fit(graph)
	elif method =='GLEE':
		model = GLEE(dimensions=128)
		model.fit(graph)
	elif method =='NodeSketch':
		model = NodeSketch(dimensions=128)
		model.fit(graph)


	X = model.get_embedding()
	# X = reduce_dimension_nmf(X, dimension=128)
	#X = np.array(pd.read_csv(path_embedding))[:, 1:]
	# X = reduce_dimension_svd(X, dimension=128)
else:
	if method == 'RWR':
		
		model = RWR(graph, alpha=alpha)
	elif method == 'CN':
		model = CN(graph)
	elif method == 'CNK':
		model = CNK(graph)
	elif method == 'JC':
		model = JC(graph)
	elif method == 'KI':
		model = KI(graph)
	elif method == 'TP':
		model = TP(graph)
	elif method == 'MTP':
		model = MTP(graph,order=order)
	elif method == 'RPR':
		alpha = 0.5
		model = RPR(graph, alpha = alpha)
	elif method == 'ACT':
		model = ACT(graph)
	elif method == 'MINE':
		model = MINE(graph, feature)
	elif method == 'UNE':
		model = UNE(graph, feature)
	elif method == 'CNA':
		model = CNA(graph, feature)
	elif method == 'JCA':
		model = JCA(graph, feature)
	elif method == 'KIA':
		model = KIA(graph, feature)
	elif method == 'RWRA':
		model = RWRA(graph, feature)
		
	X = model.get_scores_matrix() 
	#X = model.get_scores_matrix() 
	#X = X - np.sum(X,axis=-1)[:,None]
	#X = np.array(pd.read_csv(path_embedding))[:, 1:]
	#value, vector = sps.linalg.eigs(X, k=d, which='LM', return_eigenvectors=True)
	#X = vector.real
	X = softmax_inv(X, m)
	# X = X.dot(X.T)
	# X -= np.min(X)

	#X = softmax_inv(graph.number_of_nodes(), X, m)

	
	# X -= np.min(X)

	# X = TSNE(n_components=d).fit_transform(X)
	
        #X = reduce_dimension_svd(X, dimension=d)
	# X = reduce_dimension_svd(X, dimension=d)
	# X = reduce_dimension_nmf(X, dimension=128)
	# Y = reduce_dimension_KPCA(feature, dimension=128)
	#Y = reduce_dimension_svd(feature, reduction_dimensions=128)
	# X = reduce_dimension_svd(feature, reduction_dimensions=128)
	# X = np.concatenate((X, Y), axis=-1)

print('X.shape', X.shape)
print('X', X)

nodes = list(graph.nodes())
embedding = {map_idx_nodes[i]: X[i, :] for i in range(len(nodes))}
# embedding = {node_id[i]: X[i,:] for i in range(len(node_id))}

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
