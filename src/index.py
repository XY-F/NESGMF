import numpy as np
import networkx as nx
import scipy.sparse as sps
import pandas as pd
import random 
from utils import create_A_tilde, normalize_by_row, self_cosine_similarity, save_embedding
from karateclub import FSCNMF, TADW, TENE, BANE, MUSAE, FeatherNode, DeepWalk, SINE
from utils import normalize_by_row, normalize_by_column


class CN():
	def __init__(self, graph):
		self.graph = graph

	def get_scores_matrix(self):
		self.S = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()))
		self.S = sps.csr_matrix(self.S.dot(self.S.T))
		return self.S.toarray()

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2])
		return X_score


class CNK():
	def __init__(self, graph, order=5):
		self.graph = graph
		self.order = order

	def get_scores_matrix(self):
		#A = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()))
		A, D_inv = create_A_tilde(self.graph)
		A1, A11, A2, A22 = A.copy(), A.copy(), A.copy(), A.copy()
		for _ in range(self.order):
			# A11 = sps.csr_matrix(A11.dot(A11.T))
			# A1 += A11
			A22 += sps.csr_matrix(A22.dot(A))
			A2 += A22

		self.S = A2.toarray()
		# self.S = (A11 + A22).toarray()
		return self.S

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2])
		return X_score


class CNA():
	def __init__(self, graph, attribute):
		self.graph = graph
		self.attribute = attribute

	def get_scores_matrix(self):
		A = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()))
		self.S = sps.coo_matrix(A.dot(A.T))
		self.S1 = normalize_by_row(self.S)
		self.S2 = self_cosine_similarity(self.attribute)
		self.S1_norm = (self.S1 - np.min(self.S1)) / (np.max(self.S1) - np.min(self.S1))
		self.S2_norm = (self.S2 - np.min(self.S2)) / (np.max(self.S2) - np.min(self.S2))
		self.S = self.S1_norm + self.S2_norm
		return self.S

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2])
		return X_score

class ATT():
	def __init__(self, graph, attribute):
		self.graph = graph
		self.attribute = attribute

	def get_scores_matrix(self):
		self.S = self_cosine_similarity(self.attribute)

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2])
		return X_score


class JC():
	def __init__(self, graph):
		self.graph = graph

	def get_scores_matrix(self):
		A = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()))
		CN = sps.coo_matrix(A.dot(A.T))
		AN = sps.coo_matrix(A + A.T)
		# print('AN.data', AN.data)
		AN.data = np.minimum(AN.data, 1)
		# print('AN.data', AN.data)
		AN = AN.sum(-1).flatten()
		# print('AN', AN)
		self.S = np.array(CN / AN)
		return self.S

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2])
			# num = np.sum(self.S[edge1] * self.S[edge2])
			# denum = np.sum(np.maximum(1, np.minimum(self.S[edge1] + self.S[edge2], 1)))
			# X_score.append(num / denum)
		return X_score


class JCA():
	def __init__(self, graph, attribute):
		self.graph = graph
		self.attribute = attribute

	def get_scores_matrix(self):
		self.S1 = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes())).toarray()
		self.S2 = self_cosine_similarity(self.attribute)
		self.S1_norm = (self.S1 - np.min(self.S1)) / (np.max(self.S1) - np.min(self.S1))
		self.S2_norm = (self.S2 - np.min(self.S2)) / (np.max(self.S2) - np.min(self.S2))
		return self.S1_norm + self.S2_norm
		
	def get_scores(self, edges):
		
		structure_score = []
		attribute_score = []
		
		for edge1, edge2 in edges:
			num = np.sum(self.S1[edge1] * self.S1[edge2])
			denum = np.sum(np.maximum(1, np.minimum(self.S1[edge1] + self.S1[edge2], 1)))
			structure_score.append(num / denum)
		structure_score = np.asarray(structure_score)
		structure_score = (structure_score - np.min(structure_score)) / np.max(structure_score)
		
		for edge1, edge2 in edges:
			attribute_score.append(self.S2[edge1][edge2])
		attribute_score = np.asarray(attribute_score)
		attribute_score = (attribute_score - np.min(attribute_score)) / np.max(attribute_score)

		X_score = structure_score + attribute_score
		return X_score


class RPR():
	def __init__(self, graph, alpha=0.5):
		self.graph = graph
		self.alpha = 0.5

	def get_scores_matrix(self):
		A_tilde, D_inv = create_A_tilde(self.graph)
		tmp = np.linalg.inv(np.eye(self.graph.number_of_nodes()) - self.alpha * A_tilde)
		self.S = (1-self.alpha) * tmp
		self.S = np.asarray(self.S)
		print('self.S', self.S)
		print('np.where(np.sum(self.S, axis=-1)==0)', np.where(np.sum(self.S, axis=-1)==0))
		return self.S


class MTP():
	def __init__(self, graph, order=2):
		self.graph = graph
		self.order = order

	def get_scores_matrix(self):

		A_tilde, D_inv = create_A_tilde(self.graph)
		A_pool, A_hat = A_tilde.copy(), A_tilde.copy()

		for _ in range(self.order-1):
			A_tilde = sps.coo_matrix(A_tilde.dot(A_hat))
			A_pool += A_tilde
		self.S = np.array(A_pool.todense())

		return self.S

	# def get_scores_matrix(self):

	# 	A = sps.coo_matrix(nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes())))
	# 	print('1')
	# 	A_to = normalize_by_row(A)
	# 	# A_in = sps.csr_matrix(normalize_by_column(A))
	# 	# A_in = normalize_by_row(A.T)
	# 	print('2')
	# 	A_to_k = A_to.copy()
	# 	# A_in_k = A_in.copy()
	# 	print('3')
	# 	A_in_k = normalize_by_row(A_to_k.T)
	# 	A_pool = sps.coo_matrix(A_in_k.dot(A_to_k))

	# 	for _ in range(self.order-2):
	# 		print('i')
	# 		A_to_k = sps.coo_matrix(A_to_k.dot(A_to))
	# 		print('i2')
	# 		A_in_k = normalize_by_row(A_to_k.T)
	# 		print('i3')
	# 		A_pool += sps.coo_matrix(A_in_k.dot(A_to_k))
		
	# 	self.S = np.array(A_pool.todense())
	# 	return self.S



	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2]) 
		return X_score


class TP():
	def __init__(self, graph, order=2):
		self.graph = graph
		self.order = order

	def get_scores_matrix(self):
		A_tilde, D_inv = create_A_tilde(self.graph)
		A_hat = A_tilde.copy()

		for _ in range(self.order):
			A_tilde = sps.coo_matrix(A_tilde.dot(A_hat))
		self.S = A_tilde

		return self.S

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2]) 
		return X_score


class ACT():
	def __init__(self, graph):
		self.graph = graph

	def get_scores_matrix(self):
		A = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()))
		index = np.arange(self.graph.number_of_nodes())
		values = np.array([self.graph.degree[node] for node in range(self.graph.number_of_nodes())])
		shape = (self.graph.number_of_nodes(), self.graph.number_of_nodes())
		D = sps.coo_matrix((values, (index, index)), shape=shape)
		L = np.array((D - A).todense())
		print('L', L)
		ones = np.ones(shape) / self.graph.number_of_nodes()
		print('ones', ones)
		L_plus = np.linalg.inv(L - ones) + ones
		print('L_plus', L_plus)
		L_yy = np.tile(np.diagonal(L_plus), (self.graph.number_of_nodes(), 1))
		print('L_yy.shape', L_yy.shape)
		print('L_yy', L_yy)
		L_xx = L_yy.T
		self.S = 1 / (L_xx + L_yy - 2*L_plus)
		print('self.S', self.S)
		self.S[np.where(self.S == np.inf)] = 0.0
		print('self.S', self.S)
		return self.S

class RWR():
	def __init__(self, graph, alpha=0.5):
		self.graph = graph
		self.alpha = alpha

	def get_scores_matrix(self):
		A_tilde, D_inv  = create_A_tilde(self.graph)
		print('A_tilde', np.array(A_tilde.todense()))
		print('A_tilde.shape', A_tilde.shape)
		print('D_inv.shape', D_inv.shape)
		#self.S = (1-self.alpha) * np.linalg.inv((np.eye(A_tilde.shape[0]) - self.alpha * A_tilde.T)).dot(1/A_tilde.shape[0] * np.ones(A_tilde.shape))
		self.S = (1-self.alpha) * np.linalg.inv((np.eye(A_tilde.shape[0]) - self.alpha * A_tilde.T))
		self.S = np.asarray(self.S)
		A_tilde_prox = 1/self.alpha * (np.eye(A_tilde.shape[0]) - np.linalg.inv(self.S / (1-self.alpha)))
		print('A_tilde_prox', A_tilde_prox)
		self.S = self.S + self.S.T
		print('self.S', self.S)
		return self.S

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2]) 
		return X_score 

class RWRA():
	def __init__(self, graph, attribute, alpha=0.5):
		self.graph = graph
		self.attribute = attribute
		self.alpha = alpha

	def get_scores_matrix(self):
		A_tilde, D_inv  = create_A_tilde(self.graph)
		self.S1 = (1-self.alpha) * np.linalg.inv((np.eye(A_tilde.shape[0]) - self.alpha * A_tilde.T))
		self.S1 = np.asarray(self.S1)
		self.S2 = self_cosine_similarity(self.attribute)
		self.S1_norm = (self.S1 - np.min(self.S1)) / (np.max(self.S1) - np.min(self.S1))
		self.S2_norm = (self.S2 - np.min(self.S2)) / (np.max(self.S2) - np.min(self.S2))
		return self.S1_norm + self.S2_norm
		
	def get_scores(self, edges):
		structure_score = []
		attribute_score = []

		for edge1, edge2 in edges:
			structure_score.append(self.S1[edge1][edge2]) 
		structure_score = np.asarray(structure_score)
		structure_score = (structure_score - np.min(structure_score)) / np.max(structure_score)
		
		for edge1, edge2 in edges:
			attribute_score.append(self.S2[edge1][edge2])
		attribute_score = np.asarray(attribute_score)
		attribute_score = (attribute_score - np.min(attribute_score)) / np.max(attribute_score)

		X_score = structure_score + attribute_score
		return X_score

class KIA():
	def __init__(self, graph, attribute):
		self.graph = graph
		self.attribute = attribute

	def get_scores_matrix(self):
		A = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()))
		A = A.asfptype() 
		value, _ = sps.linalg.eigsh(A, k=1, which='LM')
		A = A.toarray()
		print('lambda', value[0])
		beta = 1 / (value[0] + 1)
		I = np.eye(A.shape[0])
		self.S1 = np.linalg.inv(I - beta * A) - I
		self.S2 = self_cosine_similarity(self.attribute)
		self.S1_norm = (self.S1 - np.min(self.S1)) / (np.max(self.S1) - np.min(self.S1))
		self.S2_norm = (self.S2 - np.min(self.S2)) / (np.max(self.S2) - np.min(self.S2))
		return self.S1_norm + self.S2_norm
		
	def get_scores(self,  edges):
		structure_score = []
		attribute_score = []

		for edge1, edge2 in edges:
			structure_score.append(self.S1[edge1][edge2]) 
		structure_score = np.asarray(structure_score)
		structure_score = (structure_score - np.min(structure_score)) / np.max(structure_score)
		
		for edge1, edge2 in edges:
			attribute_score.append(self.S2[edge1][edge2])
		attribute_score = np.asarray(attribute_score)
		attribute_score = (attribute_score - np.min(attribute_score)) / np.max(attribute_score)

		X_score = structure_score + attribute_score
		return X_score

class KI():
	def __init__(self, graph):
		self.graph = graph

	def get_scores_matrix(self):
		A = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()))
		A = A.asfptype() 
		value, _ = sps.linalg.eigsh(A, k=1, which='LM')
		A = A.toarray()
		print('A', A)
		print('lambda', value[0])
		beta = 1 / (value[0] + 1)
		I = np.eye(A.shape[0])
		self.S = np.linalg.inv(I - beta * A) - I
		A_prox = 1 / beta * (I - np.linalg.inv(self.S + I))
		print('A_prox', A_prox)
		return self.S

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2]) 
		return X_score


class EMB():
	def __init__(self, graph, attribute, dataset, method, edge_feature, test_percent):
		self.graph = graph
		self.attribute = attribute
		self.dataset = dataset
		self.method = method
		self.edge_feature = edge_feature
		self.test_percent = test_percent

	def get_scores_matrix(self):
		if self.method == 'FeatherNode':
			model = FeatherNode(reduction_dimensions=32, eval_points=16, order=2)
		elif self.method == 'SINE':
			model = SINE(dimensions=256)
		elif self.method == 'MUSAE':
			model = MUSAE(dimensions=128)
		elif self.method == 'TADW':
			model = TADW(dimensions=128)
		elif self.method == 'DeepWalk':
			model = DeepWalk(dimensions=256)

		path_emb = '../output/{}_{}_{}_{}.csv'.format(self.dataset, self.method, self.test_percent, random.randint(1,10000))
		if self.method == 'DeepWalk':
			model.fit(self.graph)
		elif self.method == 'SINE' or 'MUSAE':
			model.fit(self.graph, sps.coo_matrix(self.attribute))
		else:
			model.fit(self.graph, self.attribute)

		X = model.get_embedding()
		print('X.shape', X.shape)

		save_embedding(X, path_emb)
		
		if self.edge_feature == 'cosine_similarity':
			denum = np.linalg.norm(X, axis=-1)
			denum[np.where(denum == 0)] = 1.0
			X = X / denum[:,None]
			self.S = X.dot(X.T)
		elif self.edge_feature == 'hadamard':
			self.S = X
		return self.S

	def get_scores(self, edges):
		X_score = []
		if self.edge_feature == 'cosine_similarity':
			for edge1, edge2 in edges:
				X_score.append(self.S[edge1][edge2])
		elif self.edge_feature == 'hadamard':
			for edge1, edge2 in edges:
				X_score.append(self.S[edge1] * self.S[edge2])
		return X_score
