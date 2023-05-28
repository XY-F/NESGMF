import numpy as np
import networkx as nx
import scipy.sparse as sparse
import pandas as pd
import random 
from utils import cosine_similarity, save_embedding
from utils import min_max_row_normalize, sum_row_normalize, cosine_similarity


def A2H(A, X, beta):
	"""
	Creating adjacency matrix of networks extending attributes as nodes. 
	Attributed Netowrks change to Heterogeneous Networks.

	Args:
		A(Array of Numpy): Adjacency matrix of original nodes.
		X(Array of Numpy): Attributes matrix.

	Return:
		Adacency matrix of extended networks.
	"""
	A1 = np.array(sum_row_normalize(A))
	X1 = min_max_row_normalize(X)
	X1 = np.array(sum_row_normalize(X1))
	Xt = np.array(sum_row_normalize(X1.T))
	A1 = np.hstack([A1, beta*X1])
	A2 = np.hstack([beta*Xt, np.eye(X1.shape[-1])])
	AX = np.vstack([A1, A2])
	AX = np.array(sum_row_normalize(AX))
	return AX


def A2H2M(A, X, beta, gamma):
	"""
	Creating adjacency matrix of multiplex networks extended from heterogenous networks extended from attributed networks.. 
	Attributed Netowrks change to Heterogeneous Networks, then change to Multiplex Networks.

	Args:
		A(Array of Numpy): Adjacency matrix of original nodes.
		X(Array of Numpy): Attributes matrix.

	Return:
		Adacency matrix of extended networks.
	"""
	A = np.array(sum_row_normalize(A))
	Xr = min_max_row_normalize(X)
	Xc = min_max_row_normalize(X.T)
	X = Xr.dot(Xc)
	X += X.T
	X = sum_row_normalize(X)

	I = gamma * np.eye(A.shape[0])
	A1 = np.hstack([A, I])
	A2 = np.hstack([I, X])
	A = np.vstack([A1, A2])
	return A


def select_scores(S, edges):
	"""
	Selecting similarity scores by specific node pairs.

	Args:
		edges(list): List of tuples of node pairs.

	Return:
		List of similarity scroes of node pairs.
	"""
	X_score = []
	for edge1, edge2 in edges:
		X_score.append(S[edge1][edge2])
	return X_score


class CN:
	"""
	Node similarity based on common neighbors.

	Args:
		graph(Graph of Networkx): Graph object created by Networkx.
		attribute(Array of Numpy): Attributes matrix.
		s(float): The strength of links between original nodes and attributes. 
	"""
	def __init__(self, beta=1, gamma=1, attribute=None, multiple=False):
		self.beta = beta
		self.gamma = gamma
		self.attribute = attribute
		self.multiple = multiple

	def get_scores_matrix(self, graph):
		"""
		Calculating similarity matrix.

		Return type:
			Array of Numpy.
		"""
		A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))

		if self.multiple:
			A = A2H2M(A, self.attribute, self.beta, self.gamma)
		elif self.attribute is not None:
			A = A2H(A, self.attribute, self.beta)

		self.S = A.dot(A.T)
		return self.S

	def get_scores(self, edges):
		return select_scores(self.S, edges)


class CNC:
	"""
	Node similarity combining common neighbor and cosine simialrity of attributes of nodes.
	
	"""
	def __init__(self, beta=1, attribute=None):
		self.beta = beta
		self.attribute = attribute

	def get_scores_matrix(self, graph):
		A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes())) 
		S1 = A.dot(A.T).toarray()
		S1 = sum_normalize(S1)
		S2 = cosine_similarity(self.attribute, self.attribute)
		S2 = sum_row_normalize(S2)
		self.S = S1 + self.beta*S2
		return self.S

	def get_scores(self, edges):
		return select_scores(self.S, edges)


class CSA:
	"""
	Node similarity based on Cosine Simialrity of Attributes of nodes.
		
	"""
	def __init__(self, attribute):
		self.attribute = attribute

	def get_scores_matrix(self):
		self.S = cosine_similarity(self.attribute)
		return self.S

	def get_scores(self, edges):
		return select_scores(self.S, edges)


class JC:
	"""
	Node similarity based on Jaccard Index.

	"""
	def __init__(self, beta=1, gamma=1, attribute=None, multiple=False):
		self.beta = beta
		self.gamma = gamma
		self.attribute = attribute
		self.multiple = multiple

	def get_scores_matrix(self, graph):
		"""
		Jaccard Index (x,y) = (N(x) cap N(y)) / (N(x) cup N(y)), 
		where N(x) is the set of neighbor of vertex x.
		
		"""
		A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))

		if self.multiple:
			A = A2H2M(A, self.attribute, self.beta, self.gamma)
		elif self.attribute is not None:
			A = A2H(A, self.attribute, self.beta)

		CN = sparse.coo_matrix(A.dot(A.T))
		AN = sparse.coo_matrix(A + A.T)
		AN.data = np.minimum(AN.data, 1)
		AN = AN.sum(-1).flatten()
		S = np.array(CN / AN)
		self.S = S
		return self.S

	def get_scores(self, edges):
		return select_scores(self.S, edges)


class JCC:
	"""
	Node similarity combining Jaccard Index and cosine simialrity of attributes of nodes.

	"""
	def __init__(self, beta=1, attribute=None):
		self.beta = beta
		self.attribute = attribute

	def get_scores_matrix(self, graph):
		A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
		CN = sparse.coo_matrix(A.dot(A.T))
		AN = sparse.coo_matrix(A + A.T)
		AN.data = np.minimum(AN.data, 1)
		AN = AN.sum(-1).flatten()
		S1 = np.array(CN / AN)
		S1 = sum_row_normalize(S1)
		S2 = cosine_similarity(self.attribute, self.attribute)
		S2 =  sum_row_normalize(S2)
		self.S = S1 + self.beta*S2
		return self.S
		
	def get_scores(self, edges):
		return select_scores(self.S, edges)


class MTP:
	"""
	Node similarity based on Multi-scale Transition Probability.

	Args:
		order(int): The order of the transition probability matrix.
	"""
	def __init__(self, order=2,  beta=1, gamma=1, attribute=None, multiple=False):
		self.order = order
		self.beta = beta
		self.gamma = gamma
		self.attribute = attribute
		self.multiple = multiple
		
	def get_scores_matrix(self, graph):
		A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
		
		if self.multiple:
			A = A2H2M(A, self.attribute, self.beta, self.gamma)
		elif self.attribute is not None:
			A = A2H(A, self.attribute, self.beta)

		A_tilde = sum_row_normalize(A)
		A_pool, A_hat = A_tilde.copy(), A_tilde.copy()

		for _ in range(self.order-1):
			A_tilde = A_tilde.dot(A_hat)
			A_pool += A_tilde
		try:
			self.S = np.array(A_pool.todense())
		except:
			self.S = A_pool
		return self.S

	def get_scores(self, edges):
		return select_scores(self.S, edges)


class MMTP:
	"""
	Node similarity based on Multi-scale Meta-path Transition Probability.

	Args:
		k(int): The longest length of the meta-path.
	"""
	def __init__(self, k=2, s=1, attribute=None):
		self.k = k
		self.s = s
		self.attribute = attribute

	def get_scores_matrix(self, graph):
		A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
		A_tilde = np.array(sum_row_normalize(A))
		A_hat = A_tilde.copy()
		X = self.attribute
		Xr = min_max_row_normalize(X)
		Xc = min_max_row_normalize(X.T)
		X = Xr.dot(Xc)
		X += X.T
		X = sum_row_normalize(X)
		# X = sparse.csr_matrix(X)

		P = A_tilde + self.s * X
		for i in range(self.k-1):
			tmp = A_tilde.dot(X)
			P += self.s * tmp
			for j in range(i-1):
				P += self.s * tmp.dot(A_hat)
			A_tilde = A_tilde.dot(A_hat)
			P += A_tilde
		self.S = P
		return self.S

	def get_scores(self, edges):
		return select_scores(self.S, edges)


class RWR:
	"""
	Node similarity based on Random Walk with Restart.

	Args:
		graph(Graph of Networkx): Graph object created by Networkx.
		alpha(float): The probability of random walk jumping to the starting nodes.
	"""
	def __init__(self,  alpha=0.5,  beta=1,  gamma=1, attribute=None, multiple=False):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.attribute = attribute
		self.multiple = multiple

	def get_scores_matrix(self, graph):
		A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
		
		if self.multiple:
			A = A2H2M(A, self.attribute, self.beta, self.gamma)
		elif self.attribute is not None:
			A = A2H(A, self.attribute, self.beta)

		A_tilde = sum_row_normalize(A)
		S = (1-self.alpha) * np.linalg.inv((np.eye(A_tilde.shape[0]) - self.alpha * A_tilde.T))
		self.S = S + S.T
		return self.S

	def get_scores(self, edges):
		return select_scores(self.S, edges)


class RWRC:
	"""
	Node similarity combining Random Walk with Restart and cosine simialrity of attributes of nodes.

	Args:
		graph(Graph of Networkx): Graph object created by Networkx.
		attribute(Aarray of Numpy): Attribute matrix.
		alpha(float): The probability of random walk jumping to the starting nodes.
	"""
	def __init__(self, alpha=0.5, beta=1, attribute=None):
		self.alpha = alpha
		self.beta = 1
		self.attribute = attribute

	def get_scores_matrix(self, graph):
		A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
		A_tilde = sum_row_normalize(A)
		S1 = (1-self.alpha) * np.linalg.inv((np.eye(A_tilde.shape[0]) - self.alpha * A_tilde.T))
		S1 = S1 + S1.T
		S1 = sum_row_normalize(S1)
		S2 = cosine_similarity(self.attribute, self.attribute)
		S2 =  sum_row_normalize(S2)
		self.S = S1 + self.beta*S2
		return self.S
		
	def get_scores(self, edges):
		return select_scores(self.S, edges)


class KI:
	"""
	Node similarity based on Katz Index.

	Args:
		graph(Graph of Networkx): Graph object created by Networkx.
	"""
	def __init__(self, beta=1, gamma=1, attribute=None,multiple=False):
		self.beta = beta
		self.gamma = gamma
		self.attribute = attribute
		self.multiple = multiple

	def get_scores_matrix(self, graph):
		A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
		
		if self.multiple:
			A = A2H2M(A, self.attribute, self.beta, self.gamma)
		elif self.attribute is not None:
			A = A2H(A, self.attribute, self.beta)
		try:
			value, _ = sparse.linalg.eigsh(A, k=1, which='LM')
		except:
			value, _ = sparse.linalg.eigsh(A.asfptype() , k=1, which='LM')
		epsilon = 1 / (value[0] + 1)
		I = np.eye(A.shape[0])
		try:
			self.S = np.linalg.inv(I - epsilon * A.toarray()) - I
		except:
			self.S = np.linalg.inv(I - epsilon * A) - I
		return self.S

	def get_scores(self, edges):
		return select_scores(self.S, edges)


class KIC:
	"""
	Node similarity combining Katz Index and cosine simialrity of attributes of nodes.

	Args:
		graph(Graph of Networkx): Graph object created by Networkx.
		attribute(Aarray of Numpy): Attribute matrix.
	"""
	def __init__(self,  attribute, alpha=0.5, beta=1):
		self.alpha = alpha
		self.beta = 1
		self.attribute = attribute

	def get_scores_matrix(self, graph):
		A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
		value, _ = sparse.linalg.eigsh(A.asfptype() , k=1, which='LM')
		epsilon = 1 / (value[0] + 1)
		I = np.eye(A.shape[0])
		S1 = np.linalg.inv(I - epsilon * A.toarray()) - I
		S1 = sum_row_normalize(S1)
		S2 = cosine_similarity(self.attribute)
		S2 = sum_row_normalize(S2)
		self.S = S1 + self.beta*S2
		return self.S
		
	def get_scores(self, edges):
		return select_scores(self.S, edges)
