import numpy as np
import pandas as pd
import networkx as nx
from utils import normalize_by_row, normalize_by_column, create_D_inverse, create_A_tilde
from scipy import sparse
import csv
from mine import MINE
from index import CN, JC, RWR, KI, CNA, ATT, EMB, KIA, RWRA, JCA, TP
from utp import UTP
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score

random_seed = 100

class Predictor():
	def __init__(self, dataset, graph, attribute, test_percent=0.1, order=1, version=1):
		self.dataset = dataset
		self.graph = graph
		self.attribute = attribute
		self.test_percent = test_percent
		self.order = order
		self.version = version

	def get_test_edge(self):
		# get test_edge
		edges = np.array(list(self.graph.edges()))
		num_edge = len(edges)
		test_num = int(num_edge*self.test_percent)
		self.test_num = test_num

		non_edges = np.array(list(nx.non_edges(self.graph)))
		non_idx =  np.random.choice(len(non_edges), test_num)
		negative_edge = non_edges[non_idx]

		print('test_num', test_num)
		idx = np.random.choice(num_edge, test_num)
		positive_edge = edges[idx]
		
		# get a corrupted network
		self.graph.remove_edges_from(positive_edge)

		print('len(negative_edge)', len(negative_edge))
		print('len(positive_edge)', len(positive_edge)) 
		test_pos_edge = np.array(positive_edge)
		test_neg_edge = np.array(negative_edge)
		self.test_edge = np.concatenate((test_pos_edge, test_neg_edge), axis=0)

		num_test_pos = len(test_pos_edge)
		num_test_neg = len(test_neg_edge)

		self.Y_test = np.concatenate((np.ones(num_test_pos), np.zeros(num_test_neg)))


	def get_train_edge(self):
		# get train_edge for supervised models
		existent_edges = np.asarray(list(self.graph.edges()))
		num_edge = existent_edges.shape[0]
		num_train_pos = num_edge
		num_train_neg = num_edge

		idx = np.random.choice(num_edge, num_train_pos)
		train_pos_edge = existent_edges[idx]

		non_edges = np.asarray(list(nx.non_edges(self.graph)))
		idx = np.random.choice(len(non_edges), num_train_neg)
		train_neg_edge = non_edges[idx]

		self.train_edge = np.concatenate((train_pos_edge, train_neg_edge), axis=0)

		self.Y_train = np.concatenate((np.ones(num_train_pos),np.zeros(num_train_neg)))


	def train_test_scores(self, method, edge_feature):
		self.get_test_edge()
		self.get_train_edge()

		for j in range(self.graph.number_of_nodes()):
			self.graph.add_edge(j,j)

		if method == 'mine':
			model = MINE(self.graph, self.attribute, self.order, self.version)
		elif method == 'TP':
			model = TP(self.graph, self.order)
		elif method == 'UTP':
			model = UTP(self.graph, self.attribute)
		elif method == 'cn':
			model = CN(self.graph)
		elif method == 'cna':
			model = CNA(self.graph, self.attribute)
		elif method == 'jc':
			model = JC(self.graph)
		elif method == 'jca':
			model = JCA(self.graph, self.attribute)
		elif method == 'rwr':
			model = RWR(self.graph)
		elif method == 'rwra':
			model = RWRA(self.graph, self.attribute)
		elif method == 'ki':
			model = KI(self.graph)
		elif method == 'kia':
			model = KIA(self.graph, self.attribute)
		elif method == 'att':
			model = ATT(self.graph, self.attribute)
		else:
			model = EMB(self.graph, self.attribute, self.dataset, method, edge_feature, self.test_percent)
		
		model.get_scores_matrix()
	
		self.X_test = model.get_scores(self.test_edge)
		self.X_train = model.get_scores(self.train_edge)

		self.X_test = np.asarray(self.X_test)
		self.X_train = np.asarray(self.X_train)

	def predict(self, method, is_supervised, edge_feature=None):
		self.train_test_scores(method, edge_feature)
		X_train, X_test, Y_train, Y_test = self.X_train, self.X_test, self.Y_train, self.Y_test
		
		shuffle_idx = np.random.permutation(Y_test.shape[0])
		X_test = X_test[shuffle_idx]
		Y_test = Y_test[shuffle_idx]

		shuffle_idx = np.random.permutation(Y_train.shape[0])
		X_train = X_train[shuffle_idx]
		Y_train = Y_train[shuffle_idx]

		if is_supervised == False:
			print('is_supserved = ', is_supervised)
			idx = np.argsort(-X_test)
			Y_pred = np.zeros(X_test.shape)
			Y_pred[idx[:self.test_num]] = 1
			Y_prob = X_test
			print('Y_test.shape', Y_test.shape)
			print('Y_prob.shape', Y_prob.shape)
			print('Y_pred.shape', Y_pred.shape)
			# print('Y_prob', Y_prob)
			auc = roc_auc_score(Y_test, Y_prob)
			ap = average_precision_score(Y_test, Y_prob)

		else:
			print('is_supserved = ', is_supervised)
			clf = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto', probability=True))
			try:
				clf.fit(X_train[:,None], Y_train)
			except:
				clf.fit(X_train, Y_train)
			try:
				Y_pred = clf.predict(X_test[:,None])
			except:
				Y_pred = clf.predict(X_test)
			try:
				Y_prob = clf.predict_proba(X_test[:,None])
			except:
				Y_prob = clf.predict_proba(X_test)
			auc = roc_auc_score(Y_test, Y_prob[:,1])
			ap = average_precision_score(Y_test, Y_prob[:,1])

		# print('Y_test', Y_test)
		# print('Y_pred', Y_pred)
		precision = precision_score(Y_test, Y_pred)
		recall = recall_score(Y_test, Y_pred)
		f1 = f1_score(Y_test, Y_pred)
		
		print('precision', precision)
		print('recall', recall)
		print('f1', f1)
		print('auc', auc)
		print('ap', ap)
		# return auc, ap
		return precision, recall, f1, auc, ap
