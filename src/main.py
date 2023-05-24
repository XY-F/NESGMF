from scipy import sparse
import numpy as np
import pandas as pd
import networkx as nx
from param_parser import parameter_parser
from utils import tab_printer, load_graph, cosine_similarity, sum_row_normalize
from utils import softmax_inv, save_embedding, load_features, get_test_edge, reduce_dimension_svd
from utils import link_predict_without_lr, evaluate_embeddings, load_embedding, plot_embedding
from index import CN, JC, RWR, KI, MTP, MMTP, CNC, JCC, KIC, RWRC, CSA, A2H, A2H2M
import karateclub
import csv 


mine = {'rwr', 'cn', 'jc', 'ki', 'mtp', 'mmtp', 'rwrc', 'cnc', 'jcc', 'kic', 'mtpc'}

homogeneous_node_embedding = {'sociodim', 'randne', 'glee', 'diff2vec', 'nodesketch', 'netmf', 'boostne', 
	'walklets', 'grarep', 'deepwalk', 'node2vec', 'nmfadmm', 'laplacianeigenmaps', 'graphwave', 'role2vec'}

attributed_node_embedding = {'feathernode', 'ae', 'musae', 'sine', 'bane', 'tene', 'tadw', 'fscnmf', 'asne'}


def format_filename(method, attribute_process, data):

	rfile = '{}_{}'.format(data, method)
	
	if method == 'rwr':
		rfile += '_alpha_{}'.format(args.alpha)
	elif method == 'mtp':
		rfile += '_order_{}'.format(args.order)
	elif method == 'mmtp':
		rfile += '_k_{}_s_{}'.format(args.order, args.beta)
	elif method in {'deepwalk', 'node2vec', 'diff2vec', 'walklets','role2vec'}:
		window_size = args.order*2 + 1
		rfile += '_windowsize_{}'.format(window_size)
	elif method == 'node2vec':
		rfile += '_p_{}_q_{}'.format(args.p, args.q)
	elif method == 'nodesketch':
		rfile += 'iterations_{}'.format(args.order)
	elif method in {'graphwave', 'feathernode'}:
		rfile += '_thetamax_{}_evalpoints_{}'.format(args.theta_max, args.eval_points)

	if method == 'mmtp':
		attribute_process = 'a2h'

	if attribute_process == 'a2h':
		rfile += '_A2H_beta_{}'.format(args.beta)
	elif attribute_process == 'a2h2m':
		rfile += '_A2H_M_beta_{}_gamma_{}'.format(args.beta, args.gamma)
	elif attribute_process == 'csa':
		rfile += 'csa'
	
	return rfile


def output_nc_result(results, method, data, path_result, t, c=0):
	print('')
	print('train_percent', t)
	micro_f1, macro_f1 = results['micro_f1'], results['macro_f1']
	weighted_f1, sampled_f1  = results['weighted_f1'], results['samples_f1']
	micro_auc, macro_auc = results['micro_auc'], results['macro_auc']
	weighted_auc, sampled_auc = results['weighted_auc'], results['samples_auc']
	acc = results['acc']

	print('micro_f1:{:.4}   \tmacro_f1:{:.4}   \tweighted_f1:{:.4}   \tsampled_f1:{:.4}'.format(micro_f1, macro_f1, weighted_f1, sampled_f1))
	print('micro_auc:{:.4} \tmacro_auc:{:.4}  \tweighted_auc:{:.4}  \tsampled_auc:{:.4}'.format(micro_auc, macro_auc, weighted_auc, sampled_auc))
	print('acc:{:.4}'.format(acc))
	print('')

	file_nc= open(path_result, 'a', encoding='utf-8', newline='')
	writer = csv.writer(file_nc)
	writer.writerow([method, data, str(c), str(t), str(micro_f1), str(macro_f1), 
		str(weighted_f1), str(sampled_f1), str(micro_auc), str(macro_auc), 
		str(weighted_auc), str(sampled_auc), str(acc)])
	file_nc.close()


def output_lp_result(results, method, data, path_result, t, c=0):
	print('')
	print('train_percent', t)
	precision, recall, f1, auc, ap = results
	print('precision:{:.4}\trecall:{:.4}   \tf1:{:.4}\nauc:{:.4}       \tap:{:.4}'.format(precision, recall, f1, auc, ap))
	print('')

	result_file= open(path_result, 'a', encoding='utf-8', newline='')
	writer = csv.writer(result_file)
	writer.writerow([method, data, str(c), str(1-t), str(auc), str(ap), str(precision), str(recall), str(f1)])
	result_file.close()


def learn_embedding(args, graph, method, attribute_process, model, map_idx_nodes, path_output, path_attribute=None, flg=True):

	global mine, homogeneous_node_embedding, attributed_node_embedding
	num_nodes = graph.number_of_nodes()

	if method in mine:
		X = model.get_scores_matrix(graph)
		X = softmax_inv(X, args.a, args.b)
		if args.dimensions != 0:
			X = reduce_dimension_svd(X, dimensions=args.dimensions)
		if flg:
			save_embedding(X, num_nodes, path_output, map_idx_nodes)
			flg = False
	else:
		if attribute_process in {'a2h', 'a2h2m'}:
			A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
			attribute = load_features(path_attribute)
			if args.attribute_process.lower() == 'a2h':
				A = A2H(A, attribute, args.beta)
			elif args.attribute_process.lower() == 'a2h2m':
				A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
				attribute = load_features(path_attribute)
				A = A2H2M(A, attribute, args.beta, args.gamma)
			A = sparse.coo_matrix(A)
			edges = np.hstack([A.row[:,None], A.col[:,None]])
			graph_new = nx.Graph()
			graph_new.add_edges_from(edges)
			model.fit(graph)
		elif attribute_process == 'none' or 'not applicable':
			graph_new = graph
		else:
			raise Exception('Methd and attribute_process are not mismatching.')
		
		if method in homogeneous_node_embedding:
			model.fit(graph_new)
		elif method in attributed_node_embedding:
			attribute = load_features(path_attribute)
			model.fit(graph_new, sparse.coo_matrix(attribute))
		else:
			raise Exception('Please check that the method is available.')
		X = model.get_embedding()
		if flg:
			save_embedding(X, num_nodes, path_output, map_idx_nodes)
			flg = False
	return X


def link_prediction_score(X, method):
	if method == 'cos':
		S = cosine_similarity(X, X)
	elif method == 'dot':
		S = X.dot(X)
	elif method == 'origin':
		pass
	else:
		raise Exception('Please check method for calculating link prediction scores based on the embedding.')
	return S


def get_model(args, method, attribute_process, path_attribute=None):

	if method == 'mmtp':
		attribute_process = 'a2h'

	if method in mine:
		if attribute_process == 'none':
			if method == 'rwr':
				model = RWR(args.alpha)
			elif method == 'cn':
				model = CN()
			elif method == 'jc':
				model = JC()
			elif method == 'ki':
				model = KI()
			elif method == 'mtp':
				model = MTP(args.order)
			else:
				raise Exception('When attribute_process is None, method must in {RWR, CN, JC, KI, MTP}')

		elif attribute_process == 'a2h':
			attribute = load_features(path_attribute)
			if method == 'rwr':
				model = RWR(alpha=args.alpha, beta=args.beta, attribute=attribute)
			elif method == 'cn':
				model = CN(beta=args.beta, attribute=attribute)
			elif method == 'jc':
				model = JC(beta=args.beta, attribute=attribute)
			elif method == 'ki':
				model = KI(beta=args.beta, attribute=attribute)
			elif method == 'mtp':
				model = MTP(beta=args.beta, attribute=attribute)
			elif method == 'mmtp':
				model = MMTP(k=args.order, s=args.beta, attribute=attribute)
			else:
				raise Exception('When attribute_process is A2H, method must in {RWR, CN, JC, MTP, MMTP}')

		elif attribute_process == 'a2h2m':
			attribute = load_features(path_attribute)
			if method == 'rwr':
				model = RWR(alpha=args.alpha, beta=args.beta, gamma=args.gamma, attribute=attribute, multiple=True)
			elif method == 'cn':
				model = CN(beta=args.beta, gamma=args.gamma, attribute=attribute, multiple=True)
			elif method == 'jc':
				model = JC(beta=args.beta, gamma=args.gamma, attribute=attribute, multiple=True)
			elif method == 'ki':
				model = KI(beta=args.beta, gamma=args.gamma, attribute=attribute, multiple=True)
			elif method == 'mtp':
				model = MTP(beta=args.beta, gamma=args.gamma, attribute=attribute, multiple=True)
			else:
				raise Exception('When attribute_process is A2H2M, method must in {RWR, CN, JC, MTP, MTP}')
		
		elif attribute_process == 'csa':
			attribute = load_features(path_attribute)
			if method == 'rwrc':
				model = RWRC(alpha=args.alpha, beta=args.beta, attribute=attribute)
			elif method == 'cnc':
				model = CNC(beta=args.beta, attribute=attribute)
			elif method == 'jcc':
				model = JCC(beta=args.beta, attribute=attribute)
			elif method == 'mtpc':
				model = MTPC(beta=args.beta, attribute=attribute)
			elif method == 'csac':
				model = CSA(attribute)
			else:
				raise Exception('When attribute_process is CSA, method must in {RWRC, CNC, jCC, MTPC, CSA}')
		else:
			raise Exception('Please check the attribute_process.')
		
	elif method in homogeneous_node_embedding:
		if method == 'sociodim':
			model = karateclub.SocioDim(dimensions=args.dimensions)
		elif method == 'randne':
			model = karateclub.RandNE(dimensions=args.dimensions)
		elif method == 'glee':
			model = karateclub.GLEE(dimensions=args.dimensions)
		elif method == 'diff2vec':
			model = karateclub.Diff2Vec(dimensions=args.dimensions, window_size=args.order)
		elif method == 'nodesketch':
			model = karateclub.NodeSketch(dimensions=args.dimensions, iterations=args.order)
		elif method == 'netmf':
			model = karateclub.NetMF(dimensions=args.dimensions, order=args.order)
		elif method == 'boostne':
			model = karateclub.BoostNE(dimensions=args.dimensions, order=args.order)
		elif method == 'walklets':
			model = karateclub.Walklets(walk_number=args.walk_number, dimensions=args.dimensions, window_size=args.order)
		elif method == 'grarep':
			dimensions = args.dimensions // args.order
			model = karateclub.GraRep(dimensions=args.dimensions, order=args.order)
		elif method == 'deepwalk':
			model = karateclub.DeepWalk(walk_number=args.walk_number, dimensions=args.dimensions, window_size=int(args.order*2+1))
		elif method == 'node2vec':
			model = karateclub.Node2Vec(walk_number=args.walk_number, p=args.p, q=args.q, dimensions=args.dimensions, window_size=int(args.order*2+1))
		elif method == 'nmfadmm':
			dimensions = args.dimensions // args.order
			model = karateclub.NMFADMM(dimensions=dimensions)
		elif method == 'laplacianeigenmaps':
			model = karateclub.LaplacianEigenmaps(dimensions=args.dimensions)
		elif method == 'graphwave':
			model = karateclub.GraphWave(sample_number=args.dimensions, step_size=args.step_size)
		elif method == 'role2vec':
			model = karateclub.Role2Vec(walk_number=args.walk_number, dimensions=args.dimensions, window_size=args.order)
		else:
			raise Exception('Please check the homogeneous_node_embedding method.')

	elif method in attributed_node_embedding:
		if method == 'feathernode':
			dim = int(args.dimensions / args.order / 2)
			model = karateclub.FeatherNode(reduction_dimensions=dim, theta_max=args.theta_max, eval_points=args.eval_points, order=args.order)
		elif method == 'ae':
			window_size = args.order * 2 + 1
			dim = args.dimensions // 2
			model = karateclub.AE(walk_number=args.walk_number, dimensions=dim, window_size=window_size)
		elif method == 'musae':
			window_size = args.order * 2 + 1
			dim = args.dimensions // window_size
			model = karateclub.MUSAE(walk_number=args.walk_number, dimensions=dim, window_size=window_size)
		elif method == 'sine':
			model = karateclub.SINE(walk_number=args.walk_number, dimensions=args.dimensions, window_size=args.order)
		elif method == 'bane':
			model = karateclub.BANE(dimensions=args.dimensions)
		elif method == 'tene':
			dimensions = args.dimensions // 2
			model = karateclub.TENE(dimensions=dimensions)
		elif method == 'tadw':
			dimensions = args.dimensions // 2
			model = karateclub.TADW(dimensions=dimensions)
		elif method == 'fscnmf':
			dimensions = args.dimensions // 2
			model = karateclub.FSCNMF(dimensions=dimensions)
		elif method == 'asne':
			model = karateclub.ASNE(dimensions=args.dimensions)
		else:
			raise Exception('Please check the attributed_node_embedding method.')
	else:
		raise Exception('Please check embedding method.')

	return model


def check_method(method, attribute_process):
	if attribute_process == 'csa':
		if method != 'csa':
			method += 'c'
	return method


def main(args):

	flg = True
	
	if type(args.method) == str:
		args.method = [args.method]

	if type(args.dataset) == str:
		args.dataset = [args.dataset]

	if type(args.train_percent) == float:
		args.train_percent = [args.train_percent]

	for c in range(args.count):
		datasets = args.dataset
		methods = args.method
		for dataset in datasets:
			for method in methods:
				method = method.lower()
				dataset = dataset.lower()
				print(' dataset:{} '.format(dataset).center(50, '*'))
				print(' method:{} '.format(method).center(50, '*'))
				path_graph = '../dataset/node_level/{}/edges.csv'.format(dataset)
				path_target = '../dataset/node_level/{}/target.csv'.format(dataset)
				path_attribute = '../dataset/node_level/{}/features.csv'.format(dataset)

				if method == 'mmtp':
					attribute_process = 'a2h'
				elif method in attributed_node_embedding:
					attribute_process = 'not applicable'
				else:
					attribute_process = args.attribute_process.lower()
				
				print(' attribute_process:{} '.format(attribute_process).center(50, '*'))

				method = check_method(method, attribute_process)

				rfile = format_filename(method, attribute_process, dataset)

				if attribute_process != 'not applicable':
					path_result_lp = '../result/link_prediction/{}_{}_link_prediction.csv'.format(rfile, attribute_process)
					path_result_nc = '../result/node_classification/{}_{}_node_classification.csv'.format(rfile, attribute_process)
				else:
					path_result_lp = '../result/link_prediction/{}_link_prediction.csv'.format(rfile)
					path_result_nc = '../result/node_classification/{}_node_classification.csv'.format(rfile)

				if method in mine:
					rfile += '_sinv_a_{}_b_{}'.format(args.a, args.b)
				
				path_output = '../output/{}_dimension_{}.csv'.format(rfile, args.dimensions)
				path_pic = '../visual/{}_dimension_{}_visualization.png'.format(rfile, args.dimensions)

				model = get_model(args, method, attribute_process, path_attribute)
				
				
				experiments = args.experiments

				for experiment in experiments:
					print('experiment', experiment)
					experiment = experiment.lower()
					if experiment == 'node_classification':
						print('\nNode classification experiments.\n')
						graph, map_idx_nodes = load_graph(path_graph)
						num_nodes = graph.number_of_nodes()

						file_nc = open(path_result_nc, 'a', encoding='utf-8', newline='')
						writer_nc = csv.writer(file_nc)
						writer_nc.writerow(['method', 'dataset', 'cnt', 'train_percent', 'micro-f1', 'macro-f1', 'weighted-f1', 'sampled-f1', 'micro-auc', 'macro-auc', 'weighted-auc', 'sampled-auc', 'acc'])
						file_nc.close()

						X = learn_embedding(args, graph, method, attribute_process, model, map_idx_nodes, path_output, path_attribute, flg)
						
						for t in args.train_percent:
							results = evaluate_embeddings(X, path_target, t, args.classifier)
							output_nc_result(results, method, dataset, path_result_nc, t, c)

					# link prediction
					if experiment == 'link_prediction':
						print('\nLink prediction experiments.\n')
						if method not in mine and args.link_score_method.lower() == 'none':
							print('\nSkip the link prediction experiments, because the calculation of link prediction score can not be origin \
								when method is not in {CN, JC, RWR, KI, MTP, MMTP}')
						else:
							file_lp= open(path_result_lp, 'a', encoding='utf-8', newline='')
							writer_lp = csv.writer(file_lp)
							writer_lp.writerow(['method', 'dataset', 'cnt', 'train_percent', 'auc', 'ap', 'precision', 'recall', 'f1'])
							file_lp.close()
							
							train_percent = args.train_percent[::-1]
							for t in train_percent:
								p = 1 - t
								graph, map_idx_nodes = load_graph(path_graph)
								graph, positive_edge, negative_edge = get_test_edge(graph, p)
								for j in range(graph.number_of_nodes()):
									graph.add_edge(j,j)
								
								X = learn_embedding(args, graph, method, attribute_process, model, map_idx_nodes, path_output, path_attribute, flg)
								S = link_prediction_score(X, args.link_score_method.lower())

								results = link_predict_without_lr(graph, S, positive_edge, negative_edge, p)
								output_lp_result(results, method, dataset, path_result_lp, t, c)

								
					if experiment == 'visualization':
						if c == 0:
							print('\nVisualization.\n')
							if not flg:
								embedding = load_embedding(path_output)
								plot_embedding(embedding, path_target, path_pic)
							else:
								graph, map_idx_nodes = load_graph(path_graph)
								X = learn_embedding(args, graph, method, attribute_process, model, map_idx_nodes, path_output, path_attribute, flg)
								embedding = {map_idx_nodes[i]: X[i, :] for i in range(len(map_idx_nodes))}
								plot_embedding(embedding, path_target, path_pic)

				if flg:
					graph, map_idx_nodes = load_graph(path_graph)
					num_nodes = graph.number_of_nodes()
					X = learn_embedding(args, graph, method, attribute_process, model, map_idx_nodes, path_output, path_attribute, flg)
					print('X.shape', X.shape)

if __name__=='__main__':
	args = parameter_parser()
	
	tab_printer(args)
	main(args)
