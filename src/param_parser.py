"""Parameter parsing."""

import argparse

def parameter_parser():
	"""
	Parsing the parameters from the command line.
	"""
	parser=argparse.ArgumentParser(description="Run ADE")

	parser.add_argument('--dataset',
						nargs='+',
						type = str,
						default='Cora',
						help='Dataset of graph. {Cora, PubMed, CiteSeer, BlogCatalog, Youtube, \
							  Flickr, FacebookPagePage, WikipediaCrocodile, LastFMAsia}.')

	parser.add_argument('--method',
						nargs='+',
						default='RWR',
						type = str,
						help='Base method of node similarity. {RWR, CN, JC, KI, MTP, MMTP}.\
							  Homogeneous node embedding methods {sociodim, randne, glee, diff2vec,\
							  nodesketch, netmf, boostne, walklets, grarep, deepwalk, node2vec, \
							  nmfadmm, laplacianeigenmaps, graphwave, role2vec}\
							  attributed node embedding methods {feathernode, ae, musae, sine, bane,\
							  tene, tadw, fscnmf, asne}'
						)

	parser.add_argument('--dimensions',
						type=int,
						default=128,
						help='Dimensions of node embeddings.'
						)


	parser.add_argument('--attribute-process',
						nargs='?',
						default='None',
						type = str,
						help='Attributes processing method: {None, A2H, A2H2M, CSA}\
							  None: No attributes.\
							  A2H: attributed networks change to heterogeneous networks.\
							  A2H2M: attributed networks first change to heterogeneous networks, then change to multiplex networks.\
							  CSA: cosine similarity of attributes.' 
						)

	parser.add_argument('--alpha',
						type=float,
						default=0.5,
						help='Probability of random walk jumping to the starting nodes. \
							  Useful for RWR. Defult is 0.5.'
						)

	parser.add_argument('--beta',
						type=float,
						default=1.0,
						help='Connection strength coefficient of the original node and attribute in the extended network.'  
		)

	parser.add_argument('--gamma',
						type=float,
						default=1.0,
						help='Connection strength coefficient between aligned nodes in the extended network.'  
		)

	parser.add_argument('--a',
						type=float,
						default=1.0,
						help='Parameter 1 of the inverse transformation of softmax under constraints.'
						)

	parser.add_argument('--b',
						type=float,
						default=1.0,
						help='Parameter 2 of the inverse transformation of softmax under constraints'
						)

	parser.add_argument('--train-percent',
						nargs='+',
						type=float,
						default=0.8,
						help='train percent of edges. List or float.')

	parser.add_argument('--count',
						type=int,
						default=1,
						help='number of experiments.')

	parser.add_argument('--order',
						type=int,
						default=2,
						help='Order of random walk transition probability matrix.\
							  Length of the longest meta-path.\
							  Useful for MTP or MMTP. Default is 2.')

	parser.add_argument('--walk_number',
						type=int,
						default=10,
						help='Number of random walks'
						)

	parser.add_argument('--step_size',
						type=float,
						default=0.1,
						help='Step size for characteristic function sampling. '
						)
	parser.add_argument('--theta_max',
						type=float,
						default=2.5,
						help='Maximal evaluation point.'
						)

	parser.add_argument('--eval_points',
						type=int,
						default=25,
						help='Number of characteristic function evaluation points. '
						)

	parser.add_argument('--p',
						type=float,
						default=2.0,
						help='Return parameter (1/p transition probability) to move towards from previous node.'
						)

	parser.add_argument('--q',
						type=float,
						default=2.0,
						help='In-out parameter (1/q transition probability) to move away from previous node.'
						)

	parser.add_argument('--classifier',
						type=str,
						default='lr',
						help='Classifier for node classification. {lr, svc}'
						)

	parser.add_argument('--experiments',
						nargs='*',
						default=['node_classification'],
						help='Experiments conducted based on methods. {link_prediction, node_classification, visulization}'
						)

	parser.add_argument('--link_score_method',
						type=str,
						default='cos',
						help='Method for calculating link prediction scores. \
							  cos: cosine similarity.\
							  dot: Inner product of representation vectors or matrices.\
							  origin: similarity or representation matrix, only available for \
									  {rwr(c), cn(c), jc(c), mtp(c)} methods.'
						)

	return parser.parse_args()
