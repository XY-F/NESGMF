"""Parameter parsing."""

import argparse

def parameter_parser():
	"""
	Parsing the parameters from the command line.
	"""
	parser=argparse.ArgumentParser(description="Run ADE")

	parser.add_argument('--graph-input',
						nargs='?',
                                                default='cora',
						help='Input edge list csv.')

	parser.add_argument('--graph-type',
						nargs='?',
						default='undirect',
						help='Type of graph, direct or direct.',
						)

	parser.add_argument('--version',
						nargs='?',
						default=2,
						help='version 1 uses transition probability via meta path "OAO" as proximity between attributes, \
							  version 2 uses cosine similarity as proximity between attributes' 
						)

	parser.add_argument('--test-percent',
						nargs='?',
                                                default=0.9,
						help='test percent of edges')

	parser.add_argument('--is-supervised',
						nargs='?',
                                                default=False,
						help='If "False", predict links having top-k scores, else, train a supervised model to predict missing links.'
						)

	parser.add_argument('--order',
						type=int,
						default=1,
						help='Order of probability matrix powers. Default is 1.')

	return parser.parse_args()
