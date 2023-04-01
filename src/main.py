import numpy as np
import pandas as pd
import networkx as nx
from param_parser import parameter_parser
from utils import tab_printer
from utils import load_graph, load_features
from predictor import Predictor
import csv 


def main(args):
    '''
    Network embedding based on node attribute distribution.
    :param args: Arguments object parsed up.
    
    '''
    print('\n Link prediction experiments.\n')
    
    # method = 'mine'
    # method = 'TP'
    method = 'UTP'
    #method = 'TADW'
    #method = 'TENE'
    #method = 'FeatherNode'
    #method = 'FSCNMF'
    #method = 'MUSAE'
    #method = 'BANE'
    #method = 'DeepWalk'
    #method = 'ki'
    #method = 'kia'
    #method = 'jc'
    #method = 'jca'
    #method = 'rwr'
    #method = 'rwra'
    emb = False
    #emb = True
    #edge_feature = 'hadamard'
    edge_feature = 'cosine_similarity'
    args.graph_input = 'cora'
    args.is_supervised = False
    args.test_percent = 0.1
    args.version = 1
    args.order = 1

    path_graph = '../dataset/node_level/{}/edges.csv'.format(args.graph_input)
    path_attribtue = '../dataset/node_level/{}/features.csv'.format(args.graph_input)
    if method in {'mine', 'TP', 'UTP'}:
        path_result = '../result/{}_{}_{}_{}_order_{}_version_{}_supervised_{}'.format(args.graph_input, args.graph_type, args.test_percent, method, args.order, args.version, args.is_supervised)
    else:
        if emb:
            path_result = '../result/{}_{}_{}_{}_edge_{}_supervised_{}'.format(args.graph_input, args.graph_type, args.test_percent, method, edge_feature, args.is_supervised)
        else:
            path_result = '../result/{}_{}_{}_{}_supervised_{}'.format(args.graph_input, args.graph_type, args.test_percent, method, args.is_supervised)
    
    result_file= open(path_result, 'a', encoding='utf-8', newline='')
    writer = csv.writer(result_file)
    writer.writerow(['method', 'dataset', 'cnt', 'auc', 'ap', 'precision', 'recall', 'f1'])
    result_file.close()

    cnt = 10
    for i in range(cnt):
        graph, map_idx_nodes = load_graph(path_graph, args.graph_type)
        attribute = load_features(path_attribtue)
        Pred = Predictor(args.graph_input, graph, attribute, args.test_percent, args.order, args.version)
        if emb:
            precision, recall, f1, auc, ap = Pred.predict(method=method, is_supervised=args.is_supervised, edge_feature=edge_feature)
        else:
            precision, recall, f1, auc, ap = Pred.predict(method=method, is_supervised=args.is_supervised)
        result_file= open(path_result, 'a', encoding='utf-8', newline='')
        writer = csv.writer(result_file)
        writer.writerow([method, args.graph_input, str(i), str(auc), str(ap), str(auc), str(ap), str(precision), str(recall), str(f1)])
        result_file.close()



if __name__=='__main__':
    args = parameter_parser()
    tab_printer(args)
    main(args)
