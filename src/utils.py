import pandas as pd
from scipy import sparse
import numpy as np
import math
import random
from texttable import Texttable

from sklearn.decomposition import TruncatedSVD, KernelPCA, NMF, PCA
from sklearn.manifold import LocallyLinearEmbedding
import networkx as nx
from classify import Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


np.random.seed(random.randint(1,10000))

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def modify_nodes(graph):
    ''' 
    Making sure type(node) is int and arrange in order.

    '''
    nodes = sorted(list(graph.nodes))
    map_nodes_idx = {nodes[i]: i for i in range(len(nodes))}
    map_idx_nodes = {i:nodes[i] for i in range(len(nodes))}
    edges = [(map_nodes_idx[n1], map_nodes_idx[n2]) for (n1, n2) in graph.edges]
    G = nx.Graph()
    for edge in edges:
        G.add_edge(*edge)
    for j in range(G.number_of_nodes()):
        G.add_edge(j,j)
    return G, map_idx_nodes


def load_graph(graph_path):
    '''
    Reading a NetworkX graph
    :param graph_path: Path to the edge list.
    :return graph: NetworkX object.
    :return map_idx_nodes: dict mapping index to node id.
    '''
    edgelist = pd.read_csv(graph_path)
    source, target = list(edgelist.columns)[0], list(edgelist.columns)[1]
    if len(list(edgelist.columns)) > 2:
        G1 = nx.from_pandas_edgelist(edgelist, source=source, target=target, edge_attr=True, create_using=nx.Graph)
    else:
        G1 = nx.from_pandas_edgelist(edgelist, source=source, target=target, create_using=nx.Graph)
    G1, map_idx_nodes = modify_nodes(G1)
    return G1, map_idx_nodes


def load_features(features_path):
    '''
    Reading the features from drive.
    
    Arg:
        features_path(str): Location of features on drive.
    
    Return
        Attribute matrix(Array of Numpy)
    '''
    feature = pd.read_csv(features_path)
    row_idx = np.array(feature['node_id'])
    column_idx = np.array(feature['feature_id'])
    value = np.array(feature['value'])
    num_nodes = np.max(row_idx)+1
    num_features = np.max(column_idx) + 1
    features = sparse.coo_matrix((value, (row_idx, column_idx)), shape=(num_nodes, num_features)).toarray()
    return features


def save_embedding(X, num_nodes, output_path,  map_idx_nodes=None):
    '''
    Saving the node embedding.
    :param X: Node embedding array.
    :param output_path: Path for saving the node embedding.
    '''
    X = X[:num_nodes, :]
    if map_idx_nodes:
        embedding = np.concatenate((np.array([map_idx_nodes[idx] for idx in range(X.shape[0])])[:,None], X), axis=1)
    else:
        embedding = np.concatenate((np.arange(X.shape[0])[:,None], X), axis=1)
    columns = ["node_id"] + ["x_" + str(x) for x in range(X.shape[-1])]
    embedding = pd.DataFrame(embedding, columns=columns)
    embedding.to_csv(output_path, index=None)


def load_embedding(path_emb):
    embedding = pd.read_csv(path_emb)
    embedding = np.array(embedding)[:,1:]
    embedding = {i: embedding[i] for i in range(embedding.shape[0])}
    return embedding

def reduce_dimension_KPCA_sparse(X, dimension=128, seed=42):
    '''
    Reducing dimension of matrix by Kernel PCA.
    :param X: Node embedding sparse matrix.
    :param embedding_dimensions:
    :param seed: random seed
    :return X_: reduced X
    '''
    model = KernelPCA(n_components=dimension, kernel='cosine',eigen_solver='dense',  random_state=42)
    X = model.fit(X)
    v = model.eigenvalues_
    u = model.eigenvectors_
    X_ = u.dot(np.diag(v))
    return X_


def reduce_dimension_KPCA(X, dimension=128, kernel='cosine', seed=42):
    '''
    Reducing dimension of matrix by Kernel PCA.
    :param X: Node embedding array.
    :param embedding_dimensions: 
    :param seed: random seed
    :return X_: reduced X
    '''
    model = KernelPCA(n_components=dimension, kernel=kernel,eigen_solver='arpack',  random_state=42)
    X = model.fit_transform(X)
    # X = model.fit(X)
    # v = model.eigenvalues_
    # u = model.eigenvectors_
    # X_ = u.dot(np.diag(v))
    return X

def reduce_dimension_PCA(X, dimension=128, seed=42):
    pca = PCA(n_components=dimension)
    X = pca.fit_transform(X)
    return X


def reduce_dimension_svd(X, dimensions=64, svd_iterations=20, seed=42):
    '''
    Reducing dimension of matrix by SVD
    '''
    svd = TruncatedSVD(
            n_components=dimensions,
            n_iter=svd_iterations,
            random_state=seed,
        )
    svd.fit(X)
    X = svd.transform(X)
    return X


def reduce_dimension_nmf(X, dimensions=128, init='random', random_state=0):
    model = NMF(n_components=dimensions, init=init, random_state=random_state)
    W = model.fit_transform(X)
    return W


def read_node_label_from_rows(filepath, skip_head=False):
    with open(filepath, 'r')  as f:
        X = []
        Y = []
        while 1:
            if skip_head:
                f.readline()
            l = f.readline()
            if l == '':
                break
            vec = l.strip().split(' ')
            X.append(vec[0])
            Y.append(vec[1:])
    return X, Y


def read_node_label_from_csv(filepath):
    t = pd.read_csv(filepath)
    row_idx = np.array(t['id'])
    column_idx = np.array(t['target'])
    target = {}
    for i in range(len(row_idx)):
        try:
            target[row_idx[i]].append(column_idx[i])
        except:
            target[row_idx[i]] = [column_idx[i]]
    X = []
    Y = []
    keys = sorted(list(target.keys()))
    for i in keys:
        X.append(i)
        Y.append(sorted(target[i]))
    return X, Y


def evaluate_embeddings(embedding, path_label, tr_frac, classifier='LR'):
    X, Y = read_node_label_from_csv(path_label)
    # print('X', X)
    # print('type(X)', type(X))
    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
    if classifier.lower() == 'lr':
        clf = LogisticRegression(penalty='l2', solver='liblinear')
    elif classifier.lower() == 'svc':
        clf = SVC(kernel='linear', probability=True)
    clf = Classifier(embedding=embedding, clf=clf)
    results = clf.split_train_evaluate(X, Y, tr_frac)
    return results


def D_inverse(graph):
    index = np.arange(graph.number_of_nodes())
    values = np.array(
        [1.0 / graph.degree[node] for node in range(graph.number_of_nodes())]
    )
    shape = (graph.number_of_nodes(), graph.number_of_nodes())
    D_inv = sparse.coo_matrix((values, (index, index)), shape=shape)
    return D_inv


def row_normalized_A(graph):
    A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()))
    D_inv = D_inverse(graph)
    A_hat = D_inv.dot(A)
    return A_hat, D_inv


def pooling_transition_matrix(graph, order=3):
    A_hat, D_inv = row_normalized_A(graph)
    A_pool, A_tilde = A_hat.copy(), A_hat.copy()
    for _ in range(order - 1):
        A_tilde = sparse.coo_matrix(A_tilde.dot(A_hat))
        A_pool = A_pool + A_tilde
    A_pool = (A_pool / order).toarray() 
    return A_pool


def spearman(X):
    X1 = pd.DataFrame(X).T
    X2 = X1.corr(method='spearman')
    X2 = np.array(X2)
    return X2


def cross_entropy(X):
    # q(x)
    X1 = X.T
    # log(1+q(x))
    X1 = 1 / (X1)
    X1[np.where(X1 == np.inf)] = 0.99
    X1 = np.log(X1)
    # -sum_x p(x) log(1+q(x))
    X2 = X.dot(X1)
    print('X2', X2)
    return X2


def KL(X, num_nodes=0, num_graph=0):
    X1 = X.T
    X1 = 1 / X1
    # print('X1', X1)
    X1[np.where(X1 == np.inf)] = 0
    X2 = X.dot(X1)
    X2[np.where(X2 == 0)] = 1.0
    X2 = np.log(X2)
    X3 = -X.dot(X2)

    if num_nodes != 0:
        X = X3[:num_nodes, : num_nodes]
        for i in range(num_graph - 1):    
            X = np.hstack([X, X3[(i+1)*num_nodes: (i+2)*num_nodes, :]])
    else:
        X = X3
    return X

def softmax_inv(X, a=0, b=1):
    X = sum_row_normalize(X)
    X = (X + X.T)
    X = sum_row_normalize(X)

    X += 1 / (b * X.shape[0])
    X = X / np.sum(X, axis=-1)[:,None]

    logX = np.log(X)

    X = logX + (a - np.sum(logX, axis=-1)[:,None]) / X.shape[0]
 
    return X


def get_test_edge(graph, test_percent=0.1, by_degree=False):
    edges = np.array(list(graph.edges()))
    num_edge = len(edges)
    test_num = int(num_edge*test_percent)
    
    non_edges = np.array(list(nx.non_edges(graph)))
    non_idx =  np.random.choice(len(non_edges), test_num, replace=False)
    negative_edge = non_edges[non_idx]

    idx = np.random.choice(num_edge, test_num, replace=False)
    positive_edge = edges[idx]
    graph.remove_edges_from(positive_edge)

    return graph, np.array(positive_edge), np.array(negative_edge)


def min_max_row_normalize(X):
    """
    Performing min-max row normalization on the matrix.

    Args:
        X(Array of Numpy): Matrix to be processed.

    Return:
        Normalized matrix(Array of Numpy)
    """
    denum = np.max(X, axis=-1) - np.min(X, axis=-1)
    denum[np.where(denum == 0.0)] =  1.0
    return X / denum[:, None]

def sum_row_normalize(X):
    """
    Normalizing the matrix by dividing each row by its sum.
    
    Args:
        X(Array of Numpy, Compressed Sparse Row matrix)

    Return:
        Normalized matrix(Compressed Sparse Row matrix)
        
    """
    if type(X) == np.ndarray:
        denum = np.sum(X, axis=-1)
        denum[np.where(denum == 0.0)] = 1.0
        X = X / denum[:, None]
    elif type(X) == sparse.csr_matrix:
        denum = X.sum(axis=-1)
        denum[np.where(denum == 0.0)] = 1.0
        X = np.array(X /denum)
    else:
        raise Exception('The input type of the function "sum_row_normalize" is not np.ndarray or sparse.csr_matrix. ')
    return X

def cosine_similarity(X1, X2):
    """
    Computing the cosine similarity between the row vectors of X1 and X2.

    Args:
        X1(Array of Numpy)
        X2(Array of Numpy)
    
    Return:
        A matrix(Arrary of Numpy) whose elements are cosine similarities.
    """
    X = X1.dot(X2.T)
    denum = np.linalg.norm(X, axis=-1)
    denum[np.where(denum == 0.0)] = 1.0
    return X / denum[:, None]

def link_predict_without_lr(graph, embedding, test_pos_edge, test_neg_edge, test_percent):
    num_test_pos = len(test_pos_edge)
    num_test_neg = len(test_neg_edge)
    print('num_test_pos', num_test_pos)
    print('num_test_neg', num_test_neg)

    Y_test_p = np.ones(num_test_pos)
    Y_test_n = np.zeros(num_test_neg)
    Y_test = np.concatenate((Y_test_p, Y_test_n))

    test_edge = np.concatenate((test_pos_edge, test_neg_edge), axis=0)

    Y_prob = []
    Y_pred = []
    for edge1, edge2 in test_edge:
        score = embedding[edge1][edge2]
        Y_prob.append(score)
    
    Y_prob = np.array(Y_prob)
    idx_order = np.argsort(-Y_prob)[:num_test_pos]
    Y_pred = np.zeros(Y_prob.shape)
    Y_pred[idx_order] = 1.0
    
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_prob)
    ap = average_precision_score(Y_test, Y_prob)
    return precision, recall, f1, auc, ap


def plot_embedding(embeddings, path_label, path_pic):
    X, Y = read_node_label_from_csv(path_label)

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    ccc= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
    '#7f7f7f', '#bcbd22', '#17becf', '#cf9817', '#1c18f0', '#18f092', '#18f092', '#ed1876', 
    '#436a94', '#ccf58e', '#a7d0db']

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c=ccc[c], label=c)
    plt.legend()
    f = plt.gcf()
    f.savefig(path_pic)
    f.clear()