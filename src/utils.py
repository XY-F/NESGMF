import pandas as pd
from scipy import sparse
import numpy as np
import math
import random
from texttable import Texttable
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from keras.callbacks import EarlyStopping, ModelCheckpoint

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
try:
    import cupy as cp
except:
    print('Warning: can not import cupy')
try:
    from net import Net
except:
    print('Warning: can not import net')

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
    if isinstance(graph, nx.DiGraph):
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for edge in edges:
        G.add_edge(*edge)
    for j in range(G.number_of_nodes()):
        G.add_edge(j,j)
    return G, map_idx_nodes


def load_graph(graph_path, graph_type='undirect'):
    '''
    Reading a NetworkX graph
    :param graph_path: Path to the edge list.
    :param graph_type: Type of graph.
    :return graph: NetworkX object.
    :return map_idx_nodes: dict mapping index to node id.
    '''
    # print('graph_type', graph_type)
    edgelist = pd.read_csv(graph_path)
    # print('edgelist', edgelist)
    # print('type(edgelist)', type(edgelist))
    source, target = list(edgelist.columns)[0], list(edgelist.columns)[1]
    if len(list(edgelist.columns)) > 2:
        if graph_type.lower() == 'undirect':
            print('h1')
            # G1 = nx.from_edgelist(edgelist, create_using=nx.Graph)  
            G1 = nx.from_pandas_edgelist(edgelist, source=source, target=target, edge_attr=True, create_using=nx.Graph)
        elif  graph_type.lower() == 'direct':     
            print('h2')
            G1 = nx.from_pandas_edgelist(edgelist, source=source, target=target, edge_attr=True, create_using=nx.DiGraph)
        else:
            raise ValueError('graph_type must be direct or undirect.')
    else:
        if graph_type.lower() == 'undirect':
            print('h1')
            # G1 = nx.from_edgelist(edgelist, create_using=nx.Graph)  
            G1 = nx.from_pandas_edgelist(edgelist, source=source, target=target, create_using=nx.Graph)
        elif  graph_type.lower() == 'direct':     
            print('h2')
            G1 = nx.from_pandas_edgelist(edgelist,source=source, target=target,  create_using=nx.DiGraph)
        else:
            raise ValueError('graph_type must be direct or undirect.')
    G1, map_idx_nodes = modify_nodes(G1)
    print('type(G1)', type(G1))
    return G1, map_idx_nodes


def load_features(features_path):
    '''
    Reading the features from drive.
    :param features_path: Location of features on drive.
    :return features: Features Numpy array
    '''
    feature = pd.read_csv(features_path)
    row_idx = np.array(feature['node_id'])
    column_idx = np.array(feature['feature_id'])
    value = np.array(feature['value'])
    num_nodes = np.max(row_idx)+1
    num_features = np.max(column_idx) + 1
    features = sparse.coo_matrix((value, (row_idx, column_idx)), shape=(num_nodes, num_features)).toarray()
    return features


def save_embedding(X, output_path, map_idx_nodes=None):
    '''
    Saving the node embedding.
    :param X: Node embedding array.
    :param output_path: Path for saving the node embedding.
    '''
    if map_idx_nodes:
        embedding = np.concatenate((np.array([map_idx_nodes[idx] for idx in range(X.shape[0])])[:,None], X), axis=1)
    else:
        embedding = np.concatenate((np.arange(X.shape[0])[:,None], X), axis=1)
    columns = ["node_id"] + ["x_" + str(x) for x in range(X.shape[1])]
    embedding = pd.DataFrame(embedding, columns=columns)
    embedding.to_csv(output_path, index=None)


def load_embedding(path_emb):
    embedding = pd.read_csv(path_emb)
    # print('embedding.head', embedding.head)
    embedding = np.array(embedding)[:,1:]
    # print('embedding[:10]', embedding[:10])
    embedding = {i: embedding[i] for i in range(embedding.shape[0])}
    # print('embedding', embedding)
    return embedding


def min_max_normalize_by_column(X):
    X_min = np.min(X, axis=0).reshape((1, X.shape[-1]))
    X_max = np.max(X, axis=0).reshape((1, X.shape[-1]))
    denum = X_max - X_min
    denum[np.where(denum == 0.0)] = 1.0
    X = (X - X_min) / denum
    return X


def min_max_normalize_by_row( X):
        X_min = np.min(X, axis=-1).reshape((X.shape[0], 1))
        X_max = np.max(X, axis=-1).reshape((X.shape[0], 1))
        denum = X_max - X_min
        denum[np.where(denum == 0.0)] = 1.0
        X = (X - X_min) / denum
        return X

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


def self_cosine_similarity(X):
    denum = np.linalg.norm(X, axis=-1)
    denum[np.where(denum == 0)] = 1.0
    X = X / denum[:,None]
    return np.asarray(X.dot(X.T))

def normalize_by_row_np(X):
    denum = np.sum(X, axis=-1)
    denum[np.where(denum == 0.0)] = 1.0
    X = X / denum[:,None]
    return X

def normalize_by_row(X):
    row_sum = np.array(X.sum(1))
    row_sum[np.where(row_sum==0)] = 1.0
    X = sparse.coo_matrix(X / row_sum)
    return X

def normalize_by_column(X):
    row_sum = np.array(X.sum(0))
    row_sum[np.where(row_sum==0)] = 1.0
    X = sparse.coo_matrix(X / row_sum)
    return X

def create_D_inverse(graph):
    index = np.arange(graph.number_of_nodes())
    if  isinstance(graph, nx.Graph):
        values = np.array([1.0/graph.degree[node] for node in range(graph.number_of_nodes())])
        values2 = np.array([1.0/graph.degree[node] for node in range(graph.number_of_nodes())])
    else:
        values = np.array([1.0/graph.out_degree[node] for node in range(graph.number_of_nodes())])
        values2 = np.array([1.0/graph.out_degree[node] for node in range(graph.number_of_nodes())])
    shape = (graph.number_of_nodes(), graph.number_of_nodes())
    D_inv = sparse.coo_matrix((values, (index, index)), shape=shape)
    D_inv2 = sparse.coo_matrix((values2, (index, index)), shape=shape)
    return D_inv, D_inv2

def create_A_tilde(graph):
    A = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()))
    D_inv, D_inv2 = create_D_inverse(graph)
    A_tilde = sparse.coo_matrix(D_inv.dot(A))
    return A_tilde, D_inv2


def reduce_dimension_svd_cp(X, dimension=128, svd_iterations=20, seed=42):
    ''' 
    reduce dimension of matrix by SVD with cupy
    '''
    X = cp.array(X)
    u, s, v = cp.linalg.svd(X)
    X_ = cp.asnumpy(u[:dimension,:].dot(cp.diag(s[:dimension])))
    #X_ = cp.asnumpy(cp.concatenate((u, v.transpose()), axis=-1))
    return X_


def reduce_dimension_svd(X, dimension=64, svd_iterations=20, seed=42):
    '''
    Reducing dimension of matrix by SVD
    '''
    svd = TruncatedSVD(
            n_components=dimension,
            n_iter=svd_iterations,
            random_state=seed,
        )
    svd.fit(X)
    X = svd.transform(X)
    return X

def reduce_dimension_nmf(X, dimension=128, init='random', random_state=0):
    model = NMF(n_components=dimension, init=init, random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_
    print('W.shape', W.shape)
    print('H.shape', H.shape)
    # X = np.hstack([W, H])
    # print('X', X)
    # print('X.shape', X.shape)
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
    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
    if classifier == 'LR':
        clf = LogisticRegression(penalty='l2', solver='liblinear')
    elif classifier == 'SVC':
        clf = SVC(kernel='linear', probability=True)
    clf = Classifier(embedding=embedding, clf=clf)
    results = clf.split_train_evaluate(X, Y, tr_frac)
    return results


def evaluate_embeddings_nn(features, path_label, tr_frac, hidden, epochs, learning_rate,  dataset='', method='', multi_label=False):
    X, Y = read_node_label_from_csv(path_label)
    print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
    try:
        clf = Net(features, hidden, epochs, learning_rate, dataset=dataset, method=method, multi_label=multi_label)
    except:
        raise AssertionError('Please check the environment to make sure neural network can work.')
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
    # X3 = np.hstack([X3[:num_nodes,:], X3[num_nodes:, :num_nodes].T])
    # print('X3.shape', X3.shape)
    if num_nodes != 0:
        X = X3[:num_nodes, : num_nodes]
        # X2 = X[-num_attribute:, :-num_attribute]
        # X21 = X2[:, : num_nodes]
        # print('X1.shape', X1.shape)
        # print('X11.shape', X11.shape)
        # print('X2.shape', X2.shape)
        # print('X21.shape', X21.shape)
        for i in range(num_graph - 1):    
            X = np.hstack([X, X3[(i+1)*num_nodes: (i+2)*num_nodes, :]])
    else:
        X = X3
    return X

def softmax_inv(X, m=100):
    # X_row = np.sum(X, axis=-1)[:, None]
    # X_col = np.sum(X, axis=0)[None, :]
    # X = X / X_row 
    # X = X / X_col
    
    #if np.min(X) >= 1:
    #########
    # X1 = X

    denum = np.sum(X, axis=-1)[:,None]
    denum[np.where(denum == 0)] = 1.0
    X = X / denum

    # print('X',X)

    # denum = np.sum(X, axis=-1)[None,:]
    # denum[np.where(denum == 0)] = 1.0
    # X1 = X1 / denum


    X = (X + X.T)

    denum = np.sum(X, axis=-1)[:,None]
    denum[np.where(denum == 0)] = 1.0
    X = X / denum

    

    X += 1 / X.shape[0]
    X = X / np.sum(X, axis=-1)[:,None]

    logX = np.log(X)

    X = logX + (m - np.sum(logX, axis=-1)[:,None]) / X.shape[0]
 
    return X


def ppmi_matrix(graph, order=3, negative_samples=1):
    A_hat, D_inv = row_normalized_A(graph)
    A_pool, A_tilde = A_hat.copy(), A_hat.copy()
    for _ in range(order - 1):
        A_tilde = sparse.coo_matrix(A_tilde.dot(A_hat))
        A_pool = A_pool + A_tilde
    
    A_pool = sparse.coo_matrix(A_pool)
    A_pool = (graph.number_of_nodes() * A_pool) / (order * negative_samples)
    A_pool.data[A_pool.data < 1.0] = 1.0
    ppmi = sparse.coo_matri(
        (np.log(A_pool.data), (A_pool.row, A_pool.col)),
        shape=A_pool.shape,
        dtype=np.float32,
    )
    return ppmi


def get_test_edge(graph, test_percent=0.1, by_degree=False):
    edges = np.array(list(graph.edges()))
    #print('edges', edges)
    num_edge = len(edges)
    test_num = int(num_edge*test_percent)
    
    non_edges = np.array(list(nx.non_edges(graph)))
    non_idx =  np.random.choice(len(non_edges), test_num, replace=False)
    negative_edge = non_edges[non_idx]
    # negative_edge = non_edges

    print('test_num', test_num)
    idx = np.random.choice(num_edge, test_num, replace=False)
    positive_edge = edges[idx]
    graph.remove_edges_from(positive_edge)
    
    print('len(negative_edge)', len(negative_edge))
    print('len(positive_edge)', len(positive_edge)) 
    return graph, np.array(positive_edge), np.array(negative_edge)


def normalize_by_row(X):
    row_sum = np.array(X.sum(-1))
    row_sum[np.where(row_sum==0)] = 1.0
    X = sparse.coo_matrix(X / row_sum)
    return X


def link_predict_without_lr(graph, embedding, test_pos_edge, test_neg_edge, test_percent):
    # normalize_embedidng
    embedding = normalize_by_row(embedding).toarray()
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
    
    #precision = np.sum(Y_pred[:num_test_pos]) / num_test_pos
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_prob)
    ap = average_precision_score(Y_test, Y_prob)
    return precision, recall, f1, auc, ap

def link_predict_without_lr_cosine(graph, embedding, test_pos_edge, test_neg_edge, test_percent, inner_product=False):
    if not inner_product:
    # normalize_embedidng
        embedding = {x: embedding[x] / max(0.01, np.linalg.norm(embedding[x])) for x in embedding.keys()}
    num_test_pos = len(test_pos_edge)
    num_test_neg = len(test_neg_edge)
    print('num_test_pos', num_test_pos)
    print('num_test_neg', num_test_neg)
    Y_test= np.concatenate((np.ones(num_test_pos), np.zeros(num_test_neg)))
    
    test_edge = np.concatenate((test_pos_edge, test_neg_edge), axis=0)
     
    shuffle_idx = np.random.permutation(Y_test.shape[0])
    test_edge = test_edge[shuffle_idx, :]
    Y_test = Y_test[shuffle_idx]

    Y_prob = []
    for edge1, edge2 in test_edge:
        if True:
            score = embedding[edge1].dot(embedding[edge2])
            if math.isnan(score) or math.isinf(score):
                score = 0
                print('!')
            Y_prob.append(score)
            # if use lr, threshold will be auto determined 
        else:
            score = embedding[edge2][edge1]
            Y_prob.append(score)
    
    Y_prob = np.array(Y_prob)
    idx_order = np.argsort(-Y_prob) // 2
    Y_pred = np.zeros(Y_prob.shape)
    Y_pred[idx_order] = 1.0
    print('Y_test', Y_test)
    print('Y_prob', Y_prob)
    
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='micro')
    auc = roc_auc_score(Y_test, Y_prob, average='micro', multi_class='ovo')
    ap = average_precision_score(Y_test, Y_prob)
    return acc, f1, auc, ap


def link_predict_with_lr(graph, embedding, test_pos_edge, test_neg_edge, test_percent):
    edges = np.asarray(list(graph.edges()))
    num_edge = edges.shape[0]
    num_train_pos = num_edge
    #num_train_neg = int(num_edge * test_precent)
    
    idx = np.random.choice(num_edge, num_train_pos)
    train_pos_edge = edges[idx]
    
    non_edges = test_neg_edge
    random.shuffle(non_edges)
    train_neg_edge = non_edges[:-len(test_pos_edge)]
    num_train_neg = len(train_neg_edge)
    
    test_neg_edge = non_edges[-len(test_pos_edge):]
    

    #non_edges = np.asarray(list(nx.non_edges(graph)))
    #idx = np.random.choice(len(non_edges), num_train_neg)
    #train_neg_edge = non_edges[idx]
    
    num_test_pos = len(test_pos_edge)
    num_test_neg = len(test_neg_edge)

    if True:
        Y_test = np.concatenate((np.ones((num_test_pos,1)), np.zeros((num_test_neg,1))), axis=0)
        Y_train = np.concatenate((np.ones((num_train_pos,1)),np.zeros((num_train_neg,1))), axis=0)

    else:
        Y_test = np.concatenate((np.ones(num_test_pos), np.zeros(num_test_neg)))
        Y_train = np.concatenate((np.ones(num_train_pos),np.zeros(num_train_neg)))
    
    X_test = []
    if True:
        for edge1, edge2 in test_pos_edge:
            X_test.append(embedding[edge1][edge2])
        for edge1, edge2 in test_neg_edge:
            X_test.append(embedding[edge1][edge2])
    else:
        for edge1, edge2 in test_pos_edge:
            X_test.append(embedding[edge1] * embedding[edge2])
        for edge1, edge2 in test_neg_edge:
            X_test.append(embedding[edge1] * embedding[edge2])
    X_test = np.array(X_test)

    X_train = []
    if True:
        for edge1, edge2 in train_pos_edge:
            X_train.append(embedding[edge1][edge2])
            #X_train.append([embedding[edge1][edge2]])
        for edge1, edge2 in train_neg_edge:
            X_train.append(embedding[edge1][edge2])
            #X_train.append([embedding[edge1][edge2]])
    else:
        for edge1, edge2 in train_pos_edge:
            X_train.append(embedding[edge1] * embedding[edge2])
        for edge1, edge2 in train_neg_edge:
            X_train.append(embedding[edge1] * embedding[edge2])
    X_train = np.array(X_train)

    shuffle_idx = np.random.permutation(Y_test.shape[0])
    X_test = X_test[shuffle_idx]
    Y_test = Y_test[shuffle_idx]
    
    if False:
        shuffle_idx = np.random.permutation(Y_train.shape[0])
        X_train = X_train[shuffle_idx, :]
        Y_train = Y_train[shuffle_idx]


    clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
    #clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', probability=True))
    clf.fit(X_train, Y_train)  
    
    Y_pred = clf.predict(X_test)
    Y_prob = clf.predict_proba(X_test)
    
    precision = precision_score(Y_test, Y_pred)
    #acc = accuracy_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_prob[:,1])
    ap = average_precision_score(Y_test, Y_prob[:,1])

    if False:
        hidden = 10
        learning_rate = 0.1
        epochs = 50
        
        num_train_all = X_train.shape[0]
        idx = np.random.choice(num_train_all, num_train_all)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
        num_train = int(0.8*num_train_all)
        X_train, X_valid = X_train[:num_train,:], X_train[num_train:,:]
        Y_train, Y_valid = Y_train[:num_train,:], Y_train[num_train:,:]
        inputs = keras.Input(shape=(X_train.shape[-1], ))
        #x = layers.Dense(hidden,activation='relu')(inputs)
        x = layers.Dense(X_train.shape[-1],activation='tanh')(inputs)
        #x = layers.Dropout(0.5)(x)
        x = layers.Dense(1,activation='tanh')(inputs)
        #outputs = layers.Dense(Y_train.shape[-1], activation='sigmoid')(inputs)
        outputs = layers.Dense(Y_train.shape[-1], activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy()]
            )
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
        idx = str(random.randint(0, 10000))
        best_weights_filepath = './model_weights/{}_best_weights_{}.hdf5'.format('link_predict', idx)
        saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=2, save_bast_only=True, mode='auto')
        history = model.fit(
                X_train,
                Y_train,
                batch_size=32,
                epochs=epochs,
                validation_data=(X_valid, Y_valid),
                callbacks=[early_stopping, saveBestModel]
                )
        model.load_weights(best_weights_filepath)

        results = model.evaluate(X_test, Y_test, batch_size=16)
        print("test loss, test acc:", results)
        Y_prob = model.predict(X_test)
   
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average='micro')
        auc = roc_auc_score(Y_test, Y_prob)
        ap = average_precision_score(Y_test, Y_prob)

    #auc = roc_auc_score(Y_test, Y_prob[:,1])
    #ap = average_precision_score(Y_test, Y_prob[:,1])
    #ap = average_precision_score(Y_test, Y_prob)
    return precision, recall, f1, auc, ap


def plot_embeddings(embeddings, path_pic, path_target):
    X, Y = read_node_label_from_csv(path_target)

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    f = plt.gcf()
    f.savefig(path_pic)
    f.clear()


def select_features(features, map_nodes_idx_part, map_nodes_idx_all):
    nodes_part = list(map_nodes_idx_part.keys())
    part_features_idx = [map_nodes_idx_all[i] for i in nodes_part]
    part_features = features[part_features_idx]
    return part_features


def TF_IDF(X):
    # IDF
    total_D = np.tile(X.shape[0],(1, X.shape[-1]))
    D_count = np.sum(X, axis=0)[None,:]
    IDF = np.log((total_D) / (1 + D_count)) 

    check = np.sum(X, axis=0)
    print('np.where(check==0)', np.where(check==0))
    denum = np.sum(X, axis=-1)
    denum = denum[:, None]
    TF = X / denum
    TF_IDF = TF * IDF
    #TF_IDF = TF_IDF / np.linalg.norm(TF_IDF, axis=-1)[:,None]
    TF_IDF = TF_IDF / np.linalg.norm(TF_IDF, axis=0)[None,:]
    return TF_IDF
