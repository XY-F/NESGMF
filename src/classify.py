"""
Classifier for multi-class and multi-label tasks using top-k technique.

"""
import random
import numpy as np
import numpy 
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score,precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# make results are repeatable or independent
np.random.seed(random.randint(0, 10000))

class TopKRanker(OneVsRestClassifier):
    """ 
    Overwriting the method of OneVsRestClassifier to predict the top-k label(s)
    """
    def predict(self, X, top_k_list):
        """
        Predictting the top-k label(s)

        Args:
            X(Array of Numpy): Node embeddings.
            top_k_list(list): List of numbers of labels of nodes. 
        
        Returns:
            Y(Array of Numpy): Labels of nodes.
            Y_probs(Array of Numpy): Likelihoods of labels of nodes.
        """
        assert X.shape[0] == len(top_k_list)
        probs1 = np.asarray(super(TopKRanker, self).predict_proba(X))
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))

        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0 
            probs_[labels] = 1
            all_labels.append(probs_)
      
        return numpy.asarray(all_labels), probs1


class Classifier(object):
    '''
    Classifier to predict node labels based on node embeddings using top-k.

    Args:
        embedding(dict): Node embeddings
        clf(Classifier of Sklearn): Such as LogisticRegression or SVC
    '''
    def __init__(self, embedding, clf):
        self.embedding = embedding
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def evaluate(self, X, Y):
        """
        Computing evaluation metrics, including F1-score and AUC.

        Args:
            X(Array of Numpy): Node embedding.
            Y(Array of Numpy): Node labels.

        Return:
            Dict of the metrics.
        """
        Y_, Y_probs = self.predict(X, top_k_list=self.top_k_test)
  
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:     
            results[average+'_f1'] = f1_score(Y, Y_, average=average)
        for average in averages: 
            results[average+'_auc'] = roc_auc_score(Y, Y_probs, average=average, multi_class='ovr')
        results['acc'] = accuracy_score(Y, Y_)
        return results
    
    
    def predict(self, X, top_k_list=None):
        """
        Predicting labels based on embedding using the classifier.

        Args:
            X(Array of Numpy): Node embedding in the testing set, i.e. input of the classifier.
            top_k_list(list): List of the numbers of labels of nodes. 

        Returns:
            Y(Array of Numpy): Labels of nodes.
            Y_probs(Array of Numpy): Likelihoods of labels of nodes.
        """
        Y, Y_probs = self.clf.predict(X, top_k_list=top_k_list)
        return Y, Y_probs

    def split_train_evaluate(self, X, Y, train_percent=0.8):
        """
        Splitting the embedding-label pairs of nodes into training and testing sets according to a specified ratio. 
        
        Args:
            X(Array of Numpy): Node index.
            Y(Array of Numpy): Node labels.
            train_percent(float): Training set ratio.

        Return:
            Dict of the metrics.
        """
        top_k_list = np.array([len(l) for l in Y])

        self.binarizer.fit(Y)
        X_ = np.asarray([self.embedding[x] for x in X])
        
        # divide X_ Y_ by classes
        Y_all = np.concatenate([np.asarray(Y[i]) for i in range(len(Y))], axis=0)

        classes = np.unique(Y_all)
        num_class = len(classes)

        # For the i-th label, if the j-th node has the i-th label, then mark it.
        indexes = []
        for i in range(num_class):
            indexes.append([j for j in range(len(Y)) if (classes[i] in Y[j])])
        # Splitting the nodes according to their labels.
        X_c = []
        for i in range(num_class):
            X_c.append(X_[indexes[i]])
        
        Y_ = np.asarray(self.binarizer.transform(Y).todense())
        
        Y_c = []
        for i in range(num_class):
            Y_c.append(Y_[indexes[i]])
        
        top_k_c = []
        for i in range(num_class):
            top_k_c.append(top_k_list[indexes[i]])


        # divide X_ Y_ by classes
        X_train_union = []
        Y_train_union = []
        X_test_union = []
        Y_test_union = []
        
        top_k_union = []

        for i in range(num_class):
            sub_X = X_c[i]
            sub_Y = Y_c[i]
            sub_top_k = top_k_c[i]
            num_sub_X = len(sub_X)

            train_size = int(train_percent * num_sub_X)
            if train_size < 1:
                train_size = 1
            elif num_sub_X - train_size < 1:
                train_size = num_sub_X - 1

            # random shuffle
            shuffle_indices = np.random.permutation(np.arange(num_sub_X))
         
            X_train_union.append(np.asarray([sub_X[shuffle_indices[j]] for j in range(train_size)]))
            Y_train_union.append(np.asarray([sub_Y[shuffle_indices[j]] for j in range(train_size)]))
             
            X_test_union.append(np.asarray([sub_X[shuffle_indices[j]] for j in range(train_size, num_sub_X)]))
            Y_test_union.append(np.asarray([sub_Y[shuffle_indices[j]] for j in range(train_size, num_sub_X)]))
          
            top_k_union.append(np.asarray([sub_top_k[shuffle_indices[j]] for j in range(train_size, num_sub_X)])) 

        X_train = np.concatenate(X_train_union, axis=0)
        Y_train = np.concatenate(Y_train_union, axis=0)
        X_test = np.concatenate(X_test_union, axis=0)
        Y_test = np.concatenate(Y_test_union, axis=0)
        
        self.top_k_test = np.concatenate(top_k_union, axis=0)
        
        self.clf.fit(X_train, Y_train)
        return self.evaluate(X_test, Y_test)





