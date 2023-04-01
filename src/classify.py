import numpy as np
import numpy 
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score,precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


np.random.seed(100)

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
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
    
    def __init__(self, embedding, clf, seed=42):
        self.embedding = embedding
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def evaluate(self, X, Y):
        Y_, Y_probs = self.predict(X, top_k_list=self.top_k_test)
  
        # Y = self.binarizer.transform(Y).toarray()
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:     
            results[average+'_f1'] = f1_score(Y, Y_, average=average)
        for average in averages: 
            results[average+'_auc'] = roc_auc_score(Y, Y_probs, average=average, multi_class='ovr')
        results['acc'] = accuracy_score(Y, Y_)
        print('--------------------')
        precision = precision_score(Y, Y_, average=None)
        recall = recall_score(Y, Y_, average=None)
        print('precision', precision)
        print('recall', recall)
        print(results)
        return results
    
    
    def predict(self, X, top_k_list=None):
        Y, Y_probs = self.clf.predict(X, top_k_list=top_k_list)
        # print('Y_probs', Y_probs)
        return Y, Y_probs

    def split_train_evaluate(self, X, Y, train_percent=0.8):

        top_k_list = np.array([len(l) for l in Y])

        self.binarizer.fit(Y)
        X_ = np.asarray([self.embedding[x] for x in X])
        
        # divide X_ Y_ by classes
        Y_all = np.concatenate([np.asarray(Y[i]) for i in range(len(Y))], axis=0)
        # print('Y_all', Y_all)

        classes = np.unique(Y_all)
        num_class = len(classes)
        # print('num_class', num_class)

        indexes = []
        for i in range(num_class):
            indexes.append([j for j in range(len(Y)) if (classes[i] in Y[j])])
    
        X_c = []
        for i in range(num_class):
            X_c.append(X_[indexes[i]])
        # print('X_c', X_c)
        
        Y_ = np.asarray(self.binarizer.transform(Y).todense())
        
        Y_c = []
        for i in range(num_class):
            Y_c.append(Y_[indexes[i]])
        # print('Y_c', Y_c)
        
        
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
            # print('num_sub_X',num_sub_X)
            # print('train_percent', train_percent)
            train_size = int(train_percent * num_sub_X)
            if train_size < 1:
                train_size = 1
            elif num_sub_X - train_size < 1:
                train_size = num_sub_X - 1
            # print('train_size', train_size)
            shuffle_indices = np.random.permutation(np.arange(num_sub_X))
            # print('shuffle_indices', shuffle_indices)
         
            X_train_union.append(np.asarray([sub_X[shuffle_indices[j]] for j in range(train_size)]))
            Y_train_union.append(np.asarray([sub_Y[shuffle_indices[j]] for j in range(train_size)]))
             
            X_test_union.append(np.asarray([sub_X[shuffle_indices[j]] for j in range(train_size, num_sub_X)]))
            Y_test_union.append(np.asarray([sub_Y[shuffle_indices[j]] for j in range(train_size, num_sub_X)]))
          
            top_k_union.append(np.asarray([sub_top_k[shuffle_indices[j]] for j in range(train_size, num_sub_X)])) 

        # print('X_train_union', X_train_union)
        X_train = np.concatenate(X_train_union, axis=0)
        Y_train = np.concatenate(Y_train_union, axis=0)
        X_test = np.concatenate(X_test_union, axis=0)
        Y_test = np.concatenate(Y_test_union, axis=0)
        
        self.top_k_test = np.concatenate(top_k_union, axis=0)
        
        self.clf.fit(X_train, Y_train)
        return self.evaluate(X_test, Y_test)





