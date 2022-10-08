import numpy as np
import nearest_neighbors as nn


def kfold(n, n_folds):
    a = np.arange(n)
    lst = []
    for l in np.array_split(a, n_folds):
        lst.append((np.delete(a, l), l))
    return lst


def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    if kwargs.get('k', -1) != -1:
        del kwargs['k']
    if cv is None:
        cv = kfold(X.shape[0], 3)
    k_dict = {}
    for k in k_list:
        k_dict[k] = []
    for train_index, test_index in cv:
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        knn_clasifier = nn.KNNClassifier(k=k_list[-1], **kwargs)
        knn_clasifier.fit(X_train, y_train)
        A = np.zeros((X_test.shape[0], np.unique(y).shape[0]))
        if kwargs['weights'] is False:
            next_neighb = np.array([])
            prev_k = k_list[0]
            neighb = knn_clasifier.find_kneighbors(X_test,
                                                   return_distance=False)
            for k in k_list:
                if next_neighb.shape[0] == 0:
                    next_neighb = neighb[..., :k]
                else:
                    next_neighb = neighb[..., prev_k:k]
                answer = []
                for i in range(y_train[next_neighb].shape[0]):
                    values, counts = np.unique(y_train[next_neighb][i],
                                               return_counts=True)
                    for (j, c) in zip(values, counts):
                        A[i, j] += c
                y_pred = np.argmax(A, axis=1)
                if(score == 'accuracy'):
                    n_correct = np.sum(y_pred == y_test)
                    accuracy = n_correct / y_test.shape[0]
                    k_dict[k].append(accuracy)
                prev_k = k
        elif kwargs['weights'] is True:
            dists, neighb = knn_clasifier.find_kneighbors(X_test,
                                                          return_distance=True)
            weights = 1/(dists+10**(-5))
            cl = y_train[neighb]
            for k in range(k_list[-1]):
                for i in range(cl.shape[0]):
                    A[i, cl[i, k]] += weights[i, k]
                if k+1 in k_list:
                    y_pred = np.argmax(A, axis=1)
                    if(score == 'accuracy'):
                        n_correct = np.sum(y_pred == y_test)
                        accuracy = n_correct / y_test.shape[0]
                        k_dict[k+1].append(accuracy)
    for key in k_dict.keys():
        k_dict[key] = np.array(k_dict[key])
    return k_dict
