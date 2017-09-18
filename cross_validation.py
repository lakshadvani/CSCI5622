import argparse
import random
from collections import namedtuple
import numpy as np
import knn
from knn import Knearest, Numbers
from numpy  import array
import itertools

random.seed(20170830)


# READ THIS FIRST
# In n-fold cross validation, all the instances are split into n folds
# of equal sizes. We are going to run train/test n times.
# Each time, we use one fold as the testing test and train a classifier
# from the remaining n-1 folds.
# In this homework, we are going to split based on the indices
# for each instance.

# SplitIndices stores the indices for each train/test run,
# Indices for the training set and the testing set 
# are respectively in two lists named 
# `train` and `test`.

SplitIndices = namedtuple("SplitIndices", ["train", "test"])
lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
def split_cv(length, num_folds):
    """
    This function splits index [0, length - 1) into num_folds (train, test) tuples.
    """
    splits = [SplitIndices([], []) for x in range(num_folds)]
    
    indices = list(range(length))
    random.shuffle(indices)

    l= length/num_folds
    
    
    splits = lol(indices, int(l))
    print(len(splits))
    
    '''
    for i in range(len(splits)):
    	splits[i] = folds[i]
    	
    print("splits",splits)
    # Finish this function to populate `folds`. folds == splits
    # All the indices are split into num_folds folds==splits.
    # Each fold is the testing set in a split, and the remaining indices
    # are added to the corresponding training set.'''
  
    return splits


def cv_performance(x, y, num_folds, k):
    """This function evaluates average accuracy in cross validation."""
    length = len(y)
    splits = split_cv(length, num_folds)
    accuracy_array = []

    
    for split in splits:
        # Finish this function to use the training instances 
        # indexed by `split.train` to train the classifier,
        # and then store the accuracy 
        # on the testing instances indexed by `split.test`
        temp_splits = splits

        test = split
        flattened_test  = [val for sublist in test for val in test]
        test_np = array(flattened_test)

        train = temp_splits[:splits.index(split)]+temp_splits[splits.index(split)+1:]
        train_np  = [val for sublist in train for val in sublist]
        train_np = array(train_np)
        train_x = [x[idx] for idx in train_np]
        train_y = [y[idx] for idx in train_np]
        
        train_x = array(train_x)
        train_y = array(train_y)

        #train_x = train_x[: ,0 ,:]
        #train_y = train_y[: ,0]

        print(len(train_x))
        print(len(train_y))


        
        knn = Knearest(train_x,train_y,k)
        
        test_x = [x[idx] for idx in test_np]
        test_y = [y[idx] for idx in test_np]
        test_x = array(test_x)
        test_y = array(test_y)

        
        
        confusion = knn.confusion_matrix(test_x, test_y)
        accuracy = knn.acccuracy(confusion)
        accuracy_array.append(accuracy)

        
    return np.mean(accuracy_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()
    
    data = Numbers("../data/mnist.pkl.gz")
    x, y = data.train_x, data.train_y
    if args.limit > 0:
        x, y = x[:args.limit], y[:args.limit]
    best_k, best_accuracy = -1, 0
    for k in [1,3,5,7,9]:
        accuracy = cv_performance(x, y, 5, k)
        print("%d-nearest neighbor accuracy: %f" % (k, accuracy))
        if accuracy > best_accuracy:
            best_accuracy, best_k = accuracy, k
    knn = Knearest(x, y, best_k)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    accuracy = knn.acccuracy(confusion)
    print("Accuracy for chosen best k= %d: %f" % (best_k, accuracy))


