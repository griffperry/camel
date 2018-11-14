import Data_Utils
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
import numpy as np


# initializes the beginning population of test feature vectors
# inputs: number of test feature vectors desired, length of test feature vectors
# return: list of test feature vectors (list of lists)
def initialize_tfv_population(num_tfv, tfv_len):
    count = 0
    tfv = []
    tfv_list = []

    # while loop to get x number of test feature vectors
    while (count < num_tfv):
        # creates a randomly populated test feature vector with values between -10,000 and 10,000
        test_feature_vector = np.random.randint(-10000, high=10001, size=tfv_len)
        # divides test feature vector by 10,000 so the values range between -1 and 1
        for index in test_feature_vector:
            tfv.append(index / 10000.0)
        # adds a completed test feature vector to list
        tfv_list.append(tfv)
        # increments count
        count = count + 1
        # clears test feature vector
        tfv = []

    return tfv_list


# evaluates how well feature masks perform, where the rating is the accuracy
# inputs: list of feature vectors, list of test feature vectors, author list, feature mask
# return: accuracy, decision function
def evaluation_tfv(fv_list, tfv_list, author_list, fm):
    masked_fv_list = []
    masked_tfv_list = []
    decision_functions = []

    # convert feature mask to numpy array
    np_fm = np.array(fm)

    # for loop goes through each feature vector in the list
    for fv in fv_list:
        # convert feature vector to numpy array
        np_fv = np.array(fv)
        # apply mask to feature vector
        masked_fv = np_fm * np_fv
        # adds masked feature vectors to masked feature vector list
        masked_fv_list.append(masked_fv)

    # for loop goes through each test feature vector in the list
    for tfv in tfv_list:
        # convert test feature vector to numpy array
        np_tfv = np.array(tfv)
        # apply mask to test feature vector
        masked_tfv = np_fm * np_tfv
        # adds masked test feature vectors to masked test feature vector list
        masked_tfv_list.append(masked_tfv)

    # converts feature vector, test feature vector, and author lists to numpy arrays
    np_fv_list = np.array(masked_fv_list)
    np_tfv_list = np.array(masked_tfv_list)
    np_author_list = np.array(author_list)

    CU_X = np_fv_list
    Y = np_author_list

    lsvm = svm.LinearSVC()

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    fold_accuracy = []

    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = Data_Utils.DenseTransformer()

    for train, test in skf.split(CU_X, Y):
        # train split
        CU_train_data = CU_X[train]
        train_labels = Y[train]

        # test split
        CU_eval_data = CU_X[test]
        eval_labels = Y[test]

        # tf-idf
        tfidf.fit(CU_train_data)
        CU_train_data = dense.transform(tfidf.transform(CU_train_data))
        CU_eval_data = dense.transform(tfidf.transform(CU_eval_data))

        # standardization
        scaler.fit(CU_train_data)
        CU_train_data = scaler.transform(CU_train_data)
        CU_eval_data = scaler.transform(CU_eval_data)

        # normalization
        CU_train_data = normalize(CU_train_data)
        CU_eval_data = normalize(CU_eval_data)

        train_data = CU_train_data
        eval_data = CU_eval_data

        # evaluation
        lsvm.fit(train_data, train_labels)

        lsvm_acc = lsvm.score(eval_data, eval_labels)

        fold_accuracy.append(lsvm_acc)

        # decision function
        lsvm_df = lsvm.decision_function(eval_data)

        decision_functions.append(lsvm_df)

    # get the accuracy of a fold
    rating = np.mean(fold_accuracy, axis=0)

    return rating, decision_functions


# calculates evaluation function by (DF-T)^2
# inputs: list of decision functions, target function
# return: list of evaluation functions
def evaluation_function(decision_functions, target_function):
    evaluations = []

    # convert target function to numpy array
    np_tf = np.array(target_function)

    for df in decision_functions:
        # convert decision function to numpy array
        np_df = np.array(df)

        # calculate evaluation function
        np_ef = np_df - np_tf
        np.power(np_ef, 2)

        # add evaluation function to list
        evaluations.append(np_ef)

    return evaluations
