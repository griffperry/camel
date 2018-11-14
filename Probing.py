import Data_Utils
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
import numpy as np

# evaluates how well feature masks perform, where the rating is the accuracy
# inputs: list of feature vectors, list of authors, feature mask
# return: accuracy, decision function
def evaluation(fv_list, author_list, fm):
    masked_fv_list = []
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

    # converts feature vector and author lists to numpy arrays
    np_fv_list = np.array(masked_fv_list)
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
