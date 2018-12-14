import Data_Utils
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
import numpy as np


# evaluates how well feature masks perform, where the rating is the accuracy
# inputs: list of feature vectors, list of test feature vectors, author list, feature mask, target function
# return: list of test feature vectors with evaluations
def tfv_predict_2(fv_list, tfv_list, author_list, test_author_list, fm):
    masked_fv_list = []
    masked_tfv_list = []

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

    lsvm = svm.LinearSVC()

    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = Data_Utils.DenseTransformer()

    # train split
    CU_train_data = np_fv_list
    train_labels = np_author_list

    # test split
    CU_eval_data = np_tfv_list

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

    # fitting
    lsvm.fit(train_data, train_labels)

    # predictions
    lsvm_predictions = lsvm.predict(eval_data)
    print lsvm_predictions

    pass_fail = []

    # compare predictions to actual authors
    for i in range(len(lsvm_predictions)):
        if lsvm_predictions[i] == test_author_list[i]:
            pass_fail.append("Pass")
        else:
            pass_fail.append("Fail")

    return pass_fail
