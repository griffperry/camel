import random
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
# inputs: list of feature vectors, list of test feature vectors, author list, feature mask, target function
# return: list of test feature vectors with evaluations
def evaluation_tfv(fv_list, tfv_list, author_list, fm, target):
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
    #CU_eval_data = dense.transform(tfidf.transform(CU_eval_data))

    # standardization
    scaler.fit(CU_train_data)
    CU_train_data = scaler.transform(CU_train_data)
    #CU_eval_data = scaler.transform(CU_eval_data)

    # normalization
    CU_train_data = normalize(CU_train_data)
    #CU_eval_data = normalize(CU_eval_data)

    train_data = CU_train_data
    eval_data = CU_eval_data

    # fitting
    lsvm.fit(train_data, train_labels)

    # decision functions
    lsvm_df = lsvm.decision_function(eval_data)

    tfv_with_eval_list = []

    for index in range(len(tfv_list)):
        ef = evaluation_function(lsvm_df[index], target)
        tfv_with_eval_list.append([masked_tfv_list[index], ef])

    return tfv_with_eval_list


# calculates evaluation function by (DF-T)^2
# inputs: decision function, target function
# return: sum of evaluation function
def evaluation_function(decision_function, target_function):
    # convert target function to numpy array
    np_tf = np.array(target_function)

    # convert decision function to numpy array
    np_df = np.array(decision_function)

    # calculate evaluation function
    np_ef = np_df - np_tf
    squared_ef = np.power(np_ef, 2)

    # find sum of numbers in evaluation function
    ef_sum = sum(squared_ef)

    return ef_sum


# for SSGA and EGA
# randomly selects x (normally 2) number of parents to be chosen to procreate
# inputs: list of test feature vectors, number of parents, number of potential parents
# return: list of test feature vectors to act as parents
def tournament_select_parents_tfv(tfv_list, num_parent, num_potentials):
    potential_list = []
    parent_list = []

    # for loop to get x parents
    for par in range(num_parent):
        # for loop to get x potential parents
        for pot in range(num_potentials):
            # chooses a random test feature vector as a potential parent
            rand = random.randint(0, len(tfv_list) - 1)
            potential = tfv_list[rand]
            # adds potential parent to list of potential parents
            potential_list.append(potential)

        # sort potential parents by rating
        sorted_potential_list = sorted(potential_list, key=lambda tfv: tfv[1])
        # select best potential parent as a parent
        parent_list.append(sorted_potential_list[0][0])
        # clear list of potential parents
        potential_list = []

    return parent_list


# for EDA
# selects x (normally 12) number of best parents to be chosen to procreate
# inputs: list of test feature vectors, number of parents
# return: list of test feature vectors to act as parents
def select_best_parents_tfv(tfv_list, num_parent):
    parent_list = []

    # sorts list of feature masks by rating
    sorted_tfv_list = sorted(tfv_list, key=lambda tfv: tfv[1])

    # for loop to get best x parents
    for index in range(num_parent):
        # gets feature mask at index
        parent = sorted_tfv_list[index]
        # adds feature mask to the list of parents
        parent_list.append(parent)

    return parent_list


# creates x number of children from a set of parents
# inputs: list of parent test feature vectors, number of children desired
# return: list of child test feature vectors
def procreate_tfv(parent_tfv, num_children, mutation_rate):
    child_tfv = []
    child_tfv_list = []

    # something is wrong if there's fewer than two parents, so return an empty list
    if (len(parent_tfv) < 2):
        return child_tfv_list

    # first for loop creates x number of children
    for childs in range(num_children):
        # second for loop gets all the values of a child
        for index in range(len(parent_tfv[0])):
            # chooses two parents to select the value from
            #rand1 = random.randint(0, len(parent_tfv) - 1)
            #rand2 = random.randint(0, len(parent_tfv) - 1)
            # gets values from parents
            parent1 = int(parent_tfv[0][index] * 10000)
            parent2 = int(parent_tfv[1][index] * 10000)

            # gets a value near the values of the two parents
            if (parent1 > parent2):
                alpha = int(round(0.5 * (parent1 - parent2)))
                new_value = random.randint(parent2 - alpha, parent1 + alpha)
            else:
                alpha = int(round(0.5 * (parent2 - parent1)))
                new_value = random.randint(parent1 - alpha, parent2 + alpha)
            # adds value to child
            child_tfv.append(new_value / 10000.0)

        # adds finished child to list of children
        child_tfv_list.append(child_tfv)
        # clears list to start over
        child_tfv = []

    # mutates child feature masks
    mutated_child_tfv_list = mutation(child_tfv_list, mutation_rate)

    return mutated_child_tfv_list


# called by procreate()
# mutates child test feature vectors as a part of procreation
# inputs: list of child test feature vectors
# return: list of mutated child test feature vectors
def mutation(tfv_list, mutation_rate):
    # copies given feature masks to a new list
    new_tfv_list = tfv_list

    # first for loop goes through each feature mask in the list
    for tfv in new_tfv_list:
        # second for loop goes through each value of a feature mask
        for index in range(len(tfv)):
            # gives a random value x percent of the time
            if (index != 0) & (random.randint(1, 101) < mutation_rate + 1):
                tfv[index] = random.randint(-10000, 10000) / 10000

    return new_tfv_list


# sorts parent test feature vectors and child test feature vectors according to ratings,
# replaces x number of worst parents by x number of best children
# inputs: list of parent test feature vectors with rating, list of child test feature vectors with rating,
#         number of parents to be replaced (example: SSGA is 1, EGA is 24, EDA is 24)
# return: list of test feature vectors to be the next generation
def replacement_tfv(parent_tfv_list, child_tfv_list, num_replace, is_combined):
    new_gen_list = []
    new_gen_list_unrated = []

    # get the number of parents that will be kept
    num_keep = len(parent_tfv_list) - num_replace

    if is_combined:
        # creates combined list and add parent and child feature masks and ratings
        combined_tfv_list = []
        combined_tfv_list.extend(parent_tfv_list)
        combined_tfv_list.extend(child_tfv_list)

        # sorts combined list of feature masks by rating
        sorted_combined_tfv_list = sorted(combined_tfv_list, key=lambda tfv: tfv[1])

        # for loop gets the best x feature masks and adds them to the new generation
        for index in range(len(parent_tfv_list)):
            new_gen_list.append(sorted_combined_tfv_list[index])
    else:
        # sorts list of parent feature masks by rating
        sorted_parent_tfv_list = sorted(parent_tfv_list, key=lambda tfv: tfv[1])
        # sorts list of child feature masks by rating
        sorted_child_tfv_list = sorted(child_tfv_list, key=lambda tfv: tfv[1])

        # for loop gets the best x parents and adds them to the new generation
        for index in range(num_keep):
            new_gen_list.append(sorted_parent_tfv_list[index])

        # for loop gets the best x children and adds them to the new generation
        for index in range(num_replace):
            new_gen_list.append(sorted_child_tfv_list[index])

    # for loop creates a new list without ratings
    for index in range(len(new_gen_list)):
        new_gen_list_unrated.append(new_gen_list[index][0])

    return new_gen_list_unrated, new_gen_list
