import random
import Data_Utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score


# initializes the beginning population of feature masks
# inputs: number of feature masks desired, length of feature masks
# return: list of feature masks (list of lists)
def initialize_population(num_masks, mask_len):
    count = 0
    fm_list = []

    # while loop to get x number of feature masks
    while (count < num_masks):
        # creates a feature mask with 1 and 0's randomly populated
        feature_mask = np.random.randint(0, high=2, size=mask_len)
        # adds a completed feature mask to a feature mask list
        fm_list.append(feature_mask)
        # increments count
        count = count + 1

    return fm_list


# evaluates how well feature masks perform, where the ranking is the accuracy
# inputs: list of feature masks
# return: list of feature vectors, list of authors, list of feature masks with rating
#         (example: [([feature mask], rating), ([feature mask], rating), etc.])
def evaluation(fv_list, author_list, fm_list):
    fmar_list = []

    # first for loop goes through each feature mask in the list
    for fm in fm_list:
        masked_fv_list = []

        # convert feature mask to numpy array
        np_fm = np.array(fm)

        # second for loop goes through each feature vector in the list
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

        # get the accuracy of a fold
        rating = np.mean(fold_accuracy, axis=0)

        append_fm = fm
        fmar_list.append([append_fm, rating])

    return fmar_list


# for SSGA and EGA
# randomly selects x (normally 2) number of parents to be chosen to procreate
# inputs: list of feature masks, number of parents
# return: list of feature vectors to act as parents
def tournament_select_parents(fm_list, num_parent, num_potentials):
    potential_list = []
    parent_list = []

    # for loop to get x parents
    for par in range(num_parent):
        # for loop to get x potential parents
        for pot in range(num_potentials):
            # chooses a random feature mask as a potential parent
            rand = random.randint(0, len(fm_list) - 1)
            potential = fm_list[rand]
            # adds potential parent to list of potential parents
            potential_list.append(potential)

        # sort potential parents by rating
        sorted_potential_list = sorted(potential_list, key=lambda fmar: fmar[1], reverse=True)
        # select best potential parent as a parent
        parent_list.append(sorted_potential_list[0][0])
        # clear list of potential parents
        potential_list = []

    return parent_list


# for EDA
# selects x (normally 12) number of best parents to be chosen to procreate
# inputs: list of feature masks with rating (gen_evaluation output), number of parents
# return: list of feature vectors to act as parents
def select_best_parents(fmar_list, num_parent):
    parent_list = []

    # sorts list of feature masks by rating
    sorted_fmar_list = sorted(fmar_list, key=lambda fmar: fmar[1], reverse=True)

    # for loop to get best x parents
    for index in range(num_parent):
        # gets feature mask at index
        parent = sorted_fmar_list[index]
        # adds feature mask to the list of parents
        parent_list.append(parent)

    return parent_list


# creates x number of children from a set of parents
# inputs: list of parent feature masks, number of children desired
# return: list of child feature masks
def procreate(parent_vectors, num_children, mutation_rate):
    child_mask = []
    child_mask_vectors = []

    # something is wrong if there's fewer than two parents, so return an empty list
    if (len(parent_vectors) < 2):
        return child_mask_vectors

    # first for loop creates x number of children
    for childs in range(num_children):
        # second for loop gets all the values of a child
        for index in range(len(parent_vectors[0])):
            # chooses a parent to select the value from
            rand = random.randint(0, len(parent_vectors) - 1)
            # gets value from parent x
            parent = parent_vectors[rand][index]
            # adds value to child
            child_mask.append(parent)
        # adds finished child to list of children
        child_mask_vectors.append(child_mask)
        # clears list to start over
        child_mask = []

    # mutates child feature masks
    mutated_child_mask_vectors = mutation(child_mask_vectors, mutation_rate)

    return mutated_child_mask_vectors


# called by procreate()
# mutates child feature masks as a part of procreation
# inputs: list of child feature masks
# return: list of mutated child feature masks
def mutation(feature_mask_list, mutation_rate):
    # copies given feature masks to a new list
    new_feature_mask_list = feature_mask_list

    # first for loop goes through each feature mask in the list
    for fm in new_feature_mask_list:
        # second for loop goes through each value of a feature mask
        for index in range(len(fm)):
            # flips the value (0 to 1 or 1 to 0) x percent of the time
            if (random.randint(1, 101) < mutation_rate + 1):
                fm[index] = fm[index] ^ 1

    return new_feature_mask_list


# sorts parent feature vectors and child feature vectors according to ratings,
# replaces x number of worst parents by x number of best children
# inputs: list of parent feature masks with rating, list of child feature masks with rating,
#         number of parents to be replaced (example: SSGA is 1, EGA is 24, EDA is 24)
# return: list of feature vectors to be the next generation
def replacement(parent_fmar_list, child_fmar_list, num_replace, is_combined):
    new_gen_list = []
    new_gen_list_unrated = []

    # get the number of parents that will be kept
    num_keep = len(parent_fmar_list) - num_replace

    if is_combined:
        # creates combined list and add parent and child feature masks and ratings
        combined_fmar_list = []
        combined_fmar_list.extend(parent_fmar_list)
        combined_fmar_list.extend(child_fmar_list)

        # sorts combined list of feature masks by rating
        sorted_combined_fmar_list = sorted(combined_fmar_list, key=lambda fmar: fmar[1], reverse=True)

        # for loop gets the best x feature masks and adds them to the new generation
        for index in range(len(parent_fmar_list)):
            new_gen_list.append(sorted_combined_fmar_list[index])
    else:
        # sorts list of parent feature masks by rating
        sorted_parent_fmar_list = sorted(parent_fmar_list, key=lambda fmar: fmar[1], reverse=True)
        # sorts list of child feature masks by rating
        sorted_child_fmar_list = sorted(child_fmar_list, key=lambda fmar: fmar[1], reverse=True)

        # for loop gets the best x parents and adds them to the new generation
        for index in range(num_keep):
            new_gen_list.append(sorted_parent_fmar_list[index])

        # for loop gets the best x children and adds them to the new generation
        for index in range(num_replace):
            new_gen_list.append(sorted_child_fmar_list[index])

    # for loop creates a new list without ratings
    for index in range(len(new_gen_list)):
        new_gen_list_unrated.append(new_gen_list[index][0])

    return new_gen_list_unrated, new_gen_list
