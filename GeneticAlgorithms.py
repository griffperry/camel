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
    feature_mask = []
    fm_list = []

    # while loop to get x number of feature masks
    while (count < num_masks):
        # for loop to get x length of 1/0s
        for index in range(mask_len):
            # randomly gets a 1 or 0 and adds it to a feature mask
            feature_mask.append(random.randint(0, 1))
        # adds a completed feature mask to a feature mask list
        fm_list.append(feature_mask)
        # clears list to start over
        feature_mask = []
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
        # convert feature mask to numpy array
        np_fm = np.array(fm)

        masked_fv_list = []

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

        rating = np.mean(fold_accuracy, axis=0)

        append_fm = fm
        fmar_list.append([append_fm, rating])

    return fmar_list


# for SSGA and EGA
# randomly selects x (normally 2) number of parents to be chosen to procreate
# inputs: list of feature masks, number of parents
# return: list of feature vectors to act as parents
def randomly_select_parents(fm_list, num_parent):
    parent_list = []

    # for loop to get x parents
    for index in range(num_parent):
        # chooses a random feature mask as a parent
        rand = random.randint(0, len(fm_list) - 1)
        parent = fm_list[rand]
        # adds feature mask to list of parents
        parent_list.append(parent)

    return parent_list


# for EDA
# selects x (normally 12) number of best parents to be chosen to procreate
# inputs: list of feature masks with rating (gen_evaluation output), number of parents
# return: list of feature vectors to act as parents
def select_best_parents(fmar_list, num_parent):
    parent_list = []

    # sorts list of feature masks by rating
    sorted_fmar_list = sorted(fmar_list, key=lambda fmar: fmar[1])

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
def procreate(parent_vectors, num_children):
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
    mutated_child_mask_vectors = mutation(child_mask_vectors)

    return mutated_child_mask_vectors


# called by procreate()
# mutates child feature masks as a part of procreation
# inputs: list of child feature masks
# return: list of mutated child feature masks
def mutation(feature_mask_list):
    # copies given feature masks to a new list
    new_feature_mask_list = feature_mask_list

    # first for loop goes through each feature mask in the list
    for fm in new_feature_mask_list:
        # second for loop goes through each value of a feature mask
        for index in range(len(fm)):
            # flips the value (0 to 1 or 1 to 0) 5% of the time
            if (random.randint(1, 101) < 6):
                fm[index] = fm[index] ^ 1

    return new_feature_mask_list


# sorts parent feature vectors and child feature vectors according to ratings,
# replaces x number of worst parents by x number of best children
# inputs: list of parent feature masks with rating, list of child feature masks with rating,
#         number of parents to be replaced (example: SSGA is 1, EGA is 24, EDA is 24)
# return: list of feature vectors to be the next generation
def replacement(parent_fmar_list, child_fmar_list, num_replace):
    new_gen_list = []
    new_gen_list_unrated = []

    # get the number of parents that will be kept
    num_keep = len(parent_fmar_list) - num_replace

    # sorts list of parent feature masks by rating
    sorted_parent_fmar_list = sorted(parent_fmar_list, key=lambda fmar: fmar[1])
    # sorts list of child feature masks by rating
    sorted_child_fmar_list = sorted(child_fmar_list, key=lambda fmar: fmar[1])

    # for loop gets the best x parents and adds them to the new generation
    for index in range(num_keep):
        new_gen_list.append(sorted_parent_fmar_list[index])

    # for loop gets the best x children and adds them to the new generation
    for index in range(num_replace):
        new_gen_list.append(sorted_child_fmar_list[index])

    for index in range(len(new_gen_list)):
        new_gen_list_unrated.append(new_gen_list[index][0])

    return new_gen_list_unrated, new_gen_list


# EXAMPLES
'''fm_length = 10
number_of_fms = 5
number_of_parents = 2
number_of_children = 1

print "Feature Mask List:"

fm_list = initialize_population(number_of_fms, fm_length)
for fm in fm_list:
    print fm

print "\nParent Feature Mask List:"

par_list = randomly_select_parents(fm_list, number_of_parents)
for fm in par_list:
    print fm

print "\nChild Feature Mask List:"

child_list = procreate(fm_list, number_of_children)
for fm in child_list:
    print fm

# Steady State Genetic Algorithm
SSGA_fm_length = 95
SSGA_number_of_fms = 25
SSGA_number_of_parents = 2
SSGA_number_of_children = 1
SSGA_number_to_replace = 1
SSGA_iterations = 5000

# initialize the feature mask population
SSGA_generation_list = initialize_population(SSGA_number_of_fms, SSGA_fm_length)
# decrement the number of iterations by the number of feature masks created
SSGA_iterations = SSGA_iterations - SSGA_fm_length

# while loop makes sure that the correct number of evaluations have been performed
while (SSGA_iterations > 0):
    # 2 parents are randomly selected from the feature mask generation
    SSGA_parent_list = randomly_select_parents(SSGA_generation_list, SSGA_number_of_parents)
    # 1 child is created from the 2 selected parents
    SSGA_child_list = procreate(SSGA_parent_list, SSGA_number_of_children)

    # the generation is rated by accuracy
    SSGA_generation_list_rated = evaluation()
    # the child is rated by accuracy
    SSGA_child_list_rated = evaluation()

    # the child replaces the worst individual in the generation
    SSGA_generation_list = replacement(SSGA_generation_list_rated, SSGA_child_list_rated, SSGA_number_to_replace)

    # decrement the number of iterations by the number of children created
    SSGA_iterations = SSGA_iterations - len(SSGA_child_list)


# Elitist Genetic Algorithm
EGA_fm_length = 95
EGA_number_of_fms = 25
EGA_number_of_parents = 2
EGA_number_of_children = 1
EGA_number_to_replace = 24
EGA_iterations = 5000

EGA_child_list = []

# initialize the feature mask population
EGA_generation_list = initialize_population(EGA_number_of_fms, EGA_fm_length)
# decrement the number of iterations by the number of feature masks created
EGA_iterations = EGA_iterations - EGA_fm_length

# while loop makes sure that the correct number of evaluations have been performed
while (EGA_iterations > 0):
    for index in range(EGA_number_to_replace):
        # 2 parents are randomly selected from the feature mask generation
        EGA_parent_list = randomly_select_parents(EGA_generation_list, EGA_number_of_parents)
        # 1 child is created from the 2 selected parents
        EGA_child = procreate(EGA_parent_list, EGA_number_of_children)
        # child is added to child list
        EGA_child_list.append(EGA_child)

    # the generation is rated by accuracy
    EGA_generation_list_rated = evaluation()
    # the children are rated by accuracy
    EGA_child_list_rated = evaluation()

    # the child replaces the worst 24 individuals in the generation
    EGA_generation_list = replacement(EGA_generation_list_rated, EGA_child_list_rated, EGA_number_to_replace)

    # decrement the number of iterations by the number of children created
    EGA_iterations = EGA_iterations - len(EGA_child_list)

# Estimation of Distribution Algorithm
EDA_fm_length = 10
EDA_number_of_fms = 5
EDA_number_of_parents = 12
EDA_number_of_children = 24
EDA_number_to_replace = 24
EDA_iterations = 5000

EDA_child_list = []

# initialize the feature mask population
EDA_generation_list = initialize_population(EDA_number_of_fms, EDA_fm_length)
# decrement the number of iterations by the number of feature masks created
EDA_iterations = EDA_iterations - EDA_fm_length

# while loop makes sure that the correct number of evaluations have been performed
while (EDA_iterations > 0):
    # 12 best parents are selected from the feature mask generation
    EDA_parent_list = select_best_parents(EDA_generation_list, EDA_number_of_parents)
    # 24 children are created from the 12 selected parents
    EDA_child_list = procreate(EDA_parent_list, EDA_number_of_children)

    # the generation is rated by accuracy
    EDA_generation_list_rated = evaluation()
    # the children are rated by accuracy
    EDA_child_list_rated = evaluation()

    # the children replace the worst 24 individuals in the generation
    EDA_generation_list = replacement(EDA_generation_list_rated, EDA_child_list_rated, EDA_number_to_replace)

    # decrement the number of iterations by the number of children created
    EDA_iterations = EDA_iterations - len(EDA_child_list)'''

