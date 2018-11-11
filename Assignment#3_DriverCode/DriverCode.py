import Data_Utils
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
import numpy as np
from GeneticAlgorithms import replacement, evaluation, procreate, tournament_select_parents, initialize_population,\
    select_best_parents

CU_X, Y = Data_Utils.Get_Casis_CUDataset()


rbfsvm = svm.SVC()
lsvm = svm.LinearSVC()
mlp = MLPClassifier(max_iter=2000)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
fold_accuracy = []

scaler = StandardScaler()
tfidf = TfidfTransformer(norm=None)
dense = Data_Utils.DenseTransformer()

for train, test in skf.split(CU_X, Y):
    #train split
    CU_train_data = CU_X[train]
    train_labels = Y[train]
    
    #test split
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
    rbfsvm.fit(train_data, train_labels)
    lsvm.fit(train_data, train_labels)
    mlp.fit(train_data, train_labels)

    rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
    lsvm_acc = lsvm.score(eval_data, eval_labels)
    mlp_acc = mlp.score(eval_data, eval_labels)

    fold_accuracy.append((lsvm_acc, rbfsvm_acc, mlp_acc))

print fold_accuracy
print(np.mean(fold_accuracy, axis=0))

# Separating Authors and Feature Vectors
authors = Y
feature_vectors = CU_X

# create CASIS file
file = open("CASISGA.txt","w+")

# Steady State Genetic Algorithm
# No Innovations: 95, 25, 2, 2, 1, 1, 5, False, 5000, 10
SSGA_fm_length = 95
SSGA_number_of_fms = 25
SSGA_number_of_potential_parents = 10
SSGA_number_of_parents = 2
SSGA_number_of_children = 10
SSGA_number_to_replace = 1
SSGA_mutation_rate = 1
SSGA_is_replacement_combined = True
SSGA_number_of_iterations = 5000
SSGA_number_of_runs = 10
SSGA_last_generation_ratings = []

file.write("Steady State Genetic Algorithm\n")

# run SSGA x number of times
for index in range(SSGA_number_of_runs):
    SSGA_best_ratings = []
    SSGA_ratings_over_time = []
    SSGA_sum = 0

    # initialize the feature mask population
    SSGA_generation_list = initialize_population(SSGA_number_of_fms, SSGA_fm_length)

    # the initial generation is rated by accuracy
    SSGA_generation_list_rated = evaluation(feature_vectors, authors, SSGA_generation_list)

    # gets sum of all feature mask ratings
    for rating in SSGA_generation_list_rated:
        SSGA_sum = SSGA_sum + rating[1]

    # gets average of all feature mask ratings
    SSGA_average = SSGA_sum / len(SSGA_generation_list_rated)

    # adds rating average to rating list
    SSGA_ratings_over_time.append(SSGA_average)

    # initialize iteration number
    SSGA_iterations = SSGA_number_of_iterations

    # decrement the number of iterations by the number of feature masks created
    SSGA_iterations = SSGA_iterations - SSGA_number_of_fms

    # while loop makes sure that the correct number of evaluations have been performed
    while (SSGA_iterations > 0):
        SSGA_sum = 0
        #print SSGA_iterations

        # 2 parents are randomly selected from the feature mask generation
        SSGA_parent_list = tournament_select_parents(SSGA_generation_list_rated, SSGA_number_of_parents,
                                                     SSGA_number_of_potential_parents)
        # 1 child is created from the 2 selected parents
        SSGA_child_list = procreate(SSGA_parent_list, SSGA_number_of_children, SSGA_mutation_rate)

        # the child is rated by accuracy
        SSGA_child_list_rated = evaluation(feature_vectors, authors, SSGA_child_list)

        # the child replaces the worst individual in the generation
        SSGA_generation_list, SSGA_generation_list_rated = replacement(SSGA_generation_list_rated, SSGA_child_list_rated,
                                                                       SSGA_number_to_replace, SSGA_is_replacement_combined)

        # gets sum of all feature mask ratings
        for rating in SSGA_generation_list_rated:
            SSGA_sum = SSGA_sum + rating[1]

        # gets average of all feature mask ratings
        SSGA_average = SSGA_sum / len(SSGA_generation_list_rated)

        # adds rating average to rating list
        SSGA_ratings_over_time.append(SSGA_average)

        # decrement the number of iterations by the number of children created
        SSGA_iterations = SSGA_iterations - 1

    # add average rating of last generation to list
    SSGA_last_generation_ratings.append(max(SSGA_ratings_over_time))

    # write pertinent information into the CASIS file
    for rating in SSGA_ratings_over_time:
        file.write('%s' % rating)
        file.write(", ")
    file.write("\n")
    for fm in SSGA_generation_list_rated:
        print fm
        file.write('%s' % fm)
        file.write("\n")
    file.write("\n")

# get average of the final rating of each generation
SSGA_final_sum = 0
SSGA_final_max = 0

# for loop goes through each rating
for rat in SSGA_last_generation_ratings:
    # gets sum of final ratings
    SSGA_final_sum = SSGA_final_sum + rat
    # gets max value of final ratings
    if (rat > SSGA_final_max):
        SSGA_final_max = rat
# gets average of final ratings
SSGA_final_average = SSGA_final_sum / len(SSGA_last_generation_ratings)

# write pertinent information into CASIS file
file.write("\n\nSSGA Ratings for Each Generation: ")
for rating in SSGA_last_generation_ratings:
    file.write('%s' % rating)
    file.write(", ")
file.write("\nSSGA Max Rating: ")
file.write('%s' % SSGA_final_max)
file.write("\nSSGA Average Rating: ")
file.write('%s' % SSGA_final_average)

# Elitist Genetic Algorithm
# No Innovations: 95, 25, 2, 2, 1, 24, 5, False, 5000, 10

EGA_fm_length = 95
EGA_number_of_fms = 25
EGA_number_of_potential_parents = 10
EGA_number_of_parents = 2
EGA_number_of_children = 1
EGA_number_to_replace = 24
EGA_mutation_rate = 1
EGA_is_replacement_combined = True
EGA_number_of_iterations = 5000
EGA_number_of_runs = 10
EGA_last_generation_ratings = []

file.write("\n\nElitist Genetic Algorithm\n")

# run EGA x number of times
for index in range(EGA_number_of_runs):
    EGA_best_ratings = []
    EGA_ratings_over_time = []
    EGA_child_list = []
    EGA_sum = 0

    # initialize the feature mask population
    EGA_generation_list = initialize_population(EGA_number_of_fms, EGA_fm_length)

    # the initial generation is rated by accuracy
    EGA_generation_list_rated = evaluation(feature_vectors, authors, EGA_generation_list)

    # gets sum of all feature mask ratings
    for rating in EGA_generation_list_rated:
        EGA_sum = EGA_sum + rating[1]

    # gets average of all feature mask ratings
    EGA_average = EGA_sum / len(EGA_generation_list_rated)

    # adds rating average to rating list
    EGA_ratings_over_time.append(EGA_average)

    # initialize iteration number
    EGA_iterations = EGA_number_of_iterations

    # decrement the number of iterations by the number of feature masks created
    EGA_iterations = EGA_iterations - EGA_number_of_fms

    # while loop makes sure that the correct number of evaluations have been performed
    while (EGA_iterations > 0):
        EGA_sum = 0
        EGA_child_list = []
        #print EGA_iterations

        for index in range(EGA_number_to_replace):
            # 2 parents are randomly selected from the feature mask generation
            EGA_parent_list = tournament_select_parents(EGA_generation_list_rated, EGA_number_of_parents,
                                                        EGA_number_of_potential_parents)
            # 1 child is created from the 2 selected parents
            EGA_child = procreate(EGA_parent_list, EGA_number_of_children, EGA_mutation_rate)
            EGA_single_child = EGA_child[0]
            # child is added to child list
            EGA_child_list.append(EGA_single_child)

        # the children are rated by accuracy
        EGA_child_list_rated = evaluation(feature_vectors, authors, EGA_child_list)

        # the child replaces the worst 24 individuals in the generation
        EGA_generation_list, EGA_generation_list_rated = replacement(EGA_generation_list_rated, EGA_child_list_rated,
                                                                     EGA_number_to_replace, EGA_is_replacement_combined)

        # gets sum of all feature mask ratings
        for rating in EGA_generation_list_rated:
            EGA_sum = EGA_sum + rating[1]

        # gets average of all feature mask ratings
        EGA_average = EGA_sum / len(EGA_generation_list_rated)

        # adds rating average to rating list
        EGA_ratings_over_time.append(EGA_average)

        # decrement the number of iterations by the number of children created
        EGA_iterations = EGA_iterations - 24

    # add average rating of last generation to list
    EGA_last_generation_ratings.append(max(EGA_ratings_over_time))

    # write pertinent information into CASIS file
    for rating in EGA_ratings_over_time:
        file.write('%s' % rating)
        file.write(", ")
    file.write("\n")
    for fm in EGA_generation_list_rated:
        print fm
        file.write('%s' % fm)
        file.write("\n")
    file.write("\n")

# get average of the final rating of each generation
EGA_final_sum = 0
EGA_final_max = 0

# for loop goes through each rating
for rat in EGA_last_generation_ratings:
    # gets sum of final ratings
    EGA_final_sum = EGA_final_sum + rat
    # gets max value of final ratings
    if (rat > EGA_final_max):
        EGA_final_max = rat
# gets average of final ratings
EGA_final_average = EGA_final_sum / len(EGA_last_generation_ratings)

# write pertinent information into CASIS file
file.write("\n\nEGA Ratings for Each Generation: ")
for rating in EGA_last_generation_ratings:
    file.write('%s' % rating)
    file.write(", ")
file.write("\nEGA Max Rating: ")
file.write('%s' % EGA_final_max)
file.write("\nEGA Average Rating: ")
file.write('%s' % EGA_final_average)

# Estimation of Distribution Algorithm
# No Innovations: 95, 25, 12, 24, 24, 5, False, 5000, 10
EDA_fm_length = 95
EDA_number_of_fms = 25
EDA_number_of_parents = 6
EDA_number_of_children = 48
EDA_number_to_replace = 24
EDA_mutation_rate = 1
EDA_is_replacement_combined = False
EDA_number_of_iterations = 5000
EDA_number_of_runs = 10
EDA_last_generation_ratings = []

file.write("\n\nEstimation of Distribution Algorithm\n")

# run EDA x number of times
for index in range(EDA_number_of_runs):
    EDA_best_ratings = []
    EDA_ratings_over_time = []
    EDA_child_list = []
    EDA_sum = 0

    # initialize the feature mask population
    EDA_generation_list = initialize_population(EDA_number_of_fms, EDA_fm_length)

    # the initial generation is rated by accuracy
    EDA_generation_list_rated = evaluation(feature_vectors, authors, EDA_generation_list)

    # gets sum of all feature mask ratings
    for rating in EDA_generation_list_rated:
        EDA_sum = EDA_sum + rating[1]

    # gets average of all feature mask ratings
    EDA_average = EDA_sum / len(EDA_generation_list_rated)

    # adds rating average to rating list
    EDA_ratings_over_time.append(EDA_average)

    # initialize iteration number
    EDA_iterations = EDA_number_of_iterations

    # decrement the number of iterations by the number of feature masks created
    EDA_iterations = EDA_iterations - EDA_number_of_fms

    # while loop makes sure that the correct number of evaluations have been performed
    while (EDA_iterations > 0):
        EDA_sum = 0
        #print EDA_iterations

        # 12 best parents are selected from the feature mask generation
        EDA_parent_list = select_best_parents(EDA_generation_list, EDA_number_of_parents)
        # 24 children are created from the 12 selected parents
        EDA_child_list = procreate(EDA_parent_list, EDA_number_of_children, EDA_mutation_rate)

        # the children are rated by accuracy
        EDA_child_list_rated = evaluation(feature_vectors, authors, EDA_child_list)

        # the children replace the worst 24 individuals in the generation
        EDA_generation_list, EDA_generation_list_rated = replacement(EDA_generation_list_rated, EDA_child_list_rated,
                                                                     EDA_number_to_replace, EDA_is_replacement_combined)

        # gets sum of all feature mask ratings
        for rating in EDA_generation_list_rated:
            EDA_sum = EDA_sum + rating[1]

        # gets average of all feature mask ratings
        EDA_average = EDA_sum / len(EDA_generation_list_rated)

        # adds rating average to rating list
        EDA_ratings_over_time.append(EDA_average)

        # decrement the number of iterations by the number of children created
        EDA_iterations = EDA_iterations - 24

    # add average rating of last generation to list
    EDA_last_generation_ratings.append(max(EDA_ratings_over_time))

    # write pertinent information to CASIS file
    for rating in EDA_ratings_over_time:
        file.write('%s' % rating)
        file.write(", ")
    file.write("\n")
    for fm in EDA_generation_list_rated:
        file.write('%s' % fm)
        file.write("\n")
    file.write("\n")

# get average of the final rating of each generation
EDA_final_sum = 0
EDA_final_max = 0

# for loop goes through each rating
for rat in EDA_last_generation_ratings:
    # gets sum of final ratings
    EDA_final_sum = EDA_final_sum + rat
    # gets max value of final ratings
    if (rat > EDA_final_max):
        EDA_final_max = rat
# gets average of final ratings
EDA_final_average = EDA_final_sum / len(EDA_last_generation_ratings)

# write pertinent information to CASIS file
file.write("\n\nEDA Ratings for Each Generation: ")
for rating in EDA_last_generation_ratings:
    file.write('%s' % rating)
    file.write(", ")
file.write("\nEDA Max Rating: ")
file.write('%s' % EDA_final_max)
file.write("\nEDA Average Rating: ")
file.write('%s' % EDA_final_average)

# close CASIS file
file.close()
