import random


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


# evaluates how well feature masks perform
# inputs: list of feature masks
# return: list of feature masks with rating
#         (example: [([list of fms], rating), ([list of fms], rating), etc.])
#def gen_evaluation(fm_list):


# for SSGA and EGA
# randomly selects x (normally 2) number of parents to be chosen to procreate
# inputs: list of feature masks with rating (gen_evaluation output)
# return: list of feature vectors to act as parents
#def randomly_select_parents(fmar_list):


# for EDA
# selects x (normally 12) number of best parents to be chosen to procreate
# inputs: list of feature masks with rating (gen_evaluation output)
# return: list of feature vectors to act as parents
#def randomly_select_parents(fmar_list):


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
#def replacement():


# EXAMPLES
fm_length = 10
number_of_fms = 5
number_of_children = 2

fm_list = initialize_population(number_of_fms, fm_length)
for fm in fm_list:
    print fm

print " "

child_list = procreate(fm_list, number_of_children)
for fm in child_list:
    print fm
