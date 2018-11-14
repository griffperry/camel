# Leah, Steph, Griff (Unofficial group name: LSG)
# COMP 7976 Computation Intellegence with Adversarial Machine Learning
# Homework4
#
# Task: Make 3 feature vectors, that associate with 1, 2, and 3 authors.

import ast
import numpy.random as nprando

with open("Test_Normalized_feature_Vectors.txt", "r") as f:
   fvs = ast.literal_eval(f.readline())

  
# Parameters: takes a feature vector
# rounds the values to either a 0 or a 1
# returns a feature vector
def Round_To_Binary(fv):
    
    fv = ['%.f' % f for f in fv]
    
    return fv



    

# Parameters: takes in two feature vectors
# This  takes two feature vectors and returns the a vector of similiar elements 
# between the two.
# returns: a fv
def Compare_List(fv1, fv2):
    
    fv3 = set(fv1).intersection(fv2)
    
    return fv3

# Parameters: List of all the feature vectors, a target fv, the prediction vector
# Takes in this list of feature vectures and compares it to the target feature vector.
# Determine how similiar each feature vector is the target by the values 1,0 or -1
# Basically our descision function simplified
# Returns: a list of size twelve with either a 1, 0 or -1
def Predict_Author(fvs, targetfv, fv):
    return fv


# makes a random list filled with -1,0s, and 1s
def random_list_maker():
    lst = []
    
    for f in range(25):
        nprando.seed(1)
        random_list = [nprando.randint(-1,2) for r in xrange(95)]
        lst.append(random_list)
        
    return lst

print random_list_maker()


# Parameters: the single feature vector we want to change
# this function will do some modification to the feature vector in order to make
# our decision funtion choose it.
# return modified fv
def change_one(fv):
    return fv

# Parameters: A list of feature vectors 2 or 3
# this will change multiple fvs in some way but it will change them so that they
# all appear equal to the decision function.
def change_multiple(fvs):
    return fvs
    

# multiplies fv1 and fv2 together to get a new list
# fv3 = [a*b for a,b in zip(fv1,fv2)]