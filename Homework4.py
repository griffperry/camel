# Leah, Steph, Griff (Unofficial group name: LSG)
# COMP 7976 Computation Intellegence with Adversarial Machine Learning
# Homework4
#
# Task: Make 3 feature vectors, that associate with 1, 2, and 3 authors.

import ast

with open("Test_Normalized_feature_Vectors.txt", "r") as f:
   fvs = ast.literal_eval(f.readlines())

  
# Parameters: takes a feature vector
# rounds the values to either a 0 or a 1
# returns a feature vector
def Round_To_Binary(fv):
    
    fv = ['%.f' % f for f in fv]
    
    return fv

print Round_To_Binary(fvs)

    

# Parameters: takes in two feature vectors
# This  takes two feature vectors and compares them, if they are similiar 
# print a 1 if not, -1. If they are the same size 0.
# returns: a 0, 1, -1
def Compare_List(fv1, fv2):
    return 0

# Parameters: List of all the feature vectors, a target fv, the prediction vector
# Takes in this list of feature vectures and compares it to the target feature vector.
# Determine how similiar each feature vector is the target by the values 1,0 or -1
# Returns: a list of size twelve with either a 1, 0 or -1
def Predict_Author(fvs, targetfv, fv):
    return fv


# makes a random list filled with -1,0s, and 1s
# fv2 = [nprando.randint(-1,2) for r in xrange(95)]

# multiplies fv1 and fv2 together to get a new list
# fv3 = [a*b for a,b in zip(fv1,fv2)]