# Leah, Steph, Griff (Unofficial group name: LSG)
# COMP 7976 Computation Intellegence with Adversarial Machine Learning
# Homework4
#
# Task: Make 3 feature vectors, that associate with 1, 2, and 3 authors.

import ast
import numpy.random as nprando
from sklearn.cluster import KMeans


#fv means feature vector
with open("Test_Normalized_feature_Vectors.txt", "r") as f:
   fv1 = ast.literal_eval(f.readline())
    
#makes a random list filled with -1,0s, and 1s
fv2 = [nprando.randint(-1,2) for r in xrange(95)]

# multiplies fv1 and fv2 together to get a new list
fv3 = [a*b for a,b in zip(fv1,fv2)]

# suppose to read in the list of feature vectors and choose 1. I need help
def authorPredict(fv):
    aP = KMeans(n_clusters=2, init='k-means++', n_init=10, verbose=0)
    aP.fit(fv)
    
    return aP

print authorPredict(fv3)
    





     




