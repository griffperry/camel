import operator
import math

def get_weighted_neighbors(trainingSet, testInstance, resultCount):
    from homework1 import euclideanDistance
    distances = []
    length = len(testInstance)
    for character in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[character])
        distances.append((trainingSet[character], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for result in range(resultCount):
        neighbors.append(distances[result][0])
    return neighbors

def get_weighted_distance(vector_list, test_instance):
    from homework1 import euclideanDistance
    weights = []
    for vector in range(len(vector_list)):
        distance = euclideanDistance(vector_list[vector], test_instance)
        weight = 1 / (pow(distance, 5))
        weights.append(weight)
    return weights

def local_method(neighbors, test_instance):
    weights = get_weighted_distance(neighbors, test_instance)
    weight_sum = 0
    author_weight_sum = 0

    for index in range(len(weights)):
        weight_sum += weights[index]
        author = neighbors[index][-1]
        author_weight_sum += author * weights[index]
    closest = author_weight_sum / weight_sum
    rounded_closest = round(closest, 0)
    return int (rounded_closest)

def global_method(feature_vector_list, test_instance):
    weights = get_weighted_distance(feature_vector_list, test_instance)
    weight_sum = 0
    author_weight_sum = 0
    #duplicate_author = []

    for index in range(len(weights)):
        weight_sum += weights[index]
        author = feature_vector_list[index][-1]
        #for i in range(len(duplicate_author)):
            #if (author == duplicate_author[i]):
                #weights[index] = math.sqrt(weights[index])
        author_weight_sum += author * weights[index]
        #duplicate_author.append(author)
    closest = author_weight_sum / weight_sum
    rounded_closest = round(closest, 0)
    return int (rounded_closest)

def get_weighted_accuracy(testSet, predictions):
    correct = 0
    for test in range(len(testSet)):
        if testSet[test][-1] is predictions[test]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0