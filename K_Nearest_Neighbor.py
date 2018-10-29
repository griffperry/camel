import operator


def getNeighbors(trainingSet, testInstance, resultCount):
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

def getResponse(neighbors):
    close_neighbors = {}
    for neighbor in range(len(neighbors)):
        response = neighbors[neighbor][-1]
        if response in close_neighbors:
            close_neighbors[response] += 1
        else:
            close_neighbors[response] = 1
    sortedValues = sorted(close_neighbors.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedValues[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for test in range(len(testSet)):
        if testSet[test][-1] is predictions[test]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0
