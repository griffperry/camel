import math
import operator

def dist_squared(t_q, t_i):
    sum = 0.0
    for character in range(len(t_q)):
        sum += math.pow((t_q[character] - t_i[character]), 2)
    return sum

def hf(tq, ti, sigma):
    return math.exp(-dist_squared(tq, ti)/pow((2.0*sigma), 2.0))

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

def compute_fire_strengths(index, test_set, training_set, sigma):

    author1 = list_maker(d, 12, 6, 1, 0)
    author2 = list_maker(d, 12, 6, 1, 1)
    author3 = list_maker(d, 12, 6, 1, 2)
    author4 = list_maker(d, 12, 6, 1, 3)
    author5 = list_maker(d, 12, 6, 1, 4)
    author6 = list_maker(d, 12, 6, 1, 5)
    author7 = list_maker(d, 12, 6, 1, 6)
    author8 = list_maker(d, 12, 6, 1, 7)
    author9 = list_maker(d, 12, 6, 1, 8)
    author10 = list_maker(d, 12, 6, 1, 9)
    author11 = list_maker(d, 12, 6, 1, 10)
    author12 = list_maker(d, 12, 6, 1, 11)

    i = 0
    dq = []
    while i < 12:
        dq.append(0)
        i += 1

    k = 0
    practice_set = []
    if (k != 0):
        practice_set = getNeighbors(training_set, test_set, k)

    else:
        practice_set = training_set

    hfs = []
    for trainthing in practice_set:
        hfs.append(hf(test_set, trainthing, sigma))

    temp_d = list(d)
    temp_d.pop(index)
    for i in range(len(practice_set)):
        for j in range(len(dq)):
            dq[j] += hfs[i]*temp_d[i][j]

    sum_hfs = 0
    for i in range(len(practice_set)):
        sum_hfs += hfs[i]

    for j in range(len(dq)):
        if (sum_hfs != 0):
            dq[j] = dq[j]/sum_hfs
    return dq

d = []
d2 = []
def list_maker(global_list, listSize, numOfList, element, index):

    count = 0
    while(count < numOfList):
        list = [0 for n in range(listSize)]
        list.insert(index, element)
        global_list.append(list)
        count = count + 1
    return

def compute_fire_strengths_CASIS25(index, test_set, training_set, sigma):

    author1 = list_maker(d2, 25, 4, 1, 0)
    author2 = list_maker(d2, 25, 4, 1, 1)
    author3 = list_maker(d2, 25, 4, 1, 2)
    author4 = list_maker(d2, 25, 4, 1, 3)
    author5 = list_maker(d2, 25, 4, 1, 4)
    author6 = list_maker(d2, 25, 4, 1, 5)
    author7 = list_maker(d2, 25, 4, 1, 6)
    author8 = list_maker(d2, 25, 4, 1, 7)
    author9 = list_maker(d2, 25, 4, 1, 8)
    author10 = list_maker(d2, 25, 4, 1, 9)
    author11 = list_maker(d2, 25, 4, 1, 10)
    author12 = list_maker(d2, 25, 4, 1, 11)
    author13 = list_maker(d2, 25, 4, 1, 12)
    author14 = list_maker(d2, 25, 4, 1, 13)
    author15 = list_maker(d2, 25, 4, 1, 14)
    author16 = list_maker(d2, 25, 4, 1, 15)
    author17 = list_maker(d2, 25, 4, 1, 16)
    author18 = list_maker(d2, 25, 4, 1, 17)
    author19 = list_maker(d2, 25, 4, 1, 18)
    author20 = list_maker(d2, 25, 4, 1, 19)
    author21 = list_maker(d2, 25, 4, 1, 20)
    author22 = list_maker(d2, 25, 4, 1, 21)
    author23 = list_maker(d2, 25, 4, 1, 22)
    author24 = list_maker(d2, 25, 4, 1, 23)
    author25 = list_maker(d2, 25, 4, 1, 24)

    i = 0
    dq = []
    while i < 25:
        dq.append(0)
        i += 1

    k = 0
    practice_set = []
    if (k != 0):
        practice_set = getNeighbors(training_set, test_set, k)

    else:
        practice_set = training_set

    hfs = []
    for trainthing in practice_set:
        hfs.append(hf(test_set, trainthing, sigma))

    temp_d = list(d2)
    temp_d.pop(index)
    for i in range(len(practice_set)):
        for j in range(len(dq)):
            dq[j] += hfs[i]*temp_d[i][j]

    sum_hfs = 0
    for i in range(len(practice_set)):
        sum_hfs += hfs[i]

    for j in range(len(dq)):
        if (sum_hfs != 0):
            dq[j] = dq[j]/sum_hfs
    return dq


