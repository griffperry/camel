import math


printable_characters = [" ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4" , "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", """\ """, "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~"]

def read_dataset(articles):
    fh = open(articles, "r")
    writing_samples = ""
    writing_samples = fh.read()
    writing_samples.lower()
    fh.close()
    return writing_samples

def feature_vectors(article, number):
    feature_vector = []
    for char in printable_characters:
        count = 0
        count = article.count(char)
        count = math.floor(count*10)/10
        feature_vector.append(count)

    feature_vector.extend([number])
    return feature_vector

def write_to_feature_vector_file(text_file, list_of_lists):
    fh = open(text_file, "w")
    for list in list_of_lists:
        fh.write(str(list) + "\n")
    fh.close()

def normalize_fv(text_file, list_of_lists):
    fh = open(text_file, "w")
    list_of_normalized_vectors = []
    euclidean_sum = 0
    normalized_value = 0
    for list in list_of_lists:
        normalized_vector = []
        total_sum = 0;
        for value in list:
            value = math.pow((value - 0), 2)
            total_sum += value
        euclidean_sum = math.sqrt(total_sum)
        for value in list:
            normalized_value = value / euclidean_sum
            normalized_vector.append(normalized_value)
        list_of_normalized_vectors.append(normalized_vector)
    for list in list_of_normalized_vectors:
        fh.write(str(list) + "\n")
    fh.close()
    return list_of_normalized_vectors

def euclideanDistance(vector1, vector2):
    distance = 0
    for character in range(95):
        distance += pow(vector1[character] - vector2[character], 2)
    return math.sqrt(distance)

def average_chars(article1, article2, article3):
    avg = len(article1) + len(article2) + len(article3) / 3
    return avg

def average_words(article1, article2, article3):
    article1 = article1.split(" ")
    article2 = article2.split(" ")
    article3 = article3.split(" ")
    average = len(article1) + len(article2) + len(article3) / 3
    return average

def average_sentences(article1, article2, article3):
    article1_period = article1.split(".")
    article2_period = article2.split(".")
    article3_period = article3.split(".")
    article1_exclamation = article1.split("!")
    article2_exclamation = article2.split("!")
    article3_exclamation = article3.split("!")
    article1_question = article1.split("?")
    article2_question = article2.split("?")
    article3_question = article3.split("?")

    article1 = (len(article1_period) + len(article1_exclamation) + len(article1_question))
    article2 = (len(article2_period) + len(article2_exclamation) + len(article2_question))
    article3 = (len(article3_period) + len(article3_exclamation) + len(article3_question))
    average = article1 + article2 + article3 / 3
    return average