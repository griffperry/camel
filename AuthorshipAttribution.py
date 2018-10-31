#!/usr/bin/python
from K_Nearest_Neighbor import getAccuracy, getResponse, getNeighbors
from K_Weighted_Nearest_Neighbor import get_weighted_neighbors, get_weighted_distance, global_method, local_method,\
    get_weighted_accuracy
from homework1 import read_dataset, feature_vectors, write_to_feature_vector_file, normalize_fv
from GRNN import dist_squared, compute_fire_strengths, hf, compute_fire_strengths_CASIS25
from GeneticAlgorithms import replacement, evaluation, procreate, tournament_select_parents, initialize_population,\
    select_best_parents


#read in text files
week_1_author_1 = read_dataset("247Sports_Brown_TennesseevsWVA_wk1.txt")
week_2_author_1 = read_dataset("247Sports_Brown_TennesseeVsETN_wk2.txt")
week_3_author_1 = read_dataset("247Sports_Brown_TennesseeVsUTEP_wk3.txt")
week_4_author_1 = read_dataset("247Sports_Brown_TennesseeVsFLO_wk4.txt")
week_5_author_1 = read_dataset("247Sports_Brown_TennesseeVsGA_wk5.txt")
week_6_author_1 = read_dataset("247Sports_Brown_TennesseeVsAUB_wk6.txt")

week_1_author_2 = read_dataset("247Sports_Callahan_TennesseevsWVA_wk1.txt")
week_2_author_2 = read_dataset("247Sports_Callahan_TennesseeVsETN_wk2.txt")
week_3_author_2 = read_dataset("247Sports_Callahan_TennesseeVsUTEP_wk3.txt")
week_4_author_2 = read_dataset("247Sports_Callahan_TennesseeVsFLO_wk4.txt")
week_5_author_2 = read_dataset("247Sports_Callahan_TennesseeVsGA_wk5.txt")
week_6_author_2 = read_dataset("247Sports_Callahan_TennesseeVsAUB_wk6.txt")

week_1_author_3 = read_dataset("247Sports_Ramey_TennesseevsWVA_wk1.txt")
week_2_author_3 = read_dataset("247Sports_Ramey_TennesseeVsETN_wk2.txt")
week_3_author_3 = read_dataset("247Sports_Ramey_TennesseeVsUTEP_wk3.txt")
week_4_author_3 = read_dataset("247Sports_Ramey_TennesseeVsFLO_wk4.txt")
week_5_author_3 = read_dataset("247Sports_Ramey_TennesseeVsGA_wk5.txt")
week_6_author_3 = read_dataset("247Sports_Ramey_TennesseeVsAUB_wk6.txt")

week_1_author_4 = read_dataset("247Sports_Rucker_TennesseevsWVA_wk1.txt")
week_2_author_4 = read_dataset("247Sports_Rucker_TennesseeVsETN_wk2.txt")
week_3_author_4 = read_dataset("247Sports_Rucker_TennesseeVsUTEP_wk3.txt")
week_4_author_4 = read_dataset("247Sports_Rucker_TennesseeVsFLO_wk4.txt")
week_5_author_4 = read_dataset("247Sports_Rucker_TennesseeVsGA_wk5.txt")
week_6_author_4 = read_dataset("247Sports_Rucker_TennesseeVsAUB_wk6.txt")

week_1_author_5 = read_dataset("RockyTopTalk_Burlage_TennesseeVsWVA_wk1.txt")
week_2_author_5 = read_dataset("RockyTopTalk_Burlage_TennesseeVsETSU_wk2.txt")
week_3_author_5 = read_dataset("RockyTopTalk_Burlage_TennesseeVsUTEP_wk3.txt")
week_4_author_5 = read_dataset("RockyTopTalk_Burlage_TennesseeVsFL_wk4.txt")
week_5_author_5 = read_dataset("RockyTopTalk_Burlage_TennesseeVsGA_wk5.txt")
week_6_author_5 = read_dataset("RockyTopTalk_Burlage_TennesseeVsAUB_wk6.txt")

week_1_author_6 = read_dataset("CFN_Fiutak_TennesseevsWVA_wk1.txt")
week_2_author_6 = read_dataset("CFN_Fiutak_TennesseeVsETN_wk2.txt")
week_3_author_6 = read_dataset("CFN_Fiutak_TennesseeVsUTEP_wk3.txt")
week_4_author_6 = read_dataset("CFN_Fiutak_TennesseeVsFLO_wk4.txt")
week_5_author_6 = read_dataset("CFN_Fiutak_TennesseeVsGA_wk5.txt")
week_6_author_6 = read_dataset("CFN_Fiutak_TennesseeVsAUB_wk6.txt")

week_1_author_7 = read_dataset("RockyTopTalk_Knapp_TennesseeVsWVA_wk1.txt")
week_2_author_7 = read_dataset("RockyTopTalk_Knapp_TennesseeVsETSU_wk2.txt")
week_3_author_7 = read_dataset("RockyTopTalk_Knapp_TennesseeVsUTEP_wk3.txt")
week_4_author_7 = read_dataset("RockyTopTalk_Knapp_TennesseeVsFL_wk4.txt")
week_5_author_7 = read_dataset("RockyTopTalk_Knapp_TennesseeVsGA_wk5.txt")
week_6_author_7 = read_dataset("RockyTopTalk_Knapp_TennesseeVsAUB_wk6.txt")

week_1_author_8 = read_dataset("Knoxnews_Toppmeyer_TennesseevsWVA_wk1.txt")
week_2_author_8 = read_dataset("Knoxnews_Toppmeyer_TennesseeVsETN_wk2.txt")
week_3_author_8 = read_dataset("Knoxnews_Toppmeyer_TennesseeVsUTEP_wk3.txt")
week_4_author_8 = read_dataset("Knoxnews_Toppmeyer_TennesseeVsFLO_wk4.txt")
week_5_author_8 = read_dataset("Knoxnews_Toppmeyer_TennesseeVsGA_wk5.txt")
week_6_author_8 = read_dataset("Knoxnews_Toppmeyer_TennesseeVsAUB_wk6.txt")

week_1_author_9 = read_dataset("Knoxnews_Wilson_TennesseevsWVA_wk1.txt")
week_2_author_9 = read_dataset("Knoxnews_Wilson_TennesseeVsETN_wk2.txt")
week_3_author_9 = read_dataset("Knoxnews_Wilson_TennesseeVsUTEP_wk3.txt")
week_4_author_9 = read_dataset("Knoxnews_Wilson_TennesseeVsFLO_wk4.txt")
week_5_author_9 = read_dataset("Knoxnews_Wilson_TennesseeVsGA_wk5.txt")
week_6_author_9 = read_dataset("Knoxnews_Wilson_TennesseeVsAUB_wk6.txt")

week_1_author_10 = read_dataset("RockyTopTalk_Winter_TennesseevsWVA_wk1.txt")
week_2_author_10 = read_dataset("RockyTopTalk_Winter_TennesseeVsETN_wk2.txt")
week_3_author_10 = read_dataset("RockyTopTalk_Winter_TennesseeVsUTEP_wk3.txt")
week_4_author_10 = read_dataset("RockyTopTalk_Winter_TennesseeVsFLO_wk4.txt")
week_5_author_10 = read_dataset("RockyTopTalk_Winter_TennesseeVsGA_wk5.txt")
week_6_author_10 = read_dataset("RockyTopTalk_Winter_TennesseeVsAUB_wk6.txt")

week_1_author_11 = read_dataset("RockyTopTalk_Lambert_TennesseevsWVA_wk1.txt")
week_2_author_11 = read_dataset("RockyTopTalk_Lambert_TennesseeVsETSU_wk2.txt")
week_3_author_11 = read_dataset("RockyTopTalk_Lambert_TennesseeVsUTEP_wk3.txt")
week_4_author_11 = read_dataset("RockyTopTalk_Lambert_TennesseeVsFL_wk4.txt")
week_5_author_11 = read_dataset("RockyTopTalk_Lambert_TennesseeVsGA_wk5.txt")
week_6_author_11 = read_dataset("RockyTopTalk_Lambert_TennesseeVsAUB_wk6.txt")

week_1_author_12 = read_dataset("RockyTopTalk_Taylor_TennesseevsWVA_wk1.txt")
week_2_author_12 = read_dataset("RockyTopTalk_Taylor_TennesseeVsETSU_wk2.txt")
week_3_author_12 = read_dataset("RockyTopTalk_Taylor_TennesseeVsUTEP_wk3.txt")
week_4_author_12 = read_dataset("RockyTopTalk_Taylor_TennesseeVsFL_wk4.txt")
week_5_author_12 = read_dataset("RockyTopTalk_Taylor_TennesseeVsGA_wk5.txt")
week_6_author_12 = read_dataset("RockyTopTalk_Taylor_TennesseeVsAUB_wk6.txt")


#make the feature vectors and store in a list of feature vectors
feature_vector_list = []
week_1_author_1_fv = feature_vectors(week_1_author_1, 1)
week_2_author_1_fv = feature_vectors(week_2_author_1, 1)
week_3_author_1_fv = feature_vectors(week_3_author_1, 1)
week_4_author_1_fv = feature_vectors(week_4_author_1, 1)
week_5_author_1_fv = feature_vectors(week_5_author_1, 1)
week_6_author_1_fv = feature_vectors(week_6_author_1, 1)

feature_vector_list.append(week_1_author_1_fv)
feature_vector_list.append(week_2_author_1_fv)
feature_vector_list.append(week_3_author_1_fv)
feature_vector_list.append(week_4_author_1_fv)
feature_vector_list.append(week_5_author_1_fv)
feature_vector_list.append(week_6_author_1_fv)


week_1_author_2_fv = feature_vectors(week_1_author_2, 2)
week_2_author_2_fv = feature_vectors(week_2_author_2, 2)
week_3_author_2_fv = feature_vectors(week_3_author_2, 2)
week_4_author_2_fv = feature_vectors(week_4_author_2, 2)
week_5_author_2_fv = feature_vectors(week_5_author_2, 2)
week_6_author_2_fv = feature_vectors(week_6_author_2, 2)

feature_vector_list.append(week_1_author_2_fv)
feature_vector_list.append(week_2_author_2_fv)
feature_vector_list.append(week_3_author_2_fv)
feature_vector_list.append(week_4_author_2_fv)
feature_vector_list.append(week_5_author_2_fv)
feature_vector_list.append(week_6_author_2_fv)


week_1_author_3_fv = feature_vectors(week_1_author_3, 3)
week_2_author_3_fv = feature_vectors(week_2_author_3, 3)
week_3_author_3_fv = feature_vectors(week_3_author_3, 3)
week_4_author_3_fv = feature_vectors(week_4_author_3, 3)
week_5_author_3_fv = feature_vectors(week_5_author_3, 3)
week_6_author_3_fv = feature_vectors(week_6_author_3, 3)

feature_vector_list.append(week_1_author_3_fv)
feature_vector_list.append(week_2_author_3_fv)
feature_vector_list.append(week_3_author_3_fv)
feature_vector_list.append(week_4_author_3_fv)
feature_vector_list.append(week_5_author_3_fv)
feature_vector_list.append(week_6_author_3_fv)


week_1_author_4_fv = feature_vectors(week_1_author_4, 4)
week_2_author_4_fv = feature_vectors(week_2_author_4, 4)
week_3_author_4_fv = feature_vectors(week_3_author_4, 4)
week_4_author_4_fv = feature_vectors(week_4_author_4, 4)
week_5_author_4_fv = feature_vectors(week_5_author_4, 4)
week_6_author_4_fv = feature_vectors(week_6_author_4, 4)

feature_vector_list.append(week_1_author_4_fv)
feature_vector_list.append(week_2_author_4_fv)
feature_vector_list.append(week_3_author_4_fv)
feature_vector_list.append(week_4_author_4_fv)
feature_vector_list.append(week_5_author_4_fv)
feature_vector_list.append(week_6_author_4_fv)


week_1_author_5_fv = feature_vectors(week_1_author_5, 5)
week_2_author_5_fv = feature_vectors(week_2_author_5, 5)
week_3_author_5_fv = feature_vectors(week_3_author_5, 5)
week_4_author_5_fv = feature_vectors(week_4_author_5, 5)
week_5_author_5_fv = feature_vectors(week_5_author_5, 5)
week_6_author_5_fv = feature_vectors(week_6_author_5, 5)

feature_vector_list.append(week_1_author_5_fv)
feature_vector_list.append(week_2_author_5_fv)
feature_vector_list.append(week_3_author_5_fv)
feature_vector_list.append(week_4_author_5_fv)
feature_vector_list.append(week_5_author_5_fv)
feature_vector_list.append(week_6_author_5_fv)


week_1_author_6_fv = feature_vectors(week_1_author_6, 6)
week_2_author_6_fv = feature_vectors(week_2_author_6, 6)
week_3_author_6_fv = feature_vectors(week_3_author_6, 6)
week_4_author_6_fv = feature_vectors(week_4_author_6, 6)
week_5_author_6_fv = feature_vectors(week_5_author_6, 6)
week_6_author_6_fv = feature_vectors(week_6_author_6, 6)

feature_vector_list.append(week_1_author_6_fv)
feature_vector_list.append(week_2_author_6_fv)
feature_vector_list.append(week_3_author_6_fv)
feature_vector_list.append(week_4_author_6_fv)
feature_vector_list.append(week_5_author_6_fv)
feature_vector_list.append(week_6_author_6_fv)


week_1_author_7_fv = feature_vectors(week_1_author_7, 7)
week_2_author_7_fv = feature_vectors(week_2_author_7, 7)
week_3_author_7_fv = feature_vectors(week_3_author_7, 7)
week_4_author_7_fv = feature_vectors(week_4_author_7, 7)
week_5_author_7_fv = feature_vectors(week_5_author_7, 7)
week_6_author_7_fv = feature_vectors(week_6_author_7, 7)

feature_vector_list.append(week_1_author_7_fv)
feature_vector_list.append(week_2_author_7_fv)
feature_vector_list.append(week_3_author_7_fv)
feature_vector_list.append(week_4_author_7_fv)
feature_vector_list.append(week_5_author_7_fv)
feature_vector_list.append(week_6_author_7_fv)


week_1_author_8_fv = feature_vectors(week_1_author_8, 8)
week_2_author_8_fv = feature_vectors(week_2_author_8, 8)
week_3_author_8_fv = feature_vectors(week_3_author_8, 8)
week_4_author_8_fv = feature_vectors(week_4_author_8, 8)
week_5_author_8_fv = feature_vectors(week_5_author_8, 8)
week_6_author_8_fv = feature_vectors(week_6_author_8, 8)

feature_vector_list.append(week_1_author_8_fv)
feature_vector_list.append(week_2_author_8_fv)
feature_vector_list.append(week_3_author_8_fv)
feature_vector_list.append(week_4_author_8_fv)
feature_vector_list.append(week_5_author_8_fv)
feature_vector_list.append(week_6_author_8_fv)


week_1_author_9_fv = feature_vectors(week_1_author_9, 9)
week_2_author_9_fv = feature_vectors(week_2_author_9, 9)
week_3_author_9_fv = feature_vectors(week_3_author_9, 9)
week_4_author_9_fv = feature_vectors(week_4_author_9, 9)
week_5_author_9_fv = feature_vectors(week_5_author_9, 9)
week_6_author_9_fv = feature_vectors(week_6_author_9, 9)

feature_vector_list.append(week_1_author_9_fv)
feature_vector_list.append(week_2_author_9_fv)
feature_vector_list.append(week_3_author_9_fv)
feature_vector_list.append(week_4_author_9_fv)
feature_vector_list.append(week_5_author_9_fv)
feature_vector_list.append(week_6_author_9_fv)


week_1_author_10_fv = feature_vectors(week_1_author_10, 10)
week_2_author_10_fv = feature_vectors(week_2_author_10, 10)
week_3_author_10_fv = feature_vectors(week_3_author_10, 10)
week_4_author_10_fv = feature_vectors(week_4_author_10, 10)
week_5_author_10_fv = feature_vectors(week_5_author_10, 10)
week_6_author_10_fv = feature_vectors(week_6_author_10, 10)

feature_vector_list.append(week_1_author_10_fv)
feature_vector_list.append(week_2_author_10_fv)
feature_vector_list.append(week_3_author_10_fv)
feature_vector_list.append(week_4_author_10_fv)
feature_vector_list.append(week_5_author_10_fv)
feature_vector_list.append(week_6_author_10_fv)


week_1_author_11_fv = feature_vectors(week_1_author_11, 11)
week_2_author_11_fv = feature_vectors(week_2_author_11, 11)
week_3_author_11_fv = feature_vectors(week_3_author_11, 11)
week_4_author_11_fv = feature_vectors(week_4_author_11, 11)
week_5_author_11_fv = feature_vectors(week_5_author_11, 11)
week_6_author_11_fv = feature_vectors(week_6_author_11, 11)

feature_vector_list.append(week_1_author_11_fv)
feature_vector_list.append(week_2_author_11_fv)
feature_vector_list.append(week_3_author_11_fv)
feature_vector_list.append(week_4_author_11_fv)
feature_vector_list.append(week_5_author_11_fv)
feature_vector_list.append(week_6_author_11_fv)


week_1_author_12_fv = feature_vectors(week_1_author_12, 12)
week_2_author_12_fv = feature_vectors(week_2_author_12, 12)
week_3_author_12_fv = feature_vectors(week_3_author_12, 12)
week_4_author_12_fv = feature_vectors(week_4_author_12, 12)
week_5_author_12_fv = feature_vectors(week_5_author_12, 12)
week_6_author_12_fv = feature_vectors(week_6_author_12, 12)

feature_vector_list.append(week_1_author_12_fv)
feature_vector_list.append(week_2_author_12_fv)
feature_vector_list.append(week_3_author_12_fv)
feature_vector_list.append(week_4_author_12_fv)
feature_vector_list.append(week_5_author_12_fv)
feature_vector_list.append(week_6_author_12_fv)

#Test against CASIS-25 Dataset
test_1000_1 = read_dataset("1000_1.txt")
test_1000_2 = read_dataset("1000_2.txt")
test_1000_3 = read_dataset("1000_3.txt")
test_1000_4 = read_dataset("1000_4.txt")
test_1001_1 = read_dataset("1001_1.txt")
test_1001_2 = read_dataset("1001_2.txt")
test_1001_3 = read_dataset("1001_3.txt")
test_1001_4 = read_dataset("1001_4.txt")
test_1002_1 = read_dataset("1002_1.txt")
test_1002_2 = read_dataset("1002_2.txt")
test_1002_3 = read_dataset("1002_3.txt")
test_1002_4 = read_dataset("1002_4.txt")
test_1003_1 = read_dataset("1003_1.txt")
test_1003_2 = read_dataset("1003_2.txt")
test_1003_3 = read_dataset("1003_3.txt")
test_1003_4 = read_dataset("1003_4.txt")
test_1004_1 = read_dataset("1004_1.txt")
test_1004_2 = read_dataset("1004_2.txt")
test_1004_3 = read_dataset("1004_3.txt")
test_1004_4 = read_dataset("1004_4.txt")
test_1005_1 = read_dataset("1005_1.txt")
test_1005_2 = read_dataset("1005_2.txt")
test_1005_3 = read_dataset("1005_3.txt")
test_1005_4 = read_dataset("1005_4.txt")
test_1006_1 = read_dataset("1006_1.txt")
test_1006_2 = read_dataset("1006_2.txt")
test_1006_3 = read_dataset("1006_3.txt")
test_1006_4 = read_dataset("1006_4.txt")
test_1007_1 = read_dataset("1007_1.txt")
test_1007_2 = read_dataset("1007_2.txt")
test_1007_3 = read_dataset("1007_3.txt")
test_1007_4 = read_dataset("1007_4.txt")
test_1008_1 = read_dataset("1008_1.txt")
test_1008_2 = read_dataset("1008_2.txt")
test_1008_3 = read_dataset("1008_3.txt")
test_1008_4 = read_dataset("1008_4.txt")
test_1009_1 = read_dataset("1009_1.txt")
test_1009_2 = read_dataset("1009_2.txt")
test_1009_3 = read_dataset("1009_3.txt")
test_1009_4 = read_dataset("1009_4.txt")
test_1010_1 = read_dataset("1010_1.txt")
test_1010_2 = read_dataset("1010_2.txt")
test_1010_3 = read_dataset("1010_3.txt")
test_1010_4 = read_dataset("1010_4.txt")
test_1011_1 = read_dataset("1011_1.txt")
test_1011_2 = read_dataset("1011_2.txt")
test_1011_3 = read_dataset("1011_3.txt")
test_1011_4 = read_dataset("1011_4.txt")
test_1012_1 = read_dataset("1012_1.txt")
test_1012_2 = read_dataset("1012_2.txt")
test_1012_3 = read_dataset("1012_3.txt")
test_1012_4 = read_dataset("1012_4.txt")
test_1013_1 = read_dataset("1013_1.txt")
test_1013_2 = read_dataset("1013_2.txt")
test_1013_3 = read_dataset("1013_3.txt")
test_1013_4 = read_dataset("1013_4.txt")
test_1014_1 = read_dataset("1014_1.txt")
test_1014_2 = read_dataset("1014_2.txt")
test_1014_3 = read_dataset("1014_3.txt")
test_1014_4 = read_dataset("1014_4.txt")
test_1015_1 = read_dataset("1015_1.txt")
test_1015_2 = read_dataset("1015_2.txt")
test_1015_3 = read_dataset("1015_3.txt")
test_1015_4 = read_dataset("1015_4.txt")
test_1016_1 = read_dataset("1016_1.txt")
test_1016_2 = read_dataset("1016_2.txt")
test_1016_3 = read_dataset("1016_3.txt")
test_1016_4 = read_dataset("1016_4.txt")
test_1017_1 = read_dataset("1017_1.txt")
test_1017_2 = read_dataset("1017_2.txt")
test_1017_3 = read_dataset("1017_3.txt")
test_1017_4 = read_dataset("1017_4.txt")
test_1018_1 = read_dataset("1018_1.txt")
test_1018_2 = read_dataset("1018_2.txt")
test_1018_3 = read_dataset("1018_3.txt")
test_1018_4 = read_dataset("1018_4.txt")
test_1019_1 = read_dataset("1019_1.txt")
test_1019_2 = read_dataset("1019_2.txt")
test_1019_3 = read_dataset("1019_3.txt")
test_1019_4 = read_dataset("1019_4.txt")
test_1020_1 = read_dataset("1020_1.txt")
test_1020_2 = read_dataset("1020_2.txt")
test_1020_3 = read_dataset("1020_3.txt")
test_1020_4 = read_dataset("1020_4.txt")
test_1021_1 = read_dataset("1021_1.txt")
test_1021_2 = read_dataset("1021_2.txt")
test_1021_3 = read_dataset("1021_3.txt")
test_1021_4 = read_dataset("1021_4.txt")
test_1022_1 = read_dataset("1022_1.txt")
test_1022_2 = read_dataset("1022_2.txt")
test_1022_3 = read_dataset("1022_3.txt")
test_1022_4 = read_dataset("1022_4.txt")
test_1023_1 = read_dataset("1023_1.txt")
test_1023_2 = read_dataset("1023_2.txt")
test_1023_3 = read_dataset("1023_3.txt")
test_1023_4 = read_dataset("1023_4.txt")
test_1024_1 = read_dataset("1024_1.txt")
test_1024_2 = read_dataset("1024_2.txt")
test_1024_3 = read_dataset("1024_3.txt")
test_1024_4 = read_dataset("1024_4.txt")

test_list = []

test_1000_1_fv = feature_vectors(test_1000_1, 1)
test_1000_2_fv = feature_vectors(test_1000_2, 1)
test_1000_3_fv = feature_vectors(test_1000_3, 1)
test_1000_4_fv = feature_vectors(test_1000_4, 1)

test_1001_1_fv = feature_vectors(test_1001_1, 2)
test_1001_2_fv = feature_vectors(test_1001_2, 2)
test_1001_3_fv = feature_vectors(test_1001_3, 2)
test_1001_4_fv = feature_vectors(test_1001_4, 2)

test_1002_1_fv = feature_vectors(test_1002_1, 3)
test_1002_2_fv = feature_vectors(test_1002_2, 3)
test_1002_3_fv = feature_vectors(test_1002_3, 3)
test_1002_4_fv = feature_vectors(test_1002_4, 3)

test_1003_1_fv = feature_vectors(test_1003_1, 4)
test_1003_2_fv = feature_vectors(test_1003_2, 4)
test_1003_3_fv = feature_vectors(test_1003_3, 4)
test_1003_4_fv = feature_vectors(test_1003_4, 4)

test_1004_1_fv = feature_vectors(test_1004_1, 5)
test_1004_2_fv = feature_vectors(test_1004_2, 5)
test_1004_3_fv = feature_vectors(test_1004_3, 5)
test_1004_4_fv = feature_vectors(test_1004_4, 5)

test_1005_1_fv = feature_vectors(test_1005_1, 6)
test_1005_2_fv = feature_vectors(test_1005_2, 6)
test_1005_3_fv = feature_vectors(test_1005_3, 6)
test_1005_4_fv = feature_vectors(test_1005_4, 6)

test_1006_1_fv = feature_vectors(test_1006_1, 7)
test_1006_2_fv = feature_vectors(test_1006_2, 7)
test_1006_3_fv = feature_vectors(test_1006_3, 7)
test_1006_4_fv = feature_vectors(test_1006_4, 7)

test_1007_1_fv = feature_vectors(test_1007_1, 8)
test_1007_2_fv = feature_vectors(test_1007_2, 8)
test_1007_3_fv = feature_vectors(test_1007_3, 8)
test_1007_4_fv = feature_vectors(test_1007_4, 8)

test_1008_1_fv = feature_vectors(test_1008_1, 9)
test_1008_2_fv = feature_vectors(test_1008_2, 9)
test_1008_3_fv = feature_vectors(test_1008_3, 9)
test_1008_4_fv = feature_vectors(test_1008_4, 9)

test_1009_1_fv = feature_vectors(test_1009_1, 10)
test_1009_2_fv = feature_vectors(test_1009_2, 10)
test_1009_3_fv = feature_vectors(test_1009_3, 10)
test_1009_4_fv = feature_vectors(test_1009_4, 10)

test_1010_1_fv = feature_vectors(test_1010_1, 11)
test_1010_2_fv = feature_vectors(test_1010_2, 11)
test_1010_3_fv = feature_vectors(test_1010_3, 11)
test_1010_4_fv = feature_vectors(test_1010_4, 11)
8
test_1011_1_fv = feature_vectors(test_1011_1, 12)
test_1011_2_fv = feature_vectors(test_1011_2, 12)
test_1011_3_fv = feature_vectors(test_1011_3, 12)
test_1011_4_fv = feature_vectors(test_1011_4, 12)

test_1012_1_fv = feature_vectors(test_1012_1, 13)
test_1012_2_fv = feature_vectors(test_1012_2, 13)
test_1012_3_fv = feature_vectors(test_1012_3, 13)
test_1012_4_fv = feature_vectors(test_1012_4, 13)

test_1013_1_fv = feature_vectors(test_1013_1, 14)
test_1013_2_fv = feature_vectors(test_1013_2, 14)
test_1013_3_fv = feature_vectors(test_1013_3, 14)
test_1013_4_fv = feature_vectors(test_1013_4, 14)

test_1014_1_fv = feature_vectors(test_1014_1, 15)
test_1014_2_fv = feature_vectors(test_1014_2, 15)
test_1014_3_fv = feature_vectors(test_1014_3, 15)
test_1014_4_fv = feature_vectors(test_1014_4, 15)

test_1015_1_fv = feature_vectors(test_1015_1, 16)
test_1015_2_fv = feature_vectors(test_1015_2, 16)
test_1015_3_fv = feature_vectors(test_1015_3, 16)
test_1015_4_fv = feature_vectors(test_1015_4, 16)

test_1016_1_fv = feature_vectors(test_1016_1, 17)
test_1016_2_fv = feature_vectors(test_1016_2, 17)
test_1016_3_fv = feature_vectors(test_1016_3, 17)
test_1016_4_fv = feature_vectors(test_1016_4, 17)

test_1017_1_fv = feature_vectors(test_1017_1, 18)
test_1017_2_fv = feature_vectors(test_1017_2, 18)
test_1017_3_fv = feature_vectors(test_1017_3, 18)
test_1017_4_fv = feature_vectors(test_1017_4, 18)

test_1018_1_fv = feature_vectors(test_1018_1, 19)
test_1018_2_fv = feature_vectors(test_1018_2, 19)
test_1018_3_fv = feature_vectors(test_1018_3, 19)
test_1018_4_fv = feature_vectors(test_1018_4, 19)

test_1019_1_fv = feature_vectors(test_1019_1, 20)
test_1019_2_fv = feature_vectors(test_1019_2, 20)
test_1019_3_fv = feature_vectors(test_1019_3, 20)
test_1019_4_fv = feature_vectors(test_1019_4, 20)

test_1020_1_fv = feature_vectors(test_1020_1, 21)
test_1020_2_fv = feature_vectors(test_1020_2, 21)
test_1020_3_fv = feature_vectors(test_1020_3, 21)
test_1020_4_fv = feature_vectors(test_1020_4, 21)

test_1021_1_fv = feature_vectors(test_1021_1, 22)
test_1021_2_fv = feature_vectors(test_1021_2, 22)
test_1021_3_fv = feature_vectors(test_1021_3, 22)
test_1021_4_fv = feature_vectors(test_1021_4, 22)

test_1022_1_fv = feature_vectors(test_1022_1, 23)
test_1022_2_fv = feature_vectors(test_1022_2, 23)
test_1022_3_fv = feature_vectors(test_1022_3, 23)
test_1022_4_fv = feature_vectors(test_1022_4, 23)

test_1023_1_fv = feature_vectors(test_1023_1, 24)
test_1023_2_fv = feature_vectors(test_1023_2, 24)
test_1023_3_fv = feature_vectors(test_1023_3, 24)
test_1023_4_fv = feature_vectors(test_1023_4, 24)

test_1024_1_fv = feature_vectors(test_1024_1, 25)
test_1024_2_fv = feature_vectors(test_1024_2, 25)
test_1024_3_fv = feature_vectors(test_1024_3, 25)
test_1024_4_fv = feature_vectors(test_1024_4, 25)


test_list.append(test_1000_1_fv)
test_list.append(test_1000_2_fv)
test_list.append(test_1000_3_fv)
test_list.append(test_1000_4_fv)
test_list.append(test_1001_1_fv)
test_list.append(test_1001_2_fv)
test_list.append(test_1001_3_fv)
test_list.append(test_1001_4_fv)
test_list.append(test_1002_1_fv)
test_list.append(test_1002_2_fv)
test_list.append(test_1002_3_fv)
test_list.append(test_1002_4_fv)
test_list.append(test_1003_1_fv)
test_list.append(test_1003_2_fv)
test_list.append(test_1003_3_fv)
test_list.append(test_1003_4_fv)
test_list.append(test_1004_1_fv)
test_list.append(test_1004_2_fv)
test_list.append(test_1004_3_fv)
test_list.append(test_1004_4_fv)
test_list.append(test_1005_1_fv)
test_list.append(test_1005_2_fv)
test_list.append(test_1005_3_fv)
test_list.append(test_1005_4_fv)
test_list.append(test_1006_1_fv)
test_list.append(test_1006_2_fv)
test_list.append(test_1006_3_fv)
test_list.append(test_1006_4_fv)
test_list.append(test_1007_1_fv)
test_list.append(test_1007_2_fv)
test_list.append(test_1007_3_fv)
test_list.append(test_1007_4_fv)
test_list.append(test_1008_1_fv)
test_list.append(test_1008_2_fv)
test_list.append(test_1008_3_fv)
test_list.append(test_1008_4_fv)
test_list.append(test_1009_1_fv)
test_list.append(test_1009_2_fv)
test_list.append(test_1009_3_fv)
test_list.append(test_1009_4_fv)
test_list.append(test_1010_1_fv)
test_list.append(test_1010_2_fv)
test_list.append(test_1010_3_fv)
test_list.append(test_1010_4_fv)
test_list.append(test_1011_1_fv)
test_list.append(test_1011_2_fv)
test_list.append(test_1011_3_fv)
test_list.append(test_1011_4_fv)
test_list.append(test_1012_1_fv)
test_list.append(test_1012_2_fv)
test_list.append(test_1012_3_fv)
test_list.append(test_1012_4_fv)
test_list.append(test_1013_1_fv)
test_list.append(test_1013_2_fv)
test_list.append(test_1013_3_fv)
test_list.append(test_1013_4_fv)
test_list.append(test_1014_1_fv)
test_list.append(test_1014_2_fv)
test_list.append(test_1014_3_fv)
test_list.append(test_1014_4_fv)
test_list.append(test_1015_1_fv)
test_list.append(test_1015_2_fv)
test_list.append(test_1015_3_fv)
test_list.append(test_1015_4_fv)
test_list.append(test_1016_1_fv)
test_list.append(test_1016_2_fv)
test_list.append(test_1016_3_fv)
test_list.append(test_1016_4_fv)
test_list.append(test_1017_1_fv)
test_list.append(test_1017_2_fv)
test_list.append(test_1017_3_fv)
test_list.append(test_1017_4_fv)
test_list.append(test_1018_1_fv)
test_list.append(test_1018_2_fv)
test_list.append(test_1018_3_fv)
test_list.append(test_1018_4_fv)
test_list.append(test_1019_1_fv)
test_list.append(test_1019_2_fv)
test_list.append(test_1019_3_fv)
test_list.append(test_1019_4_fv)
test_list.append(test_1020_1_fv)
test_list.append(test_1020_2_fv)
test_list.append(test_1020_3_fv)
test_list.append(test_1020_4_fv)
test_list.append(test_1021_1_fv)
test_list.append(test_1021_2_fv)
test_list.append(test_1021_3_fv)
test_list.append(test_1021_4_fv)
test_list.append(test_1022_1_fv)
test_list.append(test_1022_2_fv)
test_list.append(test_1022_3_fv)
test_list.append(test_1022_4_fv)
test_list.append(test_1023_1_fv)
test_list.append(test_1023_2_fv)
test_list.append(test_1023_3_fv)
test_list.append(test_1023_4_fv)
test_list.append(test_1024_1_fv)
test_list.append(test_1024_2_fv)
test_list.append(test_1024_3_fv)
test_list.append(test_1024_4_fv)


#Write raw and normalized feature vectors to files
write_to_feature_vector_file("SEC_Sportswriters_Feature_Vectors.txt", feature_vector_list)
normalize_fv("SEC_Sportswriters_Normalized_Feature_Vectors.txt", feature_vector_list)

#Display results of K Nearest Neighbor
def leave_one_out_knn(play_list):
    k = 1
    predictions = []

    print 'K_Nearest_Neighbor Results:'

    for index in range(len(play_list)):
        temp_feature_vector_list = list(play_list)
        single_feature_vector_list = temp_feature_vector_list[index]
        temp_feature_vector_list.pop(index)
        neighbors = getNeighbors(temp_feature_vector_list, single_feature_vector_list, k)
        neighborhood = getResponse(neighbors)
        predictions.append(neighborhood)
        #print 'Predicted: ' + repr(neighborhood) + '  |  Actual: ', play_list[index][-1]
    accuracy = getAccuracy(play_list, predictions)
    print 'Accuracy: ' + repr(accuracy) + '%'

#Display results of Weighted K Nearest Neighbor
def leave_one_out_wknn(play_list):
    k = 1
    local_predictions = []
    global_predictions = []

    print 'Weighted K-Nearest Neighbor - Local Method Results:'

    for index in range(len(play_list)):
        temp_feature_vector_list = list(play_list)
        single_feature_vector_list = temp_feature_vector_list[index]
        temp_feature_vector_list.pop(index)
        neighbors = get_weighted_neighbors(temp_feature_vector_list, single_feature_vector_list, k)
        local_neighborhood = local_method(neighbors, single_feature_vector_list)
        local_predictions.append(local_neighborhood)
        #print 'Predicted: ' + repr(local_neighborhood) + '  |  Actual: ', play_list[index][-1]
    local_accuracy = getAccuracy(play_list, local_predictions)
    print 'Local Accuracy: ' + repr(local_accuracy) + '%'

    print 'Weighted K-Nearest Neighbor - Global Method Results:'
    for index in range(len(play_list)):
        temp_feature_vector_list = list(play_list)
        single_feature_vector_list = temp_feature_vector_list[index]
        temp_feature_vector_list.pop(index)
        global_neighborhood = global_method(temp_feature_vector_list, single_feature_vector_list)
        global_predictions.append(global_neighborhood)
        #print 'Predicted: ' + repr(global_neighborhood) + '  |  Actual: ', play_list[index][-1]
    global_accuracy = getAccuracy(play_list, global_predictions)
    print 'Global Accuracy: ' + repr(global_accuracy) + '%'

#Display results of GRNN
def display_results_of_GRNN_with_leave_one_out(play_list):
    print '\nGeneral Regression Neural Network Results'
    dq = []
    correct = 0
    for index in range(len(play_list)):
        temp_feature_vector_list = list(play_list)
        single_feature_vector_list = temp_feature_vector_list[index]
        temp_feature_vector_list.pop(index)
        actual = play_list[index][-1]
        dq = compute_fire_strengths(index, single_feature_vector_list, temp_feature_vector_list, 4.796)
        #print 'Predicted:', dq.index(max(dq)) + 1, '| Actual: ', actual
        if (((dq.index(max(dq)) + 1) / actual) == 1):
            correct += 1

    print 'GRNN Accuracy: ', str((correct / float(len(play_list))) * 100.0) + '%'

def CASIS25_display_results_of_GRNN_with_leave_one_out(play_list):
    print '\nGeneral Regression Neural Network Results for CASIS 25 Dataset'
    dq = []
    correct = 0
    for index in range(len(play_list)):
        temp_feature_vector_list = list(play_list)
        single_feature_vector_list = temp_feature_vector_list[index]
        temp_feature_vector_list.pop(index)
        actual = play_list[index][-1]
        dq = compute_fire_strengths_CASIS25(index, single_feature_vector_list, temp_feature_vector_list, 4.796)
        #print 'Predicted:', dq.index(max(dq)) + 1, '| Actual: ', actual
        if (((dq.index(max(dq)) + 1) / actual) == 1):
            correct += 1

    print 'GRNN Accuracy: ', str((correct / float(len(play_list))) * 100.0) + '%'

"""
print average_chars(week_1_author_1, week_2_author_1, week_3_author_1)
print average_chars(week_1_author_2, week_2_author_2, week_3_author_2)
print average_chars(week_1_author_3, week_2_author_3, week_3_author_3)
print average_chars(week_1_author_4, week_2_author_4, week_3_author_4)
print average_chars(week_1_author_5, week_2_author_5, week_3_author_5)
print average_chars(week_1_author_6, week_2_author_6, week_3_author_6)
print average_chars(week_1_author_7, week_2_author_7, week_3_author_7)
print average_chars(week_1_author_8, week_2_author_8, week_3_author_8)
print average_chars(week_1_author_9, week_2_author_9, week_3_author_9)
print average_chars(week_1_author_10, week_2_author_10, week_3_author_10)
print "\n"
print average_words(week_1_author_1, week_2_author_1, week_3_author_1)
print average_words(week_1_author_2, week_2_author_2, week_3_author_2)
print average_words(week_1_author_3, week_2_author_3, week_3_author_3)
print average_words(week_1_author_4, week_2_author_4, week_3_author_4)
print average_words(week_1_author_5, week_2_author_5, week_3_author_5)
print average_words(week_1_author_6, week_2_author_6, week_3_author_6)
print average_words(week_1_author_7, week_2_author_7, week_3_author_7)
print average_words(week_1_author_8, week_2_author_8, week_3_author_8)
print average_words(week_1_author_9, week_2_author_9, week_3_author_9)
print average_words(week_1_author_10, week_2_author_10, week_3_author_10)
print "\n"
print average_sentences(week_1_author_1, week_2_author_1, week_3_author_1)
print average_sentences(week_1_author_2, week_2_author_2, week_3_author_2)
print average_sentences(week_1_author_3, week_2_author_3, week_3_author_3)
print average_sentences(week_1_author_4, week_2_author_4, week_3_author_4)
print average_sentences(week_1_author_5, week_2_author_5, week_3_author_5)
print average_sentences(week_1_author_6, week_2_author_6, week_3_author_6)
print average_sentences(week_1_author_7, week_2_author_7, week_3_author_7)
print average_sentences(week_1_author_8, week_2_author_8, week_3_author_8)
print average_sentences(week_1_author_9, week_2_author_9, week_3_author_9)
print average_sentences(week_1_author_10, week_2_author_10, week_3_author_10)
"""

write_to_feature_vector_file("Test_Feature_Vectors.txt", test_list)
normalize_fv("Test_Normalized_Feature_Vectors.txt", test_list)

# Separating Authors and Feature Vectors
authors = []
for i in range(len(feature_vector_list)):
    authors.append(str(feature_vector_list[i][-1]))

feature_vectors = list(feature_vector_list)
for i in range(len(feature_vector_list)):
    del feature_vectors[i][-1]


# Steady State Genetic Algorithm
# No Innovations: 95, 25, 2, 2, 1, 1, False, 5000, 10
SSGA_fm_length = 95
SSGA_number_of_fms = 25
SSGA_number_of_potential_parents = 2
SSGA_number_of_parents = 2
SSGA_number_of_children = 1
SSGA_number_to_replace = 1
SSGA_is_replacement_combined = False
SSGA_number_of_iterations = 5000
SSGA_number_of_runs = 10
SSGA_last_generation_ratings = []

print "Steady State Genetic Algorithm"

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
        SSGA_child_list = procreate(SSGA_parent_list, SSGA_number_of_children)

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
        SSGA_iterations = SSGA_iterations - len(SSGA_child_list)

    # add average rating of last generation to list
    SSGA_last_generation_ratings.append(SSGA_ratings_over_time[-1])
    print SSGA_ratings_over_time

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

print "SSGA Ratings for Each Generation: ", SSGA_last_generation_ratings
print "SSGA Max Rating: ", SSGA_final_max
print "SSGA Average Rating: ", SSGA_final_average

# Elitist Genetic Algorithm
# No Innovations: 95, 25, 2, 2, 1, 24, False, 5000, 10
EGA_fm_length = 95
EGA_number_of_fms = 25
EGA_number_of_potential_parents = 2
EGA_number_of_parents = 2
EGA_number_of_children = 1
EGA_number_to_replace = 24
EGA_is_replacement_combined = False
EGA_number_of_iterations = 5000
EGA_number_of_runs = 10
EGA_last_generation_ratings = []

print "\nElitist Genetic Algorithm"

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
            EGA_child = procreate(EGA_parent_list, EGA_number_of_children)
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
        EGA_iterations = EGA_iterations - len(EGA_child_list)

    # add average rating of last generation to list
    EGA_last_generation_ratings.append(EGA_ratings_over_time[-1])
    print EGA_ratings_over_time

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

print "EGA Ratings for Each Generation: ", EGA_last_generation_ratings
print "EGA Max Rating: ", EGA_final_max
print "EGA Average Rating: ", EGA_final_average

# Estimation of Distribution Algorithm
# No Innovations: 95, 25, 12, 24, 24, False, 5000, 10
EDA_fm_length = 95
EDA_number_of_fms = 25
EDA_number_of_parents = 12
EDA_number_of_children = 24
EDA_number_to_replace = 24
EDA_is_replacement_combined = False
EDA_number_of_iterations = 5000
EDA_number_of_runs = 10
EDA_last_generation_ratings = []

print "\nEstimation of Distribution Algorithm"

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
        EDA_child_list = procreate(EDA_parent_list, EDA_number_of_children)

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
        EDA_iterations = EDA_iterations - len(EDA_child_list)

    # add average rating of last generation to list
    EDA_last_generation_ratings.append(EDA_ratings_over_time[-1])
    print EDA_ratings_over_time

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

print "EDA Ratings for Each Generation: ", EDA_last_generation_ratings
print "EDA Max Rating: ", EDA_final_max
print "EDA Average Rating: ", EDA_final_average
