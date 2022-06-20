# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.models import Model
# import numpy as np
# from os import listdir, walk
# from os.path import isfile, join
# import itertools
# from itertools import permutations, combinations, product
# from numpy import dot
# from numpy.linalg import norm
# from tqdm.auto import tqdm
# import csv
# import pandas as pd
#
# similar_list = []
#
#
# def findDifference(f1, f2):
#     # print(np.linalg.norm(f1 - f2))
#     return dot(f1, f2) / (norm(f1) * norm(f2))
#
#
# def calculations(keys_train, keys_test):
#     for k in keys_train:
#         for v in keys_test:
#             yield (k, v)
#
#
# def driver():
#     feature_vectors: dict = {}
#     feature_vectors1: dict = {}
#     model = ResNet50(weights='imagenet')
#
#     data_train = pd.read_csv(r"E:\datasets\peta\person\train.csv", header=None)
#     data_test = pd.read_csv(r"E:\datasets\peta\person\test.csv", header=None)
#
#     for j in range(len(data_train[1])):
#       data_train[1][j] = data_train[1][j][1:-1]
#       data_train[1][j] = [float(x) for x in data_train[1][j].split(" ")]
#       data_train[1][j] = np.asarray(data_train[1][j])
#
#
#     for k in range(len(data_test[1])):
#       data_test[1][k] = data_test[1][k][1:-1]
#       data_test[1][k] = [float(x) for x in data_test[1][k].split(" ")]
#       data_test[1][k] = np.asarray(data_test[1][k])
#
#     feature_vectors = pd.Series(data_train[1].values, index=data_train[0]).to_dict()
#     feature_vectors1 = pd.Series(data_test[1].values, index=data_test[0]).to_dict()
#
#     keys_train = [k for k in feature_vectors]
#     keys_test = [v for v in feature_vectors1]
#
#     for k, v in calculations(keys_train, keys_test):
#        diff = findDifference(feature_vectors[k], feature_vectors1[v])
#        if diff >= 0.98:
#          similar_list.append(v)
#          print(k, "is similar to ", v)
#
#
#
# driver()


from DeepImageSearch import Index, LoadData, SearchImage
# load the Images from the Folder (You can also import data from multiple folders in python list type)
image_list = LoadData().from_folder([r'E:\peta2\person\train\images', r'E:\peta2\person\test\images'])
# Load data from CSV file
#image_list = LoadData().from_csv(csv_file_path='your_csv_file.csv',images_column_name='column_name)

Index(image_list).Start()

SearchImage().get_similar_images(image_path=image_list[0],number_of_images=5)