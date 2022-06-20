from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
from os import listdir, walk
from os.path import isfile, join
import itertools
from itertools import permutations, combinations, product
from numpy import dot
from numpy.linalg import norm
from tqdm.auto import tqdm
import csv

similar_list = []


def getAllFilesInTrain(directoryPath: str):
    return [(directoryPath + "/" + f) for f in listdir(directoryPath) if isfile(join(directoryPath, f))]


def getAllFilesInTest(directoryPath: str):
    return [(directoryPath + "/" + f) for f in listdir(directoryPath) if isfile(join(directoryPath, f))]


def predict(img_path: str, model: Model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds


def findDifference(f1, f2):
    # print(np.linalg.norm(f1 - f2))
    # return np.linalg.norm(f1 - f2)
    # print(dot(f1, f2)/(norm(f1)*norm(f2)))
    return dot(f1, f2) / (norm(f1) * norm(f2))


def findDifferences(feature_vectors, feature_vectors1):
    keys_train = [k for k in feature_vectors]
    keys_test = [v for v in feature_vectors1]
    possible_combinations = list(product(keys_train, keys_test))
    # possible_combinations = list(itertools.combinations(keys, 2))
    # print(possible_combinations)
    # print(len(possible_combinations))
    for k, v in possible_combinations:
        diff = findDifference(feature_vectors[k], feature_vectors1[v])
        if diff >= 0.97:
            similar_list.append(v)
            print(k, "is similar to ", v)
    # print(similar_list)
    unique_similar_list = set(similar_list)
    #print(unique_similar_list)
    file = open(r'E:\datasets\peta\person\similar.csv', 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(unique_similar_list)
    #     if diff < min[k]:
    #         min[k] = diff
    #         similar[k] = v
    #         min[v] = diff
    #         similar[v] = k
    # return similar


def driver():
    feature_vectors: dict = {}
    feature_vectors1: dict = {}
    model = ResNet50(weights='imagenet')

    reader = csv.reader(open(r"E:\datasets\peta\person\train.csv", "r"))
    for rows in reader:
        k = rows[0]
        v = rows[1]
        feature_vectors[k] = v

    reader1 = csv.reader(open(r"E:\datasets\peta\person\test.csv", "r"))
    for rows in reader1:
        k1 = rows[0]
        v1 = rows[1]
        feature_vectors1[k1] = v1

    # for img_path in tqdm(getAllFilesInTrain(r"/content/drive/MyDrive/Training Dataset/data")):
    #     feature_vectors[img_path] = predict(img_path, model)[0]
    # for img_path1 in tqdm(getAllFilesInTest(r"/content/drive/MyDrive/Training Dataset/data_test")):
    #     feature_vectors1[img_path1] = predict(img_path1, model)[0]
    results = findDifferences(feature_vectors, feature_vectors1)
    print(results)
    # a_file = open("/content/train.csv", "w")
    # writer = csv.writer(a_file)
    # for key, value in feature_vectors.items():
    #     writer.writerow([key, value])
    #
    # a_file.close()
    #
    # b_file = open("/content/test.csv", "w")
    # writer = csv.writer(b_file)
    # for key, value in feature_vectors1.items():
    #     writer.writerow([key, value])
    #
    # b_file.close()
    # print(results)
    # for k, v in results.items():
    #    print(k + " is most similar to: " + v)
    # print('Predicted:', decode_predictions(preds, top=3)[0])


driver()
