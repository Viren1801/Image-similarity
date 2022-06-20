from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
from os import listdir, walk
from os.path import isfile, join
import itertools
from itertools import permutations, combinations

similar_list = []


def getAllFilesInDirectory(directoryPath: str):
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
    return np.linalg.norm(f1 - f2)



def findDifferences(feature_vectors):
    keys = [k for k, v in feature_vectors.items()]
    # for each_permutation in combinations(keys, 2):
    #   possible_combinations = list(zip(each_permutation, keys))
    possible_combinations = list(itertools.combinations(keys, 2))
    # print(possible_combinations)
    # print(len(possible_combinations))
    for k, v in possible_combinations:
        diff = findDifference(feature_vectors[k], feature_vectors[v])
        if diff <= 0.16:
            # print(k ,"is similar to ", v)
            similar_list.append(v)
    # print(similar_list)
    unique_similar_list = set(similar_list)
    #print(unique_similar_list)
    #     if diff < min[k]:
    #         min[k] = diff
    #         similar[k] = v
    #         min[v] = diff
    #         similar[v] = k
    # return similar


def driver():
    feature_vectors: dict = {}
    model = ResNet50(weights='imagenet')
    for img_path in getAllFilesInDirectory(r"E:\datasets\Training data"):
        feature_vectors[img_path] = predict(img_path, model)[0]
    #results = findDifferences(feature_vectors)
    #print(results)
    # for k, v in results.items():
    #    print(k + " is most similar to: " + v)
    # print('Predicted:', decode_predictions(preds, top=3)[0])

    for v in feature_vectors.values():
        print(type(v))




driver()
