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
import collections
import os
from functools import reduce

def getAllFilesInDirectory(directoryPath: str):
    return [(directoryPath + "/" + f) for f in listdir(directoryPath) if isfile(join(directoryPath, f))]


def predict(img_path: str, model: Model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)


def findDifference(f1, f2):
    return dot(f1, f2) / (norm(f1) * norm(f2))


def findDifferences(feature_vectors):
    similar = collections.defaultdict(list)
    parent_dir = r"E:\datasets\Training data"
    directory = "Segregated"
    path = os.path.join(parent_dir, directory)
    if os.path.exists(path):
        for k in feature_vectors.keys():
            for k1 in feature_vectors.keys():
                diff = findDifference(feature_vectors[k], feature_vectors[k1])
                if diff >= 0.90:
                    similar[k].append(k1)
        return similar
    else:
        os.mkdir(path)







    #     if diff < min[k]:
    #         min[k] = diff
    #         similar[k] = v
    #         min[v] = diff
    #         similar[v] = k
    # return similar


def driver():
    maintainer=[]
    final_dict: dict ={}
    feature_vectors: dict = {}
    model = ResNet50(weights='imagenet')
    for img_path in getAllFilesInDirectory(r"E:\datasets\Training data"):
        feature_vectors[img_path] = predict(img_path, model)[0]
    results = findDifferences(feature_vectors)
    for k, v in results.items():
        if k in v:
            v.remove(k)
    print(results)

    for k, v in results.items():
        maintainer.append(v)
        unique_values = list(reduce(lambda i, j: set(i) | set(j), maintainer))
        if k not in unique_values:
            final_dict[k] = results.get(k)
    print(final_dict)

    for k, v in final_dict.items():


    # for k, v in results.items():
    #    print(k + " is most similar to: " + v)
       #print('Predicted:', decode_predictions(preds, top=3)[0])


driver()
