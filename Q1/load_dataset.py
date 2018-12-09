import os
import numpy as np
from keras.applications import inception_resnet_v2
import scipy.sparse as sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pickle as pkl


DATA_DIR_inlier = "./Data_Q1/inlier_train/"
DATA_DIR_outlier = "./Data_Q1/outlier_train/"
DATA_DIR_test = "./Data_Q1/test/"
DATA_FEATURES_EXTRACTED = "./features/"
if "features" not in os.listdir("./"):
    os.mkdir("features")


def load_img_from_file(filepath):
    original = load_img(filepath, target_size=(299, 299))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    return image_batch


# extract features using VGG16 network
class InceptionResNetV2Extractor:
    def __init__(self):
        self.extractor = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, pooling="avg")
        self.classifier = inception_resnet_v2.InceptionResNetV2(weights='imagenet')

    def extract(self, image_array):
        # prepare the image for the VGG model
        process_image = inception_resnet_v2.preprocess_input(image_array.copy())
        # get the predicted probabilities for each class
        prediction = self.extractor.predict(process_image)
        return prediction

    def predict(self, image_array):
        print("start predict.")
        process_image = inception_resnet_v2.preprocess_input(image_array.copy())
        predict_res = self.classifier.predict(process_image)
        print(inception_resnet_v2.decode_predictions(predict_res))
        return predict_res


# input filename, output features
def get_feature_from_img_wrapper():
    counter = [0]
    feature_extractors = []
    print("feature extractor got.")

    def inner_get_features(filepath):
        if len(feature_extractors) == 0:
            print("load a model.")
            feature_extractors.append(InceptionResNetV2Extractor())
        image_batches = load_img_from_file(filepath)
        feature = feature_extractors[0].extract(image_batches)
        counter[0] += 1
        if counter[0] % 2 == 0:
            print("extracted:%d" % counter[0])
        return feature
    return inner_get_features


# return a dict:{filename: label(+1 == outlier, -1 == inlier)}
def get_labels():
    filename_label_dict = {}
    for filename in os.listdir(DATA_DIR_inlier):
        if ".jpg" in filename:
            filename_label_dict[filename] = -1
    for filename in os.listdir(DATA_DIR_outlier):
        if ".jpg" in filename:
            filename_label_dict[filename] = 1
    return filename_label_dict


# return extracted features of each pic: two dicts {filename: feature vector} for train, and that for test
def get_features(renew=False):
    train_feature_dict = {}
    test_feature_dict = {}
    get_feature_from_img = get_feature_from_img_wrapper()
    if "train_feature.pkl" not in os.listdir(DATA_FEATURES_EXTRACTED) or renew:
        for filename in os.listdir(DATA_DIR_inlier):
            if ".jpg" in filename:
                train_feature_dict[filename] = get_feature_from_img(DATA_DIR_inlier + filename)
        for filename in os.listdir(DATA_DIR_outlier):
            if ".jpg" in filename:
                train_feature_dict[filename] = get_feature_from_img(DATA_DIR_outlier + filename)
        pkl.dump(train_feature_dict, open(DATA_FEATURES_EXTRACTED + "train_feature.pkl", "wb"))
    else:
        train_feature_dict = pkl.load(open(DATA_FEATURES_EXTRACTED + "train_feature.pkl", "rb"))
    if "test_feature.pkl" not in os.listdir(DATA_FEATURES_EXTRACTED) or renew:
        for filename in os.listdir(DATA_DIR_test):
            if ".jpg" in filename:
                test_feature_dict[filename] = get_feature_from_img(DATA_DIR_test + filename)
        pkl.dump(test_feature_dict, open(DATA_FEATURES_EXTRACTED + "test_feature.pkl", "wb"))
    else:
        test_feature_dict = pkl.load(open(DATA_FEATURES_EXTRACTED + "test_feature.pkl", "rb"))
    return train_feature_dict, test_feature_dict


if __name__ == "__main__":
    image_batch = load_img_from_file(DATA_DIR_inlier + "img_0a1bda9c284b4c92a2c94c0a9d9dba72.jpg")
    feature_extractor1 = InceptionResNetV2Extractor()
    features = feature_extractor1.extract(image_batch)
    print(features.shape)
    feature_extractor1.predict(image_batch)

