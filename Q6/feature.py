import os
import numpy as np
from keras.applications import inception_resnet_v2
import scipy.sparse as sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pickle as pkl
from sklearn.decomposition import PCA


DATA_DIR_train = "./Data_Q6/train_video/"
DATA_DIR_test = "./Data_Q6/test_video/"

PIC_DIR = "./pic/"
DATA_FEATURES_EXTRACTED = "./features/"

if "features" not in os.listdir("./"):
    os.mkdir("features")

if "pic" not in os.listdir("./"):
    os.mkdir("pic")


# reduce the dimension using PCA
def PCA_reduce(train_data):
    pca_analyser = PCA(n_components=80)
    reduced_data = pca_analyser.fit_transform(train_data)
    print(reduced_data.shape)
    print("explained proportion:", pca_analyser.explained_variance_ratio_, "total:", np.sum(pca_analyser.explained_variance_ratio_))
    return reduced_data[:train_data.shape[0], :], reduced_data[train_data.shape[0]:, :]


def load_img_from_file(filepath):
    original = load_img(filepath, target_size=(299, 299))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    return image_batch


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


feature_extract_func = get_feature_from_img_wrapper()


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


class VideoDataManager:
    def __init__(self, filename, label, is_training_set):
        self.filename = filename
        self.fileid = filename.split(".")[0]
        self.n_time_slots = 0
        self.feature_vectors = []
        self.label = label
        if is_training_set:
            self.dir = DATA_DIR_train
        else:
            self.dir = DATA_DIR_test

    # generate snapshot for video, require ffmpeg under current floder
    def generate_pics(self):
        cmd = "ffmpeg -ss 0 -i %s -y -f image2 -r 0.2 -s 299x299 %s%s-%%03d.jpg" % (self.dir + self.filename, PIC_DIR, self.fileid)
        os.system(cmd)
        slot_cnt = 0
        for f in os.listdir(self.dir):
            if self.fileid in f:
                slot_cnt += 1
        self.n_time_slots = slot_cnt

    def extract_features(self):
        for i in range(self.n_time_slots):
            fig_path = PIC_DIR + "%s-%03d.jpg" % (self.fileid, i+1)
            high_dim_features = feature_extract_func(fig_path)
            self.feature_vectors.append(high_dim_features)


if __name__ == "__main__":
    label_dict = {}
    for line in open("./Data_Q6/tarin_tag.txt"):
        label_dict[line.split(",")[0]] = int(line.split(",")[1])
    train_video_managers = []
    test_video_managers = []
    for file in os.listdir(DATA_DIR_train):
        vm = VideoDataManager(file, label_dict[file], True)
        vm.generate_pics()
        train_video_managers.append(vm)
    for vm in train_video_managers:
        vm.extract_features()

    for file in os.listdir(DATA_DIR_test):
        vm = VideoDataManager(file, -1, False)
        vm.generate_pics()
        test_video_managers.append(vm)

    for vm in test_video_managers:
        vm.extract_features()
    pkl.dump(train_video_managers, open("train_vms.pkl", "wb"))
    pkl.dump(test_video_managers, open("test_vms.pkl", "wb"))


