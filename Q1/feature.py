import numpy as np
from keras.applications import vgg16
import scipy.sparse as sparse
from sklearn.decomposition import PCA


# extract features using VGG16 network
class VGGExtractor:
    def __init__(self):
        self.vgg_extractor = vgg16.VGG16(weights='imagenet', include_top=False, pooling="max")
        self.vgg_classifier = vgg16.VGG16(weights='imagenet')

    def extract(self, image_array, concat=False):
        # prepare the image for the VGG model
        process_image = vgg16.preprocess_input(image_array.copy())

        # get the predicted probabilities for each class
        prediction = self.vgg_extractor.predict(process_image)
        if concat:
            return prediction
        return prediction


# reduce the dimension using PCA
def PCA_reduce(data):
    pca_analyser = PCA(n_components=80)
    reduced_data = pca_analyser.fit_transform(data)
    print("explained:", pca_analyser.explained_variance_)
    print("explained proportion:", pca_analyser.explained_variance_ratio_, "total:", np.sum(pca_analyser.explained_variance_ratio_))
    return reduced_data


if __name__ == "__main__":
    pic_array_batch = load_img_from_file("./images/00066.jpg")

    # prepare the image for the VGG model
    processed_image = vgg16.preprocess_input(pic_array_batch.copy())

    extractor = VGGExtractor()
    features = extractor.extract(processed_image)
    print(features.shape)
    image = features[0]
    print(image.ravel().shape)
    print(np.sum(image.ravel()==0))
    image_sparse = sparse.csr_matrix(image.ravel())
    print(image_sparse.shape)
    print(image_sparse)


