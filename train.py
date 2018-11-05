import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import sys
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid 
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, trainParam):

    color_space    = trainParam.color_space
    spatial_size   = trainParam.spatial_size
    hist_bins      = trainParam.hist_bins
    orient         = trainParam.orient
    pix_per_cell   = trainParam.pix_per_cell
    cell_per_block = trainParam.cell_per_block
    hog_channel    = trainParam.hog_channel
    spatial_feat   = trainParam.spatial_feat
    hist_feat      = trainParam.hist_feat
    hog_feat       = trainParam.hog_feat

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    count = 0
    for image in imgs:
        file_features = []
        count = count + 1
        sys.stdout.write("\r{0}/{1}".format(count ,len(imgs)))
 
        sys.stdout.flush()
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

class TrainParameters:
    def __init__(self, cars, no_cars, color_space='HSV', orient=8, pix_per_cell=8, cell_per_block=2, hog_channel="ALL", spatial_size=(32,32), hist_bins=16, spatial_feat=True, hist_feat=True, hog_feat=True, y_start_stop=[None,None]):
        self.color_space    = color_space    # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient         = orient         # HOG orientations
        self.pix_per_cell   = pix_per_cell   # HOG pixels per cell
        self.cell_per_block = cell_per_block # HOG cells per block
        self.hog_channel    = hog_channel    # Can be 0, 1, 2, or "ALL"
        self.spatial_size   = spatial_size   # Spatial binning dimensions
        self.hist_bins      = hist_bins      # Number of histogram bins
        self.spatial_feat   = spatial_feat   # Spatial features on or off
        self.hist_feat      = hist_feat      # Histogram features on or off
        self.hog_feat       = hog_feat       # HOG features on or off
        self.y_start_stop   = y_start_stop   # Min and max in y to search in slide_window()
        self.X_scaler = []
        self.SVC = []
        self.feature_vector_len = None
        self.training_time  = None
        self.test_accuracy  = None
        self.cars           = cars
        self.no_cars        = no_cars

        self.__train__()

    def printParam(self):
        print("Color space               :", self.color_space)
        print("Hog orient                :", self.orient)
        print("HOG pixels per cell       :", self.pix_per_cell)
        print("HOG cells per block       :", self.cell_per_block)
        print("HOG hog_channel           :", self.hog_channel)
        print("Spatial binning dimensions:", self.spatial_size)
        print("Number of histogram bins  :", self.hist_bins)
        print("Spatial features          :", self.spatial_feat)
        print("Histogram features        :", self.hist_feat)
        print("HOG features              :", self.hog_feat)
        print("feature vector length     :", self.feature_vector_len)
        print("y_start_stop              :", self.y_start_stop)
        print("training time             :", self.training_time)
        print("test accuracy             :", self.test_accuracy)

    def __train__(self):
        # TODO play with these values to see how your classifier
        # performs under different binning scenarios
        print("Try extract car feature...")
        car_features = extract_features(self.cars, self)
        print("OK")
        print("Try extract not car feature...")
        notcar_features = extract_features(self.no_cars, self)
        print("OK")
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float32)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
            
        # Fit a per-column scaler only on the training data
        X_scaler = StandardScaler().fit(X_train)
        self.X_scaler = X_scaler
        # Apply the scaler to X_train and X_test
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
            
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X_train)
        self.X_scalar = X_scaler
        # Apply the scaler to X
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)

        self.feature_vector_len = len(X_train[0])
        # Use a linear SVC 
        self.SVC = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        self.SVC.fit(X_train, y_train)
        t2 = time.time()
        self.training_time = round(t2-t, 2)
        # Check the score of the SVC
        self.test_accuracy = round(self.SVC.score(X_test, y_test), 4)

        self.printParam()


##############################################################
# Pipeline entry point
##############################################################
def main():
    print("Load dataset...",end="")
    with open("dataset_1030.p", mode="rb") as f:
        obj = pickle.load(f)
    print("Success.")

    cars = obj["cars"]
    notcars = obj["noCars"]
 #   def __init__(self, cars, no_cars, color_space='HSV', orient=8, pix_per_cell=8, cell_per_block=2, hog_channel="ALL", spatial_size=(32,32), hist_bins=16, spatial_feat=True, hist_feat=True, hog_feat=True, y_start_stop=[None,None]):
 
    trainParam = []

    # print("Feature set 1")
    # trainParam.append(TrainParameters(cars, notcars))
    # print("\nFeature set 2")
    # trainParam.append(TrainParameters(cars, notcars,color_space='YCrCb'))
    # print("\nFeature set 3")
    # trainParam.append(TrainParameters(cars, notcars,color_space='YUV'))
    # print("\nFeature set 4")
    # trainParam.append(TrainParameters(cars, notcars,color_space='LUV'))
    # print("\nFeature set 5")
    # trainParam.append(TrainParameters(cars, notcars,color_space='HLS'))
    # print("\nFeature set 6")
    # trainParam.append(TrainParameters(cars, notcars,color_space='RGB'))
    # print("\nFeature set 7")
    # trainParam.append(TrainParameters(cars, notcars,orient=8))
    # print("\nFeature set 8")
    # trainParam.append(TrainParameters(cars, notcars,orient=9))
    # print("\nFeature set 9")
    # trainParam.append(TrainParameters(cars, notcars,orient=10))
    # print("\nFeature set 10")
    # trainParam.append(TrainParameters(cars, notcars,orient=11))
    # print("\nFeature set 11")   
    # trainParam.append(TrainParameters(cars, notcars,color_space='YCrCb',orient=8))
    # print("\nFeature set 12")
    # trainParam.append(TrainParameters(cars, notcars,color_space='YCrCb',orient=9))
    # print("\nFeature set 13")
    # trainParam.append(TrainParameters(cars, notcars,color_space='YCrCb',orient=10))
    ## Only Hog features
    print("\nFeature set 14")
    trainParam.append(TrainParameters(cars, notcars,color_space='YCrCb',orient=8,spatial_feat=False, hist_feat=False))
    print("\nFeature set 15")
    trainParam.append(TrainParameters(cars, notcars,color_space='YCrCb',orient=9,spatial_feat=False, hist_feat=False))
    print("\nFeature set 16")
    trainParam.append(TrainParameters(cars, notcars,color_space='YCrCb',orient=10,spatial_feat=False, hist_feat=False))
    print("\nFeature set 17")
    trainParam.append(TrainParameters(cars, notcars,color_space='YCrCb',orient=11,spatial_feat=False, hist_feat=False))
    print("\nFeature set 18")
    trainParam.append(TrainParameters(cars, notcars,color_space='HSV',orient=8,spatial_feat=False, hist_feat=False))
    print("\nFeature set 19")
    trainParam.append(TrainParameters(cars, notcars,color_space='HSV',orient=9,spatial_feat=False, hist_feat=False))
    print("\nFeature set 20")
    trainParam.append(TrainParameters(cars, notcars,color_space='HSV',orient=10,spatial_feat=False, hist_feat=False))
    print("\nFeature set 21")
    trainParam.append(TrainParameters(cars, notcars,color_space='HSV',orient=11,spatial_feat=False, hist_feat=False))
    print("\nFeature set 22")
    trainParam.append(TrainParameters(cars, notcars,color_space='YUV',orient=8,spatial_feat=False, hist_feat=False))
    print("\nFeature set 23")
    trainParam.append(TrainParameters(cars, notcars,color_space='YUV',orient=9,spatial_feat=False, hist_feat=False))
    print("\nFeature set 24")
    trainParam.append(TrainParameters(cars, notcars,color_space='YUV',orient=10,spatial_feat=False, hist_feat=False))
    print("\nFeature set 25")
    trainParam.append(TrainParameters(cars, notcars,color_space='YUV',orient=11,spatial_feat=False, hist_feat=False))

    with open("train1030_hog.p", mode="wb") as f:
        pickle.dump(trainParam,f)
        print("Save train model.")
   


if __name__ == "__main__":
     main()
