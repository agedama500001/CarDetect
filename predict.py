import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import sys
sys.path.append("./")
from lesson_functions import *
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, trainParam):
    
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

    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows,trainParam): 

    scaler = trainParam.X_scaler
    clf    = trainParam.SVC

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, trainParam)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


class TrainParameters:
    def __init__(self):
        self.color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 0 # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16) # Spatial binning dimensions
        self.hist_bins = 16    # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.y_start_stop = [None, None] # Min and max in y to search in slide_window()
        self.X_scaler = []
        self.SVC = []

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
        print("y_start_stop              :", self.y_start_stop)

def cvpaste(img, imgback, x, y, angle = 0, scale = 0.25, grayscale=True):  
    # x and y are the distance from the center of the background image 

    dst = np.copy(imgback)
    dst_shape = dst.shape
    # print(dst_shape)
    src = cv2.resize(img, None, fy=scale, fx=scale)
    src_shape = src.shape
    # print(src_shape)
    src_h = src_shape[0]
    src_w = src_shape[1]

    if grayscale == True:
        dst[y:y+src_h-1, x:x+src_w-1, 0] = src[0:src_h-1, 0:src_w-1]
        dst[y:y+src_h-1, x:x+src_w-1, 1] = src[0:src_h-1, 0:src_w-1]
        dst[y:y+src_h-1, x:x+src_w-1, 2] = src[0:src_h-1, 0:src_w-1]
        # print("grayscale")
    else:   
        dst[y:y+src_h-1, x:x+src_w-1, 0] = src[0:src_h-1, 0:src_w-1, 0]
        dst[y:y+src_h-1, x:x+src_w-1, 1] = src[0:src_h-1, 0:src_w-1, 1]
        dst[y:y+src_h-1, x:x+src_w-1, 2] = src[0:src_h-1, 0:src_w-1, 2]
        # print("color")

    return dst

class SearchWindowBuffer:
    def __init__(self, size, threshold):
        self.buffer = [0 for i in range(0,size)]
        self.start = 0
        self.end = 0
        self.ave = 0.0
        self.size = size
        self.thres = threshold
        self.isFind = False

    def add(self,val):
        self.buffer[self.end] = val
        self.end = (self.end + 1) % len(self.buffer)

        sum = 0.0
        for val in self.buffer:
            sum = sum + val
        self.ave = sum / self.size

        if(self.ave >= self.thres):
            self.isFind = True
        else:
            self.isFind = False

    def isActive(self):
        val = self.buffer[self.start]
        self.start =(self.start + 1) % len(self.buffer)
        #print(self.buffer, " ", self.ave , " ", self.isFind)
        return self.isFind

    def __len__(self):
        return self.end - self.start

class CarSearcher:
    def __init__(self, x_start_stop, y_start_stop, scale):
        self.x_start_stop = x_start_stop
        self.y_start_stop = y_start_stop
        self.scale        = scale
        self.isFirstSearch = False
        self.window_count_x  = 0
        self.window_count_y  = 0
        self.windowBuf = []

    def Search(self, image, trainParam):

        img    = image
        scale  = self.scale
        y_start_stop = self.y_start_stop
        x_start_stop = self.x_start_stop
        ystart = y_start_stop[0]
        ystop  = y_start_stop[1]
        xstart = x_start_stop[0]
        xstop  = x_start_stop[1]
        scale  = scale
        svc    = trainParam.SVC
        color_space = trainParam.color_space
        X_scaler = trainParam.X_scaler
        orient  = trainParam.orient
        pix_per_cell = trainParam.pix_per_cell
        cell_per_block = trainParam.cell_per_block
        is_hog_feat = trainParam.hog_feat
        is_hist_feat = trainParam.hist_feat
        is_spatial_feat = trainParam.spatial_feat
        spatial_size  = trainParam.spatial_size
        hist_bins = trainParam.hist_bins
        
        ## Load image
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        xWidthOrigin = img.shape[1]
        xOffset = xWidthOrigin - xstart
        #print(xWidthOrigin," ",xstart)

        img_tosearch = img[ystart:ystop,xstart:xstop,:]

        ctrans_tosearch = convert_color(img_tosearch, 'RGB2{0}'.format(color_space))
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 4  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        if(not self.isFirstSearch):
            self.window_count_x = nxsteps
            self.window_count_y = nysteps
            self.isFirstSearch = True
            self.windowBuf = [SearchWindowBuffer(4,0.99) for i in range(nxsteps)]
            #print(len(self.windowBuf))         

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        windows = []
        #print("xb=",nxsteps," yb=",nysteps)
        stepCount = 0
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                test_features = X_scaler.transform(hog_features.reshape(1, -1))    
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    self.windowBuf[stepCount].add(1)
                else:
                    self.windowBuf[stepCount].add(0)                   

                if self.windowBuf[stepCount].isActive() == 1:
                    # fig2 = plt.figure("Found!")
                    # fig2.gca().imshow(subimg,cmap="gray")
                    # plt.show()
                    # draw rectangle
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart, ytop_draw+win_draw+ystart),(0,0,255),6)
                    windows.append(((xbox_left+xstart, ytop_draw+ystart),(xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart)))
                    #print("Draw:",(xbox_left+xstart, ytop_draw+ystart),",",(xbox_left+xstart+win_draw,ytop_draw+win_draw+ystart),"  ",xstart)
                    #print(len(windows))

                stepCount = stepCount + 1
            cv2.rectangle(draw_img,(xstart, ystart),(xstop, ystop),(255,0,0),3)
        return draw_img, windows

####################################################
#   pipeline
####################################################
DEBUG = False
HEAT_THRES  = 1

region1 = CarSearcher([550,1279], [400,464], 1)
region2 = CarSearcher([500,1279], [416,480], 1)
region3 = CarSearcher([450,1279], [400,496], 1.5)
region4 = CarSearcher([400,1279], [432,528], 1.5)
region5 = CarSearcher([400,1279], [400,528], 2.0)
region6 = CarSearcher([400,1279], [432,560], 2.0)
region7 = CarSearcher([400,1279], [400,596], 3.5)

def pipeline(image, trainModel):
    draw_image = np.copy(image)
    heat_image = np.zeros_like(image[:,:,0]).astype(np.float)

    #trainModel = trainParam[1]
    #trainModel = trainParam[5]
    ###trainModel = trainParam[6]
    trainModel = trainParam[4]
    #trainModel = trainParam[7]

    find_car_img,  find_windows  = region1.Search(image, trainModel)
    find_car_img2, find_windows2 = region2.Search(image, trainModel)
    find_car_img3, find_windows3 = region3.Search(image, trainModel)
    find_car_img4, find_windows4 = region4.Search(image, trainModel)
    find_car_img5, find_windows5 = region5.Search(image, trainModel)
    find_car_img6, find_windows6 = region6.Search(image, trainModel)
    find_car_img7, find_windows7 = region7.Search(image, trainModel)

    find_windows_all = []
    find_windows_all.extend(find_windows)
    find_windows_all.extend(find_windows2)
    find_windows_all.extend(find_windows3)
    find_windows_all.extend(find_windows4)
    find_windows_all.extend(find_windows5)
    find_windows_all.extend(find_windows6)
    find_windows_all.extend(find_windows7)

    # draw all boxes
    draw_raw_img = draw_boxes(draw_image, find_windows_all, color=(0, 0, 255), thick=6)

    # Add heat to each box in box list
    heat_image = add_heat(heat_image, find_windows_all)

    # Apply threshold to help remove false positives
    heat_image = apply_threshold(heat_image,HEAT_THRES)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat_image, 0, 255)
    heatmapD = (heatmap.astype(float) / 5) * 255 # for debug 
    heatmapD = np.clip(heatmapD, 0, 255)

    # # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_heat_img = draw_labeled_bboxes(np.copy(image), labels)

    draw_heat_img = cvpaste(heatmapD     ,  draw_heat_img , 0,  0 , grayscale=True)
    draw_heat_img = cvpaste(draw_raw_img ,  draw_heat_img , 321,0, grayscale=False)
    draw_heat_img = cvpaste(find_car_img ,  draw_heat_img , 641,0, grayscale=False)
    draw_heat_img = cvpaste(find_car_img2,  draw_heat_img , 961,0, grayscale=False)  
    draw_heat_img = cvpaste(find_car_img3,  draw_heat_img , 0,  140 , grayscale=False)
    draw_heat_img = cvpaste(find_car_img4,  draw_heat_img , 321,140, grayscale=False)
    draw_heat_img = cvpaste(find_car_img5,  draw_heat_img , 641,140, grayscale=False)
    #draw_heat_img = cvpaste(find_car_img6,  draw_heat_img , 961,140, grayscale=False)  

    if DEBUG:
        fig = plt.figure("Vis")
        fig.gca().imshow(draw_raw_img)
        fig2 = plt.figure("heat map")
        fig2.gca().imshow(heatmap,cmap="hot")

        fig4 = plt.figure("find car")
        fig4.gca().imshow(find_car_img)
        fig5 = plt.figure("find car2")
        fig5.gca().imshow(find_car_img2)
        fig6 = plt.figure("find car3")
        fig6.gca().imshow(find_car_img3)
        fig7 = plt.figure("find car4")
        fig7.gca().imshow(find_car_img4)
        fig8 = plt.figure("find car5")
        fig8.gca().imshow(find_car_img5)
        # fig9 = plt.figure("find car6")
        # fig9.gca().imshow(find_car_img6)
        fig10 = plt.figure("find car7")
        fig10.gca().imshow(find_car_img7)
        fig11 = plt.figure("find car all")
        fig11.gca().imshow(draw_raw_img)
        plt.show()

    return draw_heat_img

def onFrame(image):
    r = pipeline(image,trainParam)
    return r

####################################################
#   main
####################################################
trainParam = []
# Load model
print("Load model with parameters...",end="")
with open("train1030_hog.p", mode="rb") as f:
    trainParam = pickle.load(f)
print("Success.")


if DEBUG:
    print("Pipeline process is DEBUG")
    src = mpimg.imread("test_images/test6.jpg")
    pipeline(src,trainParam)

else:
    print("Pipeline process is RELEASE")
    inputVideo = "./project_video.mp4"
    outputVideo = "./output_images/pipeline.mp4"

    inputVideoClip = VideoFileClip(inputVideo)#.subclip(10,15)
    outputVideoClip = inputVideoClip.fl_image(onFrame)
    outputVideoClip.write_videofile(outputVideo, audio=False)
    inputVideoClip.close()
    outputVideoClip.close()
