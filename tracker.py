''' 
Programmer: Nikola Andric
Last Eddited: Feb 21. 2022
Email: nikolazeljkoandric@gmail.com
'''
from selectors import DefaultSelector
import sys
import cv2
import numpy as np

#getting the name of the file to read in
img_name = sys.argv[1]

# taking the image in grayscale format
img = cv2.imread("images/"+img_name, cv2.IMREAD_GRAYSCALE)

# load the camera
cap = cv2.VideoCapture(0)

#track the features of the image
# use sift algorithm to detect the features on the images
sift_algorithm = cv2.xfeatures2d.SIFT_create()

#create keypoints and descriptors of the image
key_points_image, descriptors_image = sift_algorithm.detectAndCompute(img, None)

# draw the keypoints on the image
#img = cv2.drawKeypoints(img, key_points_image, img)

# Let's see what key points of the original image are present on the image of the video
# feature matching between video stream and the static image (usinig flann algorithm to match the features since it is faster than ORB match detector)
index_params = dict(algorithm = 0, trees = 5)
search_params = dict()

flann_algorithm = cv2.FlannBasedMatcher(index_params, search_params)

# run the while loop to show the video in real time
while True:
    # read the camera
    _, frame = cap.read()
    #convert frames to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # features for the grayframe
    key_points_gray_frame, descriptors_gray_frame = sift_algorithm.detectAndCompute(gray_frame, None)
    
    #find matches using flann algorithm
    matches = flann_algorithm.knnMatch(descriptors_image, descriptors_gray_frame, k=2)
 
    # considering oly good matches
    good_matches = []
    
    # iterate over both images m - original, n - gray_frame
    for m, n in matches:
        # determining how good the match is by comparing the distances.
        # the smaller the distance the better
        if m.distance < 0.5*n.distance:
            good_matches.append(m)
            
    #draw the matches between the keypoints
    #img3 = cv2.drawMatches(img, key_points_image, gray_frame, key_points_gray_frame, good_matches, gray_frame)

    #show the keypoints on the video
    #gray_frame = cv2.drawKeypoints(gray_frame, key_points_gray_frame, gray_frame)


    # find the homography and detect the object
    if len(good_matches) > 10:
        query_pts = np.float32([key_points_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        train_pts = np.float32([key_points_gray_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # using RANSAC algorithm to find the homography with outliers being removed.
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        # convert the mask to a list
        matches_mask = mask.ravel().tolist()


        # perspective transformation
        height, width = img.shape
        pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix) 

        # draw the shape around the object
        homography = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)
        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("Homography", gray_frame)
    # show the images and frames in real time
    # cv2.imshow("Image",img)
    # cv2.imshow("gray_frame",gray_frame)
    #cv2.imshow("matches",img3)
    key = cv2.waitKey(1)

    #if we press escape, we break the loop
    if key == 27:
        break

# release the camera
cap.release()
cv2.destroyAllWindows()
    

