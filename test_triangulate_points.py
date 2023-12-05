import os
from utils import load_image_data
import cv2
import matplotlib.pyplot as plt
from VO import VO

if __name__ == '__main__':
    directory = "/Users/elliotirving/Documents/ETH/VAMR/VAMR Project/data_KITTI/00/image_0" # add your local path to image_0 directory (folder of image_0 images)
    image_files = load_image_data(directory=directory)
    
    calib_file = "/Users/elliotirving/Documents/ETH/VAMR/VAMR Project/data_KITTI/00/calib.txt" # add your local path to calibration
    vo = VO(calib_file=calib_file)

    image_files = sorted(image_files) #TODO build into load_image_data() because it currently returns random array?
    
    for i in range(1, len(image_files)):
        query_img = vo.read_image_(file=os.path.join(directory, image_files[i]))
        prev_img = vo.read_image_(file=os.path.join(directory, image_files[i - 1]))

       # Extract/store previous features first
        vo.extract_features(image=prev_img, algorithm='sift')
        prev_features = vo.query_features.copy()  # store features

        # New call of VO class (we are working in relation to query image)
        vo.extract_features(image=query_img, algorithm='sift')
        query_features = vo.query_features.copy()

        matches2d2d = vo.match_features("2d2d", descriptor_prev=prev_features["descriptors"])

        # # Plot 2d2d matches
        # img3 = cv2.drawMatches(query_img, query_features["keypoints"], prev_img, prev_features["keypoints"],
        #                        matches2d2d[:], None, flags=2) # displaying all matches
        # cv2.imshow('2d2d Image Pair with Matches', img3)

        # IMPLEMENTING RANSAC + TRIANGULATION BELOW


        cv2.waitKey(50)

    cv2.destroyAllWindows()