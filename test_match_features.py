import os
from utils import load_image_data
import cv2
import matplotlib.pyplot as plt
from VO import VO
import numpy as np

if __name__ == '__main__':
    directory = "/Users/elliotirving/Documents/ETH/VAMR/VAMR Project/data_KITTI/00/image_0" # add your local path to image_0 directory (folder of image_0 images)
    image_files = load_image_data(directory=directory)
    
    calib_file = "/Users/elliotirving/Documents/ETH/VAMR/VAMR Project/data_KITTI/00/calib.txt" # add your local path to calibration
    vo = VO(calib_file=calib_file)

    image_files = sorted(image_files) #TODO build into load_image_data() because it currently returns random array?

    pt_cloud_exists = False
    
    # for i in range(1, len(image_files)):
    for i in range(1, 10): # first use 10 frames
        query_img = vo.read_image_(file=os.path.join(directory, image_files[i]))
        prev_img = vo.read_image_(file=os.path.join(directory, image_files[i - 1]))

       # Extract/store previous features first
        vo.extract_features(image=prev_img, algorithm='sift')
        prev_features = vo.query_features.copy()  # store features

        # New call of VO class (we are working in relation to query image)
        vo.extract_features(image=query_img, algorithm='sift')
        query_features = vo.query_features.copy()
        
        
        # 8pt
        if pt_cloud_exists is False:
            
            matches2d2d = vo.match_features("2d2d", descriptor_prev=prev_features["descriptors"])

            first_img_pt = np.array([prev_features["keypoints"][match.trainIdx].pt for match in matches2d2d])
            second_img_pt = np.array([query_features["keypoints"][match.queryIdx].pt for match in matches2d2d])
            E, mask_essential = cv2.findEssentialMat(points1=first_img_pt, points2=second_img_pt, cameraMatrix=vo.K, method=cv2.RANSAC, prob=0.99, threshold=1.0)
            print(f"number of inliers from findEssentialMat: {np.count_nonzero(mask_essential)}")
            # use inliers only by setting mask=mask_essential
            retval, R, t, mask_recover, triangulatedPoints = cv2.recoverPose(E, points1=first_img_pt, points2=second_img_pt, 
                                                                     cameraMatrix=vo.K, distanceThresh=50, mask=mask_essential) #tune distanceThresh
            triangulatedPoints = triangulatedPoints.T[:, :3] / triangulatedPoints.T[:, 3:4]
            
            # This section uses the mask_recovery to select only the inliers triangulated points which pass the cheirality check.
            new_3d_points = np.reshape(np.array(triangulatedPoints),(-1,3))
            pruned_points = new_3d_points[mask_recover[:, 0] == 1]


            # ~~~~~~~~~~~~TESTING HOW TO STORE 3D POINT DESCRIPTORS~~~~~~~~~~~~~~~~~~~~~
            # OLD
            # new_3d_descriptors = np.vstack([query_features["descriptor"][match.queryIdx] for match in matches2d2d]) # figure out if Im storing "cv2.descriptor" or actual 128 value descriptor
            # pruned_descriptors = new_3d_descriptors[mask_recover[:, 0] == 1] # OLD

            # TESTING
            new_3d_descriptors = [query_features["descriptors"][match.queryIdx] for match in matches2d2d]
            mask_recover_int = mask_recover[:, 0].astype(int)
            pruned_descriptors = np.array(new_3d_descriptors)[mask_recover_int == 1]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # new 3D points from triangulation, passed cheirality check, ready to vstack under existing point cloud
            pruned_new_3d = np.hstack((pruned_points, pruned_descriptors))

            # write to class
            if pt_cloud_exists is False:
                vo.world_points_3d = pruned_new_3d
            else:
                vo.world_points_3d = np.vstack((vo.world_points_3d, pruned_new_3d))
            pt_cloud_exists = True
        else:
            
            num_previous_descriptors = 50 # (Hardcoding for testing), trying to match 50 of existing 3D points to query points
            matches3d2d = vo.match_features("3d2d", num_previous_descriptors=num_previous_descriptors)
            # TODO check why failing here ^^^^^


            object_points = np.array([vo.world_points_3d[-num_previous_descriptors,:3][match.trainIdx].pt for match in matches3d2d])
            query_points = np.array([query_features["keypoints"][match.queryIdx].pt for match in matches3d2d])

            success, rvec_C_W, t_C_W, inliers = cv2.solvePnPRansac(
                object_points, query_points, cameraMatrix=vo.K, distCoeffs=np.zeros((4,1)))
            if not success:
                raise RuntimeError("RANSAC is not able to fit the model")
            
            R_C_W, _ = cv2.Rodrigues(rvec_C_W)
            t_C_W = t_C_W[:, 0]
            # TODO write to poses


        
        # Plot 2d2d matches
        img3 = cv2.drawMatches(query_img, query_features["keypoints"], prev_img, prev_features["keypoints"],
                               matches2d2d[:], None, flags=2) # displaying all matches

        cv2.imshow('2d2d Image Pair with Matches', img3)
        cv2.waitKey(500)

    cv2.destroyAllWindows()