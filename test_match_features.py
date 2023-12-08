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

    # Flags to initialize point cloud and pose databases
    pt_cloud_exists = False
    pose_exists = False
    
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

            # Use inliers only to recover [R,t] by setting mask=mask_essential
            retval, R, t, mask_recover, triangulatedPoints = cv2.recoverPose(E, points1=first_img_pt, points2=second_img_pt, 
                                                                     cameraMatrix=vo.K, distanceThresh=50, mask=mask_essential) #tune distanceThresh
            # Normalize triangulated points (up to scale)
            triangulatedPoints = triangulatedPoints.T[:, :3] / triangulatedPoints.T[:, 3:4]
            
            # This section uses the mask_recovery to select only the inliers triangulated points which pass the cheirality check.
            new_3d_points = np.reshape(np.array(triangulatedPoints),(-1,3))
            pruned_points = np.float32(new_3d_points[mask_recover[:, 0] == 1]) # NOTE store points as float32 dtype to match descriptor dtype


            # ~~~~~~~~~~~~TESTING HOW TO READ MASK, REJECT OUTLIERS, AND STORE 3D POINT DESCRIPTORS~~~~~~~~~~~~~~~~~~~~~
            # OLD
            # new_3d_descriptors = np.vstack([query_features["descriptor"][match.queryIdx] for match in matches2d2d]) # figure out if Im storing "cv2.descriptor" or actual 128 value descriptor
            # pruned_descriptors = new_3d_descriptors[mask_recover[:, 0] == 1] # OLD

            # TESTING (briefly examined output, seems to work)
            new_3d_keypoints = [query_features["keypoints"][match.queryIdx] for match in matches2d2d]
            new_3d_descriptors = [query_features["descriptors"][match.queryIdx] for match in matches2d2d]
            mask_recover = mask_recover[:, 0].astype(int)
            pruned_keypoints = np.array(new_3d_keypoints)[mask_recover == 1] # float32 dtype
            pruned_descriptors = np.array(new_3d_descriptors)[mask_recover == 1] # float32 dtype
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


            # new 3D points from triangulation, passed cheirality check, ready to vstack under existing point cloud
            pruned_new_3d = np.hstack((pruned_points, pruned_descriptors))

            # write to class
            if pt_cloud_exists is False:
                vo.world_points_3d = pruned_new_3d
            else:
                vo.world_points_3d = np.vstack((vo.world_points_3d, pruned_new_3d))
            pt_cloud_exists = True
            if pose_exists is False:
                vo.poses = np.zeros((1,3,4))
                vo.poses[0,:,:] = np.hstack((R,t))
                pose_exists = True

            # Visualize
            # Plot 2d2d matches
            img3 = cv2.drawMatches(query_img, query_features["keypoints"], prev_img, prev_features["keypoints"],
                               matches2d2d[:], None, flags=2) # displaying all matches

        else:
            
            num_previous_descriptors = 50 # (Hardcoding for testing), trying to match 50 of existing 3D points to query points
            matches3d2d = vo.match_features("3d2d", num_matches_previous=num_previous_descriptors)

            object_points = np.array([vo.world_points_3d[-num_previous_descriptors:,:3][match.trainIdx] for match in matches3d2d])
            query_points = np.array([query_features["keypoints"][match.queryIdx] for match in matches3d2d])
            query_points_uv = np.array([query_features["keypoints"][match.queryIdx].pt for match in matches3d2d]) #extracted img pts

            success, rvec_C_W, t_C_W, inliers = cv2.solvePnPRansac(
                object_points, query_points_uv, cameraMatrix=vo.K, distCoeffs=np.zeros((4,1)))
            if not success:
                raise RuntimeError("RANSAC is not able to fit the model")
            
            # Handle inliers
            num_p3p_inliers = len(inliers) # tracks number 3d-2d inliers (currently not used)
            inlier_3d_mask = inliers[:, 0].astype(int)
            # recover_matches = matches3d2d[inlier_3d_mask]
            recover_inlier_2d_points = query_points[inlier_3d_mask] # selects 2d points for plotting (not _uv)
            recover_inlier_3d_points = object_points[inlier_3d_mask] # for PLOTTING 3D points on image. We might need to store these for the visualizer
            
            R_C_W, _ = cv2.Rodrigues(rvec_C_W)
            
            # Write pose to self.poses
            build_M = np.hstack((R_C_W,t_C_W))
            if pose_exists is True:
                vo.poses = np.vstack((vo.poses,np.reshape(build_M,(1,3,4)))) # currently vstacking poses. May change to dstack
            else:
                raise ValueError("P3P attempting to write pose to uninitialized pose database. 8pt pose should have been written first")

            # Visualize
            projected_3D_points = cv2.projectPoints(objectPoints=recover_inlier_3d_points, 
                                                    rvec=vo.poses[-1,:3,:3], tvec=vo.poses[-1,:3,-1],
                                                    cameraMatrix=vo.K, distCoeffs=None) # reprojects database 3d points into image plane
            
            # extract projectPoints output to get u,v. Can probably fix this with different datastructures?
            projected_3D_points = np.hstack((projected_3D_points[0][:,:,0], projected_3D_points[0][:,:,1])) 

            # # NOTE/TODO fails, can't read projected_3D_points since not cv2.Keypoint dtype
            # img3 = cv2.drawMatches(query_img, recover_inlier_2d_points, 
            #                        prev_img, projected_3D_points,
            #                        matches3d2d, None, flags=2)

        cv2.imshow('2d2d SFM then 3d2d Matching', img3)
        cv2.waitKey(1000)

    cv2.destroyAllWindows()