from typing import *
import numpy as np
import cv2
import os

class VO():
    
    def __init__(self, calib_file: str):

        self.K = self.read_calib_file_(path=calib_file)
        self.num_features = 400
        
        # Query
        self.query_features = {} # {cv2.features, cv2.desciptors} storing 2D feature (u,v) and its HoG descriptor. Called via {"features", "descriptors"}
     

        # Database
        self.last_keyframe = {} # overwrite every time new keyframe is created. {cv2.features, cv2.desciptors, image index} 
                                # storing 2D feature (u,v) and its HoG descriptor. Called via {"features", "descriptors","index"}
        self.world_points_3d = np.ndarray() # np.ndarray(npoints x 131), where [:,:3] is the [x,y,z] world coordinate and [:,4:] is the 128 descriptor
                                # dynamic storage of triangulated 3D points in world frame and their descriptors (npoints x 128 np.ndarray)
        self.poses = np.ndarray() # np.ndarray(3 x 4 x nimages) where 3x4 is [R|t] projection matrix

        # implemented feature extraction algorithms and their hyper params
        self.feature_extraction_algorithms_config = {
            'sift': {
                'nfeatures': self.num_features
            }
        }


    def extract_features(self, image, algorithm: str = "sift"):
        '''
        Parameters
        -------
        image: np.ndarray
            image of which features are to be extracted
        algorithm: str
            feature extraction algorithm

        Comments
        -------
        I left the option to implement other feature extraction methods.
        Using self.feature_extraction_algorithms_config dict in the constructor, the hyperparams can be adjusted.

        Returns
        -------
        sift_features: list[keypoints, descriptors]
            keypoints: tuple[cv2.KeyPoint]
                tuple containing all keypoints (cv2.KeyPoint), ordered according to sift score
                cv2.KeyPoint: object containing angle: float, octave: int, pt: tuple(x,y), response: float, size: float
            descriptors: nfeatures x 128 np.ndarray
                contains all SIFT descriptors to the corresponding keyPoints
        '''

        # SIFT feature extraction
        if algorithm is list(self.feature_extraction_algorithms_config.keys())[0]:
            sift = cv2.SIFT_create(nfeatures=self.feature_extraction_algorithms_config[algorithm]['nfeatures'])
            keypoints, descriptors = sift.detectAndCompute(image, None)
        else:
            raise ValueError(f'algorithm {algorithm} not implemented')

        #TODO: When implementing other algos, make sure the output is of the same format
        features = [keypoints, descriptors]

        return features

    def match_features(self, feature, feature_prev):
        '''

        Parameters
        ----------
        feature: np.ndarray

        feature_prev: np.ndarray


        Returns
        -------
        correspondences

        '''
        pass

    def estimate_camera_pose(self, matches: List[cv2.DMatch]) -> int:
        # TODO: maybe find a way to avoid passing matches (huge list) as an
        # argument (make it member of the class?)
        '''
        Estimate the pose of the camera using p3p and RANSAC.\n
        Uses the 2D points from the query image (self.query_features) previously
        matched with 3D world points (self.world_points_3d).\n
        Append the pose to the history of camera poses (self.poses).
        
        Parameters
        ----------
            matches: List[cv2.DMatch]
                Structure containning the matching information beween
                self.query_features, and self.world_points_3d.

        Returns
        -------
            num_inliers: int
                Number of inliers used by RANSAC to compute the camera pose.
        '''
        # Retrieve matched 2D/3D points.
        matches_array = np.array(
            [(match.queryIdx, match.trainIdx) for match in matches])
        object_points = \
            self.world_points_3d[matches_array[:,1]]             # 3D pt (Nx3)
        image_points = \
            self.query_features["features"][matches_array[:,0]]  # 2D pt (Nx2)
        # Run PnP
        success, rvec_C_W, t_C_W, inliers = cv2.solvePnPRansac(
            object_points, image_points.T,
            cameraMatrix=self.K, distCoeffs=np.zeros((4,1)))
        if not success:
            raise RuntimeError("RANSAC is not able to fit the model")
        # Extract transformation matrix
        R_C_W, _ = cv2.Rodrigues(rvec_C_W)
        t_C_W = t_C_W[:, 0]
        # Add it to the list of poses
        T_C_W = np.eye(4)
        T_C_W[1:3, 1:3] = R_C_W
        T_C_W[1:3, 4] = t_C_W
        self.poses = np.append((self.poses, T_C_W[:,:,np.newaxis]), axis = 2)
        # return the number of points used to estimate the camera pose
        return inliers.size()
        
    def triangulate_point_cloud(self, query_feature, train_feature, matches):
        # TODO: Maybe these arguments should be members of the class.
        '''
        Triangulate the camera pose and a point cloud using 8-pt algorithm and
        RANSAC.\n
        Populate the 3D point cloud and append the pose to the history of
        camera poses (self.poses).
        
        Parameters
        ----------
        query_feature: np.NdArray
        train_feature: np.NdArray
        matches: List[cv2.DMatch]

        Returns
        -------

        '''
        # Retrieve matched 2D/2D points.
        matches_array = np.array(
            [(match.queryIdx, match.trainIdx) for match in matches])
        first_img_pt = train_feature[matches_array[:,1]]    # 2D pt (Nx2)
        second_img_pt = query_feature[matches_array[:,0]]   # 2D pt (Nx2)
        # Compute essential matrix.
        E, _ = cv2.findEssentialMat(first_img_pt, second_img_pt,
                                    cameraMatrix=self.K,
                                    method=cv2.RANSAC, prob=0.99, threshold=1.0)
        # Extract pose from it.
        _, R_C_W, t_C_W, mask, triangulatedPoints = \
            cv2.recoverPose(E, first_img_pt, second_img_pt,
                            cameraMatrix=self.K)
        # Populate 3D point cloud.
        # TODO: Construct the pair triangulated_point/descriptors using the
        # inlier mask and the train/query features.
        # self.world_points_3d = \
        #     np.append((self.world_points_3d, ), axis=0)


    def check_num_inliers(self):
        '''

        Returns
        -------

        '''
        pass



    def read_calib_file_(self, path: str):
        '''
        read calibration file to get camera intrisics

        Parameters
        ----------
        path: str
            path to config file

        Returns
        -------
        K: np.ndarray (3 x 3)
            contains the intrinsic camera matrix
        '''
        # read calib file
        K = np.genfromtxt(path)

        # select only first (left) cam, reshape and trim
        #todo: this is hardcoded as fuck
        K = K[0,1:].reshape((3,4))[:,:-1]

        return K

    def read_image_(self, file: str):

        image = cv2.imread(filename=file)

        if image is None:
            raise ValueError('image could not be read. Check path and filename.')

        return image

if __name__ == '__main__':
    pass
