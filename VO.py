from typing import *
import numpy as np
import cv2
import os

class VO():

    def __init__(self, calib_file: str):

        self.K = self.read_calib_file_(path=calib_file)

        # todo: whatever format?
        self.features_3D_coords = None
        self.features_2D_coords = None

        # implemented feature extraction algorithms and their hyper params
        self.feature_extraction_algorithms_config = {
            'sift': {
                'nfeatures': 200
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

    def estimate_camera_pose(self, method: str = 'p3p'):
        '''
        Estimate the pose of the camera using RANSAC.
        todo: call on function triangulate camera
        
        Parameters
        ----------
        method:
            - 'p3p': use correspondances between 2D and 3D points
        Returns
        -------

        '''
        if method == 'p3p':
            # TODO: Replace by the correct points
            object_points = np.zeros(10, 3)     # 3D points (Nx3)
            image_points = np.zeros(10, 2)      # 2D image points (Nx2)

            success, rvec_C_W, t_C_W, inliers = cv2.solvePnPRansac(
                object_points, image_points.T, cameraMatrix=self.K, distCoeffs=np.zeros((4,1)))
            if not success:
                raise RuntimeError("RANSAC is not able to fit the model")
            
            R_C_W, _ = cv2.Rodrigues(rvec_C_W)
            t_C_W = t_C_W[:, 0]
        
        elif method == '8pt':
            # TODO: Replace by the correct points
            first_img_pt = np.zeros(10, 2)      # 2D points from the first image (Nx2)
            second_img_pt = np.zeros(10, 2)      # 2D points from the second image (Nx2)

            E, mask = cv2.findEssentialMat(first_img_pt, second_img_pt,
                                           cameraMatrix=self.K,
                                           method=cv2.RANSAC, prob=0.99, threshold=1.0)
            _, R_C_W, t_C_W, mask = cv2.recoverPose(E, first_img_pt, second_img_pt,
                                                    cameraMatrix=self.K)



    def check_num_inliers(self):
        '''

        Returns
        -------

        '''
        pass

    def triangulate_camera(self):
        '''

        Returns
        -------

        '''



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
