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
        self.world_points_3d = None # np.ndarray(npoints x 131), where [:,:3] is the [x,y,z] world coordinate and [:,3:] is the 128 descriptor
                                # dynamic storage of triangulated 3D points in world frame and their descriptors (npoints x 128 np.ndarray)
        self.poses = None # np.ndarray(3 x 4 x nimages) where 3x4 is [R|t] projection matrix

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
        sift_features of the query image: {"keypoints", "descriptors"}
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
        self.query_features = {"keypoints": keypoints, "descriptors": descriptors}
        
        #TODO: store query image descriptors in self.world_points_3d ---> THIS MAY GO IN TRIANGULATE() function

    def match_features(self, method: str, descriptor_prev=None, num_matches_previous=None):
        '''
        Parameters
        ----------
        method: str
            Should be either '2d2d' or '3d2d'. Specifies the retrieval method. 
        descriptor_prev (for 2d2d only): np.ndarray
            Reference image feature descriptors.
            Used when method is '2d2d' to pass the descriptor of the previous image.
        num_matches_previous (for 3d2d only): int
            Number of 3D world points to match new query (keyframe) to.
            For example, set equal to len(self.query_features)

        Comments
        -------
        TODO Likely still need to tune num_previous_descriptors. How doo we make this dynamic?
        
        Returns
        -------
        A 1-by-N structure array with the following fields:
            - queryIdx query descriptor index (zero-based index)
            - trainIdx train descriptor index (zero-based index)
            - imgIdx train image index (zero-based index)
            - distance distance between descriptors (scalar)
        '''

        if method not in ['2d2d', '3d2d']:
            raise ValueError("Invalid retrieval method. Use '2d2d' or '3d2d'.")
        
        if method == '2d2d':
            if descriptor_prev is None:
                raise ValueError("For '2d2d' retrieval, provide descriptor_prev.")
        elif method == '3d2d':
            if num_matches_previous is None:
                raise ValueError("Please set a valid number of matches to create. Value must be positive.")
            descriptor_prev = self.world_points_3d[-num_matches_previous:, 3:]
        else:
            raise ValueError("Invalid retrieval method. Use '2d2d' or '3d2d'.")

        bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
        matches = bf.match(self.query_features["descriptors"], descriptor_prev)
        return matches

    def ransac(self, type: str = '8pt'):
        '''
        todo: call on function triangulate points
        Returns
        -------

        '''
        pass

    def check_num_inliers(self):
        '''

        Returns
        -------

        '''
        pass

    def triangulate_points(self):
        '''
        Parameters
        ----------
        projMatr1 3x4 projection matrix of the first camera.
        projMatr2 3x4 projection matrix of the second camera.
        projPoints1 2xN array of feature points in the first image. It can be also a cell array of feature points {[x,y], ...} or two-channel array of size 1xNx2 or Nx1x2.
        projPoints2 2xN array of corresponding points in the second image. It can be also a cell array of feature points {[x,y], ...} or two-channel array of size 1xNx2 or Nx1x2.

        Comments
        -------
        https://amroamroamro.github.io/mexopencv/matlab/cv.triangulatePoints.html

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
