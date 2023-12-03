from typing import *
import numpy as np
import cv2
import os

class VO():

    def __init__(self, calib_file: str):

        self.K = self.read_calib_file_(path=calib_file)

        # todo: whatever format?
        # self.features_3D_coords = None
        # self.features_2D_coords = None
        self.query_features = [] # stores {"features", "descriptors"} of the current query image
        self.world_points_3d = [] # MAYBE MAKE THIS (npoints x 131) NDARRAY, where [:,:3] is the [x,y,z] world coordinate and [:,4:] is the 128 descriptor
                                # dynamic storage of triangulated 3D points in world frame and their descriptors (npoints x 128 np.ndarray)
        

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

    def match_features(self, input: str, descriptor_prev=None, num_previous_descriptors=200):
        '''
        Parameters
        ----------
        input: str
            Specifies the retrieval method. Should be either '2d2d' or '3d2d'.
        descriptor_prev: np.ndarray of the reference image feature descriptors, optional
            Used when input is '2d2d' to pass the descriptor of the previous image.

        Returns
        -------
        correspondences between query and previous image descriptors
        '''

        if input not in ['2d2d', '3d2d']:
            raise ValueError("Invalid retrieval method. Use '2d2d' or '3d2d'.")
        
        if input == "2d2d":
            if descriptor_prev is None:
                raise ValueError("For '2d2d' retrieval, provide descriptor_prev.")
        elif input == "3d2d":
            descriptor_prev = self.world_points_3d[-num_previous_descriptors:, 4:]
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
