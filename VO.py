from typing import *
import numpy as np
import cv2 as cv

class VO:

    def __init__(self, calib_file: str):

        self.K = self.read_calib_file_(path=calib_file)

        # todo: whatever format?
        self.features_3D_coords = None
        self.features_2D_coords = None



    def extract_features(self, image):
        '''
        NOTE: see https://amroamroamro.github.io/mexopencv/matlab/cv.SIFT.detectAndCompute.html for info on SIFT in/out formats
        Parameters
        -------
        image: np.ndarray

        Returns
        -------
        sift_features: type?

        '''
        pass

    def match_features(self, descriptor, descriptor_prev):
        '''
        Parameters
        ----------
        descriptor: np.ndarray of the query image feature descriptors
        descriptor_prev: np.ndarray of the reference image feature descriptors

        Returns
        -------
        correspondences between query and preious image descriptors
        '''
        # Note: do we need self here? match_features() does not use any data stored in the VO class 
        # UNLESS we decide to store the growing number of 3D pts and features in it

        bf = cv.BFMatcher(normType=cv.NORM_L1, crossCheck=True)
        matches = bf.match(descriptor, descriptor_prev)
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

if __name__ == '__main__':

    calib_file = "D:/Master/VAMR/kitti/05/calib.txt"
    vo = VO(calibFile=calib_file)from typing import *
import numpy as np

class VO():

    def __init__(self, calib_file: str):

        self.K = self.read_calib_file_(path=calib_file)

        # todo: whatever format?
        self.features_3D_coords = None
        self.features_2D_coords = None



    def extract_features(self, image):
        '''
        Parameters
        -------
        image: np.ndarray

        Returns
        -------
        sift_features: type?

        '''
        pass

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

if __name__ == '__main__':

    calib_file = "D:/Master/VAMR/kitti/05/calib.txt"
    vo = VO(calibFile=calib_file)
