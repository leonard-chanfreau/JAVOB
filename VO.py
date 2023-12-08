from typing import *
import numpy as np
import cv2
import os

class VO():
    """
    Incremental Visual Odometry Pipeline.

    Attributes
    ----------
    K: np.ndArray (3x3)
        Calibration Matrix.
    num_feature: int
        Number of features extracted from each image.
    query_features: Dict
        Features of the query image: features (u,v) (key "features") and their
        HoG descriptor (key "descriptors").
    last_keyframe: Dict
        Features of the last keyframe: 2D features (u,v) (key "features"), their
        HoG descriptor (key "descriptors") and the index of the image (key
        "index").\n
        Overwrote every time new keyframe is created.
    world_points_3d: np.Ndarray (npoints x 131)
        Point Cloud: [x,y,z] world coordinate and the corresponding 1x128 HoG
        descriptor.\n
        Dynamically incremented every time we a new keyframe is used to
        triangulate new 3d points.
    poses: np.Ndarray (3 x 4 x nimages)
        History of the camera poses: stacked [R|t] projection matrices.
    feature_extraction_config: Dict
        Contain all config parameters required by the feature extraction
        algorithm.

    Methods
    -------
    run(img)
        Run the VO pipeline to estimate the camera pose of the new image.
    """
    
    def __init__(self, calib_file: str):
        """
        Initialize VO Pipeline.

        Parameters
        ----------
        calib_file : str
            Path to the file containing the calibration matrix.
        """
        # Parameters
        self.K = self.read_calib_file_(path=calib_file)
        self.num_features = 400
        # Query image info
        self.query_features = {}
        # Database storage
        self.last_keyframe = {}
        self.world_points_3d = None
        self.poses = None
        # Hyper parameters for the extraction algorithms
        self.feature_extraction_config = {
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
        Using self.feature_extraction_config dict in the constructor,
        the hyperparameters can be adjusted.

        Returns
        -------
        sift_features of the query image: {"keypoints", "descriptors"}
            keypoints: tuple[cv2.KeyPoint]
                tuple containing all keypoints (cv2.KeyPoint), ordered according
                to sift score.\n
                cv2.KeyPoint: object containing angle: float, octave: int,
                pt: tuple(x,y), response: float, size: float
            descriptors: nfeatures x 128 np.ndarray
                contains all SIFT descriptors to the corresponding keyPoints
        '''
        # Select the feature extraction algorithm.
        if algorithm == 'sift':
            # Extract SIFT feature.
            sift = cv2.SIFT_create(
                nfeatures =
                self.feature_extraction_config[algorithm]['nfeatures'])
            keypoints, descriptors = sift.detectAndCompute(image, None)
        else:
            raise ValueError(f'algorithm {algorithm} not implemented')
        # TODO: When implementing other algos, make sure the output is of the
        # same format.
        self.query_features = {"keypoints": keypoints,
                               "descriptors": descriptors}

    def match_features(self, method: str,
                       descriptor_prev = None, num_matches_previous = None):
        '''
        Parameters
        ----------
        method: str
            Retrieval method ('2d2d' or '3d2d').
        descriptor_prev (for 2d2d only): np.ndarray
            Train image feature descriptors.\n
            Used when method is '2d2d' to pass the descriptor of the train
            image.
        num_matches_previous (for 3d2d only): int
            Number of 3D world points to match new query (keyframe) to.\n
            For example, set equal to len(self.query_features)

        Comments
        --------
        TODO Likely still need to tune num_previous_descriptors.
        How do we make this dynamic?
        
        Returns
        -------
        matches: List[cv2.DMatch]
            Matches between descriptors from the 2d train image (resp. 3d point
            cloud) and from the 2d query image.\n
            Each cv2.DMatch object contains the following information:\n
                - queryIdx: The index of the query descriptor (zero-based index).\n
                - trainIdx: The index of the train descriptor (zero-based index).\n
                - imgIdx: The index of the train image (zero-based index).\n
                - distance: The distance between the descriptors (scalar).\n
        '''
        # Select the correct method.
        if method == '2d2d':
            # Descriptors from a previous train image are expected.
            if descriptor_prev is None:
                raise ValueError(
                    "For '2d2d' retrieval, provide descriptor_prev.")
        elif method == '3d2d':
            # Select num_matches_previous descriptors from the 3d point cloud
            if num_matches_previous is None:
                raise ValueError(
                    "Please set a valid number of matches to create. Value must"
                    " be positive.")
            descriptor_prev = self.world_points_3d[-num_matches_previous:, 3:]
        else:
            raise ValueError("Invalid retrieval method. Use '2d2d' or '3d2d'.")
        # Match descriptors with query image descriptors
        bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
        return bf.match(self.query_features["descriptors"], descriptor_prev)

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
                Structure containing the matching information between
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
            self.world_points_3d[matches_array[:,1], 0:3]        # 3D pt (Nx3)
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
        
    def triangulate_point_cloud(
            self, query_feature: dict, train_feature: dict,
            matches: List[cv2.DMatch]):
        # TODO: Maybe these arguments should be members of the class.
        '''
        Triangulate the camera pose and a point cloud using 8-pt algorithm and
        RANSAC.\n
        Populate the 3D point cloud and append the pose to the history of
        camera poses (self.poses).
        
        Parameters
        ----------
        query_feature: dict of "features" and "descriptor"
        train_feature: dict of "features" and "descriptor"
        matches: List[cv2.DMatch]

        Returns
        -------

        '''
        # Retrieve matched 2D/2D points.
        matches_array = np.array(
            [(match.queryIdx, match.trainIdx) for match in matches])
        first_img_pt = train_feature["features"][matches_array[:,1]]    # 2D pt (Nx2)
        second_img_pt = query_feature["features"][matches_array[:,0]]   # 2D pt (Nx2)
        # Compute essential matrix.
        E, essential_inliers = cv2.findEssentialMat(
            first_img_pt, second_img_pt,
            cameraMatrix=self.K,
            method=cv2.RANSAC, prob=0.99, threshold=1.0)
        # Extract pose from it.
        _, R_C_W, t_C_W, inlier_mask, triangulatedPoints = \
            cv2.recoverPose(E, first_img_pt, second_img_pt,
                            cameraMatrix=self.K,
                            mask=essential_inliers)
        # Create 3D feature concatenating normalized triangulated points and
        # descriptors.
        norm_triangulated_points = \
            triangulatedPoints[:, 1:3] / triangulatedPoints[:, 4]
        feature_3d = np.hstack((norm_triangulated_points,
                                query_feature["descriptor"]))
        # Remove outliers and populate point cloud
        feature_3d = feature_3d[inlier_mask[:,0].astype(np.bool)]
        self.world_points_3d = \
            np.vstack((self.world_points_3d, norm_triangulated_points))
        # Append pose to history
        self.poses = \
            np.dstack((self.poses, np.vstack(R_C_W, t_C_W)))

    def check_num_inliers(self):
        '''

        Returns
        -------

        '''
        pass



    def read_calib_file_(self, path: str):
        '''
        read calibration file to get camera intrinsics

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
