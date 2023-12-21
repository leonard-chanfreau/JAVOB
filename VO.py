from typing import *
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# For pose
# https://dfki-ric.github.io/pytransform3d/install.html
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.rotations as pr
import pytransform3d.camera as pc
from scipy.spatial.transform import Rotation as R
from cycler import cycle
from mpl_toolkits.mplot3d import proj3d



class VO():
    """
    Incremental Visual Odometry Pipeline.

    Attributes
    ----------
    K: np.ndArray (3x3)
        Calibration Matrix.
    num_feature: int
        Number of features extracted from each image.
    max_distance_threshold: float
        Maximum distance to triangulate new points
    query_image: np.array
        Input image
    query_features: Dict
        Features of the query image: features (u,v) (key "features") and their
        HoG descriptor (key "descriptors").
    keyframe_image: np.array
        Image of the last keyframe.
    keyframe_pos: np.array
        Camera pose at the last keyframe (stack [R|t])
    keyframe_features: 
        Features of the keyframe image: features (u,v) (key "features") and
        their HoG descriptor (key "descriptors").
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
    num_points_3d: int
        number of 3d points in the database.
    poses: np.Ndarray (3 x 4 x nimages)
        History of the camera poses: stacked [R|t] matrices.
    iteration: int
        Counter for number of call to run.
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
        self.num_features = 1000
        self.max_distance_thresh = 50
        # Query image info
        self.query_image = None
        self.query_features = {}
        # Last keyframe image info
        self.keyframe_image = None
        self.keyframe_pos = None
        self.keyframe_features = {}
        # Database storage
        self.world_points_3d = None
        self.num_points_3d = None
        self.num_matches_previous = 300 # TODO: verify if it works and doc
        self.poses = [np.hstack([np.eye(3), np.zeros((3,1))])]
        # internal state
        self.iteration = 0
        # Hyper parameters for the extraction algorithms
        self.feature_extraction_config = {
            'sift': {
                'nfeatures': self.num_features
            }
        }

    def run(self, image: np.ndarray):
        '''
        Treat new incoming frame (image) to continue building the trajectory
        using the VO pipeline.
        Parameters
        ----------
        image: np.ndarray
            New incoming frame.
        '''
        # Set parameters
        # inliers threshold -> activate re-triangulation
        thresh_inliers = 60
        # Increment the call-counter
        self.iteration += 1
        # ================== INITIALIZATION ================== #
        # Set very first frame as keyframe and world frame reference
        if self.keyframe_image is None:
            self.keyframe_image = image
            self.keyframe_features = self.extract_features(image,
                                                           algorithm="sift")
            return
        # The next incoming frames are treated as query
        self.query_image = image
        # If the point cloud is not yet set.
        if self.world_points_3d is None:
            # Use frame to initialize the point cloud and set new keyframe
            self.initialize_point_cloud(image)
            return
        # ================== CONTINUOUS RUN ================== #
        # Extract 2d features from query image
        self.query_features = self.extract_features(image, algorithm="sift")
        # Find matches between those features and last few points of
        # world_points_3d (CURRENT keyframe points)
        matches = self.match_features(method="3d2d")
        # Estimate the camera pose from those matches
        pose, num_inliers = self.estimate_camera_pose(matches=matches)
        self.poses.append(pose)
        
        # Visualization
        print(f"#Inliers : {num_inliers}")
        self.display_matches(matches)
        self.display_traj_2d()

        # Check if enough inliers are found,
        if num_inliers < thresh_inliers: # TODO: Maybe rather a percentage of point cloud
            # Expand point cloud using new keyframe (query) and old keyframe
            matches = self.match_features("2d2d")
            self.triangulate_new_points(pose, matches)   #TODO: Maybe a wrapper as initialize point cloud that checks if camera are distant enough
            self.keyframe_image = image
            self.keyframe_pos = self.poses[-1]
            self.keyframe_features = self.query_features
            return

    def initialize_point_cloud(self, image: np.ndarray):
        """
        Initialize point cloud if depth uncertainty is below threshold (i.e.
        frames are sufficiently far apart)
        
        Parameters
        ----------
        image: np.ndarray
            Query image
        """
        # Set threshold for average depth
        # TODO hyperparam dict
        thresh = 0.10
        # Extract query features
        self.query_features = self.extract_features(image=image)
        # Match query and keyframe features
        matches = self.match_features(method="2d2d")
        # Initialize preliminary point cloud
        # TODO: remove self.... as args?
        pose, points_3d = self.compute_pose_and_3D_pts(
            self.query_features, self.keyframe_features, matches)
        # Get Z coord, average depth and baseline
        depth = points_3d[:,2]
        average_depth = np.mean(depth)
        baseline = np.linalg.norm(pose[:,-1])
        # Check average depth criterion
        if baseline/average_depth > thresh:
            # Uncertainty is below threshold: set point cloud and new keyframe
            self.world_points_3d = points_3d
            self.num_points_3d = points_3d.shape[0]
            self.poses.append(pose)
            self.keyframe_image = image #TODO: maybe small method to set keyframe
            self.keyframe_pos = self.poses[-1]
            self.keyframe_features = self.extract_features(image=image, algorithm="sift")

    def extract_features(self, image, algorithm: str = "sift"):
        '''
        Extract 2d features from an image using the chosen algorithm (SIFT by
        default).

        Parameters
        -------
        image: np.ndarray
            Image of which features are to be extracted.
        algorithm: str
            Feature extraction algorithm (default: SIFT).

        Comments
        -------
        I left the option to implement other feature extraction methods.
        Uses self.feature_extraction_config dict in the constructor.
        The hyperparameters can be adjusted.

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
        # Return results.
        #TODO: When implementing other algos, make sure the output is of the
        # same format.
        return {"keypoints": keypoints, "descriptors": descriptors}
        #TODO: store query image descriptors in self.world_points_3d ---> THIS MAY GO IN TRIANGULATE() function


    def match_features(
        self, method: str, descriptor_prev=None) -> List[cv2.DMatch]:
        '''
        Match query features with train features (= 3d features of the database
        or 2d features from the keyframe passed as arg) based on their proximity.
        
        Parameters
        ----------
        method: str
            Retrieval method ('2d2d' or '3d2d').
        descriptor_prev (for 2d2d only): np.ndarray
            Train image feature descriptors.\n
            Used when method is '2d2d' to pass the descriptor of the train
            image.

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
            # Select keyframe descriptors.
            descriptor_prev = self.keyframe_features['descriptors']
        elif method == '3d2d':
            # Select num_matches_previous descriptors from the 3d point cloud
            descriptor_prev = \
                self.world_points_3d[-self.num_matches_previous:, 3:]
        else:
            raise ValueError("Invalid retrieval method. Use '2d2d' or '3d2d'.")
        # Match descriptors with query image descriptors
        bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
        return bf.match(self.query_features["descriptors"], descriptor_prev)

    def estimate_camera_pose(self, matches: Tuple[cv2.DMatch]) -> int:
        # TODO: maybe find a way to avoid passing matches (huge list) as an
        # argument (make it member of the class?)
        '''
        Estimate the pose of the camera using p3p and RANSAC.\n
        Uses the 2D points from the query image (self.query_features) previously
        matched with 3D world points (self.world_points_3d).\n
        Append the pose to the history of camera poses (self.poses).
        
        Parameters
        ----------
        matches: Tuple[cv2.DMatch]
            Structure containing the matching information between query_features
            and world_points_3d.

        Returns
        -------
        num_inliers: int
            Number of inliers used by RANSAC to compute the camera pose.
        TODO engelbracht: idea: just return the pose, its appended in run?
        '''
        # Retrieve matched 2D/3D points.
        # Compute the offset in matching due to the fact that we only compare
        # the query features with the N last features of the database.
        matching_train_offset = self.num_points_3d - self.num_matches_previous
        matches_array = np.array(
            [(match.queryIdx, match.trainIdx + matching_train_offset)
             for match in matches])
        train_object_points = \
            self.world_points_3d[matches_array[:,1], 0:3]          # 3D pt (Nx3)
        query_image_points = np.array(
            [self.query_features['keypoints'][match.queryIdx].pt
             for match in matches])                                # 2D pt (Nx2)
        # Run PnP algorithm
        success, rvec_C_W, t_C_W, inliers = cv2.solvePnPRansac(
            train_object_points, query_image_points,
            cameraMatrix=self.K, distCoeffs=np.zeros((4,1)))
        # Catch errors
        if not success:
            raise RuntimeError("RANSAC is not able to fit the model")
        # Extract rotation matrix from rotation vec resulting of PnP
        R_C_W, _ = cv2.Rodrigues(rvec_C_W)
        # Invert to have camera pose in world frame
        R_W_C = R_C_W.T
        t_W_C = -R_W_C @ t_C_W
        # Construct pose matrix
        T_W_C = np.hstack([R_W_C, t_W_C])
        # Return the pose and the number of points used to estimate the latter
        return T_W_C, inliers.size

    def compute_pose_and_3D_pts(
            self, query_feature: Dict[tuple[cv2.Feature2D], np.ndarray],
            train_feature: Dict[tuple[cv2.Feature2D], np.ndarray],
            matches: tuple[cv2.DMatch]) -> (np.ndarray, np.ndarray):
        # TODO: Maybe these arguments should be members of the class.
        '''
        Compute the camera pose and construct a point cloud using 8-pt algorithm
        and RANSAC (SFM).\n
        Populate the 3D point cloud and append the pose to the history of
        camera poses (self.poses).
        
        Parameters
        ----------
        query_feature: dict
            Features of the query image ("keypoints" and "descriptor")
        train_feature: dict
            Features of the train image ("keypoints" and "descriptor")
        matches: List[cv2.DMatch]
            Structure containing the matching information between query_features
            and world_points_3d.

        Returns
        -------
        pose: np.ndarray [3 x 4]
            Query camera pose (transformation between the two cameras).
        features_3d: np.ndarray [N x 131]
            Triangulated 3d features.

        '''
        # Retrieve matched 2D/2D points.
        train_img_pt = np.array(
            [train_feature['keypoints'][match.trainIdx].pt for match in matches])
        query_img_pt = np.array(
            [query_feature['keypoints'][match.queryIdx].pt for match in matches])
        # Retrieve query image descriptors
        matches_array = np.array(
            [(match.queryIdx, match.trainIdx) for match in matches])
        query_img_descriptor = query_feature["descriptors"][matches_array[:, 0]]
        # Compute essential matrix from correspondences using 8-pt algorithm.
        E, essential_inliers = cv2.findEssentialMat(
            points1=train_img_pt, points2=query_img_pt,
            cameraMatrix=self.K,
            method=cv2.RANSAC, prob=0.99, threshold=1.0)
        # Extract the pose from it.
        retVal, R_C_W, t_C_W, inlier_mask, points_4d = cv2.recoverPose(
                            E=E,
                            points1=train_img_pt,
                            points2=query_img_pt,
                            cameraMatrix=self.K,
                            mask=essential_inliers,
                            distanceThresh=self.max_distance_thresh)
        # Convert it to camera pose in world frame.
        R_W_C = R_C_W.T
        t_W_C = -R_W_C @ t_C_W
        pose = np.hstack((R_W_C, t_W_C))
        # Create 3D feature concatenating normalized triangulated points and
        # descriptors.
        points_3d = (points_4d[:3, :] / points_4d[3, :]).T
        feature_3d = np.hstack((points_3d, query_img_descriptor))
        # Remove outliers.
        feature_3d = feature_3d[np.where(inlier_mask[:,0] == 1)].astype("float32")
        # Return camera pose and triangulated 3d features.
        return pose, feature_3d

    def triangulate_new_points(
        self, query_pose: np.ndarray, matches: Tuple[cv2.DMatch]):
        '''
        Triangulates new 3D points in the scene using the last keyframe and the
        query image.\n
        Same working principle as stereo cameras.
        
        Parameters
        ----------
        query_pose: np.ndarray [3 x 4]
            Pose of the first camera (Last Keyframe).
        matches: Tuple[cv2.DMatch]
            Structure containing the matching information between query_features
            and world_points_3d.
            
        Comments
        -------
        Should not be called  at every frame, only when we are triangulating new
        3D features.
        '''
        # Construct projections matrices = project 3D world point into the image
        keyframe_R_C_W = self.keyframe_pos[:,:3].T
        keyframe_t_C_W = - keyframe_R_C_W @ self.keyframe_pos[:,3:]
        M_keyframe = self.K @ np.hstack((keyframe_R_C_W, keyframe_t_C_W))
        query_R_C_W = query_pose[:,:3].T
        query_t_C_W = - query_R_C_W @ query_pose[:,3:]
        M_query = self.K @ np.hstack((query_R_C_W, query_t_C_W))
        # Construct matched 2D point list
        matches = sorted(matches, key=lambda x: x.distance)
        p_keyframe = np.array(
            [self.keyframe_features['keypoints'][match.trainIdx].pt
             for match in matches[:250]]).T
        p_query = np.array(
            [self.query_features['keypoints'][match.queryIdx].pt
             for match in matches[:250]]).T
        # Triangulate those points using projection matrices.
        points_4d = cv2.triangulatePoints(
            M_keyframe, M_query, p_keyframe, p_query)
        # Un-homogenize points in world frame (-> verified)
        points_3d = (points_4d[:3, :] / points_4d[3, :]).T
        # Discard points with positive z value
        # TODO: in fact rather the points "in front / behind" not necessary z axis
        mask = np.where((points_3d[:, 2] > query_pose[2,3]) &
                        (points_3d[:, 2] - query_pose[2,3] < self.max_distance_thresh))
        points_3d = points_3d[mask]
        # Package 3D points and their descriptors
        new_3d_points = np.zeros((points_3d.shape[0], 131)).astype(np.float32)
        new_3d_points[:,:3] = points_3d
        new_3d_points[:,3:] = np.array(
            [self.query_features["descriptors"][match.queryIdx]
             for match in matches])[mask]
        # Augment database.
        self.world_points_3d = np.vstack((self.world_points_3d, new_3d_points))
        self.num_points_3d = points_3d.shape[0]

    def reprojectPoints(
        self, P: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
        '''
        Reproject 3D points given a projection matrix (c.f. ex02 (PnP)).
        
        Parameters
        ----------
        P: np.ndarray [N x 3]
            Coordinates of the 3d points in the world frame
        pose: np.ndarray [3 x 4]
            Camera Pose
        K: np.ndarray [3 x 3]
            Calibration matrix

        Returns
        -------
        points_2d: np.ndarray [n x 2]
            Image coordinates of the reprojected points
        '''
        p_homo = (K @ pose @ np.r_[P.T, np.ones((1, P.shape[0]))]).T
        return p_homo[:,:2]/p_homo[:,2,np.newaxis]

    def display_matches(
        self, matches: Tuple[cv2.DMatch], project_points:bool =False):
        '''
        Display the query image inlier feature matches overlaid on image.
        Parameters
        -------
        matches: Tuple[cv2.DMatch]
            Required for 'match' mode.
            Structure containing the matching information between query_features
            and world_points_3d.
        project_points (optional): bool
            Default is False. If True, project all self.world_3d_points into 
            the camera frame. \n
            TODO: trim down number of 3D points to project into the scene each 
            time this is called

        Comments
        -------
        - Should be called AFTER query pose is written. This is so that when
        projecting 3D world points into the image, we use the query image pose.
        
        TODO for future development
        - 'mask' mode (maybe helpful to visualize specific masks)
        - Plot dashboard: (some of these are nice-to-have)
            Image plots
                Query image with points
                    Which points? Display only features that were matched with 
                    prev image or keyframe?
            Local trajectory + 3D world points
                Keep current pose at the center of graph
            Global trajectory
                Also ground truth if available.
            Number of keypoints
                World 3D points in frame?
                    Others plot number of new keypoints, tracked keypoints, 
                    and keypoint candidates on one plot
        - Make dashboard/subplots
        - Add ground truth argument later if we need
            gt (optional): str
                    Path to config file 
                    (check format of ground truth)
        '''
        # Give an id to the figure to not overwrite other figs.
        match_fig_id = 1
        # Create figure.
        plt.figure(match_fig_id)
        # Clear last "features_all" points.
        plt.figure(match_fig_id).clf()
        # Activate interactive mode for dynamic updating.
        plt.ion()
        # Display image
        im = self.query_image
        plt.imshow(im)
        # Create 2d points arrays.
        matched_keypoints = np.array(
            [self.query_features['keypoints'][match.queryIdx].pt
             for match in matches])
        all_keypoints = np.asarray(
            cv2.KeyPoint_convert(self.query_features['keypoints']))
        # Plot all keypoints in red and matched features in green.
        plt.scatter(x=all_keypoints[:,0], y=all_keypoints[:,1],
                    c='r', s=9, marker='x')
        plt.scatter(x=matched_keypoints[:,0], y=matched_keypoints[:,1],
                    c='lime', s=9, marker='x')
        if project_points:
            # Projects ALL point cloud features. TODO: maybe use cv func
            points = self.reprojectPoints(self.world_points_3d[:,:3], 
                                            self.poses[:,:,-1], 
                                            self.K)
            # Plot them as cyan circle into frame.
            plt.scatter(x=points[:,0], y=points[:,1], 
                        s=20, facecolors='none', edgecolors='cyan')
        # Add whitespace padding
        padding = 50
        plt.xlim(-padding, im.shape[1] + padding)
        plt.ylim(im.shape[0] + padding, -padding)
        plt.draw()
        plt.pause(0.001)
        # Set interactive mode to off after updating
        plt.ioff()
        
    def display_traj_2d(self):
        '''
        Display 2D top-down view of trajectory and point cloud.
        '''
        # Give an id to the figure to not overwrite other figs.
        traj_fig_number = 2
        if not plt.fignum_exists(traj_fig_number):
            # Turn on interactive mode for dynamic updating.
            plt.ion()
            fig = plt.figure(num=traj_fig_number, figsize=(7, 6))
        else:
            # If figure exists, clear plot.
            plt.figure(traj_fig_number).clf()
            fig = plt.gcf()
        traj = fig.add_subplot(111)
        # Plot projected poses in 2D top-down view.
        poses = np.array(self.poses)
        traj.plot(poses[:, 0, 3], poses[:, 2, 3], label='Trajectory')
        traj.set_xlabel('X')
        traj.set_ylabel('Z')
        # Plot projected 3D database points in 2D top-down view.
        x_world = self.world_points_3d[:, 0]
        z_world = self.world_points_3d[:, 2]
        traj.scatter(x_world, z_world, color='red', label='World Points')
        # Draw figure.
        traj.legend()
        plt.draw()
        plt.pause(0.001)


    def read_calib_file_(self, path: str) -> np.ndarray:
        '''
        Read calibration file to get camera intrinsics

        Parameters
        ----------
        path: str
            path to config file

        Returns
        -------
        K: np.ndarray [3 x 3]
            contains the intrinsic camera matrix
        '''
        # Read calib file
        K = np.genfromtxt(path)
        # Select only first (left) cam, reshape and trim
        # TODO: this is hardcoded as fuck
        K = K[0,1:].reshape((3,4))[:,:-1]
        return K

    def read_image_(self, file: str):
        image = cv2.imread(filename=file)
        if image is None:
            raise ValueError('image could not be read. Check path and filename.')
        return image
