from typing import *
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class VO():
    
    def __init__(self, calib_file: str):

        self.K = self.read_calib_file_(path=calib_file)
        self.num_features = 1000
        
        # Query
        self.query_features = {} # {cv2.features, cv2.desciptors} storing 2D feature (u,v) and its HoG descriptor. Called via {"features", "descriptors"}

        self.keyframe = None
        self.query_frame = None

        # Database
        self.keyframe_features = {} # overwrite every time new keyframe is created. {cv2.features, cv2.desciptors}
                                # storing 2D feature (u,v) and its HoG descriptor. Called via {"features", "descriptors"}
        self.world_points_3d = None # np.ndarray(npoints x 131), where [:,:3] is the [x,y,z] world coordinate and [:,3:] is the 128 descriptor
                                # dynamic storage of triangulated 3D points in world frame and their descriptors (npoints x 128 np.ndarray)
        self.num_keyframe_points_3d = None

        self.poses = [np.hstack([np.eye(3), np.zeros((3,1))])]# list(np.ndarray(3 x 4 x nimages)) where 3x4 is [R|t] projection matrix

        # interna state
        self.iteration = 0

        # implemented feature extraction algorithms and their hyper params
        self.feature_extraction_algorithms_config = {
            'sift': {
                'nfeatures': self.num_features
            }
        }

    def extract_features(self, image, algorithm: str = "sift"):
        '''
        I left the option to implement other feature extraction methods.
        Using self.feature_extraction_algorithms_config dict in the constructor, the hyperparams can be adjusted.

        Parameters
        -------
        image: np.ndarray
            image of which features are to be extracted
        algorithm: str
            feature extraction algorithm

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
        # self.query_features = {"keypoints": keypoints, "descriptors": descriptors}
        return {"keypoints": keypoints, "descriptors": descriptors}
        #TODO: store query image descriptors in self.world_points_3d ---> THIS MAY GO IN TRIANGULATE() function

    def initialize_point_cloud(self, image: np.ndarray, mode: str = "init"):
        """
        initialize point cloud if depth uncertainty is below threshold (i.e. frames are sufficiently far apart)

        Parameters
        ----------
        image:
            query image

        Returns
        -------

        """

        if mode not in ["init", "expand"]:
            raise ValueError("Invalid mode. Use 'init' or 'expand'.")

        # extract and set query features
        self.query_features = self.extract_features(image=image)

        # match query and keyframe features
        matches = self.match_features(method="2d2d")

        # initialize preliminary point cloud
        # todo remove self.... as args?
        pose, points_3d = self.triangulate_point_cloud(query_feature=self.query_features, train_feature=self.keyframe_features, matches=matches)

        if mode == "init":
            # todo hyperparam dict
            thresh = 0.10

            # get Z coord and average depth and baseline
            depth = points_3d[:,2]
            average_depth = np.mean(depth)
            baseline = np.linalg.norm(pose[:,-1])

            # check average depth criterion
            if baseline/average_depth > thresh:
                self.world_points_3d = points_3d
                self.num_keyframe_points_3d = points_3d.shape[0]
                return True
            else:
                return False

        if mode == "expand":
            # todo: i would not concatenate, list append might be better here, array grows quickly -> slow
            self.world_points_3d = np.concatenate((self.world_points_3d, points_3d), axis=0)
            self.num_keyframe_points_3d = points_3d.shape[0]
            return True

    def run(self, image: np.ndarray):
        '''

        Parameters
        ----------
        image

        Returns
        -------

        '''

        # match threshold
        thresh_matches = 30

        # set very first keyframe
        if self.keyframe is None:
            self.keyframe = image
            self.keyframe_features = self.extract_features(image=image, algorithm="sift")
            self.iteration += 1
            return

        # initialize first point cloud and set new keyframe
        if self.world_points_3d is None:
            self.query_frame = image
            if self.initialize_point_cloud(image=image, mode="init"):
                self.keyframe = image
                self.keyframe_features = self.extract_features(image=image, algorithm="sift")
            self.iteration += 1
            return

        # continuous run
        # extract features and find matches between self.query_features
        # and self.world_points_3d[-self.num_keyframe_points_3d:, 3:] (CURRENT keyframe points)
        self.query_frame = image
        self.query_features = self.extract_features(image=image, algorithm="sift")
        matches = self.match_features(method="3d2d")

        # check if enough matches can be produced,
        # else reinit new keyframe and expand point cloud using new keyframe and old keyframe
        if len(matches) < thresh_matches:
            if self.initialize_point_cloud(image=image, mode="expand"):
                self.keyframe = image
                self.keyframe_features = self.extract_features(image=image, algorithm="sift")
            self.iteration += 1
            return

        # estimate cam pose
        pose, inliers = self.estimate_camera_pose(matches=matches)
        
        print(f"#Inliers : {inliers}")

        self.poses.append(pose)

        a = 2
        self.iteration += 1



    def match_features(self, method: str, descriptor_prev=None):
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
            descriptor_prev = self.keyframe_features['descriptors']
        elif method == '3d2d':
            if self.num_keyframe_points_3d is None:
                raise ValueError("Please set a valid number of matches to create. Value must be positive.")
            descriptor_prev = self.world_points_3d[-self.num_keyframe_points_3d:, 3:]
            a = 2
        else:
            raise ValueError("Invalid retrieval method. Use '2d2d' or '3d2d'.")

        bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
        matches = bf.match(self.query_features["descriptors"], descriptor_prev)
        return matches

    def estimate_camera_pose(self, matches: Tuple[cv2.DMatch]):
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
                Structure containning the matching information beween
                self.query_features, and self.world_points_3d.

        Returns
        -------
            num_inliers: int
                Number of inliers used by RANSAC to compute the camera pose.
            todo. engelbracht: idea: just return the pose, its appended in run?
        '''
        # Retrieve matched 2D/3D points.
        matches_array = np.array([(match.queryIdx, match.trainIdx) for match in matches])
        object_points = self.world_points_3d[matches_array[:,1], 0:3]       # 3D pt (Nx3)
        # image_points = self.query_features["keypoints"][matches_array[:,0]]  # 2D pt (Nx2)
        image_points = np.array([self.query_features['keypoints'][match.queryIdx].pt for match in matches])

        # Run PnP
        success, rvec_C_W, t_C_W, inliers = cv2.solvePnPRansac(
            object_points, image_points,
            cameraMatrix=self.K, distCoeffs=np.zeros((4,1)))
        if not success:
            raise RuntimeError("RANSAC is not able to fit the model")

        # Extract transformation matrix
        R_C_W, _ = cv2.Rodrigues(rvec_C_W)
        # t_C_W = t_C_W[:, 0]

        # Add it to the list of poses
        # T_C_W = np.eye(4)
        # T_C_W[1:3, 1:3] = R_C_W
        # T_C_W[1:3, 4] = t_C_W
        T_C_W = np.hstack([R_C_W, t_C_W])
        # self.poses = np.append((self.poses, T_C_W[:,:,np.newaxis]), axis = 2)
        # return the number of points used to estimate the camera pose
        # self.poses =
        # return inliers.size()
        return T_C_W, inliers.size

    def triangulate_point_cloud(
            self, query_feature: Dict[tuple[cv2.Feature2D], np.ndarray], train_feature: Dict[tuple[cv2.Feature2D], np.ndarray],
            matches: tuple[cv2.DMatch]):
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
        distance_thresh = 50
        # Retrieve matched 2D/2D points.
        matches_array = np.array(
            [(match.queryIdx, match.trainIdx) for match in matches])
        # first_img_pt = np.expand_dims(np.array(train_feature["keypoints"])[matches_array[:,1]],1)  # 2D pt (Nx2)
        # second_img_pt = np.expand_dims(np.array(query_feature["keypoints"])[matches_array[:,0]],1)  # 2D pt (Nx2)
        first_img_pt = np.array([train_feature['keypoints'][match.trainIdx].pt for match in matches])
        second_img_pt = np.array([query_feature['keypoints'][match.queryIdx].pt for match in matches])
        second_img_descriptor = query_feature["descriptors"][matches_array[:, 0]]

        # Compute essential matrix.
        E, essential_inliers = cv2.findEssentialMat(
            points1=first_img_pt, points2=second_img_pt,
            cameraMatrix=self.K,
            method=cv2.RANSAC, prob=0.99, threshold=1.0)

        # Extract pose from it.
        retVal, R_C_W, t_C_W, inlier_mask, triangulatedPoints = cv2.recoverPose(
                            E=E,
                            points1=first_img_pt,
                            points2=second_img_pt,
                            cameraMatrix=self.K,
                            mask=essential_inliers,
                            distanceThresh=distance_thresh)

        # Create 3D feature concatenating normalized triangulated points and
        # descriptors.
        norm_triangulated_points = (triangulatedPoints[:-1] / triangulatedPoints[3, :]).T
        feature_3d = np.hstack((norm_triangulated_points,
                               second_img_descriptor))

        # Remove outliers and populate point cloud
        feature_3d = feature_3d[np.where(inlier_mask[:,0] == 1)].astype("float32")

        # if self.world_points_3d is None:
        #     world_points_3d = feature_3d
        # else:
        #     self.world_points_3d = np.vstack((self.world_points_3d, feature_3d))

        # Append pose to history
        # todo: maybve buffer?
        # self.poses = np.dstack((self.poses, np.hstack((R_C_W, t_C_W))))

        pose = np.hstack((R_C_W, t_C_W))
        return pose, feature_3d

    def check_num_inliers(self):
        '''

        Returns
        -------

        '''
        pass

    def reprojectPoints(self, P, M, K):
        '''
        Reproject 3D points given a projection matrix. From ex02 (PnP)\n
        
        Parameters
        ----------
            P: [N x 3] 
                Coordinates of the 3d points in the world frame
            M: [3 x 4] projection matrix
            K: [3 x 3] camera matrix

        Returns
        -------
            [n x 2] image coordinates of the reprojected points
        '''
        p_homo = (K @ M @ np.r_[P.T, np.ones((1, P.shape[0]))]).T
        return p_homo[:,:2]/p_homo[:,2,np.newaxis]

    def visualize(self, mode: str, matches=None, project_points=False):
        '''
        Parameters
        -------
        mode: str
            'match': 
                - Visualizes the query image inlier feature 
                matches overlaid on image. 
                - Must provide matches: Tuple[cv2.DMatch].
                - Setting project_points=True allowed to visualize world 3D point 
                projections in camera
            
        matches: Tuple[cv2.DMatch]
            - Required for 'match' mode. 
            - List of correspondances between. 
            vizualize() only reads the queryIdx of the correspondances in order
            to determine u,v locations for inliers.

        project_points (optional): bool
            Default is False. If True, project all self.world_3d_points into 
            the camera frame.
            TODO: trim down number of 3D points to project into the scene each 
            time this is called


        Comments
        -------
        NOTE: vizualize() should be called AFTER query pose is written. This is
        so that when projecting 3D world points into the image, we use the
        query image pose.

        Poses are in world frame (first image frame). (TODO: agree on this)

        NOTE for future development
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
        
        Returns
        -------
        '''

        # # Plot pose
        # num_poses = len(self.poses)
        # pose_figsize=(7, 8)
        # _, ax = plt.subplots(figsize=pose_figsize)
        
        # for i, pose in enumerate(self.poses):
        #     # Extract translation
        #     xy = pose[:,-1]

        #     # Ground plane in camera xz plane, 3D points in world frame

        # Alternative Poses code (come back to this)
        # ground_truth = np.zeros((len(poses), 3, 4))
        # for i in range(len(poses)):
        #     ground_truth[i] = np.array(poses.iloc[i]).reshape((3, 4))
        # # matplotlib widget
        # fig = plt.figure(figsize=(7,6))
        # traj = fig.add_subplot(111, projection='3d')
        # traj.plot(ground_truth[:,:,3][:,0], ground_truth[:,:,3][:,1], ground_truth[:,:,3][:,2])
        # traj.set_xlabel('x')
        # traj.set_ylabel('y')
        # traj.set_zlabel('z')

        # Quick visualizer for query frame with features
        if mode == 'match':
            if matches is None:
                raise ValueError("Please input match correspondances \
                                 (Tuple[cv2.DMatch]) when using 'match' mode.")
            
            im = self.query_frame
            plt.clf() # clears last "features_all" points
            implot = plt.imshow(im)
            padding = 50 # whitespace padding when visualizing data

            features_matched = np.array([self.query_features['keypoints'][match.queryIdx].pt for match in matches])
            features_all = np.asarray(cv2.KeyPoint_convert(self.query_features['keypoints']))
            
            # All features in red, matched features in green
            plt.scatter(x=features_all[:,0], y=features_all[:,1], c='r', s=9, marker='x')
            plt.scatter(x=features_matched[:,0], y=features_matched[:,1], c='lime', s=9, marker='x')

            # Optional param. 
            # Projects ALL point cloud features as cyan circle into frame
            if project_points is True:
                # Our own reprojectPoints() function works
                points = self.reprojectPoints(self.world_points_3d[:,:3], 
                                              self.poses[:,:,-1], 
                                              self.K)
                plt.scatter(x=points[:,0], 
                            y=points[:,1], 
                            s=20, 
                            facecolors='none', 
                            edgecolors='cyan')

                # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # # Leonard's implementation. 
                # # Still fails, projected_points returns NaN
                # projected_points, _ = cv2.projectPoints(\
                #     objectPoints=self.world_points_3d[:,:3].copy(), 
                #     rvec=self.poses[:,:3,-1], 
                #     tvec=self.poses[:,-1,-1], 
                #     cameraMatrix=self.K,
                #     distCoeffs=np.empty((1,4)))
                # projected_points = np.squeeze(projected_points)
                # plt.scatter(x=projected_points[:,0], y=projected_points[:,1], \
                #             marker='x', s=20, edgecolors='fuchsia')
                # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            # Visualize image with whitespace padding
            plt.xlim(-padding, im.shape[1] + padding)
            plt.ylim(im.shape[0] + padding, -padding)
            plt.show()
            plt.pause(0.01)
            
        else:
            raise ValueError("Please input a valid mode for. See visualize() definition.")



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
