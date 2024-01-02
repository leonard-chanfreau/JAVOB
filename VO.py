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
    
    def __init__(self, calib_file: str):

        self.K = self.read_calib_file_(path=calib_file)
        self.num_features = 1000

        self.max_distance_thresh = 50
        
        # Query
        self.query_features = {} # {cv2.features, cv2.desciptors} storing 2D feature (u,v) and its HoG descriptor. Called via {"keypoints", "descriptors"}

        self.keyframe = None
        self.keyframe_pos = None
        self.query_frame = None

        # KLT
        self.prev_frame = None

        # Database
        self.keyframe_features = {} # overwrite every time new keyframe is created. {cv2.features, cv2.desciptors}
                                # storing 2D feature (u,v) and its HoG descriptor. Called via {"keypoints", "descriptors"}
        self.world_points_3d = None # np.ndarray(npoints x 131), where [:,:3] is the [x,y,z] world coordinate and [:,3:] is the 128 descriptor
                                # dynamic storage of triangulated 3D points in world frame and their descriptors (npoints x 128 np.ndarray)
        self.num_keyframe_points_3d = None

        self.poses = [np.hstack([np.eye(3), np.zeros((3,1))])]# list(3x4 array) where 3x4 is [R_W_C|t_W_C] projection matrix

        # internal state
        self.iteration = 0
        self.retriangulate_calls = 0


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
    
    def klt_track(self):
        '''
        Track features from previous frame.

        Parameters
        ----------

        Returns
        -------
        Good points. In continuous run, if number of good points < threshold, triangulate new w.r.t keyframe.
        '''
        pass
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it
        p0 = self.query_features["keypoints"]
        # Create a mask image for drawing purposes
        mask = np.zeros_like(self.query_frame)

        # TODO here belowwwwwwwww~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # while(1):
        #     # calculate optical flow
        #     p1, st, err = cv.calcOpticalFlowPyrLK(self.prev_frame, self.query_frame, p0, None, **lk_params)
        #     # Select good points
        #     if p1 is not None:
        #         good_new = p1[st==1]
        #         good_old = p0[st==1]
        #     # draw the tracks
        #     for i, (new, old) in enumerate(zip(good_new, good_old)):
        #         a, b = new.ravel()
        #         c, d = old.ravel()
        #         mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        #         frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        #     img = cv.add(frame, mask)
        #     cv.imshow('frame', img)
        #     k = cv.waitKey(30) & 0xff
        #     if k == 27:
        #         break
        #     # Now update the previous frame and previous points
        #     old_gray = frame_gray.copy()
        #     p0 = good_new.reshape(-1, 1, 2)


    def initialize_point_cloud(self, image: np.ndarray):
        """
        initialize point cloud if depth uncertainty is below threshold (i.e. frames are sufficiently far apart)

        Parameters
        ----------
        image:
            query image

        Returns
        -------

        """

        # extract and set query features
        self.query_features = self.extract_features(image=image)

        # match query and keyframe features
        matches = self.match_features(method="2d2d")

        # initialize preliminary point cloud
        # todo remove self.... as args?
        pose, points_3d = self.compute_pose_and_3D_pts(query_feature=self.query_features, train_feature=self.keyframe_features, matches=matches)

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
            self.poses.append(pose)
            return True
        else:
            return False

    def run_normal(self, image: np.ndarray):
        '''

        Parameters
        ----------
        image
        mode
            'normal'
            'klt'

        Returns
        -------

        '''
        # match threshold
        thresh_inliers = 60

        # set very first keyframe
        if self.keyframe is None:
            self.keyframe = image
            self.keyframe_features = self.extract_features(image=image, algorithm="sift")
            self.iteration += 1
            return

        # initialize first point cloud and set new keyframe
        if self.world_points_3d is None:
            self.query_frame = image
            if self.initialize_point_cloud(image=image):
                self.keyframe = image
                self.keyframe_pos = self.poses[-1]
                self.keyframe_features = self.extract_features(image=image, algorithm="sift")
            self.iteration += 1
            return

        # continuous run
        # extract features and find matches between self.query_features
        # and self.world_points_3d[-self.num_keyframe_points_3d:, 3:] (CURRENT keyframe points)
        self.query_frame = image
        self.query_features = self.extract_features(image=image, algorithm="sift")
        matches = self.match_features(method="3d2d")

        # estimate cam pose
        pose, num_inliers = self.estimate_camera_pose(matches=matches)
        self.poses.append(pose)

        # DEBUG
        last_pose = np.array(self.poses)[-1,:,:]
        before_last_pose = np.array(self.poses)[-2,:,:]
        raw_travel = before_last_pose[:,3] - last_pose[:,3]
        # print(f"raw travel: {raw_travel}")
        travel = np.linalg.norm(before_last_pose[:,3] - last_pose[:,3])
        # print(f"dist travel: {travel}")
        if travel > 1:
            b=2
        
        print(f"#Inliers : {num_inliers}")
        self.iteration += 1
        
        self.visualize(mode='match', matches=matches)
        self.visualize(mode="traj2d", centered=True)
        b=2
        
        #if self.iteration > 25:
            #self.plot_map()

        # Check if enough inliers are found,
        # else reinit new keyframe and expand point cloud using new keyframe and old keyframe
        if (num_inliers < thresh_inliers) or (self.iteration % 4 == 0): # TODO: Maybe rather a percentage of point cloud
            matches = self.match_features("2d2d")
            self.triangulate_new_points(pose, matches)   #TODO: Maybe a wrapper as initialize point cloud that checks if camera are distant enough
            self.keyframe = image
            self.keyframe_pos = self.poses[-1]
            self.keyframe_features = self.query_features
            return
        
    def run_klt(self, image: np.ndarray):
    
        # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # match threshold
        thresh_inliers = 60

        # set very first keyframe
        if self.keyframe is None:
            self.keyframe = image
            self.keyframe_features = self.extract_features(image=image, algorithm="sift")
            self.iteration += 1
            return

        # initialize first point cloud and set new keyframe
        if self.world_points_3d is None:
            self.query_frame = image
            if self.initialize_point_cloud(image=image):
                self.keyframe = image
                self.keyframe_pos = self.poses[-1]
                self.keyframe_features = self.extract_features(image=image, algorithm="sift")
            self.iteration += 1
            return

        # continuous run
        # extract features and find matches between self.query_features
        # and self.world_points_3d[-self.num_keyframe_points_3d:, 3:] (CURRENT keyframe points)
        self.query_frame = image
        self.query_features = self.extract_features(image=image, algorithm="sift")
        # matches = self.match_features(method="3d2d")
        # TODO TRACK FEATURES
        good_keypoints = self.klt_track()
        self.prev_frame = image # TODO check if this goes here

        threshold = 60
        if len(good_keypoints) < threshold:
            # Extract SIFT features in the previous frame and current frame using SIFT toolbox
            # Match keypoints in these two frames using SIFT toolbox
            # Estimate fundamental matrix by Matlab function estimateFundamentalMatrix
            # Estimate the current camera pose by decomposing Fundamental Matrix
            # Triangulate Landmarks
        
        # TODO here belowwwwwwwww~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # estimate cam pose
        pose, num_inliers = self.estimate_camera_pose(matches=matches)
        self.poses.append(pose)
        
        print(f"#Inliers : {num_inliers}")
        self.iteration += 1
        
        self.visualize(mode='match', matches=matches)
        self.visualize(mode="traj2d", centered=True)
        b=2
        
        #if self.iteration > 25:
            #self.plot_map()

        # Check if enough inliers are found,
        # else reinit new keyframe and expand point cloud using new keyframe and old keyframe
        if num_inliers < thresh_inliers: # TODO: Maybe rather a percentage of point cloud
            matches = self.match_features("2d2d")
            self.triangulate_new_points(pose, matches)   #TODO: Maybe a wrapper as initialize point cloud that checks if camera are distant enough
            self.keyframe = image
            self.keyframe_pos = self.poses[-1]
            self.keyframe_features = self.query_features
            return




    def match_features(self, method: str, descriptor_prev=None, ratio_test=False):
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
        ratio_test: bool
            NOTE: BEWARE using knn leads to early implosion of VO!!!???
            True, uses KNN matching and apply ratio test for matching.
            False runs Brute Force matching with no ratio test.

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
            descriptor_prev = self.world_points_3d[-self.num_keyframe_points_3d:, 3:]   #TODO: this induce a shift -> see where impacted
            a = 2
        else:
            raise ValueError("Invalid retrieval method. Use '2d2d' or '3d2d'.")

        if ratio_test:
            bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=False) # False crossCheck
            knn_output = bf.knnMatch(self.query_features["descriptors"], descriptor_prev,k=2)
            # Apply ratio test
            threshold = 0.8
            matches = []
            for m,n in knn_output:
                if m.distance < threshold * n.distance:
                    matches.append(m)
        else:
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
        matches_array = np.array([(match.queryIdx, match.trainIdx+len(self.world_points_3d)-self.num_keyframe_points_3d) for match in matches])
        train_object_points = self.world_points_3d[matches_array[:,1], 0:3]       # 3D pt (Nx3)
        # image_points = self.query_features["keypoints"][matches_array[:,0]]  # 2D pt (Nx2)
        query_image_points = np.array([self.query_features['keypoints'][match.queryIdx].pt for match in matches])

        # Run PnP
        success, rvec_C_W, t_C_W, inliers = cv2.solvePnPRansac(
            train_object_points, query_image_points,
            cameraMatrix=self.K, distCoeffs=np.zeros((4,1)))
        
        # if self.iteration > 26:
        #     R = self.keyframe_pos[:, :3].T
        #     t = -R@self.keyframe_pos[:, 3].reshape(3, 1)
        #     projected_points, _ = cv2.projectPoints(train_object_points, R, t, self.K, None)
        #     projected_points = np.squeeze(projected_points)
        #     matched_image = cv2.drawMatches(
        #         self.query_frame, np.array([self.query_features["keypoints"][match.queryIdx] for match in matches]),
        #         self.keyframe, np.array([cv2.KeyPoint(projected_points[i,0], projected_points[i,1], 5) for i in range(projected_points.shape[0])]),
        #         [cv2.DMatch(i[0], i[0], 0) for i in inliers][:10],
        #         None,
        #         matchColor=(0, 255, 0),  # Green color for inliers
        #         singlePointColor=(255, 0, 0)  # Blue color for keypoints
        #     )

        #     plt.imshow(matched_image)
        #     plt.show()
        
        if not success:
            raise RuntimeError("RANSAC is not able to fit the model")

        # Extract rotation matrix
        R_C_W, _ = cv2.Rodrigues(rvec_C_W)
        # Invert to have camera pose in world frame
        R_W_C = R_C_W.T
        t_W_C = -R_W_C @ t_C_W
        # Construct pose matrix
        T_W_C = np.hstack([R_W_C, t_W_C])
        
        return T_W_C, inliers.size

    def compute_pose_and_3D_pts(
            self, query_feature: Dict[tuple[cv2.Feature2D], np.ndarray], train_feature: Dict[tuple[cv2.Feature2D], np.ndarray],
            matches: tuple[cv2.DMatch]):
        # TODO: Maybe these arguments should be members of the class.
        '''
        Compute the camera pose and construct a point cloud using 8-pt algorithm
        and RANSAC (SFM).\n
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
                            distanceThresh=self.max_distance_thresh)
        # Convert it to camera pose in world frame
        R_W_C = R_C_W.T
        t_W_C = -R_W_C @ t_C_W

        # Create 3D feature concatenating normalized triangulated points and
        # descriptors.
        norm_triangulated_points = (triangulatedPoints[:-1] / triangulatedPoints[3, :]).T
        feature_3d = np.hstack((norm_triangulated_points,
                               second_img_descriptor))

        # Remove outliers and populate point cloud
        feature_3d = feature_3d[np.where(inlier_mask[:,0] == 1)].astype("float32")
        
        pose = np.hstack((R_W_C, t_W_C))
        
        return pose, feature_3d

    def check_num_inliers(self):
        '''

        Returns
        -------

        '''
        pass

    def triangulate_new_points(self, query_pose, matches):
        '''
        Parameters
        ----------
        - M_keyframe: 3x4 projection matrix of the first camera (Last Keyframe).
        - M_query: 3x4 projection matrix of the second camera (Query image).
        - P_keyframe: 2xN array of feature points in the first image (Last Keyframe). 
            - It can be also a cell array of feature points {[x,y], ...}
        - P_query: 2xN array of corresponding points in the second image (Query image).
            - It can be also a cell array of feature points {[x,y], ...}
        Comments
        -------
        Triangulates new 3D points in the scene using the last keyframe and the query image. Should not be called 
        every frame, only when we are triangulating new 3D features.
        Returns
        -------
        points_3d: 4xN array of reconstructed points in homogeneous coordinates [[x;y;z;w], ...]
        '''
        # Construct projections matrices: project 3D world point into the images
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
        # TODO: dont know if all matches are reliable
        
        # matches = sorted(matches, key=lambda x: x.distance)
        # matched_image = cv2.drawMatches(
        #     self.query_frame, np.array(
        #         [self.query_features['keypoints'][match.queryIdx] for match in matches]),
        #     self.keyframe, np.array(
        #         [self.keyframe_features['keypoints'][match.trainIdx] for match in matches]),
        #     [cv2.DMatch(i, i, 0) for i in range(len(matches))][:100],
        #     None,
        #     matchColor=(0, 255, 0),  # Green color for inliers
        #     singlePointColor=(255, 0, 0)  # Blue color for keypoints
        #     )

        # plt.imshow(matched_image)
        # plt.show()
        
        #self.plot_map()

        # testing last frame

        points_4d = cv2.triangulatePoints(M_keyframe, M_query, p_keyframe, p_query)
        # Un-homogenize points
        points_3d = (points_4d[:3, :] / points_4d[3, :]).T      # In world frame (verified)
        
        # Project into camera view
        points_3d_c = np.transpose(query_pose[:3,:3] @ points_3d.T  + np.expand_dims(query_pose[:,-1],1))
        
        # mask = np.where((points_3d[:, 2] > query_pose[2,3]) &
        #                 (points_3d[:, 2] - query_pose[2,3] < self.max_distance_thresh)) #TODO: in fact rather the points "in front" not necessary z axis
        
        # This mask works best. 
        # Discards points with negative z value IN CAMERA VIEW
        mask = np.where((points_3d_c[:, 2] > 0) & 
                        (points_3d_c[:, 2] < self.max_distance_thresh))
        
        points_3d = points_3d[mask]
        
        # Package 3D points and their descriptors
        new_3d_points = np.zeros((points_3d.shape[0], 131)).astype(np.float32)
        new_3d_points[:,:3] = points_3d
        new_3d_points[:,3:] = np.array(
            [self.query_features["descriptors"][match.queryIdx]
             for match in matches])[mask]
        self.world_points_3d = np.vstack((self.world_points_3d, new_3d_points))
        self.num_keyframe_points_3d = points_3d.shape[0]
        #self.plot_map()

        # DEBUGGING
        # Report number of retriangulation calls
        self.retriangulate_calls += 1
        print(f"NEW TRIANGULATION CALL {self.retriangulate_calls}")
        print(f"Number of new points: {new_3d_points.shape[0]}")
        
        

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

    def visualize(self, mode: str, matches=None, project_points=False, centered=False):
        '''
        Parameters
        -------
        mode: str
            'match': 
                Visualizes the query image inlier feature 
                matches overlaid on image. 
                Must provide matches: Tuple[cv2.DMatch].
                Setting project_points=True allowed to visualize world 3D point 
                projections in camera
            'traj2d'
                RECOMMENDED
                2D top-down view of trajectory and point cloud.
            'traj3d' 
                NOTE: NOT recommended
                Attempt at using pytransform3d library. Somewhat works but can't
                see global view of traj and pt cloud.
                Visualizes the query frame camera pose and the trajectory of
                previous poses
            
        matches: Tuple[cv2.DMatch]
            Required for 'match' mode. 
            List of correspondances between. 
            vizualize() only reads the queryIdx of the correspondances in order
            to determine u,v locations for inliers.

        project_points (optional): bool
            Optional for 'match' mode.
            Default is False. If True, project all self.world_3d_points into 
            the camera frame.
            TODO: trim down number of 3D points to project into the scene each 
            time this is called
        
        centered (optional): bool
            Optional for 'traj2d' mode.
            Default is False. If True, keeps the query pose in the frame center
            when plotting trajecotry and point cloud.


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

        if mode == 'match':
            if matches is None:
                raise ValueError("Please input match correspondances \
                                 (Tuple[cv2.DMatch]) when using 'match' mode.")
            match_plot_num = 1 # to not overwrite other figs

            plt.figure(match_plot_num)
            plt.figure(match_plot_num).clf() # clears last "features_all" points
            plt.ion()  # interactive mode on for dynamic updating

            im = self.query_frame
            implot = plt.imshow(im)
            padding = 50 # whitespace padding when visualizing data

            features_matched = np.array([self.query_features['keypoints'][match.queryIdx].pt for match in matches])
            features_all = np.asarray(cv2.KeyPoint_convert(self.query_features['keypoints']))
            
            # All features in red, matched features in green
            plt.scatter(x=features_all[:,0], y=features_all[:,1], c='r', s=9, marker='x')
            plt.scatter(x=features_matched[:,0], y=features_matched[:,1], c='lime', s=9, marker='x')

            # Optional param. 
            # Projects ALL point cloud features as cyan circle into frame

            # TODO rotation causes points to reproj wrong way.
                # Also only project points in FRONT of frame??
            
            if project_points is True:
                # Our own reprojectPoints() function works

                # TODO new feature: reproject 3D points that fall within a bounding sphere
                # of the current pose

                points = self.reprojectPoints(self.world_points_3d[:,:3], 
                                              np.array(self.poses)[-1,:,:], 
                                              self.K)
                plt.scatter(x=points[:,0], 
                            y=points[:,1], 
                            s=20, 
                            facecolors='none', 
                            edgecolors='cyan')

                # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # # Attempting cv2.projectPoints implementation. 
                # # STILL FAILS, returns NaN
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
            plt.draw()
            plt.pause(0.001)
            plt.ioff()  # interactive mode off after updating
        
        elif mode == 'traj2d':
            #TODO later: plot new 3d points in blue, old 3d points in different shade.
            
            traj_plot_number = 2 # to not overwrite 'match'

            if not plt.fignum_exists(traj_plot_number):
                plt.ion()  # Turn on interactive mode for dynamic updating
                fig = plt.figure(num=traj_plot_number, figsize=(7, 6))
                traj = fig.add_subplot(111)
            else:
                # If figure exists, clear plot
                plt.figure(traj_plot_number).clf()
                fig = plt.gcf()
                traj = fig.add_subplot(111)

            R_T = np.array(self.poses)

            # XZ
            traj.plot(R_T[:, 0, 3], R_T[:, 2, 3], label='Trajectory') # Plots X,Z
            traj.set_xlabel('X')
            traj.set_ylabel('Z')

            # # XY
            # traj.plot(R_T[:, 0, 3], R_T[:, 1, 3], label='Trajectory') # Plots X,Y
            # traj.set_xlabel('X')
            # traj.set_ylabel('Y')

            # Plot point cloud
            x_world = self.world_points_3d[:, 0]
            y_world = self.world_points_3d[:, 1]
            z_world = self.world_points_3d[:, 2]

            # XZ
            traj.scatter(x_world, z_world, 
                         color='coral', label='World Points',
                         s=8) 
            
            # # XY
            # traj.scatter(x_world, y_world, 
            #              color='coral', label='World Points',
            #              s=8)
            
            # Plot optical axis arrow (x,y,z)
            optical_axis = R_T[-1,:,:3] @ np.array((0,0,1)).T # 
            # NOTE: XZ! b/c in camera frame TODO CHECK THIS
            x = R_T[-1,0,-1]
            z = R_T[-1,2,-1]
            traj.arrow(x, z, optical_axis[0], optical_axis[2],
                       width=0.1)

            if centered:
                # Plots XZ
                traj.set_xlim(-15 + R_T[-1,0,-1], 15 + R_T[-1,0,-1])
                traj.set_ylim(-15 + R_T[-1,-1,-1], 15 + R_T[-1,-1,-1])

                # # Plots XY
                # traj.set_xlim(-15 + R_T[-1,0,-1], 15 + R_T[-1,0,-1])
                # traj.set_ylim(-15 + R_T[-1,1,-1], 15 + R_T[-1,1,-1])

            traj.legend()

            plt.draw()
            plt.pause(0.001)
        
        elif mode == 'traj3d':

            # TESTING PYTRANSFROM3D ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # transpose() changes dstack (3x3xN) to vstack (Nx3x3)
            pose_rot_mat = R.from_matrix(np.transpose(self.poses[:3,:3,:], (2, 0, 1))) 
            # Convert [R|t] to quaternion (Nx4)
            quat = R.as_quat(pose_rot_mat)

            quat = quat[:, (3,0,1,2)] # xyzw to wxyz format
            # does same as pr.quaternion_wxyz_from_xyzw()
            
            # Creating N x (x, y, z, qw, qx, qy, qz)
            pose_quat = np.hstack((np.transpose(self.poses[:3,-1,:]), quat))
    
            cam2world_trajectory = ptr.transforms_from_pqs(pose_quat)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

            # Plot camera trajectory
            ax = pt.plot_transform(ax, s=0.3)  # Use the same axis for the camera trajectory
            ax = ptr.plot_trajectory(ax, P=pose_quat, s=0.1, n_frames=10)

            # image_size = np.array([1920, 1440])
            key_frames_indices = np.linspace(0, len(pose_quat) - 1, 10, dtype=int)
            colors = cycle("rgb")
            for i, c in zip(key_frames_indices, colors):
                pc.plot_camera(ax, self.K, cam2world_trajectory[i],
                            sensor_size=self.query_frame.shape, virtual_image_distance=0.2, c=c)

            # Set limits and view for camera trajectory
            pos_min = np.min(pose_quat[:, :3], axis=0)
            pos_max = np.max(pose_quat[:, :3], axis=0)
            center = (pos_max + pos_min) / 2.0
            max_half_extent = max(pos_max - pos_min) / 2.0
            ax.set_xlim((center[0] - max_half_extent, center[0] + max_half_extent))
            ax.set_ylim((center[1] - max_half_extent, center[1] + max_half_extent))
            ax.set_zlim((center[2] - max_half_extent, center[2] + max_half_extent))

            latest_pose = pose_quat[-1, :3]  # Extract translation from the latest pose
            ax.view_init(azim=-90, elev=0)  # Top-down (XZ) view

            # Plot 3D point cloud on the same plot centered around the latest pose
            x = self.world_points_3d[:, 0] - latest_pose[0]
            y = self.world_points_3d[:, 1] - latest_pose[1]
            z = self.world_points_3d[:, 2] - latest_pose[2]
            ax.scatter(x, y, z)

            plt.show()
            # END TESTING PYTRANSFROM3D ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        else:
            raise ValueError("Please input a valid mode for. See visualize() definition.")
        
    def plot_map(self):
        p_W_landmarks = np.array(self.poses)[:,:,3].T #TODO resize it to 3*N
        p_W_frames = self.world_points_3d[:, :3].T
        
        
        fig = plt.figure()
        ax = fig.add_axes(111)
        ax.plot(p_W_landmarks[2, :], -p_W_landmarks[0, :], '.')
        ax.plot(p_W_frames[2, :], -p_W_frames[0, :], 'rx', linewidth=3)

        ax.axis('equal')
        ax.set_title("Trajectory and point cloud")
        plt.show()
        plt.pause(0.1)


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
