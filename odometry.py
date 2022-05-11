from glob import glob
import cv2, skimage, os, scipy
import scipy.spatial
import numpy as np
from base import VO

class Odometry(VO):
    def __init__(self, frame_path) -> None:
        super().__init__(frame_path)

    def get_sift_data(self, img):
        """
        detect the keypoints and compute their SIFT descriptors with opencv library
        """
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        kp = np.array([k.pt for k in kp])
        return kp, des

    def get_best_matches(self, img1, img2, num_matches=4000):
        """
        Get the top n matches using SIFT to calculate fundamental matrix

        Returns: 
        ndarray -- 2d pts pairs (num_matches x 4)
        """
        _kp1, des1 = self.get_sift_data(img1)
        _kp2, des2 = self.get_sift_data(img2)

        # find distance between descriptors in images
        dist = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')
        
        # get the matches corresponding to the top 8/num_matches dist
        rows = dist.shape[0]
        cols = dist.shape[1]
        idx = np.argsort(dist.flatten())
        results = []
        kp1 = []
        kp2 = []
        for i in range(num_matches):
            _kp_1 = _kp1[idx[i]//cols, :]
            _kp_2 = _kp2[idx[i]%cols, :]
            # results.append([_kp_1, _kp_2])
            kp1.append(_kp_1)
            kp2.append(_kp_2)

        # results = np.array(results).reshape(-1, 4)
        kp1 = np.array(kp1).reshape(-1, 2)
        kp2 = np.array(kp2).reshape(-1, 2)

        # return results
        return kp1, kp2

    def get_fundamental_matrix(self, matches):
        """
        Compute fundamental matrix with matches
        Param:
        - matches: n x 4 matrix, where n >= 8

        Return fundamental matrix F
        """
        assert(matches.shape[0] >= 8)
        U = []
        for x1, y1, x2, y2 in matches[:8]:
            U.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
        U = np.array(U).reshape(8, -1)
        _, _, V = np.linalg.svd(U)
        F = V[-1,:].squeeze()
        # normalization
        F = F / F[8]
        F = F.reshape(3, 3)

        return F
    
    def normalize(self, kp):
        tx, ty = np.average(kp, axis=0)
        mc = kp - np.array([tx, ty])
        # get scale factor
        s = np.sqrt(np.sum(mc**2))/len(kp)
        T = np.array([[s, 0, -s*tx],
                      [0, s, -s*ty],
                      [0, 0, 1]])
        
        kp = np.hstack([kp, np.ones((len(kp), 1))])
        kp = kp.dot(T.T)

        return T, kp

    def get_fundamental_matrix(self, kp1, kp2):
        """
        Compute fundamental matrix with matches
        Param:
        - matches: n x 4 matrix, where n >= 8

        Return fundamental matrix F
        """
        T1, kp1 = self.normalize(kp1)
        T2, kp2 = self.normalize(kp2)
        matches = np.hstack([kp1, kp2])
        assert(matches.shape[0] >= 8 and matches.shape[1] == 6)
        U = []
        # for x1, y1, x2, y2 in matches[:8]:
        for x1, y1, _, x2, y2, _ in matches[:8]:
            U.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
        U = np.array(U).reshape(8, -1)
        _, _, V = np.linalg.svd(U)
        F = V[-1,:].squeeze()
        # normalization
        F = F / F[8]
        F = F.reshape(3, 3)
        F = (T1.T).dot(F).dot(T2)

        return F

    def get_camera_matrix(self):
        camera_matrix = np.array([[self.focal_length, 0, self.pp[0]],
                                    [0, self.focal_length, self.pp[1]],
                                    [0, 0, 1]])

        return camera_matrix

    def get_essential_matrix(self, kp1, kp2, thres=0.1, iter=300, p=0.9):
        F, kp1, kp2 = self.ransac(kp1, kp2, max_iters=iter, thres=thres, p=p)
        K = self.get_camera_matrix()

        return K.T.dot(F).dot(K), kp1, kp2

    def featureTracking(self, img_1, img_2, p1):
        lk_params = dict(winSize  = (21,21),
                        maxLevel = 3,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
        st = st.reshape(st.shape[0])
        ##find good one
        p1 = p1[st==1]
        p2 = p2[st==1]

        return p1, p2

    def run(self):
        """
        Uses the video frame to predict the path taken by the camera

        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """

        for i in range(len(self.frames)):
            img = self.imread_bw(self.frames[i])
            self.images.append(img)
        
        #find the detector
        detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        kp1 = detector.detect(self.imread(self.frames[0]))
        kp1 = np.array([ele.pt for ele in kp1],dtype='float32')

        kp1, kp2 = self.featureTracking(self.images[0], self.images[1], kp1)
        # calculate essential matrix with given matches
        # E, kp2, kp1 = self.get_essential_matrix(kp2, kp1, iter=100, p=0.9)
        E, mask = cv2.findEssentialMat(kp2, kp1, self.focal_length, self.pp, cv2.RANSAC,0.999,1.0)
        
        # extract rotation matrix R and translation t
        _, R, t, mask = cv2.recoverPose(E, kp2, kp1, focal=self.focal_length, pp=self.pp)

        # initialize variables
        kp_t_1 = kp2
        I_t_1 = self.images[1]
        R_cur = R
        t_cur = t
        err = 0

        x0, y0, z0 = self.get_gt(0).flatten()
        x1, y1, z1 = t_cur.flatten()
        coords = [[.0, .0, .0], [x1, y1, z1]]
        iter = 100
        p = 0.8
        
        for idx in range(2, len(self.frames)):

            if (len(kp_t_1) < 2000):
                kp   = detector.detect(I_t_1)
                kp_t_1 = np.array([ele.pt for ele in kp],dtype='float32')
                # iter = 200

            I_t = self.images[idx]

            kp_t_1, kp_t = self.featureTracking(I_t_1, I_t, kp_t_1)

            E, _ = cv2.findEssentialMat(kp_t, kp_t_1, self.focal_length, self.pp, cv2.RANSAC, 0.999, 1.0)
            _, R, t, _ = cv2.recoverPose(E, kp_t, kp_t_1, focal=self.focal_length, pp=self.pp)
            
            mask = np.where(mask == 1)[0]
            kp_t_1 = kp_t_1[mask, :].reshape(-1, 2)
            kp_t = kp_t[mask, :].reshape(-1, 2)

            coord = self.get_gt(idx)
            scale = self.get_scale(idx)

            if scale > 0.1:  
                t_cur = t_cur + scale * R_cur.dot(t)
                R_cur = R.dot(R_cur)

            I_t_1 = I_t
            kp_t_1 = kp_t
            # iter = 100

            err = np.linalg.norm(coord - t_cur)
            print("Error = ", err)

            coords.append(t_cur.flatten().tolist())
        
        coords = np.array(coords).reshape(-1, 3)
        np.save("predictions", coords)
        
        return coords
        
