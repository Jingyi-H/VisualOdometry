from glob import glob
import cv2, skimage, os, scipy
import scipy.spatial
import numpy as np
from base import VO

class Odometry(VO):
    def __init__(self, frame_path) -> None:
        super().__init__(frame_path)

    def get_camera_matrix(self):
        camera_matrix = np.array([[self.focal_length, 0, self.pp[0]],
                                    [0, self.focal_length, self.pp[1]],
                                    [0, 0, 1]])

        return camera_matrix

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
            
            # remove invalid key points
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
        
