from glob import glob
import cv2, skimage, os, scipy
import scipy.spatial
import numpy as np


class VO:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.images = []
        self.pos = []
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]

    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname)

    def imread_bw(self, fname):
        """
        read image as gray scale format
        """
        return cv2.cvtColor(self.imread(fname), cv2.COLOR_BGR2GRAY)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

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

    def get_essential_matrix(self, kp1, kp2, thres=0.1, iter=300):
        F, kp1, kp2 = self.ransac(kp1, kp2, max_iters=iter, thres=thres)
        K = self.get_camera_matrix()
        return K.T.dot(F).dot(K), kp1, kp2

    def calcLucasKanade(self, img1: np.array, img2: np.array)->np.array:
        '''
        referenced from lab5_soln.ipynb
        Takes two consecutive frames from a video and computes Optical flow using LucasKanade algorithm.
        
        :param img1: frame at t-1
        :param img2: frame at t
        '''

        #Flattens last two dimensions in a 4D array
        flattenLast = lambda img: np.reshape(img, (img.shape[0], img.shape[1], -1))

        # Flattens first two dimension in a 3D array
        flattenFirst = lambda img: np.reshape(img, (-1, img.shape[-1]))

        # kernel for x derivative
        dX = np.array([[1, 0, -1]])

        # Kernel for y derivative
        dY = dX.T
        
        img1, img2 = cv2.GaussianBlur(img1, (5, 5), 0.5), cv2.GaussianBlur(img2, (5, 5), 0.5) 
        
        # Get X, Y and t derivatives
        Ix = sig.convolve2d(img1, dX, mode='same')
        Iy = sig.convolve2d(img2, dY, mode='same')
        It = img2-img1
        
        # Gets Sliding windows of size 5x5 for the dervatives
        Ix = sliding_window_view(Ix, (3, 3))
        Iy = sliding_window_view(Iy, (3, 3))
        It = sliding_window_view(It, (3, 3))
        
        # Flatten for broadcasting matrix multiplications
        Ix = flattenFirst(flattenLast(Ix))
        Iy = flattenFirst(flattenLast(Iy))
        It = flattenFirst(flattenLast(It))
        
        # Setup the equations
        A = np.dstack((Ix, Iy))
        At = np.transpose(A, (0, 2, 1))
        It = It[:, :, None]
        
        # Compute flows
        Ainv = np.stack(list(map(np.linalg.inv, At@A+np.random.rand(2,2)*1e-6))) # calculate each matrix inverse
        buf = At@It

        flow = Ainv@buf
        flow = np.reshape(flow[:, :, 0], (img1.shape[0]-2, img1.shape[1]-2, 2))
        
        return flow

    def calcOpticalFlowLK(self, img1, img2, kp1):
        flow = self.calcLucasKanade(img1, img2)
        u = flow[:, :, 0].T
        v = flow[:, :, 1].T
        assert(kp1.shape[1] == 2 and img1.shape == img2.shape)
        
        # threshold kp1, discard pts that is out of flow map
        thres = np.array([flow.shape[1], flow.shape[0]])
        status = np.all(kp1 < thres, axis=1).reshape(-1, 1)
        status = np.hstack([status, status])
        kp1 = kp1[status].reshape(-1, 2)

        # translation for kp1
        row = kp1.astype(np.int32)[:, 0]
        col = kp1.astype(np.int32)[:, 1]
        du = u[tuple([row.tolist(), col.tolist()])].reshape(-1, 1)
        dv = v[tuple([row.tolist(), col.tolist()])].reshape(-1, 1)
        kp2 = kp1 + np.hstack([du, dv])
        kp2 = np.around(kp2)
        
        # discard invisible pts
        thres = np.array([img1.shape[1], img1.shape[0]])
        status = np.all((kp2 < thres) & (kp2 >= 0), axis=1).reshape(-1, 1)
        status = np.hstack([status, status])

        kp1 = kp1[status].reshape(-1, 2)
        kp2 = kp2[status].reshape(-1, 2)

        kp1, kp2 = self.ransac(kp1, kp2, thres=1)
        
        return kp1, kp2
    
    def ransac(self, kp1, kp2, min_inliers=10, max_iters=100, thres=1):
        opt = -1
        optim_inliers = 0 # optimal number of inliers
        optim_F = []   # homography corresponding to the optimal number of inliers
        assert(kp1.shape[0] == kp2.shape[0])
        n = kp1.shape[0]
        idx = []
        for iter in range(max_iters):
            # randomly select 8 matches
            randIdx = np.random.randint(0, n, 8)
            F = self.get_fundamental_matrix(kp1[randIdx, :], kp2[randIdx, :])
            # test on all input matches
            _kp1 = np.hstack([kp1, np.ones((n, 1))])
            _kp2 = np.hstack([kp2, np.ones((n, 1))])
            prod = _kp1.dot(F).dot(_kp2.T)
            prod = np.abs(prod.diagonal())
            norm = np.linalg.norm(prod) / n
            num_inliers = np.where(prod < thres)[0].size
            if num_inliers < min_inliers:
                continue
            if num_inliers > optim_inliers:
                # update optim_inliers if current number of inliers is larger than optimal
                optim_inliers = num_inliers
                optim_F = F
                opt = norm
                idx = np.where(prod < thres)
            if norm < thres:
                break

        kp1 = kp1[idx, :]
        kp2 = kp2[idx, :]

        return optim_F, kp1, kp2

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

        kp1, kp2 = self.calcOpticalFlowLK(self.images[0], self.images[1], kp1)
        # calculate essential matrix with given matches
        E, kp2, kp1 = self.get_essential_matrix(kp2, kp1, iter=200, thres=0.01)
        
        # extract rotation matrix R and translation t
        _, R, t, _ = cv2.recoverPose(E, kp2, kp1, focal=self.focal_length, pp=self.pp)

        # initialize variables
        kp_t_1 = kp2
        I_t_1 = self.images[1]
        R_cur = R
        t_cur = t
        err = 0

        x0, y0, z0 = self.get_gt(0).flatten()
        x1, y1, z1 = t_cur.flatten()
        coords = [[.0, .0, .0], [x1, y1, z1]]
        
        for idx in range(2, len(self.frames)):

            if (len(kp_t_1) < 1500):
                kp   = detector.detect(I_t_1)
                kp_t_1 = np.array([ele.pt for ele in kp],dtype='float32')

            I_t = self.images[idx]

            kp_t_1, kp_t = self.calcOpticalFlowLK(I_t_1, I_t, kp_t_1)
            E, kp_t, kp_t_1 = self.get_essential_matrix(kp_t, kp_t_1, iter=200, thres=0.01)

            _, R, t, _ = cv2.recoverPose(E, kp_t, kp_t_1, focal=self.focal_length, pp=self.pp)

            coord = self.get_gt(idx)
            scale = self.get_scale(idx)

            if scale > 0.1:  
                t_cur = t_cur + scale * R_cur.dot(t)
                R_cur = R.dot(R_cur)

            I_t_1 = I_t
            kp_t_1 = kp_t

            err = np.linalg.norm(coord - t_cur)
            print("Error = ", err)

            coords.append(t_cur.flatten().tolist())
        
        coords = np.array(coords).reshape(-1, 3)
        np.save("predictions", coords)
        
        return coords