import os
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

from Vec3 import *

mp_pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)


class Skeleton:
    """ class with a skeleton """
    reduce_indice  = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    dim = 33
    full_dim      = 3*dim
    reduced_dim   = len(reduce_indice)*2

    colors_rgb = np.array([
                    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                    [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
                    [0, 0, 128], [255, 128, 0], [128, 0, 128], [128, 128, 0],
                    [0, 128, 128], [128, 128, 128], [192, 192, 192], [255, 165, 0],
                    [165, 42, 42], [0, 128, 192], [128, 0, 128], [128, 0, 0],
                    [128, 128, 0], [0, 128, 0], [0, 128, 128], [0, 0, 128],
                    [0, 165, 255], [165, 42, 42], [255, 140, 0], [0, 250, 154],
                    [75, 0, 130], [0, 255, 255], [218, 112, 214], [210, 105, 30],
                    [240, 230, 140], [255, 20, 147],
                ], dtype=np.uint8)

    def __init__(self, ske=None):
        if ske is not None:
            self.ske = ske
        else:
            self.ske = np.empty(Skeleton.dim, dtype=Vec3)
            for i in range(Skeleton.dim):
                self.ske[i] = Vec3(0,0,0)

    def __str__(self):          
        return str(self.ske)

    def __array__(self, dtype=None, reduced=False):
        """ return skeleton as a numpy array of float, if reduced is True, keep only 13 minimals joints """
        if reduced:
            return np.vstack(self.ske[self.reduce_indice]).astype(float)[:, :2]
        else:
            return np.vstack(self.ske).astype(float)
    
    def toarray(self, reduced=False):
        if reduced:
            return np.vstack(self.ske[self.reduce_indice]).astype(float)[:, :2]
        else:
            return np.vstack(self.ske).astype(float)

    def reduce(self):
        return self.toarray(reduced=True)
    
    def fromImage(self, image):     
        results = mp_pose_detector.process(image)
        if results.pose_landmarks is None:
            return False
        if results.pose_landmarks:
            for index, landmark in enumerate(results.pose_landmarks.landmark):                    
                self.ske[index] = Vec3(landmark.x, landmark.y, landmark.z)
        ok = len(results.pose_landmarks.landmark) == Skeleton.dim
        return ok

    def crop(self, x,y,w,h):
        for i in range(Skeleton.dim):
            self.ske[i].x = (self.ske[i].x - x) / w
            self.ske[i].y = (self.ske[i].y - y) / h

    def boundingBox(self):
        minx, maxx = 1, 0
        miny, maxy = 1, 0
        for i in range(Skeleton.dim):
            minx = min(minx, self.ske[i].x)
            maxx = max(maxx, self.ske[i].x)
            miny = min(miny, self.ske[i].y)
            maxy = max(maxy, self.ske[i].y)
        return minx, miny, maxx, maxy

    def distance(self, ske):
        d = 0.0
        if isinstance(ske, np.ndarray):
            ske_array = np.vstack(ske).astype(float)
            self_array = np.vstack(self.ske).astype(float)
            d = np.linalg.norm(self_array - ske_array)
        else:
            for i in range(Skeleton.dim):
                d += np.sqrt((self.ske[i].x - ske.ske[i].x)**2 + (self.ske[i].y - ske.ske[i].y)**2 + (self.ske[i].z - ske.ske[i].z)**2)
        return d

    def draw(self, image):
        image.flags.writeable = True
        height,width,_ = image.shape
        for i in range(Skeleton.dim):
            x, y = int(self.ske[i].x * width), int(self.ske[i].y * height)
            cv2.circle(image, (x,y), 3, Skeleton.colors_rgb[i].tolist() , -1)
        Skeleton.draw_reduced(self.reduce(), image)

    def CoM(self,w=1,h=1):
        moyenne = self.ske.mean()
        moyenne = np.array( [moyenne[0] * w, moyenne[1] * h ] )
        return moyenne.astype(int)

    @staticmethod
    def neck(ske,w,h):
        ls = np.array( [int(ske[1][0] * w), int(ske[1][1] * h) ], dtype=int )
        rs = np.array( [int(ske[2][0] * w), int(ske[2][1] * h) ], dtype=int )
        return (0.5*(ls+rs)).astype(int)

    @staticmethod
    def pelvis(ske,w,h):
        lh = np.array( [ int(ske[7][0] * w), int(ske[7][1] * h) ], dtype=int )
        rh = np.array( [ int(ske[8][0] * w), int(ske[8][1] * h) ], dtype=int )
        return (0.5*(lh+rh)).astype(int)

    @staticmethod
    def joint(ske,w,h,idx):
        return np.array( [ ske[idx][0] * w, ske[idx][1] * h ], dtype=int )

    @staticmethod
    def color(idx):
        return ( int(Skeleton.colors_rgb[idx][0]),  int(Skeleton.colors_rgb[idx][1]),  int(Skeleton.colors_rgb[idx][2]) )

    @staticmethod
    def draw_reduced(skr, image):
        image.flags.writeable = True
        h,w,_ = image.shape
        # Protection contre des coordonnées hors limites ou NaN
        if np.isnan(skr).any(): return
        
        pelvis = tuple(Skeleton.pelvis(skr,w,h))
        neck   = tuple(Skeleton.neck(skr,w,h))
        
        lines = [
            (pelvis, neck, 0),
            (Skeleton.joint(skr,w,h,3), Skeleton.joint(skr,w,h,1), 1),
            (Skeleton.joint(skr,w,h,5), Skeleton.joint(skr,w,h,3), 2),
            (Skeleton.joint(skr,w,h,2), Skeleton.joint(skr,w,h,4), 3),
            (Skeleton.joint(skr,w,h,4), Skeleton.joint(skr,w,h,6), 4),
            (neck, Skeleton.joint(skr,w,h,1), 5),
            (neck, Skeleton.joint(skr,w,h,2), 6),
            (neck, Skeleton.joint(skr,w,h,0), 7),
            (Skeleton.joint(skr,w,h,7), pelvis, 8),
            (Skeleton.joint(skr,w,h,8), pelvis, 9),
            (Skeleton.joint(skr,w,h,8), Skeleton.joint(skr,w,h,10), 10),
            (Skeleton.joint(skr,w,h,10), Skeleton.joint(skr,w,h,12), 11),
            (Skeleton.joint(skr,w,h,7), Skeleton.joint(skr,w,h,9), 12),
            (Skeleton.joint(skr,w,h,9), Skeleton.joint(skr,w,h,11), 13),
        ]
        
        for p1, p2, col_idx in lines:
            # Conversion explicite en tuple d'entiers pour OpenCV
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.line(image, pt1, pt2, Skeleton.color(col_idx), 2)

class SkeletonSmoother:
    """ Average moving filter for skeleton stability """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def update(self, ske):
        # On stocke uniquement la version réduite (numpy array)
        if isinstance(ske, Skeleton):
            current = ske.toarray(reduced=True)
        else:
            current = ske
            
        self.history.append(current)
        
        # Moyenne temporelle
        arr = np.array(self.history)
        smoothed = np.mean(arr, axis=0)
        return smoothed

if __name__ == '__main__':
    s = Skeleton()
    print("Current Working Directory:", os.getcwd())
    image = cv2.imread("../data/raw/image14000.jpg")
    if image is None:
        print('Lecture de l\'image a échoué.')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    s.fromImage(image)
    #print(s)
    print( "landmarks:", s )
    print( "landmarks as np:", s.__array__() )
    print( "landmarks as np:", s.__array__(reduced=True) )

    s.draw(image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()