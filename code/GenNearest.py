import numpy as np
import cv2
import os
import sys

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        """ generator of image from skeleton """
        # Recherche du squelette le plus proche dans la base de données cible
        best_dist = float('inf')
        best_idx = -1
        
        # Convertir le squelette en array si ce n'est pas déjà fait
        if isinstance(ske, np.ndarray):
            ske_array = np.vstack(ske).astype(float)
        else:
            ske_array = np.vstack(ske.ske).astype(float)
        
        # On parcourt tous les squelettes de la vidéo cible
        for i in range(self.videoSkeletonTarget.skeCount()):
            target_ske = self.videoSkeletonTarget.ske[i]
            
            # Convertir le squelette cible en array
            if isinstance(target_ske, np.ndarray):
                target_array = np.vstack(target_ske).astype(float)
            else:
                target_array = np.vstack(target_ske.ske).astype(float)
            
            # Calculer la distance
            dist = np.linalg.norm(ske_array - target_array)
            
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx != -1:
            # On retourne l'image correspondante
            return self.videoSkeletonTarget.readImage(best_idx)
        else:
            # Image vide par défaut si échec
            return np.ones((64, 64, 3), dtype=np.uint8) * 255