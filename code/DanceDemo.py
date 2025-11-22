import numpy as np
import cv2
import os
import sys

from VideoSkeleton import VideoSkeleton
from VideoSkeleton import combineTwoImages
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenNearest import GenNeirest
from GenVanillaNN import *
from GenGAN import *

class DanceDemo:
    """ class that run a demo of the dance. """
    def __init__(self, filename_src, typeOfGen=1, filename_tgt=None):
        # --- MODIF: Gestion flexible de la cible ---
        if filename_tgt is None:
            # Défaut historique, mais risqué si le fichier n'existe pas
            filename_tgt = "data/processed/taichi1.pkl"
            
        print(f"DanceDemo: Source={filename_src} Cible={filename_tgt}")
        
        # On charge le dataset cible (qui doit exister dans data/processed)
        if not os.path.exists(filename_tgt):
             print(f"ERREUR: Fichier cible introuvable: {filename_tgt}")
             # Fallback sur ce qui existe
             import glob
             pkls = glob.glob("data/processed/*.pkl")
             if pkls:
                 filename_tgt = pkls[0]
                 print(f" -> Utilisation de {filename_tgt} à la place.")
        
        self.target = VideoSkeleton(filename_tgt)
        self.source = VideoReader(filename_src)
        
        if self.target.skeCount() == 0:
            raise ValueError("Dataset vide ! Lancez l'extraction (Option 1) sur la vidéo cible.")

        if typeOfGen==1:
            print("Generator: GenNeirest")
            self.generator = GenNeirest(self.target)
        elif typeOfGen==2:
            print("Generator: GenVanillaNN (Vector)")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=1)
        elif typeOfGen==3:
            print("Generator: GenVanillaNN (Image)")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=2)
        elif typeOfGen==4:
            print("Generator: GenGAN")
            self.generator = GenGAN(self.target, loadFromFile=True)
        else:
            print("DanceDemo: typeOfGen error!!!")

    def draw(self, skip_frames=4, wait_ms=1):
        """
        skip_frames: traiter 1 frame sur X (4 = plus rapide, 1 = toutes les frames)
        wait_ms: délai entre frames (1 = rapide, 30 = normal)
        """
        ske = Skeleton()
        # Image d'erreur rouge
        image_err = np.zeros((256, 256, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)
        
        print(f"[DEMO] Vitesse: 1 frame sur {skip_frames}, délai {wait_ms}ms")

        for i in range(self.source.getTotalFrames()):
            image_src = self.source.readFrame()
            if image_src is None: break
            
            # On peut accélérer en sautant des frames
            if i % skip_frames == 0:
                # On adapte les dims cibles à la source pour l'extraction
                self.target.setWidthHeigh(image_src, -1, 1.0)
                
                isSke, image_src_crop, ske = self.target.cropAndSke(image_src, ske)
                if isSke:
                    # Visualisation: squelette sur source
                    ske.draw(image_src_crop)
                    
                    # GENERATION
                    image_tgt = self.generator.generate(ske)
                    
                    # Mise à l'échelle pour affichage
                    image_tgt = cv2.resize(image_tgt, (128, 128))
                    image_src_crop = cv2.resize(image_src_crop, (128, 128))
                    
                    image_combined = combineTwoImages(image_src_crop, image_tgt)
                    
                else:
                    image_src_resized = cv2.resize(image_src, (128, 128))
                    image_err_resized = cv2.resize(image_err, (128, 128))
                    image_combined = combineTwoImages(image_src_resized, image_err_resized)
                
                image_combined = cv2.resize(image_combined, (512, 256))
                cv2.imshow('Dance Demo', image_combined)
                
                key = cv2.waitKey(wait_ms)
                if key & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Pour test autonome
    ddemo = DanceDemo("data/raw/taichi2.mp4", 1)
    ddemo.draw()