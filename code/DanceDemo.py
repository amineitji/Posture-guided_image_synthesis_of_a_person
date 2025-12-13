import numpy as np
import cv2
import os
import time

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton, SkeletonSmoother
from GenNearest import GenNeirest
from GenVanillaNN import GenVanillaNN
from GenGAN import GenGAN

class DanceDemo:
    def __init__(self, filename_src, typeOfGen=1, filename_tgt=None):
        if filename_tgt is None: filename_tgt = "data/processed/taichi1.pkl"
        
        print(f"[DanceDemo] Chargement...")
        self.target = VideoSkeleton(filename_tgt,cropRatio=1.3,modFrame=3)
        self.source = VideoReader(filename_src)
        self.smoother = SkeletonSmoother(window_size=5)

        if typeOfGen == 1:
            self.generator = GenNeirest(self.target)
            self.model_name = "Nearest"
        elif typeOfGen == 2:
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=1)
            self.model_name = "VanillaVec"
        elif typeOfGen == 3:
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=2)
            self.model_name = "VanillaUnet"
        elif typeOfGen == 4:
            self.generator = GenGAN(self.target, loadFromFile=True)
            self.model_name = "WGAN-GP"
        else:
            raise ValueError("Type inconnu")

    def draw(self):
        ske = Skeleton()
        display_size = 256
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # --- REGLAGE VITESSE ---
        # 1 = Toutes les frames (Lent)
        # 3 = Vitesse normale (Saute des frames pour garder le rythme)
        speed_factor = 3  
        
        print(f"[DEMO] Vitesse x{speed_factor}. Appuyez sur 'q' pour quitter.")
        
        fps_time = time.time()
        frames_processed = 0
        fps_display = 0

        while True:
            # 1. AVANCE RAPIDE (Frame Skipping)
            # On "consomme" les images précédentes sans les décoder pour aller vite
            for _ in range(speed_factor - 1):
                self.source.cap.grab() 

            # 2. Lecture de la frame actuelle
            image_src = self.source.readFrame()
            if image_src is None: break
            
            # 3. Traitement
            self.target.setWidthHeigh(image_src, -1, 1.0)
            isSke, image_src_crop, ske = self.target.cropAndSke(image_src, ske)
            
            if isSke:
                # Lissage
                smoothed_coords = self.smoother.update(ske)
                
                # Mise à jour squelette pour génération
                for i, idx_full in enumerate(Skeleton.reduce_indice):
                    ske.ske[idx_full].x = smoothed_coords[i, 0]
                    ske.ske[idx_full].y = smoothed_coords[i, 1]

                # Génération IA
                image_gen = self.generator.generate(ske)
                
                # --- INTERFACE GRAPHIQUE ---
                # Redimensionnement
                p1 = cv2.resize(image_src_crop, (display_size, display_size))
                p3 = cv2.resize(image_gen, (display_size, display_size))
                
                # Création image squelette (noir)
                p2 = np.zeros((display_size, display_size, 3), dtype=np.uint8)
                Skeleton.draw_reduced(smoothed_coords, p2)
                
                # Assemblage
                combined = np.hstack((p1, p2, p3))
                
                # Bandeau Header
                header = np.zeros((35, combined.shape[1], 3), dtype=np.uint8)
                final_display = np.vstack((header, combined))
                
                # Textes
                col = (255, 255, 255)
                cv2.putText(final_display, f"FPS: {fps_display}", (10, 22), font, 0.5, (0, 255, 0), 1)
                cv2.putText(final_display, "SOURCE VIDEO", (80, 22), font, 0.5, col, 1)
                cv2.putText(final_display, "SQUELETTE", (80+display_size, 22), font, 0.5, col, 1)
                cv2.putText(final_display, f"GENERATION ({self.model_name})", (60+display_size*2, 22), font, 0.5, col, 1)

                cv2.imshow('Deep Dance Transfer', final_display)
                
                # Calcul FPS Réel
                frames_processed += 1
                if time.time() - fps_time >= 1.0:
                    fps_display = frames_processed
                    frames_processed = 0
                    fps_time = time.time()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                pass # Si pas de squelette, on continue sans afficher (évite clignotement)

        cv2.destroyAllWindows()