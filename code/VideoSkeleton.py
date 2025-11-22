import numpy as np
import cv2
import os
import pickle
import sys
import math
import gc

from VideoReader import VideoReader
from Skeleton import Skeleton

def filename_change_ext(filename_full, nouvelle_extension):
    # --- MODIF: Force la sauvegarde dans data/processed ---
    base = os.path.basename(filename_full)
    name_no_ext = os.path.splitext(base)[0]
    
    # On suppose qu'on est à la racine du projet
    processed_dir = "data/processed"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    # Le fichier pkl sera dans data/processed/nom.pkl
    nouveau_nom_fichier = os.path.join(processed_dir, name_no_ext + nouvelle_extension)
    
    # Le dossier des images sera data/processed/
    path = processed_dir
    
    return nouveau_nom_fichier, path, name_no_ext

def combineTwoImages(image1, image2):
    height = max(image1.shape[0], image2.shape[0])
    combined_width = image1.shape[1] + image2.shape[1]
    combined_image = np.zeros((height, combined_width, 3), dtype=np.uint8)
    combined_image[:image1.shape[0], :image1.shape[1]] = image1
    # Resize si hauteurs différentes (protection)
    if image2.shape[0] != height:
        w = int(image2.shape[1] * (height / image2.shape[0]))
        image2 = cv2.resize(image2, (w, height))
        combined_width = image1.shape[1] + w
        combined_image = np.zeros((height, combined_width, 3), dtype=np.uint8)
        combined_image[:image1.shape[0], :image1.shape[1]] = image1
        combined_image[:height, image1.shape[1]:] = image2
    else:
        combined_image[:image2.shape[0], image1.shape[1]:] = image2
    return combined_image

def crop_with_padding(image, x_start, y_start, width, height):
    img_height, img_width = image.shape[:2]
    x_end = x_start + width
    y_end = y_start + height
    pad_top = max(0, -y_start)
    pad_bottom = max(0, y_end - img_height)
    pad_left = max(0, -x_start)
    pad_right = max(0, x_end - img_width)
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    cropped_image = padded_image[y_start:y_start+height, x_start:x_start+width]
    return cropped_image

class VideoSkeleton:
    """ 
    Class that associate a skeleton to each frame of a video
    """
    def __init__(self, filename, forceCompute=False, modFrame=10, newVideoWidth=256, cropRatio=1.0, isCrop=True):
        self.mod_frame = modFrame
        
        # --- MODIF: Recherche automatique dans data/raw ---
        if not os.path.exists(filename):
            raw_path = os.path.join("data/raw", os.path.basename(filename))
            if os.path.exists(raw_path):
                filename = raw_path
            else:
                print(f"Erreur: Vidéo {filename} introuvable.")
                return

        filename_pkl, filename_dir, filename_base = filename_change_ext(filename, ".pkl")
        
        # Dossier spécifique pour les images : data/processed/nom_video
        self.images_dir = os.path.join(filename_dir, filename_base)
        print(f"directory output: {self.images_dir}")
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        self.path = self.images_dir # Important pour readImage
        
        if os.path.exists(filename_pkl) and not forceCompute:
            # Vérifions si le fichier n'est pas vide/corrompu
            try:
                print(f"===== read precompute: {filename_pkl}")
                vs = VideoSkeleton.load(filename_pkl)
                if vs.ske is not None and len(vs.ske) > 0:
                    self.ske = vs.ske
                    self.im = vs.im
                    self.path = self.images_dir # Force le chemin correct
                    image = self.readImage(0)
                    if image is not None:
                        self.setWidthHeigh(image, newVideoWidth, cropRatio)
                        return
                else:
                    print("Cache vide. Recalcul...")
            except Exception as e:
                print(f"Erreur cache: {e}. Recalcul...")
    
        print(f"===== compute: {filename}")
        video = VideoReader(filename)
        total_frames = video.getTotalFrames()
        print(f"[EXTRACTION] Vidéo: {filename}")
        print(f"[EXTRACTION] Total frames: {total_frames}")
        print(f"[EXTRACTION] Frames à traiter (1 sur {self.mod_frame}): {total_frames // self.mod_frame}")
        print("-" * 60)
        
        self.ske = [] 
        self.im = []
        widthIsSet = False
        frames_processed = 0
        frames_kept = 0
        
        for i in range(total_frames):
            image = video.readFrame()
            if image is None: continue

            if widthIsSet == False:
                self.setWidthHeigh(image, newVideoWidth, cropRatio)
                widthIsSet = True
                print(f"[EXTRACTION] Résolution vidéo: {image.shape}")
                print(f"[EXTRACTION] Nouvelle taille: {self.new_video_width}x{self.new_video_height}")
                print(f"[EXTRACTION] Taille crop: {self.ske_width_crop}x{self.ske_height_crop}")
                print("-" * 60)

            if (i%self.mod_frame == 0):
                frames_processed += 1
                ske = Skeleton()
                isSke, image_res, ske = self.cropAndSke(image, ske, isCrop)
                if isSke:
                    filename_im = f"image{i}.jpg"
                    filename_imsave = os.path.join(self.images_dir, filename_im)
                    cv2.imwrite(filename_imsave, image_res)
                    self.ske.append(ske)
                    self.im.append(filename_im)
                    frames_kept += 1
                    
                    if frames_kept % 50 == 0:
                        print(f"[EXTRACTION] Frame {i}/{total_frames} - {frames_kept} squelettes extraits")

        video.release()
        self.ske = np.array(self.ske, dtype=Skeleton)
        self.im = np.array(self.im)
        print("-" * 60)
        print(f"[EXTRACTION] Extraction terminée!")
        print(f"[EXTRACTION] Frames traitées: {frames_processed}")
        print(f"[EXTRACTION] Squelettes gardés: {frames_kept}")
        print(f"[EXTRACTION] Taux de réussite: {100*frames_kept/frames_processed:.1f}%")
        self.save(filename_pkl)

    def setWidthHeigh(self, image, newVideoWidth, cropRatio):
        if newVideoWidth == -1:
            self.new_video_width = image.shape[1]
            self.new_video_height = image.shape[0]
        else:
            self.new_video_width = newVideoWidth
            self.new_video_height = int(image.shape[0] * self.new_video_width / image.shape[1])
        self.ske_width_crop = int(cropRatio*self.new_video_width)
        self.ske_height_crop = int(cropRatio*self.new_video_height)    

    def cropAndSke(self, image, ske, isCrop=True):
        image = cv2.resize(image, (self.new_video_width, self.new_video_height))
        if ske.fromImage(image):
            if isCrop:
                xm, ym, xM, yM = ske.boundingBox()
                center_x = self.new_video_width * (xm + xM) / 2
                center_y = self.new_video_height * (ym + yM) / 2
                xm = int(center_x-self.ske_width_crop/2)
                xM = int(center_x+self.ske_width_crop/2)
                ym = int(center_y-self.ske_height_crop/2)
                yM = int(center_y+self.ske_height_crop/2)
                image = crop_with_padding(image, xm, ym, self.ske_width_crop, self.ske_height_crop)
                ske.crop(xm/self.new_video_width, ym/self.new_video_height, 
                         self.ske_width_crop/self.new_video_width, self.ske_height_crop/self.new_video_height)
            return True, image, ske
        else:
            return False, image, ske        

    def save(self, filename):
        with open(filename, "wb") as fichier:
            pickle.dump(self, fichier)
        print(f"save: {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as fichier:
            objet_charge = pickle.load(fichier)
        return objet_charge

    def imagePath(self, idx):
        return os.path.join(self.path, self.im[idx])

    def readImage(self, idx):
        return cv2.imread(self.imagePath(idx))
    
    def skeCount(self):
        return self.ske.shape[0] if self.ske is not None else 0

    def draw(self):
        for i in range(self.skeCount()):
            empty = np.zeros((self.ske_height_crop, self.ske_width_crop, 3), dtype=np.uint8)
            im = self.readImage(i)
            if im is None: continue
            self.ske[i].draw(empty)
            resim = combineTwoImages(im, empty)
            cv2.imshow('Image', resim)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Test rapide
    filename = "data/raw/taichi1.mp4"
    s = VideoSkeleton(filename, forceCompute=True, modFrame=50)
    s.draw()