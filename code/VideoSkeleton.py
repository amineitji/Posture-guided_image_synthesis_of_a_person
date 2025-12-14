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

    # Calcul des manques (padding nécessaire)
    pad_top = max(0, -y_start)
    pad_left = max(0, -x_start)
    pad_bottom = max(0, (y_start + height) - img_height)
    pad_right = max(0, (x_start + width) - img_width)

    # Si on est totalement hors champ (sécurité)
    if pad_top >= height or pad_left >= width:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # 1. On applique le padding (Remplissage intelligent 'edge' au lieu de noir)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        # mode='edge' répète le bord. mode='reflect' fait un miroir.
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')

    # 2. Maintenant que l'image est agrandie, on découpe la zone voulue
    # Les coordonnées changent car on a ajouté des pixels en haut et à gauche
    new_y = y_start + pad_top
    new_x = x_start + pad_left

    return image[new_y:new_y + height, new_x:new_x + width]

class VideoSkeleton:
    """ 
    Class that associate a skeleton to each frame of a video
    """
    def __init__(self, filename, forceCompute=False, modFrame=3, newVideoWidth=256, cropRatio=1.3, isCrop=True):
        self.mod_frame = modFrame
        self.cropRatio = cropRatio  # <--- AJOUTER CETTE LIGNE
        self.new_video_width = newVideoWidth  # <--- AJOUTER CETTE LIGNE
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
        # 1. On garde l'image ORIGINALE (Haute qualité) pour les calculs
        # On ne fait PAS de cv2.resize(image) ici !
        img_height, img_width = image.shape[:2]

        # Note : Pour que ske.fromImage marche, il faut s'assurer que le squelette
        # a été détecté sur cette taille d'image, ou mis à l'échelle.
        # (Supposons ici que ske correspond à l'image fournie en entrée)

        if ske.fromImage(image):
            if isCrop:
                # On calcule la taille du crop basée sur un ratio de la hauteur originale
                # Ex: Si la vidéo fait 1080p, crop_size sera ~1000px
                dim_ref = min(img_width, img_height)
                crop_size = int(dim_ref * self.cropRatio)  # Il faut que cropRatio soit dispo ici

                xm, ym, xM, yM = ske.boundingBox()
                center_x = (xm + xM) * img_width / 2
                center_y = (ym + yM) * img_height / 2

                tl_x = int(center_x - crop_size / 2)
                tl_y = int(center_y - crop_size / 2)

                # Clamping (éviter de sortir de l'image)
                tl_x = max(0, min(tl_x, img_width - crop_size))
                tl_y = max(0, min(tl_y, img_height - crop_size))

                # Découpe Haute Résolution
                image_crop = image[tl_y: tl_y + crop_size, tl_x: tl_x + crop_size]

                # --- C'EST ICI QU'ON UTILISE newVideoWidth ---
                # On redimensionne le CROP final vers la taille voulue (ex: 128x128)
                # On force un carré.
                image_final = cv2.resize(image_crop, (self.new_video_width, self.new_video_width))

                # Mise à jour du squelette (Normalisation classique)
                real_h, real_w = image_crop.shape[:2]
                norm_x = tl_x / img_width
                norm_y = tl_y / img_height
                norm_w = real_w / img_width
                norm_h = real_h / img_height
                ske.crop(norm_x, norm_y, norm_w, norm_h)

                return True, image_final, ske

            # Si pas de crop, on resize tout brutalement en carré
            image_final = cv2.resize(image, (self.new_video_width, self.new_video_width))
            return True, image_final, ske

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
    # filename = "../data/raw/taichi1.mp4"
    # s = VideoSkeleton(filename, forceCompute=True, modFrame=1000, newVideoWidth=256, cropRatio=2.0)
    # image_noire = np.zeros((s.ske_height_crop, s.ske_width_crop, 3), dtype=np.uint8)
    # s.ske[2].draw(image_noire)

    # 1. Configuration
    filename = "../data/raw/taichi1.mp4"

    print("-" * 30)
    print("DEMARRAGE DU TEST VISUEL")

    # 2. Chargement (forceCompute=False pour utiliser ce qui est déjà calculé)
    # Assure-toi que cropRatio est bien à 2.0 ici
    s = VideoSkeleton(filename, forceCompute=False, newVideoWidth=256, cropRatio=1.3)

    # 3. Vérification du nombre de squelettes
    count = s.skeCount()
    print(f"Nombre de squelettes trouvés : {count}")

    if count > 0:
        # On prend le premier squelette (index 0) pour être sûr qu'il existe
        idx = 0

        # Création image noire (H, W, 3)
        print(f"Dimensions de l'image (H, W) : {s.ske_height_crop} x {s.ske_width_crop}")
        image_noire = np.zeros((s.ske_height_crop, s.ske_width_crop, 3), dtype=np.uint8)

        # Dessin
        print(f"Dessin du squelette n°{idx}...")
        try:
            s.ske[idx].draw(image_noire)

            # 4. AFFICHAGE (Le point critique)
            cv2.imshow('Verification Squelette', image_noire)

            print(">>> FENETRE OUVERTE. Appuie sur une touche du clavier pour quitter. <<<")
            # waitKey(0) BLOQUE le programme indéfiniment jusqu'à une touche
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Erreur pendant le dessin : {e}")
            print("Vérifie que tu as bien corrigé la classe VideoSkeleton (numpy object) !")
    else:
        print("ERREUR : Aucun squelette dans la liste.")
        print("1. Vérifie le chemin de la vidéo.")
        print("2. Essaie de mettre forceCompute=True pour recalculer.")