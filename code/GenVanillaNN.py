import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)

class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size
        # Import ici pour éviter les problèmes de référence locale
        from Skeleton import Skeleton
        self.Skeleton = Skeleton

    def __call__(self, ske):
        # Création d'une image noire pour dessiner le squelette
        image = np.zeros((self.imsize, self.imsize, 3), dtype=np.uint8)
        
        # Gestion de deux cas : objet Skeleton ou numpy array
        if isinstance(ske, np.ndarray):
            # Convertir les Vec3 en float explicitement
            ske_array = np.vstack(ske[self.Skeleton.reduce_indice]).astype(float)[:, :2]
        else:
            ske_array = ske.toarray(reduced=True)
        
        # Utilisation du dessin réduit (sticks)
        self.Skeleton.draw_reduced(ske_array, image)
        return Image.fromarray(image)

class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset: 
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        # preprocess skeleton (input)
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        
        # preprocess image (output/target)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image
    
    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            # Si pas de transformation, on convertit en vecteur (Tensor)
            # Gestion de deux cas : ske peut être un objet Skeleton OU déjà un numpy array
            if isinstance(ske, np.ndarray):
                # Cas où ske est déjà un array numpy (chargé depuis pkl)
                # Il peut contenir des objets Vec3, il faut convertir en float
                from Skeleton import Skeleton
                if self.ske_reduced:
                    # Extraire les indices réduits et convertir en float
                    ske_array = np.vstack(ske[Skeleton.reduce_indice]).astype(float)[:, :2]
                else:
                    ske_array = np.vstack(ske).astype(float)
                ske = torch.from_numpy(ske_array.flatten())
            else:
                # Cas où ske est un objet Skeleton
                ske = torch.from_numpy(ske.toarray(reduced=self.ske_reduced).flatten())
            
            ske = ske.to(torch.float32)
            # Reshape pour compatibilité (ex: conv1d ou dense)
            ske = ske.reshape(ske.shape[0], 1, 1)
        return ske

    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().cpu().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # Denormalization: [-1, 1] => [0, 255]
        numpy_image = (numpy_image * 0.5 + 0.5) * 255
        # passage a des images cv2 pour affichage (RGB -> BGR si nécessaire, ici on garde RGB pour matplotlib ou conversion ultérieure)
        # Si affichage OpenCV direct, conversion BGR nécessaire :
        # numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        return numpy_image.astype(np.uint8)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class GenNNSke26ToImage(nn.Module):
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton_dim26)->Image
    """
    def __init__(self):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim # 26
        
        # Partie Fully Connected pour projeter le vecteur 26D vers un volume spatial
        self.fc = nn.Sequential(
            nn.Linear(26, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 256 * 4 * 4), # On vise une feature map de 4x4 avec 256 channels
            nn.BatchNorm1d(256 * 4 * 4),
            nn.LeakyReLU(0.2)
        )
        
        # Partie Convolutive (Décodeur)
        self.conv = nn.Sequential(
            # Input: 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # -> 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # -> 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # -> 3 x 64 x 64
            nn.Tanh() # Sortie entre -1 et 1 pour correspondre à la normalisation
        )
        print(self.fc)
        print(self.conv)

    def forward(self, z):
        # Aplatir l'entrée si nécessaire : (Batch, 26, 1, 1) -> (Batch, 26)
        z = z.view(z.size(0), -1)
        x = self.fc(z)
        # Reshape pour les convolutions
        x = x.view(-1, 256, 4, 4)
        img = self.conv(x)
        return img

class GenNNSkeImToImage(nn.Module):
    """ class that Generate a new image from from THE IMAGE OF the new skeleton posture
       SkeletonImage is an image with the skeleton drawed on it
       Fonc generator(SkeletonImage)->Image
    """
    def __init__(self):
        super().__init__()
        # Architecture type U-Net simplifié ou AutoEncoder
        
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1) # 64x32x32
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1) # 128x16x16
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1) # 256x8x8
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # 128x16x16
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, 2, 1) # 64x32x32
        self.dec3 = nn.ConvTranspose2d(64, 3, 4, 2, 1) # 3x64x64
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn128 = nn.BatchNorm2d(128)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn256 = nn.BatchNorm2d(256)

    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.bn128(self.enc2(e1)))
        e3 = self.relu(self.bn256(self.enc3(e2)))
        
        # Decoder avec Skip Connections (addition)
        d1 = self.relu(self.bn128(self.dec1(e3))) + e2
        d2 = self.relu(self.bn64(self.dec2(d1))) + e1
        d3 = self.tanh(self.dec3(d2))
        return d3

class GenVanillaNN():
    """ class that Generate a new image from a new skeleton posture
        Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optSkeOrImage = optSkeOrImage
        
        if optSkeOrImage==1:        # skeleton_dim26 to image
            self.netG = GenNNSke26ToImage().to(self.device)
            src_transform = None
            self.filename = 'models/DanceGenVanillaFromSke26.pth'
        else:                       # skeleton_image to image
            self.netG = GenNNSkeImToImage().to(self.device)
            src_transform = transforms.Compose([ 
                                                 SkeToImageTransform(image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
            self.filename = 'models/DanceGenVanillaFromSkeim.pth'

        tgt_transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            self.netG.load_state_dict(torch.load(self.filename, map_location=self.device))

    def train(self, n_epochs=20):
        print(f"[TRAIN] Début de l'entraînement sur {self.device}")
        print(f"[TRAIN] Dataset: {len(self.dataset)} frames")
        print(f"[TRAIN] Batch size: {self.dataloader.batch_size}")
        print(f"[TRAIN] Nombre de batches: {len(self.dataloader)}")
        print(f"[TRAIN] Epochs demandés: {n_epochs}")
        print("-" * 60)
        
        optimizer = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.MSELoss() # On utilise MSE pour la régression d'image pixel par pixel

        for epoch in range(n_epochs):
            self.netG.train()
            running_loss = 0.0
            batch_count = 0
            
            for i, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.netG(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
                
                # Affichage toutes les 10 batches
                if (i + 1) % 10 == 0:
                    print(f"  [Epoch {epoch+1}/{n_epochs}] Batch {i+1}/{len(self.dataloader)} - Loss: {loss.item():.5f}")
            
            avg_loss = running_loss / batch_count
            print(f"[EPOCH {epoch+1}/{n_epochs} TERMINÉ] Loss moyenne: {avg_loss:.5f}")
            print("-" * 60)
        
        # Sauvegarde du modèle
        torch.save(self.netG.state_dict(), self.filename)
        print(f"[TRAIN] Modèle sauvegardé dans {self.filename}")
        print(f"[TRAIN] Entraînement terminé avec succès!")

    def generate(self, ske):
        """ generator of image from skeleton """
        self.netG.eval()
        with torch.no_grad():
            # Préparation de l'input selon le mode (vecteur ou image)
            if self.optSkeOrImage == 1:
                ske_input = torch.from_numpy(ske.toarray(reduced=True).flatten()).float()
                ske_input = ske_input.unsqueeze(0).to(self.device) # Batch dimension
            else:
                transform = transforms.Compose([
                    SkeToImageTransform(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                ske_input = transform(ske).unsqueeze(0).to(self.device)
            
            output = self.netG(ske_input)
            
            # Conversion en image affichable
            res = self.dataset.tensor2image(output[0])
            return res

if __name__ == '__main__':
    force = False
    optSkeOrImage = 2           
    n_epoch = 200
    train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/raw/taichi1.mp4" 

    print("GenVanillaNN: Filename=", filename)
    targetVideoSke = VideoSkeleton(filename)

    if train:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=optSkeOrImage)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True, optSkeOrImage=optSkeOrImage)    

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i] )
        # Resize pour mieux voir
        image = cv2.resize(image, (256, 256))
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)