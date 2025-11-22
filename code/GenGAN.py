import numpy as np
import cv2
import os
import pickle
import sys
import math

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
from GenVanillaNN import * 

class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super().__init__()
        self.ngpu = ngpu
        # Architecture classique DCGAN Discriminator
        # Input: 3 x 64 x 64
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), # -> 64 x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1), # -> 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1), # -> 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1), # -> 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: 1 (Probabilité Real/Fake)
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input).view(-1, 1).squeeze(1)

class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # On utilise le générateur Vanilla (Vecteur -> Image) comme base
        self.netG = GenNNSke26ToImage().to(self.device)
        self.netD = Discriminator().to(self.device)
        
        self.filenameG = 'models/DanceGenGAN_G.pth'
        self.filenameD = 'models/DanceGenGAN_D.pth'
        
        tgt_transform = transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        
        # GAN utilise ici le squelette réduit en vecteur (mode 26 dim)
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        
        if loadFromFile and os.path.isfile(self.filenameG):
            print("GenGAN: Load=", self.filenameG)
            self.netG.load_state_dict(torch.load(self.filenameG, map_location=self.device))
            # On pourrait charger D aussi si on voulait continuer l'entrainement

    def train(self, n_epochs=20):
        # Hyperparamètres
        lr = 0.0002
        beta1 = 0.5
        
        criterion = nn.BCELoss()
        
        # Optimiseurs séparés
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        
        print(f"[GAN TRAIN] Début de l'entraînement GAN sur {self.device}")
        print(f"[GAN TRAIN] Dataset: {len(self.dataset)} frames")
        print(f"[GAN TRAIN] Batch size: {self.dataloader.batch_size}")
        print(f"[GAN TRAIN] Nombre de batches: {len(self.dataloader)}")
        print(f"[GAN TRAIN] Epochs demandés: {n_epochs}")
        print("-" * 60)
        
        for epoch in range(n_epochs):
            epoch_lossD = 0.0
            epoch_lossG = 0.0
            batch_count = 0
            
            for i, (ske_batch, real_img_batch) in enumerate(self.dataloader):
                batch_size = ske_batch.size(0)
                
                ske_batch = ske_batch.to(self.device)
                real_img_batch = real_img_batch.to(self.device)
                
                # Labels
                label_real = torch.ones(batch_size, device=self.device)
                label_fake = torch.zeros(batch_size, device=self.device)
                
                # ---------------------
                # 1. Entraîner D (Discriminateur)
                # ---------------------
                self.netD.zero_grad()
                
                # a. Entraîner avec des vraies images
                output_real = self.netD(real_img_batch)
                errD_real = criterion(output_real, label_real)
                errD_real.backward()
                
                # b. Entraîner avec des fausses images (générées par G)
                fake_img = self.netG(ske_batch)
                output_fake = self.netD(fake_img.detach()) # .detach() pour ne pas propager le gradient vers G ici
                errD_fake = criterion(output_fake, label_fake)
                errD_fake.backward()
                
                errD = errD_real + errD_fake
                optimizerD.step()
                
                # ---------------------
                # 2. Entraîner G (Générateur)
                # ---------------------
                self.netG.zero_grad()
                # On veut que D se trompe et prédise "Vrai" (1) pour nos fausses images
                output_fake_for_G = self.netD(fake_img)
                errG = criterion(output_fake_for_G, label_real)
                errG.backward()
                optimizerG.step()
                
                epoch_lossD += errD.item()
                epoch_lossG += errG.item()
                batch_count += 1
                
                # Affichage toutes les 10 batches
                if (i + 1) % 10 == 0:
                    print(f"  [Epoch {epoch+1}/{n_epochs}] Batch {i+1}/{len(self.dataloader)} - Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
            
            avg_lossD = epoch_lossD / batch_count
            avg_lossG = epoch_lossG / batch_count
            print(f"[EPOCH {epoch+1}/{n_epochs} TERMINÉ] Loss_D moyenne: {avg_lossD:.4f} | Loss_G moyenne: {avg_lossG:.4f}")
            print("-" * 60)
            
        # Sauvegarde
        torch.save(self.netG.state_dict(), self.filenameG)
        torch.save(self.netD.state_dict(), self.filenameD)
        print("[GAN TRAIN] Modèles GAN sauvegardés.")
        print(f"  - Générateur: {self.filenameG}")
        print(f"  - Discriminateur: {self.filenameD}")
        print("[GAN TRAIN] Entraînement terminé avec succès!")

    def generate(self, ske):           
        """ generator of image from skeleton """
        self.netG.eval()
        with torch.no_grad():
            # Préparation du squelette en vecteur
            ske_t = torch.from_numpy(ske.toarray(reduced=True).flatten()).float()
            ske_t = ske_t.unsqueeze(0).to(self.device)
            
            # Génération
            output = self.netG(ske_t)
            
            # Conversion image
            res = self.dataset.tensor2image(output[0])
            return res

if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/raw/taichi1.mp4"
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(50) 
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)