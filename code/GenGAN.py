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
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        print("[Discriminator] Architecture AMELIOREE chargée")

    def forward(self, input):
        return self.model(input).view(-1, 1).squeeze(1)

class GenGAN():
    def __init__(self, videoSke, loadFromFile=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[GPU] Device: {self.device}")
        if torch.cuda.is_available():
            print(f"[GPU] Name: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.netG = GenNNSke26ToImage().to(self.device)
        self.netD = Discriminator().to(self.device)
        
        self.filenameG = 'models/DanceGenGAN_G.pth'
        self.filenameD = 'models/DanceGenGAN_D.pth'
        
        tgt_transform = transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.CenterCrop(64),
                            transforms.RandomHorizontalFlip(p=0.3),
                            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                            transforms.RandomRotation(5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        
        if loadFromFile and os.path.isfile(self.filenameG):
            print("GenGAN: Load=", self.filenameG)
            try:
                state_dict_g = torch.load(self.filenameG, map_location=self.device)
                
                if any(key.startswith('_orig_mod.') for key in state_dict_g.keys()):
                    new_state_dict = {}
                    for key, value in state_dict_g.items():
                        new_key = key.replace('_orig_mod.', '')
                        new_state_dict[new_key] = value
                    state_dict_g = new_state_dict
                
                self.netG.load_state_dict(state_dict_g)
                print("[✓] Générateur chargé avec succès")
            except Exception as e:
                print(f"[✗] Erreur chargement générateur: {e}")
        
        print(f"[GenGAN] Data Augmentation activée!")

    def train(self, n_epochs=20):
        lr_g = 0.0001
        lr_d = 0.0004
        beta1 = 0.5
        
        criterion_bce = nn.BCELoss()
        criterion_l1 = nn.L1Loss()
        lambda_l1 = 10.0
        
        train_size = int(0.85 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        batch_size = min(32, max(8, train_size // 20))
        
        num_workers = 4 if torch.cuda.is_available() else 2
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr_g, betas=(beta1, 0.999))
        
        schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
        schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
        
        use_amp = torch.cuda.is_available()
        scaler_G = torch.cuda.amp.GradScaler() if use_amp else None
        scaler_D = torch.cuda.amp.GradScaler() if use_amp else None
        
        print(f"[GAN TRAIN] Début de l'entraînement GAN AMELIORE sur {self.device}")
        print(f"[GAN TRAIN] Train: {train_size} | Val: {val_size}")
        print(f"[GAN TRAIN] Batch size: {batch_size}")
        print(f"[GAN TRAIN] Workers: {num_workers} threads")
        print(f"[GAN TRAIN] Nombre de batches: {len(train_loader)}")
        print(f"[GAN TRAIN] Epochs demandés: {n_epochs}")
        print(f"[GAN TRAIN] Lambda L1: {lambda_l1}")
        if use_amp:
            print("[OPTIM] Mixed Precision (AMP) activée")
        print("-" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            epoch_lossD = 0.0
            epoch_lossG = 0.0
            epoch_lossG_l1 = 0.0
            batch_count = 0
            
            self.netG.train()
            self.netD.train()
            
            for i, (ske_batch, real_img_batch) in enumerate(train_loader):
                batch_size_actual = ske_batch.size(0)
                
                ske_batch = ske_batch.to(self.device, non_blocking=True)
                real_img_batch = real_img_batch.to(self.device, non_blocking=True)
                
                label_real = torch.ones(batch_size_actual, device=self.device) * 0.9
                label_fake = torch.zeros(batch_size_actual, device=self.device) * 0.1
                
                self.netD.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output_real = self.netD(real_img_batch)
                        errD_real = criterion_bce(output_real, label_real)
                    
                    scaler_D.scale(errD_real).backward()
                    
                    with torch.cuda.amp.autocast():
                        fake_img = self.netG(ske_batch)
                        output_fake = self.netD(fake_img.detach())
                        errD_fake = criterion_bce(output_fake, label_fake)
                    
                    scaler_D.scale(errD_fake).backward()
                    scaler_D.step(optimizerD)
                    scaler_D.update()
                else:
                    output_real = self.netD(real_img_batch)
                    errD_real = criterion_bce(output_real, label_real)
                    errD_real.backward()
                    
                    fake_img = self.netG(ske_batch)
                    output_fake = self.netD(fake_img.detach())
                    errD_fake = criterion_bce(output_fake, label_fake)
                    errD_fake.backward()
                    optimizerD.step()
                
                errD = errD_real + errD_fake
                
                self.netG.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output_fake_for_G = self.netD(fake_img)
                        errG_adv = criterion_bce(output_fake_for_G, torch.ones(batch_size_actual, device=self.device))
                        errG_l1 = criterion_l1(fake_img, real_img_batch)
                        errG = errG_adv + lambda_l1 * errG_l1
                    
                    scaler_G.scale(errG).backward()
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
                    scaler_G.step(optimizerG)
                    scaler_G.update()
                else:
                    output_fake_for_G = self.netD(fake_img)
                    errG_adv = criterion_bce(output_fake_for_G, torch.ones(batch_size_actual, device=self.device))
                    errG_l1 = criterion_l1(fake_img, real_img_batch)
                    errG = errG_adv + lambda_l1 * errG_l1
                    
                    errG.backward()
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
                    optimizerG.step()
                
                epoch_lossD += errD.item()
                epoch_lossG += errG_adv.item()
                epoch_lossG_l1 += errG_l1.item()
                batch_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  [Epoch {epoch+1}/{n_epochs}] Batch {i+1}/{len(train_loader)}")
                    print(f"    Loss_D: {errD.item():.4f} | Loss_G_adv: {errG_adv.item():.4f} | Loss_G_L1: {errG_l1.item():.4f}")
            
            avg_lossD = epoch_lossD / batch_count
            avg_lossG = epoch_lossG / batch_count
            avg_lossG_l1 = epoch_lossG_l1 / batch_count
            
            self.netG.eval()
            val_loss = 0.0
            val_count = 0
            
            with torch.no_grad():
                for ske_batch, real_img_batch in val_loader:
                    ske_batch = ske_batch.to(self.device, non_blocking=True)
                    real_img_batch = real_img_batch.to(self.device, non_blocking=True)
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            fake_img = self.netG(ske_batch)
                            loss = criterion_l1(fake_img, real_img_batch)
                    else:
                        fake_img = self.netG(ske_batch)
                        loss = criterion_l1(fake_img, real_img_batch)
                    
                    val_loss += loss.item()
                    val_count += 1
            
            avg_val_loss = val_loss / val_count if val_count > 0 else 0
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.netG.state_dict(), self.filenameG)
                torch.save(self.netD.state_dict(), self.filenameD)
                print(f"  [✓] Meilleur modèle sauvegardé (val_L1: {avg_val_loss:.4f})")
            
            schedulerD.step()
            schedulerG.step()
            
            print(f"[EPOCH {epoch+1}/{n_epochs} TERMINÉ]")
            print(f"  Train - D: {avg_lossD:.4f} | G_adv: {avg_lossG:.4f} | G_L1: {avg_lossG_l1:.4f}")
            print(f"  Val   - L1: {avg_val_loss:.4f}")
            print(f"  LR    - D: {schedulerD.get_last_lr()[0]:.6f} | G: {schedulerG.get_last_lr()[0]:.6f}")
            print("-" * 60)
            
        torch.save(self.netG.state_dict(), self.filenameG)
        torch.save(self.netD.state_dict(), self.filenameD)
        print("[GAN TRAIN] Modèles GAN sauvegardés.")
        print(f"  - Générateur: {self.filenameG}")
        print(f"  - Discriminateur: {self.filenameD}")
        print(f"[GAN TRAIN] Meilleur Val L1 Loss: {best_val_loss:.4f}")
        print("[GAN TRAIN] Entraînement terminé avec succès!")

    def generate(self, ske):
        self.netG.eval()
        with torch.no_grad():
            ske_t = torch.from_numpy(ske.toarray(reduced=True).flatten()).float()
            ske_t = ske_t.unsqueeze(0).to(self.device)
            
            output = self.netG(ske_t)
            
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

    if True:
        gen = GenGAN(targetVideoSke, False)
        gen.train(50) 
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)