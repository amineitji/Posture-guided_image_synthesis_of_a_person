import numpy as np
import cv2
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from VideoSkeleton import VideoSkeleton
# On réutilise les outils du VanillaNN
from GenVanillaNN import GenNNSkeImToImage, VideoSkeletonDataset, SkeToImageTransform

class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super().__init__()
        self.ngpu = ngpu
        
        # PatchGAN-like discriminator (sans Sigmoid final pour WGAN)
        self.model = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1),      # -> 64 x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),    # -> 128 x 16 x 16
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),   # -> 256 x 8 x 8
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 1, 1),   # -> 512 x 7 x 7
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Sortie brute (Logits) pour WGAN
            nn.Conv2d(512, 1, 4, 1, 0),     # -> 1 x 4 x 4 (Valid padding)
        )
        print("[Discriminator] WGAN-GP Critic Architecture loaded")

    def forward(self, input):
        return self.model(input)

class GenGAN():
    def __init__(self, videoSke, loadFromFile=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # --- STYLE VANILLA : Infos GPU ---
        print(f"[GPU] Device: {self.device}")
        if torch.cuda.is_available():
            print(f"[GPU] Name: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        self.netG = GenNNSkeImToImage().to(self.device)
        self.netD = Discriminator().to(self.device)
        
        self.filenameG = 'models/DanceGenGAN_G.pth'
        self.filenameD = 'models/DanceGenGAN_D.pth'
        
        image_size = 64
        
        # Transformations
        src_transform = transforms.Compose([
            SkeToImageTransform(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        tgt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, 
                                          source_transform=src_transform, 
                                          target_transform=tgt_transform)
        
        if loadFromFile and os.path.isfile(self.filenameG):
            print(f"GenGAN: Load= {self.filenameG}")
            try:
                self.netG.load_state_dict(torch.load(self.filenameG, map_location=self.device))
                if os.path.isfile(self.filenameD):
                    self.netD.load_state_dict(torch.load(self.filenameD, map_location=self.device))
                print("[✓] Modèles WGAN chargés avec succès")
            except Exception as e:
                print(f"[✗] Erreur chargement: {e}")
        
        print(f"[GenGAN] Mode: WGAN-GP (Squelette -> Image Réaliste)")
        print(f"[GenGAN] Data Augmentation activée!")

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.netD(interpolates)
        
        fake = torch.ones(d_interpolates.shape, device=self.device, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, n_epochs=20):
        # Paramètres WGAN
        lr = 0.0001
        b1 = 0.5
        b2 = 0.999
        lambda_gp = 10
        n_critic = 3
        
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(b1, b2))
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(b1, b2))
        
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True, num_workers=2)
        
        # --- STYLE VANILLA : Header ---
        print(f"[TRAIN] Début de l'entraînement sur {self.device}")
        print(f"[TRAIN] Dataset: {len(self.dataset)} paires (Squelette/Image)")
        print(f"[TRAIN] Batch size: 32")
        print(f"[TRAIN] Nombre de batches: {len(dataloader)}")
        print(f"[TRAIN] Epochs demandés: {n_epochs}")
        print("[INFO] Appuyez sur Ctrl+C pour arrêter proprement.")
        print("-" * 60)

        try:
            for epoch in range(n_epochs):
                self.netG.train()
                self.netD.train()
                
                # Pour calculer la moyenne de l'epoch
                running_d_loss = 0.0
                running_g_loss = 0.0
                batch_count = 0
                
                for i, (ske_imgs, real_imgs) in enumerate(dataloader):
                    ske_imgs = ske_imgs.to(self.device)
                    real_imgs = real_imgs.to(self.device)
                    
                    # ---------------------
                    #  1. Train Discriminator
                    # ---------------------
                    optimizerD.zero_grad()
                    fake_imgs = self.netG(ske_imgs)
                    real_validity = self.netD(real_imgs)
                    fake_validity = self.netD(fake_imgs.detach())
                    gradient_penalty = self.compute_gradient_penalty(real_imgs.data, fake_imgs.data)
                    
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                    d_loss.backward()
                    optimizerD.step()
                    
                    running_d_loss += d_loss.item()

                    # -----------------
                    #  2. Train Generator
                    # -----------------
                    if i % n_critic == 0:
                        optimizerG.zero_grad()
                        gen_imgs = self.netG(ske_imgs)
                        fake_validity = self.netD(gen_imgs)
                        
                        g_loss_adv = -torch.mean(fake_validity)
                        g_loss_l1 = torch.nn.functional.l1_loss(gen_imgs, real_imgs) * 100
                        g_loss = g_loss_adv + g_loss_l1
                        
                        g_loss.backward()
                        optimizerG.step()
                        
                        running_g_loss += g_loss.item()
                    else:
                        # Si on ne met pas à jour G, on garde la loss précédente pour la moyenne
                        # (Approximation pour l'affichage)
                        pass

                    batch_count += 1

                    # --- STYLE VANILLA : Log par batch ---
                    if (i + 1) % 10 == 0:
                        # On affiche la dernière G Loss connue si disponible
                        g_val = g_loss.item() if 'g_loss' in locals() else 0.0
                        print(f"  [Epoch {epoch+1}/{n_epochs}] Batch {i+1}/{len(dataloader)} - D_Loss: {d_loss.item():.4f} | G_Loss: {g_val:.4f}")

                # Fin d'epoch : Moyennes
                avg_d_loss = running_d_loss / batch_count
                avg_g_loss = running_g_loss / (batch_count / n_critic) # Approximation
                
                # --- Sauvegarde et Feedback ---
                
                # 1. Sauvegarde périodique (Standard)
                if (epoch+1) % 5 == 0:
                    torch.save(self.netG.state_dict(), self.filenameG)
                    torch.save(self.netD.state_dict(), self.filenameD)
                    print(f"  [✓] Checkpoint sauvegardé (Epoch {epoch+1})")

                # 2. Sauvegarde historique (Archive)
                if (epoch+1) % 50 == 0:
                    name_G_hist = f"models/DanceGenGAN_G_epoch_{epoch+1}.pth"
                    name_D_hist = f"models/DanceGenGAN_D_epoch_{epoch+1}.pth"
                    torch.save(self.netG.state_dict(), name_G_hist)
                    torch.save(self.netD.state_dict(), name_D_hist)
                    print(f"  [+] Archive créée : {name_G_hist}")

                # --- STYLE VANILLA : Résumé Epoch ---
                print(f"[EPOCH {epoch+1}/{n_epochs} TERMINÉ] Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f}")
                print("-" * 60)
        
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("[!] INTERRUPTION (Ctrl+C)")
            print("[!] Sauvegarde d'urgence des modèles en cours...")
            torch.save(self.netG.state_dict(), self.filenameG)
            torch.save(self.netD.state_dict(), self.filenameD)
            print(f"[✓] Modèles sauvegardés dans {self.filenameG}")
            print("="*60)
            sys.exit(0)

        print(f"[TRAIN] Entraînement terminé avec succès!")
        print(f"[TRAIN] Derniers modèles sauvegardés dans {self.filenameG}")

    def generate(self, ske):
        self.netG.eval()
        transform = transforms.Compose([
            SkeToImageTransform(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        with torch.no_grad():
            ske_input = transform(ske).unsqueeze(0).to(self.device)
            output = self.netG(ske_input)
            res = self.dataset.tensor2image(output[0])
            return res