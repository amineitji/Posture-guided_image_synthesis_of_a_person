import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from VideoSkeleton import VideoSkeleton
from GenVanillaNN import GenNNSkeImToImage, VideoSkeletonDataset, SkeToImageTransform


# ============================================================
#                     DISCRIMINATOR (OPTIMISÉ)
# ============================================================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0),
        )

        print("[Discriminator] Optimized WGAN-GP Critic loaded (GroupNorm + no Sigmoid)")

    def forward(self, x):
        return self.model(x.contiguous())


# ============================================================
#                        WGAN-GP OPTIMISÉ
# ============================================================

class GenGAN():
    def __init__(self, videoSke, loadFromFile=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[GPU] {self.device}")
        if torch.cuda.is_available():
            print(f"[GPU] Name: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

        # Réseaux
        self.netG = GenNNSkeImToImage().to(self.device)
        self.netD = Discriminator().to(self.device)

        self.filenameG = "models/DanceGenGAN_G.pth"
        self.filenameD = "models/DanceGenGAN_D.pth"

        # Transformations
        img_size = 64
        src_transform = transforms.Compose([
            SkeToImageTransform(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

        tgt_transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

        self.dataset = VideoSkeletonDataset(
            videoSke, ske_reduced=True,
            source_transform=src_transform,
            target_transform=tgt_transform
        )

        print(f"[INFO] {len(self.dataset)} images disponibles pour l'entraînement.")
        print("[INFO] Initialisation : WGAN-GP (Architecture GAN Stable)")

    # ============================================================
    #              GRADIENT PENALTY (toujours en FP32)
    # ============================================================
    def compute_gradient_penalty(self, real, fake):
        real = real.float()
        fake = fake.float()

        alpha = torch.rand(real.size(0), 1, 1, 1, device=self.device)
        interpol = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        d_interpol = self.netD(interpol)
        grad_outputs = torch.ones_like(d_interpol)

        gradients = torch.autograd.grad(
            outputs=d_interpol,
            inputs=interpol,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
        return penalty

    # ============================================================
    #                          TRAINING
    # ============================================================
    def train(self, n_epochs=20):

        batch_size = 512
        num_workers = 12
        n_critic = 1
        lambda_gp = 10
        lr = 0.0001

        optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        print(f"[TRAIN] Dataset: {len(self.dataset)}")
        print(f"[TRAIN] Batch: {batch_size}")
        print(f"[TRAIN] Workers: {num_workers}")
        print(f"[TRAIN] Batches per epoch: {len(dataloader)}")

        scaler = torch.cuda.amp.GradScaler()

        # tracking
        best_g_loss = float("inf")

        for epoch in range(n_epochs):
            d_loss_epoch, g_loss_epoch = 0, 0

            for i, (ske, real) in enumerate(dataloader):
                ske = ske.to(self.device, non_blocking=True)
                real = real.to(this.device, non_blocking=True)

                # ====================================================
                #                    TRAIN D
                # ====================================================
                optimizerD.zero_grad()

                with torch.cuda.amp.autocast():
                    fake = self.netG(ske)
                    d_real = self.netD(real)
                    d_fake = self.netD(fake.detach())
                    d_loss = -(d_real.mean()) + d_fake.mean()

                gp = self.compute_gradient_penalty(real, fake) * lambda_gp

                d_total = d_loss + gp
                scaler.scale(d_total).backward()
                scaler.step(optimizerD)

                d_loss_epoch += d_total.item()

                # ====================================================
                #                    TRAIN G
                # ====================================================
                if i % n_critic == 0:
                    optimizerG.zero_grad()

                    with torch.cuda.amp.autocast():
                        fake = self.netG(ske)
                        d_fake_for_g = self.netD(fake)
                        g_loss_adv = -d_fake_for_g.mean()
                        g_loss_l1 = nn.functional.l1_loss(fake, real) * 100
                        g_loss = g_loss_adv + g_loss_l1

                    scaler.scale(g_loss).backward()
                    scaler.step(optimizerG)

                    g_loss_epoch += g_loss.item()

                scaler.update()

            print(f"[EPOCH {epoch+1}/{n_epochs}]  D_loss={d_loss_epoch:.4f} | G_loss={g_loss_epoch:.4f}")

            # ============================================================
            #                    🔥 SAUVEGARDES 🔥
            # ============================================================

            # 1) always save latest
            torch.save(self.netG.state_dict(), "models/latest_G.pth")
            torch.save(self.netD.state_dict(), "models/latest_D.pth")


            # 3) archive automatique tous les 200 epochs
            if (epoch + 1) % 200 == 0:
                archG = f"models/archive_G_epoch_{epoch+1}.pth"
                archD = f"models/archive_D_epoch_{epoch+1}.pth"
                torch.save(self.netG.state_dict(), archG)
                torch.save(self.netD.state_dict(), archD)
                print(f"   [🗄️] Archive sauvegardée : {archG}")

        print("[TRAIN] Entraînement terminé.")
        torch.save(self.netG.state_dict(), self.filenameG)
        torch.save(self.netD.state_dict(), self.filenameD)
