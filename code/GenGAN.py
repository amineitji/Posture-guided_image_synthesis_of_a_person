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
#                  MODULES D'ATTENTION
# ============================================================

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key   = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)   # B x (HW) x (C/8)
        proj_key   = self.key(x).view(B, -1, H * W)                      # B x (C/8) x (HW)
        energy     = torch.bmm(proj_query, proj_key)                     # B x (HW) x (HW)
        attention  = torch.softmax(energy, dim=-1)

        proj_value = self.value(x).view(B, C, H * W)                     # B x C x (HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))          # B x C x (HW)
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


# ============================================================
#                     DISCRIMINATOR (OPTIMISÉ + ATTENTION)
# ============================================================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)     # 64x32x32
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)   # 128x16x16
        self.gn2   = nn.GroupNorm(32, 128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)  # 256x8x8
        self.gn3   = nn.GroupNorm(32, 256)

        self.conv4 = nn.Conv2d(256, 512, 4, 1, 1)  # 512x7x7
        self.gn4   = nn.GroupNorm(32, 512)

        # Attention + Channel attention sur les features profonds
        self.att3 = SelfAttention(256)
        self.att4 = SelfAttention(512)
        self.se3  = SEBlock(256)
        self.se4  = SEBlock(512)

        self.out = nn.Conv2d(512, 1, 4, 1, 0)      # 1x4x4 (logits pour WGAN-GP)

        print("[Discriminator] WGAN-GP Critic avec Self-Attention + SEBlock chargé")

    def forward(self, x):
        x = x.contiguous()

        x = self.lrelu(self.conv1(x))          # 64x32x32

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.lrelu(x)                      # 128x16x16

        x = self.conv3(x)
        x = self.gn3(x)
        x = self.lrelu(x)                      # 256x8x8
        x = self.att3(x)
        x = self.se3(x)

        x = self.conv4(x)
        x = self.gn4(x)
        x = self.lrelu(x)                      # 512x7x7
        x = self.att4(x)
        x = self.se4(x)

        x = self.out(x)                        # 1x4x4
        return x


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

        self.filenameG = "models/latest_G.pth"
        self.filenameD = "models/latest_D.pth"

        if loadFromFile and os.path.exists(self.filenameG):
            print(f"[LOAD] Chargement du générateur depuis {self.filenameG}")
            state_dict = torch.load(self.filenameG, map_location=self.device)
            self.netG.load_state_dict(state_dict, strict=False)

        if loadFromFile and os.path.exists(self.filenameD):
            print(f"[LOAD] Chargement du discriminateur depuis {self.filenameD}")
            state_dict = torch.load(self.filenameD, map_location=self.device)
            self.netD.load_state_dict(state_dict, strict=False)

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
        print("[INFO] Initialisation : WGAN-GP (Architecture GAN Stable + Attention)")

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

        batch_size  = 512
        num_workers = 12
        n_critic    = 1
        lambda_gp   = 10
        lr          = 0.0001

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

        for epoch in range(n_epochs):
            d_loss_epoch, g_loss_epoch = 0, 0

            for i, (ske, real) in enumerate(dataloader):
                ske  = ske.to(self.device, non_blocking=True)
                real = real.to(self.device, non_blocking=True)

                # ====================================================
                #                    TRAIN D
                # ====================================================
                optimizerD.zero_grad()

                with torch.cuda.amp.autocast():
                    fake     = self.netG(ske)
                    d_real   = self.netD(real)
                    d_fake   = self.netD(fake.detach())
                    d_loss   = -(d_real.mean()) + d_fake.mean()

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
                        fake          = self.netG(ske)
                        d_fake_for_g  = self.netD(fake)
                        g_loss_adv    = -d_fake_for_g.mean()
                        g_loss_l1     = nn.functional.l1_loss(fake, real) * 100
                        g_loss        = g_loss_adv + g_loss_l1

                    scaler.scale(g_loss).backward()
                    scaler.step(optimizerG)

                    g_loss_epoch += g_loss.item()

                scaler.update()

            print(f"[EPOCH {epoch+1}/{n_epochs}]  D_loss={d_loss_epoch:.4f} | G_loss={g_loss_epoch:.4f}")

            # ============================================================
            #                    🔥 SAUVEGARDES 🔥
            # ============================================================

            # always save latest
            torch.save(self.netG.state_dict(), "models/latest_G.pth")
            torch.save(self.netD.state_dict(), "models/latest_D.pth")

            # archive tous les 200 epochs
            if (epoch + 1) % 200 == 0:
                archG = f"models/archive_G_epoch_{epoch+1}.pth"
                archD = f"models/archive_D_epoch_{epoch+1}.pth"
                torch.save(self.netG.state_dict(), archG)
                torch.save(self.netD.state_dict(), archD)
                print(f"   [🗄️] Archive sauvegardée : {archG}")

        print("[TRAIN] Entraînement terminé.")
        torch.save(self.netG.state_dict(), self.filenameG)
        torch.save(self.netD.state_dict(), self.filenameD)
    # ============================================================
    #                     GENERATION (Inference)
    # ============================================================
    def generate(self, ske):
        """
        Génère une image à partir d’un skeleton (inférence, pas entraînement)
        """
        self.netG.eval()

        with torch.no_grad():
            # Transformer le squelette en image 64x64 (comme dans train)
            transform = transforms.Compose([
                SkeToImageTransform(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
            ])

            # Si ske est un objet Skeleton → on le transforme
            if not torch.is_tensor(ske):
                ske = transform(ske).unsqueeze(0).to(self.device)
            else:
                ske = ske.unsqueeze(0).to(self.device)

            # Génération
            fake = self.netG(ske)[0]

            # Dé-normalisation
            img = fake.detach().cpu()
            img = (img * 0.5 + 0.5).clamp(0,1)

            # Tensor → numpy HWC 0-255
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype("uint8")

            return img_np
