import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import cv2
from GenVanillaNN import GenNNSkeImToImage, VideoSkeletonDataset, SkeToImageTransform, VGGPerceptualLoss
from torch.utils.data import DataLoader, Dataset

# ============================================================
#  🚀 OPTIMISATIONS RTX 4000 (TF32 & BENCHMARK)
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ============================================================
#                  MODULES D'ATTENTION
# ============================================================

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)

        proj_value = self.value(x).view(B, C, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
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
#                     DISCRIMINATOR
# ============================================================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Etage 1 : 128 -> 64
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)  # 64x64
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Etage 2 : 64 -> 32
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # 32x32
        # InstanceNorm est plus rapide et standard pour les GANs image-to-image
        self.gn2 = nn.InstanceNorm2d(128, affine=True)

        # Etage 3 : 32 -> 16
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)  # 16x16
        self.gn3 = nn.InstanceNorm2d(256, affine=True)
        # OPTIMISATION : On supprime self.att3 (trop lourd ici)
        # On garde juste SEBlock (coût négligeable)
        self.se3 = SEBlock(256)

        # Etage 4 : 16 -> 8
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)  # 8x8 (Note le stride 2 pour descendre à 8px)
        self.gn4 = nn.InstanceNorm2d(512, affine=True)

        # On garde l'attention ICI seulement (sur du 8x8, c'est très rapide)
        self.att4 = SelfAttention(512)
        self.se4 = SEBlock(512)

        # Output : 8x8 -> 1
        self.out = nn.Conv2d(512, 1, 4, 1, 0)

        print("[Discriminator] Version Light chargée (Attention uniquement en couche finale)")

    def forward(self, x):
        # x: 128x128
        x = self.lrelu(self.conv1(x))  # -> 64x64

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.lrelu(x)  # -> 32x32

        x = self.conv3(x)
        x = self.gn3(x)
        x = self.lrelu(x)
        # Pas d'attention ici
        x = self.se3(x)  # -> 16x16

        x = self.conv4(x)
        x = self.gn4(x)
        x = self.lrelu(x)
        x = self.att4(x)  # OK car image petite (8x8)
        x = self.se4(x)  # -> 8x8

        x = self.out(x)
        return x


# ============================================================
#                        DATASET RAM
# ============================================================

class RAMDataset(Dataset):
    def __init__(self, original_dataset):
        self.data = []
        n_images = len(original_dataset)
        print(f"[RAM] Chargement de {n_images} images en mémoire... (Veuillez patienter)")

        for i in range(n_images):
            ske, real = original_dataset[i]
            self.data.append((ske, real))
            if (i + 1) % 1000 == 0:
                print(f"   ... {i + 1} images chargées")

        print("[RAM] Chargement terminé.")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# ============================================================
#               FONCTION D'INITIALISATION (AJOUTÉE)
# ============================================================
def weights_init(m):
    """
    Initialisation des poids pour le GAN (Normal 0.02).
    Gère Conv2d, ConvTranspose2d, BatchNorm, InstanceNorm, GroupNorm et Linear.
    """
    # 1. Convolutions et Linear (Linear est utilisé dans SEBlock !)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    # 2. Normalisations
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# ============================================================
#                        WGAN-GP OPTIMISÉ
# ============================================================

class GenGAN():
    def __init__(self, videoSke, loadFromFile=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 128

        print(f"[GPU] {self.device} (Optimisé Channels Last)")

        # --- 1. Modèles ---
        # Passage en Channels Last pour RTX
        self.netG = GenNNSkeImToImage().to(self.device, memory_format=torch.channels_last)
        self.netD = Discriminator().to(self.device, memory_format=torch.channels_last)

        # Initialisation
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        self.filenameG = "models/latest_G.pth"
        self.filenameD = "models/latest_D.pth"

        if loadFromFile:
            if os.path.exists(self.filenameG):
                print(f"[LOAD] G: {self.filenameG}")
                self.netG.load_state_dict(torch.load(self.filenameG, map_location=self.device))
            if os.path.exists(self.filenameD):
                print(f"[LOAD] D: {self.filenameD}")
                self.netD.load_state_dict(torch.load(self.filenameD, map_location=self.device))

        # --- 2. Dataset & Transformations ---
        src_transform = transforms.Compose([
            SkeToImageTransform(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        tgt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        base_dataset = VideoSkeletonDataset(
            videoSke, ske_reduced=True,
            source_transform=src_transform,
            target_transform=tgt_transform
        )

        self.dataset = RAMDataset(base_dataset)

    def compute_gradient_penalty(self, real, fake):
        real = real.float()
        fake = fake.float()
        alpha = torch.rand(real.size(0), 1, 1, 1, device=self.device)
        interpol = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        d_interpol = self.netD(interpol)

        fake_grad = torch.ones(d_interpol.shape, device=self.device, requires_grad=False)

        gradients = torch.autograd.grad(
            outputs=d_interpol,
            inputs=interpol,
            grad_outputs=fake_grad,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def train(self, n_epochs=200):
        # Configuration Optimisée RTX 4060
        batch_size = 16
        lr = 0.0001
        n_critic = 3
        lambda_gp = 10

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.0, 0.9))
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.0, 0.9))

        criterion_L1 = nn.L1Loss()
        criterion_VGG = VGGPerceptualLoss().to(self.device)
        scaler = torch.cuda.amp.GradScaler()

        print("[TRAIN] Début WGAN-GP (RAM + Channels Last)")

        for epoch in range(n_epochs):
            for i, (ske, real) in enumerate(dataloader):

                # OPTIMISATION RTX : CHANNELS LAST
                ske = ske.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                real = real.to(self.device, non_blocking=True, memory_format=torch.channels_last)

                # ====================================================
                #            1. TRAIN DISCRIMINATOR
                # ====================================================
                optimizerD.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    fake = self.netG(ske).detach()
                    d_real = self.netD(real)
                    d_fake = self.netD(fake)
                    d_loss_adv = -torch.mean(d_real) + torch.mean(d_fake)

                gp = self.compute_gradient_penalty(real, fake) * lambda_gp
                d_loss = d_loss_adv + gp

                scaler.scale(d_loss).backward()
                scaler.step(optimizerD)

                # ====================================================
                #            2. TRAIN GENERATOR
                # ====================================================
                if i % n_critic == 0:
                    optimizerG.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast():
                        fake = self.netG(ske)
                        d_fake = self.netD(fake)

                        g_loss_adv = -torch.mean(d_fake)
                        g_loss_l1 = criterion_L1(fake, real)
                        g_loss_vgg = criterion_VGG(fake, real)

                        # Tes pondérations personnalisées
                        g_loss = 0.3 * g_loss_adv + 6 * g_loss_l1 + 2.0 * g_loss_vgg

                    scaler.scale(g_loss).backward()
                    scaler.step(optimizerG)

                    if i % 50 == 0:
                        print(f"\r[Ep {epoch}][Bt {i}] D:{d_loss.item():.4f} G:{g_loss.item():.4f}", end="")

                scaler.update()

            if (epoch + 1) % 5 == 0:
                torch.save(self.netG.state_dict(), self.filenameG)
                torch.save(self.netD.state_dict(), self.filenameD)
                with torch.no_grad():
                    img = self.generate(ske[0])
                    cv2.imwrite(f"images_suivi/epoch_{epoch}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def generate(self, ske):
        self.netG.eval()
        with torch.no_grad():
            transform = transforms.Compose([
                SkeToImageTransform(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            if not torch.is_tensor(ske):
                ske = transform(ske).unsqueeze(0).to(self.device)
            else:
                ske = ske.unsqueeze(0).to(self.device)

            # Application Memory Format pour inference
            ske = ske.to(memory_format=torch.channels_last)

            with torch.cuda.amp.autocast():
                fake = self.netG(ske)[0]

            img = fake.float().detach().cpu().permute(1, 2, 0).numpy()
            img = ((img * 0.5 + 0.5).clip(0, 1) * 255).astype("uint8")
            return img