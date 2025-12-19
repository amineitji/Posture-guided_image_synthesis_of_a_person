import os
import contextlib  # Nécessaire pour gérer le contexte CPU/GPU

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import cv2
from GenVanillaNN import GenNNSkeImToImage, VideoSkeletonDataset, SkeToImageTransform, VGGPerceptualLoss
from torch.utils.data import DataLoader, Dataset

# ============================================================
#         OPTIMISATIONS RTX 4000 & HYBRIDE CPU/GPU
# ============================================================

# On détecte si le GPU est dispo AVANT d'appliquer les optimisations
use_cuda = torch.cuda.is_available()

if use_cuda:
    # Optimisations spécifiques RTX (TF32 & Benchmark)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"[INIT] Mode GPU activé (Optimisations RTX 4000)")
else:
    print(f"[INIT] Mode CPU activé (Optimisations GPU désactivées)")


# ============================================================
# UTILS POUR LA COMPATIBILITÉ CPU
# ============================================================
class CpuScaler:
    """Classe 'Fake' qui imite le GradScaler pour le CPU (ne fait rien)"""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


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

        # Utilisation de .reshape() au lieu de .view()
        # Cela permet de gérer le format channels_last
        proj_query = self.query(x).reshape(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).reshape(B, -1, H * W)

        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)

        proj_value = self.value(x).reshape(B, C, H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        # Ici aussi, .reshape() pour remettre en forme
        out = out.reshape(B, C, H, W)

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

        # Définition du Dropout (0.3 = 30% des neurones désactivés)
        # On le définit une fois et on l'appelle plusieurs fois
        self.dropout = nn.Dropout(0.3)

        # Etage 1 : 128 -> 64
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        # PAS DE DROPOUT ICI (On garde l'input intact)

        # Etage 2 : 64 -> 32
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.gn2 = nn.InstanceNorm2d(128, affine=True)
        # DROPOUT ICI : Oui

        # Etage 3 : 32 -> 16
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.gn3 = nn.InstanceNorm2d(256, affine=True)
        self.se3 = SEBlock(256)
        # DROPOUT ICI : Oui

        # Etage 4 : 16 -> 8
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.gn4 = nn.InstanceNorm2d(512, affine=True)

        self.att4 = SelfAttention(512)
        self.se4 = SEBlock(512)
        # PAS DE DROPOUT ICI (On veut que l'Attention travaille sur des features stables)

        # Output : 8x8 -> 1
        self.out = nn.Conv2d(512, 1, 4, 1, 0)

        print("[Discriminator] Version Light + Dropout (0.3) chargée")

    def forward(self, x):
        # x: 128x128

        # Bloc 1
        x = self.lrelu(self.conv1(x))

        # Bloc 2 + Dropout
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.lrelu(x)
        x = self.dropout(x)  # <--- AJOUT STRATÉGIQUE 1

        # Bloc 3 + Dropout
        x = self.conv3(x)
        x = self.gn3(x)
        x = self.lrelu(x)
        x = self.se3(x)
        x = self.dropout(x)  # <--- AJOUT STRATÉGIQUE 2

        # Bloc 4 (Pas de dropout avant l'attention pour ne pas la casser)
        x = self.conv4(x)
        x = self.gn4(x)
        x = self.lrelu(x)
        x = self.att4(x)
        x = self.se4(x)

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
#               FONCTION D'INITIALISATION
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
#                        WGAN-GP OPTIMISÉ (HYBRIDE)
# ============================================================

class GenGAN():
    def __init__(self, videoSke, loadFromFile=False):
        # Détection automatique du device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 128

        print(f"[DEVICE] Utilisation de : {self.device}")
        if self.device.type == 'cuda':
            print("       (Optimisé Channels Last activé)")

        # --- 1. Modèles ---
        self.netG = GenNNSkeImToImage()
        self.netD = Discriminator()

        # Passage en Channels Last pour RTX UNIQUEMENT SI CUDA
        if self.device.type == 'cuda':
            self.netG = self.netG.to(memory_format=torch.channels_last)
            self.netD = self.netD.to(memory_format=torch.channels_last)

        # Envoi sur le device
        self.netG = self.netG.to(self.device)
        self.netD = self.netD.to(self.device)

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

        # On s'assure que alpha est sur le bon device
        alpha = torch.rand(real.size(0), 1, 1, 1, device=self.device)

        # L'interpolation garde le format channels_last si real/fake le sont
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

        #  .reshape() au lieu de .view() car gradients est en channels_last
        gradients = gradients.reshape(gradients.size(0), -1)

        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def train(self, n_epochs=200):
        # --- CONFIGURATION HYPER-PARAMÈTRES ---
        batch_size = 64
        lr = 0.0001
        n_critic = 4  # Le Discriminateur s'entraîne 4 fois plus que le Générateur
        lambda_gp = 10  # Force de la pénalité de gradient (Standard WGAN-GP)

        # --- DATALOADER ---
        # pin_memory n'est utile que si on a un GPU
        use_pin = (self.device.type == 'cuda')

        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=use_pin
        )
        total_batches = len(dataloader)

        # --- OPTIMIZERS & LOSS ---
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.0, 0.9))
        optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.0, 0.9))

        criterion_L1 = nn.L1Loss()
        criterion_VGG = VGGPerceptualLoss().to(self.device)

        # --- CONFIGURATION MIXED PRECISION (HYBRIDE) ---
        if self.device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            amp_context = torch.amp.autocast("cuda")
        else:
            scaler = CpuScaler()
            amp_context = contextlib.nullcontext()

        print(f"[TRAIN] Démarrage WGAN-GP sur {self.device}...")
        print(f"        Images: {len(self.dataset)} | Batchs: {total_batches}")

        # ============================================================
        # 1. CRÉATION DU "FIXED BATCH" (POUR SUIVRE LA PROGRESSION)
        # ============================================================
        print("[INIT] Sélection de 5 poses fixes pour le suivi...")
        data_iter = iter(dataloader)
        first_batch = next(data_iter)
        # On garde les 5 premiers squelettes. .clone() protège les données.
        fixed_ske = first_batch[0][:5].to(self.device, non_blocking=use_pin).clone()
        if self.device.type == 'cuda':
            fixed_ske = fixed_ske.to(memory_format=torch.channels_last)
        # ============================================================

        # --- BOUCLE D'ENTRAÎNEMENT ---
        for epoch in range(n_epochs):
            for i, (ske, real) in enumerate(dataloader):

                # Optimisation mémoire
                ske = ske.to(self.device, non_blocking=use_pin)
                real = real.to(self.device, non_blocking=use_pin)

                # Optimisation Channels Last (GPU Seulement)
                if self.device.type == 'cuda':
                    ske = ske.to(memory_format=torch.channels_last)
                    real = real.to(memory_format=torch.channels_last)

                # ---------------------------------------
                #  ETAPE 1 : ENTRAÎNEMENT DU DISCRIMINATOR
                # ---------------------------------------
                optimizerD.zero_grad(set_to_none=True)

                with amp_context:
                    fake = self.netG(ske).detach()  # .detach() pour ne pas toucher au Générateur
                    d_real = self.netD(real)
                    d_fake = self.netD(fake)
                    d_loss_adv = -torch.mean(d_real) + torch.mean(d_fake)

                    # Calcul de la pénalité de gradient (cœur du WGAN-GP)
                    # GP est calculé hors autocast généralement ou géré par autograd
                    # Ici on reste simple, si ça plante sur CPU on peut désactiver le GP ou le simplifier
                    gp = self.compute_gradient_penalty(real, fake) * lambda_gp
                    d_loss = d_loss_adv + gp

                scaler.scale(d_loss).backward()
                scaler.step(optimizerD)

                # ---------------------------------------
                #  ETAPE 2 : ENTRAÎNEMENT DU GENERATOR
                # ---------------------------------------
                # On entraîne le G seulement tous les 'n_critic' pas
                if i % n_critic == 0:
                    optimizerG.zero_grad(set_to_none=True)

                    with amp_context:
                        fake = self.netG(ske)
                        d_fake = self.netD(fake)

                        # Pertes composites
                        g_loss_adv = -torch.mean(d_fake)  # Tromper le discriminateur
                        g_loss_l1 = criterion_L1(fake, real)  # Ressembler aux pixels réels
                        g_loss_vgg = criterion_VGG(fake, real)  # Avoir la même texture (Perceptuel)

                        # Poids des pertes : Adv=3, L1=8, VGG=0.3
                        g_loss = 3 * g_loss_adv + 8.0 * g_loss_l1 + 0.3 * g_loss_vgg

                    scaler.scale(g_loss).backward()
                    scaler.step(optimizerG)

                    # --- Print  ---
                    print(
                        f"\r[Ep {epoch + 1}/{n_epochs}] [Bt {i + 1}/{total_batches}] "
                        f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f}",
                        end=""
                    )

                scaler.update()

            # Fin de l'époque
            print()

            # ============================================================
            # 3. SAUVEGARDE & VISUALISATION MULTI-POSE
            # ============================================================
            if (epoch + 1) % 5 == 0:
                # Sauvegarde des modèles
                torch.save(self.netG.state_dict(), self.filenameG)
                torch.save(self.netD.state_dict(), self.filenameD)

                # Génération de l'image témoin
                with torch.no_grad():
                    images_list = []
                    for k in range(len(fixed_ske)):
                        # Générer chaque pose fixe
                        img_gen = self.generate(fixed_ske[k])
                        images_list.append(img_gen)

                    # Concaténer horizontalement (H-Concat)
                    if len(images_list) > 0:
                        combined_image = cv2.hconcat(images_list)
                        cv2.imwrite(f"images_suivi/epoch_{epoch}_multi.jpg", combined_image)
                        print(f"   [SAVE] Checkpoint et Image multi-poses sauvés.")

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

            if self.device.type == 'cuda':
                ske = ske.to(memory_format=torch.channels_last)

            # Contexte Autocast Hybride
            ctx = torch.amp.autocast("cuda") if self.device.type == 'cuda' else contextlib.nullcontext()

            with ctx:
                fake = self.netG(ske)[0]

            # Tensor (RGB) -> Numpy (RGB)
            img = fake.float().detach().cpu().permute(1, 2, 0).numpy()
            img = ((img * 0.5 + 0.5).clip(0, 1) * 255).astype("uint8")

            # CORRECTION COULEUR ICI
            # On convertit  RGB -> BGR pour OpenCV
            # img[..., ::-1] inverse l'ordre des canaux (R,G,B -> B,G,R)
            img_bgr = img[..., ::-1].copy()

            return img_bgr