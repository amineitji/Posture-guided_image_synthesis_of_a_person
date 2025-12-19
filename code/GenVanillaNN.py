import numpy as np
import cv2
import os
import contextlib  # Nécessaire pour gérer le contexte CPU/GPU proprement

from PIL import Image
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from VideoSkeleton import VideoSkeleton
from Skeleton import Skeleton

# =============================================================================
#  PARTIE 1 : OPTIMISATIONS & CONFIGURATION (CPU / GPU)
# =============================================================================

torch.set_default_dtype(torch.float32)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# On détecte si le GPU est dispo AVANT d'appliquer les optimisations
use_cuda = torch.cuda.is_available()

if use_cuda:
    # Optimisations spécifiques pour les cartes RTX (TensorFloat-32)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    print(f"[INIT] Mode GPU activé (Optimisations RTX)")
else:
    print(f"[INIT] Mode CPU activé (Optimisations GPU désactivées)")


# --- CLASSE UTILITAIRE POUR LE CPU ---
class CpuScaler:
    """Classe 'Fake' qui imite le GradScaler pour le CPU (ne fait rien)"""
    def scale(self, loss): return loss
    def step(self, optimizer): optimizer.step()
    def update(self): pass


# =============================================================================
#  PARTIE 2 : DATASET & TRANSFORMATION DES DONNÉES
# =============================================================================

class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size
        from Skeleton import Skeleton
        self.Skeleton = Skeleton

    def __call__(self, ske):
        image = np.zeros((self.imsize, self.imsize, 3), dtype=np.uint8)

        if isinstance(ske, np.ndarray):
            ske_array = np.vstack(ske[self.Skeleton.reduce_indice]).astype(float)[:, :2]
        else:
            ske_array = ske.toarray(reduced=True)

        self.Skeleton.draw_reduced(ske_array, image)
        return Image.fromarray(image)


class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced

        print(f"[DATASET] Chargement des données en RAM...")
        self.cached_images = []
        self.cached_skeletons = []

        # --- PRÉ-CHARGEMENT EN RAM ---
        # On lit TOUTES les images maintenant. Ça prendra quelques secondes au début,
        # mais ensuite l'entraînement sera ultra-fluide.
        for idx in range(self.videoSke.skeCount()):
            img = Image.open(self.videoSke.imagePath(idx)).convert('RGB')
            self.cached_images.append(img)

            ske = self.videoSke.ske[idx]
            ske_tensor = self.preprocessSkeleton(ske)
            self.cached_skeletons.append(ske_tensor)

        print(f"[DATASET] {len(self.cached_images)} images chargées en mémoire RAM.")

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        image = self.cached_images[idx]
        ske = self.cached_skeletons[idx]

        if self.target_transform:
            image = self.target_transform(image)

        return ske, image

    def preprocessSkeleton(self, ske):

        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            if isinstance(ske, np.ndarray):
                from Skeleton import Skeleton
                if self.ske_reduced:
                    ske_array = np.vstack(ske[Skeleton.reduce_indice]).astype(float)[:, :2]
                else:
                    ske_array = np.vstack(ske).astype(float)
                ske = torch.from_numpy(ske_array.flatten())
            else:
                ske = torch.from_numpy(ske.toarray(reduced=self.ske_reduced).flatten())

            ske = ske.to(torch.float32)
            ske = ske.reshape(ske.shape[0], 1, 1)
        return ske

    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().cpu().float().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = (numpy_image * 0.5 + 0.5) * 255
        return numpy_image.clip(0, 255).astype(np.uint8)


# =============================================================================
#  PARTIE 3 : BLOCS DE CONSTRUCTION (ATTENTION & RESNET)
# =============================================================================

class SEBlock(nn.Module):
    """Bloc Squeeze-and-Excitation : Améliore l'importance des canaux"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SelfAttention(nn.Module):
    """Attention Spatiale : Permet au réseau de voir l'ensemble de l'image"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    """Bloc Résiduel simple pour le modèle Vectoriel"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


# =============================================================================
#  PARTIE 4 : GÉNÉRATEUR SIMPLE (VECTEUR -> IMAGE)
# =============================================================================

class GenNNSke26ToImage(nn.Module):
    """Prend un vecteur de 26 coordonnées et génère une image"""
    def __init__(self, latent_dim=26):
        super(GenNNSke26ToImage, self).__init__()
        self.fc_init = nn.Linear(latent_dim, 512 * 4 * 4)
        self.bn_init = nn.InstanceNorm2d(512)

        # --- UPSAMPLING (Avec SEBlock intégré) ---

        # 4x4 -> 8x8
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(256)
        )

        # 8x8 -> 16x16
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(128)
        )

        # --- ATTENTION (Placée à 16x16) ---
        # assez de résolution pour voir la forme,
        # mais assez petit pour ne pas exploser la mémoire.
        self.attention = SelfAttention(128)

        # 16x16 -> 32x32
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(64)
        )

        # 32x32 -> 64x64
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(32)
        )

        # --- RESIDUAL BLOCKS (Raffinage des détails à 64x64) ---
        # Permet de nettoyer la texture avant l'agrandissement final
        self.res_blocks = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        # --- FINAL (64x64 -> 128x128 -> RGB) ---
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Sortie RGB
            nn.Conv2d(16, 3, 3, 1, 1, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        # 1. (Batch, 26)
        x = x.view(x.size(0), -1)

        # 2. Projection Latente
        x = self.fc_init(x)
        x = x.view(-1, 512, 4, 4)
        x = F.leaky_relu(self.bn_init(x), 0.2)

        # 3. Montée en résolution
        x = self.up1(x)  # -> 8x8
        x = self.up2(x)  # -> 16x16

        # 4. Attention Map
        x = self.attention(x)

        # 5. Suite Montée
        x = self.up3(x)  # -> 32x32
        x = self.up4(x)  # -> 64x64

        # 6. Raffinage Structurel
        x = self.res_blocks(x)

        # 7. Rendu Final
        x = self.final(x)  # -> 128x128

        return x


# =============================================================================
#  PARTIE 5 : FONCTION DE PERTE (LOSS)
# =============================================================================

class VGGPerceptualLoss(nn.Module):
    """Calcule la différence de 'style' et de 'texture' en utilisant un VGG pré-entraîné"""
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # On utilise VGG16 (léger et efficace pour la texture)
        # On prend .features pour avoir les convs, pas le classifier
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # On découpe le réseau proprement
        self.blocks = nn.ModuleList([
            vgg[:4],    # Niveau 1
            vgg[4:9],   # Niveau 2
            vgg[9:16],  # Niveau 3
            vgg[16:23]  # Niveau 4
        ])

        # On gèle tout
        for param in self.parameters():
            param.requires_grad = False

        self.eval() # On met VGG en mode évaluation (désactive dropout interne)

        # OPTIMISATION : On utilise register_buffer
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # 1. Gestion du Noir & Blanc (si nécessaire)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # 2. On passe de [-1, 1] à [0, 1]
        # Car Tanh sort du [-1, 1], mais la normalisation ImageNet veut du [0, 1]
        input = (input + 1) / 2
        target = (target + 1) / 2

        # 3. Normalisation ImageNet standard
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x = input
        y = target

        # 4. Calcul de la loss par bloc
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)

        return loss


# =============================================================================
#  PARTIE 6 : GÉNÉRATEUR AVANCÉ (U-NET / PIX2PIX)
# =============================================================================

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=False), # Bias=False avant InstanceNorm
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(dim)
        )

        # AJOUT DU SEBLOCK
        self.se = SEBlock(dim)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.se(out)  # Application du SE Block
        return x + out      # Skip Connection

class GenNNSkeImToImage(nn.Module):
    def __init__(self):
        super().__init__()

        # --- ENCODER (Downsampling) ---
        # Le standard: Conv -> InstanceNorm -> ReLU
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU(True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128), nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(256), nn.ReLU(True)
        )
        # On descend jusqu'à 512 filtres
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(512), nn.ReLU(True)
        )

        # --- BOTTLENECK (ResNet + SE) ---
        # 4 blocs résiduels à 512 canaux
        self.bottleneck = nn.Sequential(
            ResnetBlock(512),
            ResnetBlock(512),
            ResnetBlock(512),
            ResnetBlock(512),
        )

        # --- DECODER (Upsampling) ---
        # Upsample + Conv pour éviter le checkerboard artifacts
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Lisse
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        # Up 2 : 256 -> 128
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256 + 256, 128, 3, 1, 0, bias=False), # +256 pour le Skip-Co
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )

        # Up 3 : 128 -> 64
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128 + 128, 64, 3, 1, 0, bias=False), # +128 pour le Skip-Co
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        # Final Layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64 + 64, 3, 7, 1, 0), # +64 pour le Skip-Co
            nn.Tanh() # Sortie entre -1 et 1
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)  # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512

        # Bottleneck
        b = self.bottleneck(x4)  # 512

        # Decoder + Skip Connections (U-Net)

        # Pas de skip sur le niveau le plus profond (classique Pix2Pix)
        u1 = self.up1(b)  # 256

        u1_cat = torch.cat([u1, x3], dim=1)  # On recolle x3
        u2 = self.up2(u1_cat)  # 128

        u2_cat = torch.cat([u2, x2], dim=1)  # On recolle x2
        u3 = self.up3(u2_cat)  # 64

        u3_cat = torch.cat([u3, x1], dim=1)  # On recolle x1
        output = self.final(u3_cat)

        return output


def weights_init(m):
    """Fonction d'initialisation des poids (Normal distribution)"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1 or classname.find('GroupNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# =============================================================================
#  PARTIE 7 : CLASSE PRINCIPALE (GESTION DE L'ENTRAINEMENT)
# =============================================================================

class GenVanillaNN:
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        self.image_size = 128
        image_size = 128

        # --- CONFIGURATION DEVICE (GPU/CPU) ---
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[DEVICE] Utilisation de : {self.device}")

        self.optSkeOrImage = optSkeOrImage

        # Choix du modèle selon l'option
        if optSkeOrImage == 1:
            self.netG = GenNNSke26ToImage().to(self.device)
            src_transform = None
            self.filename = "models/DanceGenVanillaFromSke26.pth"
        else:
            self.netG = GenNNSkeImToImage()
            # Optimisation "Channels Last" uniquement si GPU
            if self.device.type == 'cuda':
                self.netG = self.netG.to(memory_format=torch.channels_last)

            self.netG = self.netG.to(self.device)
            self.netG.apply(weights_init)

            src_transform = transforms.Compose([
                SkeToImageTransform(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.filename = "models/modelsDanceGenVanillaFromSkeim.pth"

        # Transformations pour l'image cible (Data Augmentation)
        tgt_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ])

        # Création du Dataset
        self.dataset = VideoSkeletonDataset(
            videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform
        )

        # Chargement d'un modèle existant
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            try:
                state_dict = torch.load(self.filename, map_location=self.device)
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                self.netG.load_state_dict(new_state_dict)
                print("[✓] Modèle chargé avec succès")
            except Exception as e:
                print(f"[✗] Erreur chargement: {e}")

    def generate(self, ske):
        """Fonction pour générer une image à partir d'un squelette"""
        self.netG.eval()
        with torch.no_grad():
            if self.optSkeOrImage == 1:
                ske_input = torch.from_numpy(ske.toarray(reduced=True).flatten()).float()
                ske_input = ske_input.unsqueeze(0).to(self.device)
            else:
                transform = transforms.Compose([
                    SkeToImageTransform(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

                ske_input = transform(ske).unsqueeze(0).to(self.device)
                if self.device.type == 'cuda':
                    ske_input = ske_input.to(memory_format=torch.channels_last)

            # Gestion du mode mixte (CPU/GPU)
            ctx = torch.amp.autocast("cuda") if self.device.type == 'cuda' else contextlib.nullcontext()

            with ctx:
                output = self.netG(ske_input)

            rgb_image = self.dataset.tensor2image(output[0])
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            return bgr_image

    def train(self, n_epochs=200):
        """Boucle principale d'entraînement"""
        print(f"[TRAIN] Début de l'entraînement sur {self.device}")

        # Configuration des Dataloaders
        batch_size = 64
        train_size = int(0.85 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

        use_pin = (self.device.type == 'cuda')
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                  pin_memory=use_pin)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                pin_memory=use_pin)

        # Image témoin pour le suivi
        fixed_sample_iter = iter(val_loader)
        fixed_sample = next(fixed_sample_iter)
        fixed_ske = fixed_sample[0][0:1].to(self.device, non_blocking=True)
        if self.device.type == 'cuda':
            fixed_ske = fixed_ske.to(memory_format=torch.channels_last)

        if not os.path.exists("images_suivi"):
            os.makedirs("images_suivi")

        # Configuration Loss & Optimizer
        criterion_l1 = nn.L1Loss()
        print("[TRAIN] Chargement du VGG...")
        criterion_vgg = VGGPerceptualLoss().to(self.device)

        optimizer = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        # Scaler pour le Mixed Precision (GPU uniquement)
        if self.device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            amp_context = torch.amp.autocast("cuda")
        else:
            scaler = CpuScaler()
            amp_context = contextlib.nullcontext()

        best_val_loss = float("inf")

        # --- BOUCLE DES EPOCHS ---
        for epoch in range(n_epochs):
            self.netG.train()
            running_loss = 0.0
            batch_count = 0

            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device, non_blocking=use_pin)
                targets = targets.to(self.device, non_blocking=use_pin)

                if self.device.type == 'cuda':
                    inputs = inputs.to(memory_format=torch.channels_last)
                    targets = targets.to(memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)

                # Forward Pass + Loss Calculation
                with amp_context:
                    outputs = self.netG(inputs)
                    loss_l1_val = criterion_l1(outputs, targets)
                    loss_vgg_val = criterion_vgg(outputs, targets)
                    # Formule : 10*L1 + 0.5*VGG
                    loss = 10.0 * loss_l1_val + 0.5 * loss_vgg_val

                # Backward Pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                batch_count += 1

                if i % 10 == 0:
                    # On affiche les deux détails (L1 et VGG) pour comprendre ce qui se passe
                    print(
                        f"\rEpoch {epoch + 1} [{i}/{len(train_loader)}] Loss: {loss.item():.4f} (L1:{loss_l1_val:.3f} VGG:{loss_vgg_val:.3f})",
                        end="")

            avg_train_loss = running_loss / batch_count

            # Validation
            self.netG.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device, non_blocking=use_pin)
                    targets = targets.to(self.device, non_blocking=use_pin)
                    if self.device.type == 'cuda':
                        inputs = inputs.to(memory_format=torch.channels_last)
                        targets = targets.to(memory_format=torch.channels_last)

                    with amp_context:
                        outputs = self.netG(inputs)
                        l1 = criterion_l1(outputs, targets)
                        vgg = criterion_vgg(outputs, targets)
                        loss = 10.0 * l1 + 0.5 * vgg

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            # Sauvegarde d'une image témoin tous les 2 epochs
            if (epoch + 1) % 2 == 0:
                with torch.no_grad():
                    fake_img_tensor = self.netG(fixed_ske)
                    img_to_save = self.dataset.tensor2image(fake_img_tensor[0])
                    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                    filename = f"images_suivi/epoch_{epoch + 1}.jpg"
                    cv2.imwrite(filename, img_to_save)
                    print(f"  [Disk] Image sauvegardée : {filename}")
            # ---------------------------------------------------------

            # Sauvegarde du modèle si meilleur
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.netG.state_dict(), self.filename)
                print(f"\n[✓] Save: {avg_val_loss:.4f}")
            else:
                print(f"\n[ ] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    @staticmethod
    def verify_alignment(filename, crop_ratio):
        print(f"--- VÉRIFICATION ALIGNEMENT (Crop Ratio: {crop_ratio}) ---")

        # 1. On charge la vidéo et on force le recalcul pour être sûr
        # forceCompute=True est vital ici pour voir tes dernières modifs (Shift, Crop, etc.)
        video_ske = VideoSkeleton(filename, cropRatio=crop_ratio, forceCompute=True,modFrame=100)

        # 2. On crée le Dataset (comme pour l'entraînement)
        # ske_reduced=True car c'est ce que tu utilises pour l'entrainement
        dataset = VideoSkeletonDataset(video_ske, ske_reduced=True)

        print(f"Nombre d'images chargées : {len(dataset)}")
        print("Appuie sur 'Espace' pour passer à l'image suivante.")
        print("Appuie sur 'Q' pour quitter.")

        # On parcourt quelques images (pas forcément toutes)
        for i in range(len(dataset)):
            # Récupération des données brutes du dataset
            ske_tensor, pil_image = dataset[i]

            # --- A. PRÉPARATION DE L'IMAGE ---
            # Le dataset renvoie une image PIL, on la convertit en OpenCV (BGR)
            image_cv = np.array(pil_image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

            # --- B. PRÉPARATION DU SQUELETTE ---
            # Le dataset renvoie un Tensor plat (ex: 26 valeurs).
            # On doit le remettre en forme (13 points, 2 coordonnées x,y)
            ske_numpy = ske_tensor.numpy()

            # On reforme le tableau (N points, 2 coordonnées)
            # Si ske_reduced=True, on a 26 valeurs -> 13 points
            ske_reshaped = ske_numpy.reshape((-1, 2))

            # --- C. DESSIN ET VERIFICATION ---
            # On utilise la fonction statique de ta classe Skeleton pour dessiner
            # Elle attend des coordonnées normalisées (0-1) et va les multiplier par la taille de l'image
            Skeleton.draw_reduced(ske_reshaped, image_cv)

            # On redimensionne  (zoom x2)
            h, w = image_cv.shape[:2]
            image_zoom = cv2.resize(image_cv, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)

            # Affichage
            cv2.imshow("Verification Alignement", image_zoom)

            key = cv2.waitKey(0)  # Attend une touche indéfiniment
            if key & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()




if __name__ == "__main__":
    video_file = "../data/raw/taichi1.mp4"
    # test alignement skeleton avec crop_ratio 1.3
    GenVanillaNN.verify_alignment(video_file, crop_ratio=1.3)