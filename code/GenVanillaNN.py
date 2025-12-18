import numpy as np
import cv2
import os


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
# OPTIMISATIONS RTX 4060 (Ada Lovelace)
# =============================================================================
torch.set_default_dtype(torch.float32)

# 1. Active le TensorFloat-32 (TF32) : Indispensable sur RTX 4000
# 'high' = très rapide, perte précision minime. 'medium' = encore plus rapide.
torch.set_float32_matmul_precision('high')

# 2. Benchmark CuDNN pour trouver l'algo de convolution le plus rapide
torch.backends.cudnn.benchmark = True

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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

        print(f"[DATASET] Chargement des données en RAM pour accélérer l'entraînement...")
        self.cached_images = []
        self.cached_skeletons = []

        # --- PRÉ-CHARGEMENT EN RAM ---
        # On lit TOUTES les images maintenant. Ça prendra quelques secondes au début,
        # mais ensuite l'entraînement sera ultra-fluide.
        for idx in range(self.videoSke.skeCount()):
            # 1. Charger l'image
            # On convertit tout de suite en RGB pour éviter de le refaire 1000 fois
            img = Image.open(self.videoSke.imagePath(idx)).convert('RGB')
            self.cached_images.append(img)

            # 2. Préparer le squelette
            # On fait le calcul numpy -> torch tout de suite aussi
            ske = self.videoSke.ske[idx]
            ske_tensor = self.preprocessSkeleton(ske)
            self.cached_skeletons.append(ske_tensor)

        print(f"[DATASET] {len(self.cached_images)} images chargées en mémoire RAM. Prêt !")

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        # ACCÈS INSTANTANÉ (Plus de lecture disque)
        image = self.cached_images[idx]
        ske = self.cached_skeletons[idx]

        # On applique les transformations (Data Augmentation) à la volée
        # C'est important de le faire ici pour que le "Random" fonctionne à chaque epoch
        if self.target_transform:
            image = self.target_transform(image)

        return ske, image

    def preprocessSkeleton(self, ske):
        # Cette fonction ne change pas, elle est juste appelée dans le __init__ maintenant
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

class SEBlock(nn.Module):
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

# ============================================================
#           GENERATEUR VECTORIEL (26 -> 128x128)
# ============================================================
class SelfAttention(nn.Module):
    """
    Mécanisme d'attention pour relier des pixels distants.
    """

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


# ============================================================
# 2. MODULE RESIDUAL BLOCK (Optimisé Conv2d)
# ============================================================
class ResidualBlock(nn.Module):
    """
    Bloc résiduel classique mais avec 'ReflectionPad' intégré
    via padding_mode='reflect' pour une meilleure gestion des bords.
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            # Conv 1
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv 2
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


# ============================================================
# 3. GÉNÉRATEUR FINAL (SKELETON -> IMAGE)
# ============================================================
class GenNNSke26ToImage(nn.Module):
    def __init__(self, latent_dim=26):
        super(GenNNSke26ToImage, self).__init__()

        # --- BLOC INITIAL (Latent -> 4x4) ---
        # Projection linéaire + Reshape (Plus stable que ConvTranspose sur vecteur 1D)
        self.fc_init = nn.Linear(latent_dim, 512 * 4 * 4)
        self.bn_init = nn.InstanceNorm2d(512)

        # --- UPSAMPLING (Avec SEBlock intégré) ---

        # 4x4 -> 8x8
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, 1, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(256)  # Utilisation de ta classe existante
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
        # C'est le meilleur endroit : assez de résolution pour voir la forme,
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
        # 1. Input Handling : On s'assure que c'est plat (Batch, 26)
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


class VGGPerceptualLoss(nn.Module):
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

        self.eval() # Important : On met VGG en mode évaluation (désactive dropout interne)

        # OPTIMISATION : On utilise register_buffer
        # Comme ça, mean et std vont sur le GPU tout seuls quand tu fais model.to('cuda')
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # 1. Gestion du Noir & Blanc (si nécessaire)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # 2. CORRECTION CRUCIALE : On passe de [-1, 1] à [0, 1]
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





# ============================================================
#                     GENERATEUR PRINCIPAL
# ============================================================

class ResnetBlock(nn.Module):
    # Le bloc standard des papiers CycleGAN / Pix2PixHD
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            # CORRECTION : on spécifie stride=1 et padding=0 explicitement
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0,bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            # CORRECTION ICI AUSSI
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0,bias=False),
            nn.InstanceNorm2d(dim)
        )

        # AJOUT DU SEBLOCK
        self.se = SEBlock(dim)

    def forward(self, x):
        # On applique le SEBlock sur le chemin résiduel avant l'addition
        out = self.conv_block(x)
        out = self.se(out)
        return x + out

class GenNNSkeImToImage(nn.Module):
    def __init__(self):
        super().__init__()

        # --- ENCODER (Downsampling) ---
        # Le standard: Conv -> InstanceNorm -> ReLU
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, 1, 0,bias=False),  # Grosse convolution au début pour capter l'ensemble
            nn.InstanceNorm2d(64), nn.ReLU(True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1,bias=False),  # Stride 2 = Downsample
            nn.InstanceNorm2d(128), nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1,bias=False),
            nn.InstanceNorm2d(256), nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1,bias=False),
            nn.InstanceNorm2d(512), nn.ReLU(True)
        )

        # --- BOTTLENECK (ResNet) ---
        # Pix2PixHD recommande 6 à 9 blocs ici pour bien comprendre la structure
        # C'est ici que la magie opère
        self.bottleneck = nn.Sequential(
            ResnetBlock(512),
            ResnetBlock(512),
            ResnetBlock(512),
            ResnetBlock(512),  # On en met 4 pour rester léger sur ta 4060
        )

        # --- DECODER (Upsampling "Resize-Conv") ---
        # C'est LA solution "selon les papiers" pour éviter la pixelisation

        # Up 1 : 512 -> 256
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Lisse
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, 1, 0,bias=False),  # Raffine
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        # Up 2 : 256 -> 128
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256 + 256, 128, 3, 1, 0,bias=False),  # +256 à cause du Skip Connection (cat)
            nn.InstanceNorm2d(128),
            nn.ReLU(True,)
        )

        # Up 3 : 128 -> 64
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128 + 128, 64, 3, 1, 0,bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True,)
        )

        # Final
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64 + 64, 3, 7, 1, 0),
            nn.Tanh()
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
        # Attention aux concaténations : on recolle les morceaux de l'encodeur

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
    classname = m.__class__.__name__

    # 1. Convolutions ET Linear (pour les SEBlocks !)
    # On vérifie si c'est une Conv ou une Linear
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

        # Gestion propre du biais (certaines couches n'en ont pas si bias=False)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    # 2. Normalisations (BatchNorm, InstanceNorm, GroupNorm)
    # On ajoute GroupNorm au cas où tu voudrais tester plus tard
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1 or classname.find('GroupNorm') != -1:
        # On ne touche que si affine=True (c'est-à-dire s'il y a des poids)
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class GenVanillaNN:
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        # 128x128 est un bon compromis pour 8GB VRAM. 256x256 peut passer avec batch=8.
        self.image_size = 128
        image_size = 128
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[GPU] Device: {self.device}")

        self.optSkeOrImage = optSkeOrImage

        if optSkeOrImage == 1:
            self.netG = GenNNSke26ToImage().to(self.device)
            src_transform = None
            self.filename = "models/DanceGenVanillaFromSke26.pth"
        else:
            self.netG = GenNNSkeImToImage()
            # 3. OPTIMISATION : Channels Last Memory Format
            # Les RTX 4000 sont beaucoup plus rapides si la mémoire est agencée en (N, H, W, C)
            self.netG = self.netG.to(memory_format=torch.channels_last)
            self.netG = self.netG.to(self.device)
            self.netG.apply(weights_init)
            src_transform = transforms.Compose([
                SkeToImageTransform(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.filename = "models/test123.pth"

        tgt_transform = transforms.Compose([
            # 1. Jitter : On change un peu la lumière/contraste (sur l'image PIL)
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),

            # 2. Conversion en Tenseur
            transforms.ToTensor(),

            # 3. Normalisation : On passe les pixels entre -1 et 1
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

            # 4. ANTI-PAR-CŒUR : Random Erasing (Doit être APRES ToTensor)
            # p=0.5             : 1 chance sur 2 d'effacer une zone
            # scale=(0.02, 0.1) : Le trou fait entre 2% et 10% de l'image
            # value=0           : On remplit avec du gris moyen (0 est le milieu car on est entre -1 et 1)
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ])

        self.dataset = VideoSkeletonDataset(
            videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform
        )

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
        self.netG.eval()
        with torch.no_grad():
            if self.optSkeOrImage == 1:
                ske_input = torch.from_numpy(ske.toarray(reduced=True).flatten()).float()
                ske_input = ske_input.unsqueeze(0).to(self.device)
            else:
                # CORRECTION : On utilise self.image_size (ex: 128) et pas 256 en dur !
                transform = transforms.Compose([
                    SkeToImageTransform(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

                # Le passage en .unsqueeze(0) ajoute la dimension du Batch (1, 3, 128, 128)
                ske_input = transform(ske).unsqueeze(0).to(self.device, memory_format=torch.channels_last)

            with torch.amp.autocast("cuda"):
                output = self.netG(ske_input)

            # Récupération en RGB
            rgb_image = self.dataset.tensor2image(output[0])

            # Conversion RGB -> BGR pour OpenCV
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            return bgr_image

    def train(self, n_epochs=200):  # Je conseille 200 car VGG affine doucement
        print(f"[TRAIN] Début de l'entraînement (L1 + VGG Perceptual) sur {self.device}")

        # --- CONFIGURATION ---
        batch_size = 64  #
        train_size = int(0.85 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

        num_workers = 0

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        # --- VISU: On récupère un squelette fixe UNE SEULE FOIS avant la boucle ---
        fixed_sample_iter = iter(val_loader)
        fixed_sample = next(fixed_sample_iter)
        fixed_ske = fixed_sample[0][0:1].to(self.device, non_blocking=True, memory_format=torch.channels_last)
        print("[TRAIN] Image témoin chargée.")

        if not os.path.exists("images_suivi"):
            os.makedirs("images_suivi")

        # --- INITIALISATION DES LOSS ---
        criterion_l1 = nn.L1Loss()
        print("[TRAIN] Chargement du VGG...")
        criterion_vgg = VGGPerceptualLoss().to(self.device)

        optimizer = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        scaler = torch.cuda.amp.GradScaler()

        best_val_loss = float("inf")

        for epoch in range(n_epochs):
            self.netG.train()
            running_loss = 0.0
            batch_count = 0

            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(self.device, non_blocking=True, memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda"):
                    outputs = self.netG(inputs)

                    # --- NOUVELLE FORMULE DE LOSS ---
                    loss_l1_val = criterion_l1(outputs, targets)
                    loss_vgg_val = criterion_vgg(outputs, targets)

                    # Formule magique : Structure (L1) + Texture (VGG)
                    # On met un poids fort sur L1 pour garder la pose correcte
                    loss = 10.0 * loss_l1_val + 0.5 * loss_vgg_val

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
                    inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                    targets = targets.to(self.device, non_blocking=True, memory_format=torch.channels_last)

                    with torch.amp.autocast("cuda"):
                        outputs = self.netG(inputs)
                        # On valide avec la même métrique
                        l1 = criterion_l1(outputs, targets)
                        vgg = criterion_vgg(outputs, targets)
                        loss = 10.0 * l1 + 0.5 * vgg

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            # --- VISU: Sauvegarde de l'image témoin TOUTES LES 2 EPOCHS (Comme tu voulais) ---
            if (epoch + 1) % 2 == 0:
                with torch.no_grad():
                    fake_img_tensor = self.netG(fixed_ske)
                    img_to_save = self.dataset.tensor2image(fake_img_tensor[0])
                    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                    filename = f"images_suivi/epoch_{epoch + 1}.jpg"
                    cv2.imwrite(filename, img_to_save)
                    print(f"  [Disk] Image sauvegardée : {filename}")
            # ---------------------------------------------------------

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
    GenVanillaNN.verify_alignment(video_file, crop_ratio=1.3,)