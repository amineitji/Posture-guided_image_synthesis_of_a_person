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




class GenNNSke26ToImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim
        self.fc = nn.Sequential(
            nn.Linear(26, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512 * 4 * 4), nn.BatchNorm1d(512 * 4 * 4), nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), -1)
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        img = self.conv(x)
        return img


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # On utilise VGG16 (plus léger que le 19 pour ta 4060)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # On découpe le réseau pour récupérer les textures à différents niveaux
        self.blocks = nn.ModuleList([
            vgg[:4],  # Niveau 1 : Traits fins
            vgg[4:9],  # Niveau 2 : Formes
            vgg[9:16],  # Niveau 3 : Textures complexes
            vgg[16:23]  # Niveau 4 : Structure globale
        ])

        # On gèle les poids (VGG ne s'entraîne pas, il juge seulement)
        for param in self.parameters():
            param.requires_grad = False

        # Normalisation standard ImageNet (Indispensable pour que VGG comprenne l'image)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, input, target):
        # Si l'image est en Noir & Blanc, on la duplique pour faire du RGB
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # On normalise les images
        input = (input - self.mean.to(input.device)) / self.std.to(input.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        loss = 0.0
        x = input
        y = target

        # On compare les "features" à chaque étape
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)

        return loss


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


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            # OPTIMISATION : padding_mode='reflect' intégré + bias=False
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),  # On garde ReLU dans le bottleneck (standard ResNet)

            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GenNNSkeImToImage(nn.Module):
    def __init__(self):
        super().__init__()

        # --- ENCODER (LeakyReLU + bias=False) ---

        # Init : 3 -> 64
        self.enc1 = nn.Sequential(
            # Conv 7x7 avec reflection padding intégré
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)  # LeakyReLU est mieux pour l'encodeur
        )

        # Down 1 : 64 -> 128
        self.down1 = nn.Sequential(
            # Stride 2 = Downsample
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Down 2 : 128 -> 256
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Down 3 : 256 -> 512
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- BOTTLENECK ---
        # 4 blocs pour rester léger
        self.bottleneck = nn.Sequential(
            ResnetBlock(512),
            ResnetBlock(512),
            ResnetBlock(512),
            ResnetBlock(512),
        )

        # --- DECODER (ReLU + bias=False) ---
        # On garde ReLU ici pour nettoyer le signal avant la sortie

        # Up 1 : 512 -> 256
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Up 2 : 256 (cat) -> 128
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # Input 512 car concatenation (256 venant du haut + 256 venant de x3)
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Up 3 : 128 (cat) -> 64
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # Input 256 car concatenation (128 venant du haut + 128 venant de x2)
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final : 64 (cat) -> 3 RGB
        self.final = nn.Sequential(
            # Input 128 car concatenation (64 venant du haut + 64 venant de x1)
            # Pas de bias=False ici car pas de Norm derrière (Tanh direct)
            nn.Conv2d(128, 3, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
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

        # Decoder + Skip Connections
        u1 = self.up1(b)  # 256

        u1_cat = torch.cat([u1, x3], dim=1)
        u2 = self.up2(u1_cat)  # 128

        u2_cat = torch.cat([u2, x2], dim=1)
        u3 = self.up3(u2_cat)  # 64

        u3_cat = torch.cat([u3, x1], dim=1)
        output = self.final(u3_cat)

        return output

def weights_init(m):
    classname = m.__class__.__name__
    # Pour les Convolutions, on met une loi normale (moyenne 0, écart-type 0.02)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    # Pour le BatchNorm/InstanceNorm, on initialise à 1 (pas de changement) et biais 0
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        # Note : InstanceNorm n'a parfois pas de paramètres learnable,
        # mais si 'affine=True' est mis, cela aide.
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
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
            # Si VideoSkeleton sort déjà du 128x128, ceci est juste une sécurité
            #transforms.Resize((image_size, image_size)),

            # On garde le Jitter car il aide l'IA à gérer l'éclairage sans casser la pose
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),

            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

        # --- ATTENTION : Batch Size réduit pour VGG ---
        # VGG consomme beaucoup de VRAM. Sur RTX 4060, on met 16.
        # Si tu as une erreur "Out of Memory", descends à 8.
        batch_size =64

        train_size = int(0.85 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

        # num_workers=0 car tes données sont en RAM (Dataset optimisé)
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
        print("[TRAIN] Image témoin chargée pour la visualisation")

        # Création du dossier de suivi s'il n'existe pas
        if not os.path.exists("images_suivi"):
            os.makedirs("images_suivi")
        # ---------------------------------------------------------

        # --- INITIALISATION DES LOSS ---
        criterion_l1 = nn.L1Loss()

        # On initialise le Juge VGG
        print("[TRAIN] Chargement du VGG pour la netteté...")
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
                    loss = 10.0 * loss_l1_val + 1.0 * loss_vgg_val

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
                        loss = 10.0 * l1 + 1.0 * vgg

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            # --- VISU: Sauvegarde de l'image témoin TOUTES LES 2 EPOCHS (Comme tu voulais) ---
            if (epoch + 1) % 2 == 0:
                with torch.no_grad():
                    # 1. Génération
                    fake_img_tensor = self.netG(fixed_ske)

                    # 2. Conversion Tensor -> Numpy RGB
                    img_to_save = self.dataset.tensor2image(fake_img_tensor[0])

                    # 3. Conversion RGB -> BGR (Obligatoire pour OpenCV)
                    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)

                    # 4. Sauvegarde directe
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

            # On redimensionne un peu pour bien voir sur ton écran (zoom x2)
            h, w = image_cv.shape[:2]
            image_zoom = cv2.resize(image_cv, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)

            # Affichage
            cv2.imshow("Verification Alignement", image_zoom)

            key = cv2.waitKey(0)  # Attend une touche indéfiniment
            if key & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Mets ici le chemin de ta vidéo
    video_file = "../data/raw/taichi1.mp4"

    # Mets ici LE MÊME ratio que celui que tu veux utiliser (ex: 1.3 ou 1.5)
    GenVanillaNN.verify_alignment(video_file, crop_ratio=1.3)