import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
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

torch.set_default_dtype(torch.float32)

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
        print(
            "VideoSkeletonDataset: ",
            "ske_reduced=",
            ske_reduced,
            "=(",
            Skeleton.reduced_dim,
            " or ",
            Skeleton.full_dim,
            ")",
        )

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)

        image = Image.open(self.videoSke.imagePath(idx))
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
        numpy_image = normalized_image.detach().cpu().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = (numpy_image * 0.5 + 0.5) * 255
        return numpy_image.astype(np.uint8)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GenNNSke26ToImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim

        self.fc = nn.Sequential(
            nn.Linear(26, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

        print("[GenNNSke26ToImage] Architecture AMELIOREE chargée")

    def forward(self, z):
        z = z.view(z.size(0), -1)
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        img = self.conv(x)
        return img


class GenNNSkeImToImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.enc1_bn = nn.BatchNorm2d(64)

        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc2_bn = nn.BatchNorm2d(128)

        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.enc3_bn = nn.BatchNorm2d(256)

        self.enc4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.enc4_bn = nn.BatchNorm2d(512)

        self.dec1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec1_bn = nn.BatchNorm2d(256)

        self.dec2 = nn.ConvTranspose2d(256 + 256, 128, 4, 2, 1)
        self.dec2_bn = nn.BatchNorm2d(128)

        self.dec3 = nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1)
        self.dec3_bn = nn.BatchNorm2d(64)

        self.dec4 = nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1)
        self.dec4_bn = nn.BatchNorm2d(32)

        self.final = nn.Conv2d(32, 3, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        print("[GenNNSkeImToImage] Architecture U-Net AMELIOREE chargée")

    def forward(self, x):
        e1 = self.lrelu(self.enc1_bn(self.enc1(x)))
        e2 = self.lrelu(self.enc2_bn(self.enc2(e1)))
        e3 = self.lrelu(self.enc3_bn(self.enc3(e2)))
        e4 = self.lrelu(self.enc4_bn(self.enc4(e3)))

        d1 = self.relu(self.dec1_bn(self.dec1(e4)))
        d1 = torch.cat([d1, e3], dim=1)

        d2 = self.relu(self.dec2_bn(self.dec2(d1)))
        d2 = torch.cat([d2, e2], dim=1)

        d3 = self.relu(self.dec3_bn(self.dec3(d2)))
        d3 = torch.cat([d3, e1], dim=1)

        d4 = self.relu(self.dec4_bn(self.dec4(d3)))

        out = self.tanh(self.final(d4))
        return out


class GenVanillaNN:
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[GPU] Device: {self.device}")
        if torch.cuda.is_available():
            print(f"[GPU] Name: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        self.optSkeOrImage = optSkeOrImage

        if optSkeOrImage == 1:
            self.netG = GenNNSke26ToImage().to(self.device)
            src_transform = None
            self.filename = "models/DanceGenVanillaFromSke26.pth"
        else:
            self.netG = GenNNSkeImToImage().to(self.device)
            src_transform = transforms.Compose(
                [
                    SkeToImageTransform(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            self.filename = "models/DanceGenVanillaFromSkeim.pth"

        tgt_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dataset = VideoSkeletonDataset(
            videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform
        )

        # Dataloader “général” (peu utilisé dans ton train(), mais on le met propre)
        base_workers = 12 if torch.cuda.is_available() else 2
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=512,
            shuffle=True,
            num_workers=base_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            try:
                state_dict = torch.load(self.filename, map_location=self.device)

                if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace("_orig_mod.", "")
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict

                self.netG.load_state_dict(state_dict)
                print("[✓] Modèle chargé avec succès")
            except Exception as e:
                print(f"[✗] Erreur chargement: {e}")
                print("[!] Le modèle sera réentraîné")

        print(f"[GenVanillaNN] Mode: {'Vecteur->Image' if optSkeOrImage == 1 else 'Image->Image'}")
        print(f"[GenVanillaNN] Data Augmentation activée!")

    def train(self, n_epochs=20):
        print(f"[TRAIN] Début de l'entraînement sur {self.device}")
        print(f"[TRAIN] Dataset: {len(self.dataset)} frames")

        train_size = int(0.85 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

        batch_size = 512
        print(f"[TRAIN] Train: {train_size} | Val: {val_size}")
        print(f"[TRAIN] Batch size: {batch_size}")

        num_workers = 12 if torch.cuda.is_available() else 2

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        print(f"[TRAIN] Nombre de batches train: {len(train_loader)}")
        print(f"[TRAIN] Workers: {num_workers} threads")
        print(f"[TRAIN] Epochs demandés: {n_epochs}")
        print("-" * 60)

        criterion_mse = nn.MSELoss()
        criterion_l1 = nn.L1Loss()

        optimizer = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )

        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        if use_amp:
            print("[OPTIM] Mixed Precision (AMP) activée")

        best_val_loss = float("inf")

        for epoch in range(n_epochs):
            self.netG.train()
            running_loss = 0.0
            batch_count = 0

            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.netG(inputs)
                        loss_mse = criterion_mse(outputs, targets)
                        loss_l1 = criterion_l1(outputs, targets)
                        loss = 0.6 * loss_mse + 0.4 * loss_l1

                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.netG(inputs)
                    loss_mse = criterion_mse(outputs, targets)
                    loss_l1 = criterion_l1(outputs, targets)
                    loss = 0.6 * loss_mse + 0.4 * loss_l1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
                    optimizer.step()

                running_loss += loss.item()
                batch_count += 1

                if (i + 1) % 10 == 0:
                    print(
                        f"  [Epoch {epoch+1}/{n_epochs}] "
                        f"Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.5f}"
                    )

            avg_train_loss = running_loss / batch_count

            # ---------- Validation ----------
            self.netG.eval()
            val_loss = 0.0
            val_count = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            outputs = self.netG(inputs)
                            loss_mse = criterion_mse(outputs, targets)
                            loss_l1 = criterion_l1(outputs, targets)
                            loss = 0.6 * loss_mse + 0.4 * loss_l1
                    else:
                        outputs = self.netG(inputs)
                        loss_mse = criterion_mse(outputs, targets)
                        loss_l1 = criterion_l1(outputs, targets)
                        loss = 0.6 * loss_mse + 0.4 * loss_l1

                    val_loss += loss.item()
                    val_count += 1

            avg_val_loss = val_loss / val_count

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.netG.state_dict(), self.filename)
                print(f"  [✓] Meilleur modèle sauvegardé (val_loss: {avg_val_loss:.5f})")

            print(
                f"[EPOCH {epoch+1}/{n_epochs} TERMINÉ] "
                f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}"
            )
            print("-" * 60)

        print("[TRAIN] Entraînement terminé avec succès!")
        print(f"[TRAIN] Meilleur Val Loss: {best_val_loss:.5f}")
        print(f"[TRAIN] Modèle sauvegardé dans {self.filename}")

    def generate(self, ske):
        self.netG.eval()
        with torch.no_grad():
            if self.optSkeOrImage == 1:
                ske_input = torch.from_numpy(ske.toarray(reduced=True).flatten()).float()
                ske_input = ske_input.unsqueeze(0).to(self.device)
            else:
                transform = transforms.Compose(
                    [
                        SkeToImageTransform(64),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
                ske_input = transform(ske).unsqueeze(0).to(self.device)

            output = self.netG(ske_input)
            res = self.dataset.tensor2image(output[0])
            return res


if __name__ == "__main__":
    force = False
    optSkeOrImage = 2
    n_epoch = 200
    train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/raw/taichi1.mp4"

    print("GenVanillaNN: Filename=", filename)
    targetVideoSke = VideoSkeleton(filename)

    if train:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=optSkeOrImage)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True, optSkeOrImage=optSkeOrImage)

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        image = cv2.resize(image, (256, 256))
        cv2.imshow("Image", image)
        key = cv2.waitKey(-1)
