from torchvision.datasets import ImageFolder
from PIL import Image
import os
import torch
import random
from torchvision import transforms

class FourChannelImageFolder(ImageFolder):
    def __init__(self, root, mask_root, train=False):
        super().__init__(root)
        self.mask_root = mask_root
        self.train = train

        # 공통: Resize
        self.resize = transforms.Resize((224, 224))

        # 증강 (train 전용)
        self.augment = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.2),
        ])

        # ToTensor + Normalize (4채널 기준)
        self.to_tensor_and_norm = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], 
                                 std=[0.229, 0.224, 0.225, 0.5])
        ])

    def __getitem__(self, index):
        path, target = self.samples[index]

        # Load image and mask
        img = Image.open(path).convert('RGB')
        rel_path = os.path.relpath(path, self.root)
        mask_path = os.path.join(self.mask_root, rel_path)
        mask = Image.open(mask_path).convert('L')

        # 공통 resize
        img = self.resize(img)
        mask = self.resize(mask)

        if self.train:
            # 동일한 seed로 증강 동기화
            seed = random.randint(0, 99999)
            random.seed(seed)
            img = self.augment(img)
            random.seed(seed)
            mask = self.augment(mask)

        img = transforms.ToTensor()(img)       # (3, H, W)
        mask = transforms.ToTensor()(mask)     # (1, H, W)
        img_4ch = torch.cat((img, mask), dim=0)  # (4, H, W)

        img_4ch = self.to_tensor_and_norm(img_4ch)
        return img_4ch, target