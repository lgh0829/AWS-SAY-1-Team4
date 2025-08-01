import torch.nn as nn
import torch
from torch.utils.data import Dataset
from pathlib import Path
import kornia.augmentation as K




class PTTensorDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        # .pt 파일 리스트 (서브폴더명이 레이블)
        self.files   = list(Path(root_dir).rglob('*.pt'))
        self.augment = augment
        if augment:
            # 배치 단위로 처리하는 Kornia 증강기 선언
            self.augs = nn.Sequential(
                K.Resize((224,224)),
                K.RandomRotation(15.0, p=0.5),
                K.RandomAffine(degrees=0.0, translate=(0.1,0.1), p=0.5),
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1) .pt에서 [4, H, W] 텐서와 레이블 로드
        tensor = torch.load(self.files[idx])           # float Tensor
        label  = int(self.files[idx].parent.name)      # 폴더명이 클래스

        # 2) 증강 (학습 시만)
        if self.augment:
            x = tensor.unsqueeze(0)    # [1,4,H,W]
            x = self.augs(x)           # [1,4,224,224]
            tensor = x.squeeze(0)      # [4,224,224]

        # 3) 채널별 정규화
        #   - RGB 채널: ImageNet mean/std
        #   - mask 채널: mean=0, std=1 (변경 없이)
        mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=tensor.device)
        std  = torch.tensor([0.229, 0.224, 0.225, 1.0], device=tensor.device)
        tensor = (tensor - mean[:,None,None]) / std[:,None,None]

        return tensor, label