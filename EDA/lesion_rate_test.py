import os
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms as transforms


# 클래스 이름
class_names = ['Normal', 'Pneumonia', 'Other abnormal']

# 3. Guided ReLU
class GuidedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        ctx.save_for_backward(positive_mask)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        positive_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[positive_mask == 0] = 0
        grad_input[grad_output < 0] = 0
        return grad_input

class GuidedReLUWrapper(nn.Module):
    def forward(self, x):
        return GuidedReLU.apply(x)

def replace_relu_with_guided_relu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, GuidedReLUWrapper())
        else:
            replace_relu_with_guided_relu(module)

# 4. Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx

# 5. Guided Backprop
class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        replace_relu_with_guided_relu(self.model)

    def __call__(self, x, class_idx=None):
        x = x.requires_grad_()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        return x.grad[0].cpu().numpy()

# 6. 시각화
def visualize(original_image, guided_gradcam, predicted_class_name):
    guided_gradcam = np.moveaxis(guided_gradcam, 0, -1)
    guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (guided_gradcam.max() - guided_gradcam.min())
    guided_gradcam = np.uint8(255 * guided_gradcam)
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.imshow(guided_gradcam)
    plt.title(f"Guided Grad-CAM\n(Predicted: {predicted_class_name})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# 이미지 전처리
def preprocess_image(img_path):
    # 1. Grayscale
    img = Image.open(img_path).convert("L")

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    np_img = np.array(img)
    equalized = clahe.apply(np_img)
    clahe_img = Image.fromarray(equalized)

    # 3. Gaussian Blur
    blurred = clahe_img.filter(ImageFilter.GaussianBlur(radius=1.0))

    # 4. Sharpening
    sharpened = blurred.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


    # 5. Convert to 3-channel RGB
    rgb_img = Image.merge("RGB", (sharpened, sharpened, sharpened))

    # 6. Resize & Tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(rgb_img).unsqueeze(0), rgb_img



def overlay_cam_on_image(original_image, cam):
    """
    original_image: PIL.Image (RGB)
    cam: 2D numpy array (normalized between 0~1)
    """
    # Grad-CAM 크기를 원본 이미지와 맞춤
    cam_resized = cv2.resize(cam, original_image.size)

    # 색상 맵 적용 (Jet 컬러맵 → 빨-노-파)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 원본 이미지와 겹치기 (50%씩 반영)
    original_np = np.array(original_image)
    overlay = np.uint8(0.5 * heatmap + 0.5 * original_np)
    return overlay


def visualize_overlay(original_image, cam, predicted_class_name):
    """
    original_image: PIL.Image (RGB)
    cam: Grad-CAM 결과 (2D numpy array)
    predicted_class_name: 예측된 클래스명 (str)
    """
    overlay = overlay_cam_on_image(original_image, cam)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"Guided Grad-CAM Overlay\n(Predicted: {predicted_class_name})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    
def overlay_cam_on_mask(mask_img, cam, threshold=0.5):
    """
    mask_img: 폐 마스크 이미지 (PIL Image or np.ndarray)
    cam: normalized cam (0~1)
    threshold: 보여줄 최소 임계값 (0.0~1.0)
    """
    # PIL → numpy 변환
    if isinstance(mask_img, Image.Image):
        mask_np = np.array(mask_img.convert("L"))
    else:
        mask_np = mask_img

    # float형일 수 있으니 확실하게 정수형으로 변환
    if mask_np.max() <= 1.0:
        mask_np = (mask_np * 255).astype(np.uint8)

    # CAM resize
    cam_resized = cv2.resize(cam, (mask_np.shape[1], mask_np.shape[0]))  # (width, height)

    # 임계값 이하 제거
    cam_thresh = np.where(cam_resized >= threshold, cam_resized, 0)

    # ColorMap 적용
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_thresh), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # 흑백 mask → 3채널 gray RGB
    gray_rgb = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)

    # Overlay
    overlay = np.uint8(0.5 * heatmap + gray_rgb)
    

    return overlay


def visualize_overlay_on_mask(mask_img, cam, predicted_class_name, threshold=0.5):
    overlay = overlay_cam_on_mask(mask_img, cam, threshold)
    plt.figure(figsize=(4, 4))
    plt.imshow(overlay, cmap='gray')
    # plt.title(f"Grad-CAM on Mask (≥ {threshold})\nPredicted: {predicted_class_name}")
    # plt.axis("off")
    plt.tight_layout()
    plt.show()
