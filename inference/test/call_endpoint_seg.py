import boto3
import json
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 설정
endpoint_name = "say1-4team-hardseg-25-08-01-13-13"
runtime = boto3.client("sagemaker-runtime")

# 1. 이미지 불러오기
image_path = '00000710_003.png'
image = Image.open(image_path).convert('RGB')
# 원본 해상도 그대로 전송 → inference.py에서 Resize 수행
buffer = io.BytesIO()
image.save(buffer, format="JPEG")
image_bytes = buffer.getvalue()

# # 1-1. 512x512 RGB 더미 이미지 생성
# dummy_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
# dummy_image = Image.fromarray(dummy_array, mode='RGB')

# # 1-2. JPEG 바이트로 변환
# buffer = io.BytesIO()
# dummy_image.save(buffer, format="JPEG")
# image_bytes = buffer.getvalue()

# 2. JSON payload 생성 (바이트 배열을 리스트로 변환)
payload = json.dumps({
    "image": list(image_bytes)  # 바이트 배열을 list of ints로 변환
})

# 3. 엔드포인트 호출
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=payload
)

# 4. 결과 파싱
result = json.loads(response['Body'].read().decode())
print("Inference result:", result)
print("Mask shape:", np.array(result["mask"]).shape)

# 마스크 두 채널 합성 (OR 연산 방식)
mask = [np.array(result["mask"][0]), np.array(result["mask"][1])]
combined_mask = ((mask[0] + mask[1]) > 0).astype(np.uint8)  # 이진 마스크 (0 또는 1)

# 저장을 위해 0~255로 변환
masked_image = combined_mask * 255

# 이미지 저장
cv2.imwrite("output_mask.png", masked_image)
print("Saved: output_mask.png")

## 5. 시각화 (원본 이미지와 마스크 오버레이)
# 원본 이미지 로드 (1024x1024 assumed)
original_image = Image.open(image_path).convert("RGB")
original_np = np.array(original_image).astype(np.float32) / 255.0  # (1024, 1024, 3)

# 마스크 로드 (512x512 → float32)
# mask_array: shape (2, 512, 512), 이미 받아온 상태
combined_mask = ((mask0 + mask1) > 0).astype(np.uint8)  # shape: (512, 512)

# 마스크 업샘플링 to 1024x1024
mask_resized = cv2.resize(combined_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
mask_norm = mask_resized.astype(np.float32)  # 0.0 또는 1.0

# 3채널 마스크 생성
mask_3ch = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)

# 오버레이 생성
overlay = 0.6 * original_np + 0.4 * mask_3ch
overlay = np.clip(overlay, 0, 1)

# 6. 저장
cv2.imwrite("overlay_on_original.png", (overlay * 255).astype(np.uint8))


# masked = np.array((result["mask"][0] + result["mask"][1]) * 127)  # 마스크 합성
# cv2.imwrite("output_mask.png", masked.astype(np.uint8))

# mask_array = np.array(result["mask"][1])  # [1]: 폐 영역

# # 5. 시각화 (원본 이미지와 마스크 오버레이)
# # image_np = np.array(dummy_image.resize((512, 512))).astype(np.float32) / 255.0
# image_np = np.array(image.resize((512, 512))).astype(np.float32) / 255.0
# mask_norm = np.expand_dims(mask_array, axis=2)
# mask_3ch = np.repeat(mask_norm, 3, axis=2)
# overlay = (0.6 * image_np + 0.4 * mask_3ch)
# overlay = np.clip(overlay, 0, 1)

# plt.figure(figsize=(8, 8))
# plt.title("Segmentation Mask Overlay")
# plt.imshow(overlay)
# plt.axis('off')
# plt.savefig("segmentation_overlay.png", bbox_inches='tight')
# print("✅ 시각화 저장 완료: segmentation_overlay.png")