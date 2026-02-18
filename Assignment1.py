import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("Welcome to Smart Document Scanner & Quality Analysis System")
os.makedirs("outputs", exist_ok=True)

# ---------- Image Acquisition ----------
img = cv2.imread("document1.jpg")   
img = cv2.resize(img, (512, 512))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("outputs/grayscale.jpg", gray)

# ---------- Image Sampling ----------
high_res = gray.copy()

medium = cv2.resize(gray, (256, 256))
medium_res = cv2.resize(medium, (512, 512))

low = cv2.resize(gray, (128, 128))
low_res = cv2.resize(low, (512, 512))

cv2.imwrite("outputs/high_res.jpg", high_res)
cv2.imwrite("outputs/medium_res.jpg", medium_res)
cv2.imwrite("outputs/low_res.jpg", low_res)

# ---------- Image Quantization ----------
quant_256 = gray
quant_16 = np.floor(gray / 16) * 16
quant_4 = np.floor(gray / 64) * 64

cv2.imwrite("outputs/quant_256.jpg", quant_256)
cv2.imwrite("outputs/quant_16.jpg", quant_16)
cv2.imwrite("outputs/quant_4.jpg", quant_4)

titles = [
    "Original", "Grayscale",
    "512x512", "256x256", "128x128",
    "256 Levels", "16 Levels", "4 Levels"
]

images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gray,
    high_res, medium_res, low_res,
    quant_256, quant_16, quant_4
]

plt.figure(figsize=(12, 8))

for i in range(len(images)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/comparison.png")
plt.show()
print("\nQuality Observations:")
print("• High resolution preserves sharp edges and clear text.")
print("• Medium resolution slightly reduces clarity but remains readable.")
print("• Low resolution causes major text blur and poor OCR performance.")
print("• Lower gray levels create banding and remove fine details.")
print("• Best OCR quality occurs at high resolution with 256 gray levels.")
