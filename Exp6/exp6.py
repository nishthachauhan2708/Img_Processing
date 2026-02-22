import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Load image
image = cv2.imread('image.tif', 0)
image = np.float32(image)

# 2️⃣ Add Periodic Noise
rows, cols = image.shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)

periodic_noise = 30 * np.sin(2 * np.pi * X / 30)
noisy_image = image + periodic_noise

# 3️⃣ FFT
F = np.fft.fft2(noisy_image)
Fshift = np.fft.fftshift(F)

# 4️⃣ Assume simple degradation function
H = np.ones_like(Fshift)
H[rows//2-5:rows//2+5, cols//2-5:cols//2+5] = 0.5

# ----------------------------
# 🔵 Inverse Filtering
# ----------------------------
H_inv = np.where(H == 0, 0.01, H)
F_inverse = Fshift / H_inv
img_inverse = np.abs(np.fft.ifft2(np.fft.ifftshift(F_inverse)))

# ----------------------------
# 🔴 Wiener Filtering
# ----------------------------
K = 0.01
H_conj = np.conj(H)
F_wiener = (H_conj / (np.abs(H)**2 + K)) * Fshift
img_wiener = np.abs(np.fft.ifft2(np.fft.ifftshift(F_wiener)))

# 5️⃣ Display
plt.figure(figsize=(12,8))

plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(222)
plt.imshow(noisy_image, cmap='gray')
plt.title("Periodic Noisy Image")
plt.axis('off')

plt.subplot(223)
plt.imshow(img_inverse, cmap='gray')
plt.title("Inverse Filter Output")
plt.axis('off')

plt.subplot(224)
plt.imshow(img_wiener, cmap='gray')
plt.title("Wiener Filter Output")
plt.axis('off')

plt.tight_layout()
plt.show()
