import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("Welcome to Image Restoration System for Surveillance Images")
os.makedirs("outputs", exist_ok=True)

img = cv2.imread("surveillance1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/original.jpg", gray)
# Gaussian Noise
gaussian_noise = gray + np.random.normal(0, 25, gray.shape)
gaussian_noise = np.clip(gaussian_noise, 0, 255).astype(np.uint8)

# Salt & Pepper Noise
sp_noise = gray.copy()
prob = 0.02
rand = np.random.rand(*gray.shape)

sp_noise[rand < prob/2] = 0
sp_noise[rand > 1 - prob/2] = 255

cv2.imwrite("outputs/gaussian_noise.jpg", gaussian_noise)
cv2.imwrite("outputs/sp_noise.jpg", sp_noise)

# Mean Filter
mean_gauss = cv2.blur(gaussian_noise, (3, 3))
mean_sp = cv2.blur(sp_noise, (3, 3))

# Median Filter
median_gauss = cv2.medianBlur(gaussian_noise, 3)
median_sp = cv2.medianBlur(sp_noise, 3)

# Gaussian Filter
gauss_gauss = cv2.GaussianBlur(gaussian_noise, (3, 3), 0)
gauss_sp = cv2.GaussianBlur(sp_noise, (3, 3), 0)

cv2.imwrite("outputs/mean_gauss.jpg", mean_gauss)
cv2.imwrite("outputs/median_gauss.jpg", median_gauss)
cv2.imwrite("outputs/gauss_gauss.jpg", gauss_gauss)

cv2.imwrite("outputs/mean_sp.jpg", mean_sp)
cv2.imwrite("outputs/median_sp.jpg", median_sp)
cv2.imwrite("outputs/gauss_sp.jpg", gauss_sp)


def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(m))

print("\n--- Performance Metrics ---")

filters = {
    "Mean (Gaussian Noise)": mean_gauss,
    "Median (Gaussian Noise)": median_gauss,
    "Gaussian (Gaussian Noise)": gauss_gauss,
    "Mean (S&P Noise)": mean_sp,
    "Median (S&P Noise)": median_sp,
    "Gaussian (S&P Noise)": gauss_sp,
}

for name, img_restored in filters.items():
    print(f"{name} → MSE: {mse(gray, img_restored):.2f}, PSNR: {psnr(gray, img_restored):.2f} dB")
titles = [
    "Original", "Gaussian Noise", "Salt & Pepper Noise",
    "Mean Restored", "Median Restored", "Gaussian Restored"
]

images = [
    gray, gaussian_noise, sp_noise,
    median_gauss, median_sp, gauss_sp
]

plt.figure(figsize=(10, 6))

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/comparison.png")
plt.show()

print("\nAnalysis:")
print("• Median filter performs best for Salt-and-Pepper noise.")
print("• Gaussian filter performs well for Gaussian noise.")
print("• Mean filter causes more blurring and detail loss.")
