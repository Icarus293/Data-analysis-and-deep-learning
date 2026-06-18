import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

print("🚀 Downloading Fashion MNIST...")
# Tải bộ dữ liệu Fashion MNIST
data_folder = './data'
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets

print(f"✓ Loaded: {tr_images.shape}")
print(f'Classes: {fmnist.classes}')

# Vẽ 10 mẫu từ mỗi lớp
print("📊 Creating plot...")
fig, ax = plt.subplots(10, 10, figsize=(10, 10))
for label_class, plot_row in enumerate(ax):
    label_x_rows = np.where(tr_targets == label_class)[0]
    for plot_cell in plot_row:
        plot_cell.grid(False)
        plot_cell.axis('off')
        ix = np.random.choice(label_x_rows)
        plot_cell.imshow(tr_images[ix], cmap='gray')

plt.tight_layout()
plt.savefig('fashion_mnist_samples.png', dpi=100, bbox_inches='tight')
print("✓ Saved: fashion_mnist_samples.png", flush=True)
plt.show()
