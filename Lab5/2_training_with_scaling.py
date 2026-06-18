import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"
data_folder = './data'

fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images, tr_targets = fmnist.data, fmnist.targets

val_fmnist = datasets.FashionMNIST(data_folder, download=True, train=False)
val_images, val_targets = val_fmnist.data, val_fmnist.targets

class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float() / 255
        x = x.view(-1, 784)
        self.x, self.y = x, y
    
    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)
    
    def __len__(self):
        return len(self.x)

train_dl = DataLoader(FMNISTDataset(tr_images, tr_targets), batch_size=32, shuffle=True)
val_dl = DataLoader(FMNISTDataset(val_images, val_targets), batch_size=len(val_images), shuffle=False)

model = nn.Sequential(nn.Linear(784, 1000), nn.ReLU(), nn.Linear(1000, 10)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-2)

train_losses, train_accs, val_losses, val_accs = [], [], [], []

print("🚀 Training started...")
for epoch in range(5):
    print(f"Epoch {epoch+1}/5...", flush=True)
    model.train()
    epoch_losses = []
    for x, y in train_dl:
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_losses.append(loss.item())
    
    train_losses.append(np.mean(epoch_losses))
    
    model.eval()
    with torch.no_grad():
        train_correct = []
        for x, y in train_dl:
            pred = model(x)
            train_correct.extend((pred.argmax(-1) == y).cpu().numpy())
        train_accs.append(np.mean(train_correct))
        
        for x, y in val_dl:
            pred = model(x)
            val_loss_val = loss_fn(pred, y)
            val_correct = (pred.argmax(-1) == y).cpu().numpy()
        val_losses.append(val_loss_val.item())
        val_accs.append(np.mean(val_correct))
    
    print(f"  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2%}", flush=True)
    print(f"  Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.2%}", flush=True)

epochs = np.arange(5) + 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs, train_losses, 'o-', label='Train'), ax1.plot(epochs, val_losses, 's-', label='Val')
ax1.set(title='Loss', xlabel='Epoch', ylabel='Loss'), ax1.legend(), ax1.grid(True, alpha=0.3)
ax2.plot(epochs, train_accs, 'o-', label='Train'), ax2.plot(epochs, val_accs, 's-', label='Val')
ax2.set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy'), ax2.legend(), ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_with_scaling.png', dpi=100)
print("✓ Saved: training_with_scaling.png", flush=True)
plt.show()
