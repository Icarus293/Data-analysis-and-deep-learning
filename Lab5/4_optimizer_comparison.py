import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
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

def train_model(opt_name, lr):
    print(f"  Training with {opt_name} (lr={lr})...")
    train_dl = DataLoader(FMNISTDataset(tr_images, tr_targets), batch_size=32, shuffle=True)
    val_dl = DataLoader(FMNISTDataset(val_images, val_targets), batch_size=len(val_images), shuffle=False)
    
    model = nn.Sequential(nn.Linear(784, 1000), nn.ReLU(), nn.Linear(1000, 10)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr) if opt_name == 'SGD' else Adam(model.parameters(), lr=lr)
    
    val_losses, val_accs = [], []
    for epoch in range(5):
        model.train()
        for x, y in train_dl:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x)
                val_losses.append(loss_fn(pred, y).item())
                val_accs.append((pred.argmax(-1) == y).float().mean().item())
    
    return val_losses[::len(val_losses)//5], val_accs[::len(val_accs)//5]

print("🚀 Comparing Optimizers...")
configs = [('SGD', 0.1), ('SGD', 0.01), ('Adam', 0.01), ('Adam', 0.001)]
results = {f"{opt}_{lr}": train_model(opt, lr) for opt, lr in configs}

print("📊 Creating plots...")
epochs = np.arange(5) + 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for key, (losses, accs) in results.items():
    ax1.plot(epochs, losses, 'o-', label=key)
    ax2.plot(epochs, accs, 'o-', label=key)
ax1.set(title='Validation Loss', xlabel='Epoch', ylabel='Loss'), ax1.legend(), ax1.grid(True, alpha=0.3)
ax2.set(title='Validation Accuracy', xlabel='Epoch', ylabel='Accuracy'), ax2.legend(), ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=100)
print("✓ Saved: optimizer_comparison.png", flush=True)
plt.show()
