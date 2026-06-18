import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
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

def train_model(lr):
    print(f"  Training with learning_rate={lr}...")
    train_dl = DataLoader(FMNISTDataset(tr_images, tr_targets), batch_size=32, shuffle=True)
    val_dl = DataLoader(FMNISTDataset(val_images, val_targets), batch_size=len(val_images), shuffle=False)
    
    model = nn.Sequential(nn.Linear(784, 1000), nn.ReLU(), nn.Linear(1000, 10)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    
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
    
    return model, val_losses[::len(val_losses)//5], val_accs[::len(val_accs)//5]

print("🚀 Comparing Learning Rates...")
lrs = [0.1, 0.01, 0.001, 0.0001]
results = {lr: train_model(lr) for lr in lrs}

print("📊 Creating plots...")
epochs = np.arange(5) + 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for lr in lrs:
    model, losses, accs = results[lr]
    ax1.plot(epochs, losses, 'o-', label=f'LR={lr}')
    ax2.plot(epochs, accs, 'o-', label=f'LR={lr}')
ax1.set(title='Validation Loss', xlabel='Epoch', ylabel='Loss'), ax1.legend(), ax1.grid(True, alpha=0.3)
ax2.set(title='Validation Accuracy', xlabel='Epoch', ylabel='Accuracy'), ax2.legend(), ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_rate_comparison.png', dpi=100)
print("✓ Saved: learning_rate_comparison.png", flush=True)

# Weight distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, (lr, (model, _, _)) in enumerate(results.items()):
    ax = axes[i//2, i%2]
    weights = model[0].weight.data.cpu().numpy().flatten()
    ax.hist(weights, bins=50, edgecolor='black', alpha=0.7)
    ax.set_title(f'Weight Distribution - LR={lr}')
    ax.set_xlabel('Weight Value'), ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('weight_distribution.png', dpi=100)
print("✓ Saved: weight_distribution.png", flush=True)
plt.show()
