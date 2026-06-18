import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
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

def train_model(batch_size):
    print(f"  Training with batch_size={batch_size}...")
    train_dl = DataLoader(FMNISTDataset(tr_images, tr_targets), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(FMNISTDataset(val_images, val_targets), batch_size=len(val_images), shuffle=False)
    
    model = nn.Sequential(nn.Linear(784, 1000), nn.ReLU(), nn.Linear(1000, 10)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)
    
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    for epoch in range(5):
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
                train_correct.extend((model(x).argmax(-1) == y).cpu().numpy())
            train_accs.append(np.mean(train_correct))
            
            for x, y in val_dl:
                pred = model(x)
                val_correct = (pred.argmax(-1) == y).cpu().numpy()
            val_losses.append(loss_fn(pred, y).item())
            val_accs.append(np.mean(val_correct))
    
    return train_losses, train_accs, val_losses, val_accs

print("🚀 Comparing Batch Sizes...")
batch_sizes = [32, 128, 512]
results = {bs: train_model(bs) for bs in batch_sizes}

print("📊 Creating plots...")
epochs = np.arange(5) + 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for bs in batch_sizes:
    ax1.plot(epochs, results[bs][2], 'o-', label=f'BS={bs}')
    ax2.plot(epochs, results[bs][3], 'o-', label=f'BS={bs}')
ax1.set(title='Validation Loss', xlabel='Epoch', ylabel='Loss'), ax1.legend(), ax1.grid(True, alpha=0.3)
ax2.set(title='Validation Accuracy', xlabel='Epoch', ylabel='Accuracy'), ax2.legend(), ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('batch_size_comparison.png', dpi=100)
print("✓ Saved: batch_size_comparison.png", flush=True)
plt.show()
