import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vae_model import ConvVAE
import os

# --- CONFIGURATION ---
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models/vae_mnist_conv.pth'

def loss_function(recon_x, x, mu, logvar):
    # BCE = Reconstruction Loss (how blurry?)
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KLD = Regularization Loss (how organized is the latent space?)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train():
    # 1. Setup Data
    print("⬇Checking for MNIST dataset...")
    transform = transforms.ToTensor()
    # num_workers=0 is safer for Windows to avoid multiprocessing crashes
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 2. Setup Model
    model = ConvVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Train Loop
    model.train()
    print(f"Starting training on {DEVICE} for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f"   Epoch {epoch+1}/{EPOCHS} | Avg Loss: {train_loss / len(train_loader.dataset):.4f}")

    # 4. Save
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training Complete! Model saved to: {MODEL_PATH}")

if __name__ == '__main__':
    train()