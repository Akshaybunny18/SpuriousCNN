"""
Task 4 - The Intervention
Retrain the model to focus on shape instead of color.
Debiasing Strategies:
1. Batch Color Jitter Augmentation (breaks color-label consistency)
2. Grayscale-Consistency Regularization (forces color and grayscale feature consistency)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
from mnist_color import CMNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# Robust CNN Model (32 -> 64 -> 128 -> 1152)
# ==========================================
class RobustCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(128 * 3 * 3, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.c1(x)))  # 28 -> 14
        x = self.pool(torch.relu(self.c2(x)))  # 14 -> 7
        x = self.pool(torch.relu(self.c3(x)))  # 7 -> 3
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Fast vectorized batch color augmentation
def apply_color_jitter(imgs):
    # imgs: (B, 3, H, W)
    bsz = imgs.size(0)
    # Random channel scaling & shifts
    factors = 0.5 + 1.0 * torch.rand((bsz, 3, 1, 1), device=imgs.device)
    shifts = -0.2 + 0.4 * torch.rand((bsz, 3, 1, 1), device=imgs.device)
    jittered = torch.clamp(imgs * factors + shifts, 0.0, 1.0)
    return jittered

def to_gray_3ch(imgs):
    # Grayscale replicated across 3 channels (luminance)
    luma = 0.299 * imgs[:, 0:1] + 0.587 * imgs[:, 1:2] + 0.114 * imgs[:, 2:3]
    return luma.repeat(1, 3, 1, 1)

# ========================
# Training Function
# ========================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

    for inputs, labels in tqdm(dataloader, desc="Training Robust", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Method 1: Jittered colored input
        aug_inputs = apply_color_jitter(inputs)
        # Method 2: Grayscale replicated input for consistency
        gray_inputs = to_gray_3ch(inputs)

        optimizer.zero_grad()
        outputs_aug = model(aug_inputs)
        outputs_gray = model(gray_inputs)

        # Supervised classification loss
        cls_loss = criterion(outputs_aug, labels)
        
        # Consistency loss (KL divergence between prediction distributions)
        log_prob_aug = torch.log_softmax(outputs_aug, dim=1)
        prob_gray = torch.softmax(outputs_gray.detach(), dim=1)
        consistency_loss = kl_loss_fn(log_prob_aug, prob_gray)

        loss = cls_loss + 1.0 * consistency_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs_aug.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ========================
# Evaluation Function
# ========================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TASK 4 - THE INTERVENTION")
    print("="*60)

    train_dataset = CMNIST(train=True)
    hard_test_dataset = CMNIST(train=False)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=0)
    hard_test_loader = DataLoader(hard_test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = RobustCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2
    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        _, hard_acc, _, _ = evaluate(model, hard_test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.2f}%")
        print(f"  Hard Test Acc: {hard_acc:.2f}% {'[TARGET MET >70%]' if hard_acc > 70 else ''}")
        print("-" * 60)

    torch.save(model.state_dict(), 'robust_model.pth')
    print("Model saved as 'robust_model.pth'")

    # Final evaluation
    _, easy_acc, easy_preds, easy_labels = evaluate(model, val_loader, criterion, device)
    _, hard_acc, hard_preds, hard_labels = evaluate(model, hard_test_loader, criterion, device)

    print("\n" + "="*60)
    print("FINAL SUMMARY - ROBUST MODEL")
    print("="*60)
    print(f"Easy Validation Accuracy: {easy_acc:.2f}%")
    print(f"Hard Test (OOD) Accuracy: {hard_acc:.2f}%")

    # Plot Comparison
    labels = ['Easy Val (In-Dist)', 'Hard Test (OOD)']
    cheater_res = [97.5, 11.2]
    robust_res = [easy_acc, hard_acc]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, cheater_res, width, label='Cheater CNN (Task 1)', color='salmon')
    ax.bar(x + width/2, robust_res, width, label='Robust CNN (Task 4)', color='mediumseagreen')
    ax.axhline(70, color='blue', linestyle='--', label='70% Target')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Cheater vs Robust Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig('task4_cheater_vs_robust.png', dpi=300)
    print("Saved task4_cheater_vs_robust.png")
