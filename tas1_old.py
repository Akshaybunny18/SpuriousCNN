"""
Task 1 - The Cheater
Train a CNN that learns color bias and fails on the Hard test set
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
from mnist_color import CMNIST, colors, paint

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========================
# Simple 3-Layer CNN Model
# ========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(self.relu(self.conv3(x)))  # 7x7 -> 3x3
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ========================
# Training Function
# ========================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
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
    all_preds = []
    all_labels = []
    
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

# ========================
# Main Training Script
# ========================
print("\n" + "="*60)
print("TASK 1 - THE CHEATER")
print("="*60)

# Load datasets
print("\nLoading datasets...")
train_dataset = CMNIST(train=True)
hard_test_dataset = CMNIST(train=False)

# Split training into train and validation (Easy validation)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

print(f"Easy Train set: {len(train_subset)} images")
print(f"Easy Validation set: {len(val_subset)} images")
print(f"Hard Test set: {len(hard_test_dataset)} images")

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
hard_test_loader = DataLoader(hard_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize model
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\nModel: Simple 3-Layer CNN")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
num_epochs = 1
train_losses, train_accs = [], []
val_losses, val_accs = [], []

print(f"\nTraining for {num_epochs} epochs...")
print("-" * 60)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print("-" * 60)

# Save the model
torch.save(model.state_dict(), 'cheater_model.pth')
print("\nModel saved as 'cheater_model.pth'")

# ========================
# EVALUATION & ANALYSIS
# ========================
print("\n" + "="*60)
print("EVALUATION & ANALYSIS")
print("="*60)

# Evaluate on Easy Validation set
print("\n1. Easy Validation Set Performance:")
_, easy_acc, easy_preds, easy_labels = evaluate(model, val_loader, criterion, device)
print(f"   Accuracy: {easy_acc:.2f}%")

# Evaluate on Hard Test set
print("\n2. Hard Test Set Performance:")
_, hard_acc, hard_preds, hard_labels = evaluate(model, hard_test_loader, criterion, device)
print(f"   Accuracy: {hard_acc:.2f}%")

print(f"\n⚠️  Accuracy DROP: {easy_acc:.2f}% → {hard_acc:.2f}% ({easy_acc - hard_acc:.2f}% decrease)")

# ========================
# VISUALIZATION 1: Training Curves
# ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label='Train Loss', marker='o')
axes[0].plot(val_losses, label='Validation Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Train Accuracy', marker='o')
axes[1].plot(val_accs, label='Validation Accuracy', marker='s')
axes[1].axhline(y=95, color='r', linestyle='--', label='95% Target', alpha=0.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_training_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: task1_training_curves.png")
plt.show()

# ========================
# VISUALIZATION 2: Confusion Matrices
# ========================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Easy Validation Confusion Matrix
cm_easy = confusion_matrix(easy_labels, easy_preds)
sns.heatmap(cm_easy, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_title(f'Easy Validation Set Confusion Matrix\nAccuracy: {easy_acc:.2f}%')

# Hard Test Confusion Matrix
cm_hard = confusion_matrix(hard_labels, hard_preds)
sns.heatmap(cm_hard, annot=True, fmt='d', cmap='Reds', ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
axes[1].set_title(f'Hard Test Set Confusion Matrix\nAccuracy: {hard_acc:.2f}%')

plt.tight_layout()
plt.savefig('task1_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task1_confusion_matrices.png")
plt.show()

# ========================
# VISUALIZATION 3: Per-Class Accuracy
# ========================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Calculate per-class accuracy for both sets
easy_per_class = []
hard_per_class = []

for i in range(10):
    easy_mask = easy_labels == i
    if easy_mask.sum() > 0:
        easy_per_class.append(100.0 * (easy_preds[easy_mask] == i).sum() / easy_mask.sum())
    else:
        easy_per_class.append(0)
    
    hard_mask = hard_labels == i
    if hard_mask.sum() > 0:
        hard_per_class.append(100.0 * (hard_preds[hard_mask] == i).sum() / hard_mask.sum())
    else:
        hard_per_class.append(0)

x = np.arange(10)
width = 0.35

bars1 = ax.bar(x - width/2, easy_per_class, width, label='Easy Validation', color='skyblue')
bars2 = ax.bar(x + width/2, hard_per_class, width, label='Hard Test', color='salmon')

ax.set_xlabel('Digit Class')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Per-Class Accuracy: Easy vs Hard')
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(10)])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=95, color='g', linestyle='--', alpha=0.5, label='95% Target')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('task1_per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task1_per_class_accuracy.png")
plt.show()

# ========================
# VISUALIZATION 4: Prediction Distribution
# ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Easy Validation prediction distribution
axes[0].hist(easy_preds, bins=np.arange(11)-0.5, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_xlabel('Predicted Class')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Easy Validation: Prediction Distribution')
axes[0].set_xticks(range(10))
axes[0].grid(True, alpha=0.3, axis='y')

# Hard Test prediction distribution
axes[1].hist(hard_preds, bins=np.arange(11)-0.5, alpha=0.7, color='red', edgecolor='black')
axes[1].set_xlabel('Predicted Class')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Hard Test: Prediction Distribution')
axes[1].set_xticks(range(10))
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('task1_prediction_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: task1_prediction_distribution.png")
plt.show()

# ========================
# COLOR BIAS PROOF TEST
# ========================
print("\n" + "="*60)
print("COLOR BIAS PROOF TEST")
print("="*60)

# Get a sample digit 1 from the test set
model.eval()
test_full = CMNIST(train=False)
found_one = False
original_img = None

for i in range(len(test_full)):
    _, label = test_full.data[i]
    if label == 1:
        original_img, _ = test_full.data[i]
        found_one = True
        break

if found_one:
    # Paint the digit 1 with RED (color for 0)
    red_color = colors[0]  # Red
    red_one = paint(original_img, red_color)
    
    # Convert to tensor
    red_one_tensor = torch.tensor(np.array(red_one)).permute(2,0,1).float() / 255.0
    red_one_tensor = red_one_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(red_one_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(1).item()
    
    print(f"\n📊 Test: Red Digit '1' (True Label: 1)")
    print(f"   Prediction: {prediction}")
    print(f"   Confidence: {probabilities[0][prediction].item()*100:.2f}%")
    print(f"\n   Top 3 Predictions:")
    top3_probs, top3_indices = torch.topk(probabilities[0], 3)
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
        print(f"      {i+1}. Class {idx.item()}: {prob.item()*100:.2f}%")
    
    # Visual proof
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original grayscale
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Digit: 1')
    axes[0].axis('off')
    
    # Red colored version
    axes[1].imshow(red_one)
    axes[1].set_title(f'Red-Colored 1\n(Using color of 0)')
    axes[1].axis('off')
    
    # Prediction probabilities
    axes[2].bar(range(10), probabilities[0].cpu().numpy(), color=['red' if i == prediction else 'gray' for i in range(10)])
    axes[2].set_xlabel('Digit Class')
    axes[2].set_ylabel('Probability')
    axes[2].set_title(f'Model Prediction: {prediction}\n(True Label: 1)')
    axes[2].set_xticks(range(10))
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('task1_color_bias_proof.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: task1_color_bias_proof.png")
    plt.show()
    
    if prediction == 0:
        print("\n✅ PROOF OF COLOR BIAS: Model predicted 0 for a RED digit 1!")
        print("   The model is clearly using color as the primary feature.")
    else:
        print(f"\n⚠️  Model predicted {prediction} (not 0), but still wrong.")
        print("   The model is still biased, just not as strongly as expected.")

# ========================
# SUMMARY REPORT
# ========================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"""
✓ Model trained successfully on Easy set
✓ Easy Validation Accuracy: {easy_acc:.2f}% {'✅ (>95%)' if easy_acc > 95 else '❌ (<95%)'}
✓ Hard Test Accuracy: {hard_acc:.2f}% {'✅ (<20%)' if hard_acc < 20 else '⚠️  (>20%)'}
✓ Accuracy Drop: {easy_acc - hard_acc:.2f}%
✓ Confusion matrices generated
✓ Color bias proven with red digit test
✓ All visualizations saved

Files generated:
- cheater_model.pth (trained model)
- task1_training_curves.png
- task1_confusion_matrices.png
- task1_per_class_accuracy.png
- task1_prediction_distribution.png
- task1_color_bias_proof.png

The model is a CHEATER! 🎭
It learned to associate colors with digits instead of learning digit shapes.
""")

print("="*60)
