"""
Task 5 - The Invisible Cloak
Targeted Adversarial Attack: Turn a digit 7 into a predicted 3 with >90% confidence.
Constraint: Max perturbation epsilon <= 0.05 (invisible to human eye).
Compare Cheater CNN vs Robust CNN susceptibility to adversarial noise.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mnist_color import CMNIST, colors, paint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CheaterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 4, 3, padding=1)
        self.c2 = nn.Conv2d(4, 6, 3, padding=1)
        self.c3 = nn.Conv2d(6, 12, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(12*3*3, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        x = self.pool(torch.relu(self.c3(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class RobustCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(128 * 3 * 3, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        x = self.pool(torch.relu(self.c3(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

def targeted_attack(model, img_tensor, target_class=3, epsilon=0.05, steps=300, lr=0.01):
    model.eval()
    # Optimize perturbation delta bounded in [-epsilon, +epsilon]
    delta = torch.zeros_like(img_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    target = torch.tensor([target_class], device=device)
    criterion = nn.CrossEntropyLoss()

    best_conf = 0.0
    best_adv = None

    for step in range(steps):
        optimizer.zero_grad()
        adv_img = torch.clamp(img_tensor + delta, 0.0, 1.0)
        outputs = model(adv_img)
        probs = torch.softmax(outputs, dim=1)
        conf = probs[0, target_class].item()

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # Project delta into L-infinity ball of radius epsilon
        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)

        if conf > best_conf:
            best_conf = conf
            best_adv = adv_img.detach().clone()

        if conf >= 0.90:
            print(f"  Target >90% reached at step {step+1}! Confidence: {conf*100:.2f}%")
            break

    return best_adv, best_conf, delta.detach()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TASK 5 - TARGETED ADVERSARIAL ATTACK (7 -> 3)")
    print("="*60)

    cheater = CheaterCNN().to(device)
    cheater.load_state_dict(torch.load('cheater_model.pth', map_location=device))

    robust = RobustCNN().to(device)
    robust.load_state_dict(torch.load('robust_model.pth', map_location=device))

    # Find sample of digit 7
    ds = CMNIST(train=True)
    img_7, label_7 = next((ds.data[i][0], 7) for i in range(len(ds)) if ds.data[i][1] == 7)
    
    # Paint digit 7 with its dominant color
    painted_7 = paint(img_7, colors[7])
    img_tensor = torch.tensor(np.array(painted_7)).permute(2,0,1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    print("\n1. Attacking Cheater CNN...")
    adv_cheater, conf_cheater, delta_c = targeted_attack(cheater, img_tensor, target_class=3, epsilon=0.05)
    print(f"  Cheater Result: Target '3' Conf = {conf_cheater*100:.2f}%")

    print("\n2. Attacking Robust CNN...")
    adv_robust, conf_robust, delta_r = targeted_attack(robust, img_tensor, target_class=3, epsilon=0.05)
    print(f"  Robust Result: Target '3' Conf = {conf_robust*100:.2f}%")

    # Visualize Attack Results
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    
    # Original
    axes[0,0].imshow(np.array(painted_7))
    axes[0,0].set_title("Original (7)")
    axes[0,0].axis('off')

    # Cheater Perturbation & Adv
    axes[0,1].imshow(((delta_c.squeeze().permute(1,2,0).cpu().numpy() + 0.05) / 0.1).clip(0,1))
    axes[0,1].set_title("Noise (eps=0.05)")
    axes[0,1].axis('off')

    axes[0,2].imshow(adv_cheater.squeeze().permute(1,2,0).cpu().numpy())
    axes[0,2].set_title(f"Cheater Output: 3\nConf: {conf_cheater*100:.1f}%")
    axes[0,2].axis('off')

    # Robust Row
    axes[1,0].imshow(np.array(painted_7))
    axes[1,0].set_title("Original (7)")
    axes[1,0].axis('off')

    axes[1,1].imshow(((delta_r.squeeze().permute(1,2,0).cpu().numpy() + 0.05) / 0.1).clip(0,1))
    axes[1,1].set_title("Noise (eps=0.05)")
    axes[1,1].axis('off')

    axes[1,2].imshow(adv_robust.squeeze().permute(1,2,0).cpu().numpy())
    axes[1,2].set_title(f"Robust Output: {'3' if conf_robust > 0.5 else '7'}\nConf 3: {conf_robust*100:.1f}%")
    axes[1,2].axis('off')

    plt.suptitle("Task 5: Targeted Adversarial Attack (eps <= 0.05)", fontsize=13)
    plt.tight_layout()
    plt.savefig('task5_adversarial_proof.png', dpi=300)
    plt.close()
    print("\nSaved task5_adversarial_proof.png")
