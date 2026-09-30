"""
Task 3 - The Interrogation
Implement Grad-CAM (Gradient-weighted Class Activation Mapping) from scratch.
No external libraries (no pytorch-gradcam).
Proves where the Cheater CNN vs Robust CNN looks when classifying images.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from mnist_color import CMNIST, colors, paint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# Model Architectures
# ========================
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
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ========================
# Grad-CAM from Scratch
# ========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        # Backward pass on target class score
        score = output[0, target_class]
        score.backward()

        # Global average pooling on gradients to get importance weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to image size (28x28) and normalize to [0, 1]
        cam = F.interpolate(cam, size=(28, 28), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class, output

def visualize_gradcam(model, original_img, color_rgb, title, filename):
    painted = paint(original_img, color_rgb)
    img_tensor = torch.tensor(np.array(painted)).permute(2,0,1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    gradcam = GradCAM(model, model.c3)
    heatmap, pred_class, logits = gradcam.generate(img_tensor)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(np.array(painted))
    axes[0].set_title(f"Input Image\nPred: {pred_class}")
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    # Overlay
    rgb_np = np.array(painted) / 255.0
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    overlay = 0.5 * rgb_np + 0.5 * heatmap_colored
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TASK 3 - GRAD-CAM FROM SCRATCH")
    print("="*60)

    model = CheaterCNN().to(device)
    model.load_state_dict(torch.load('cheater_model.pth', map_location=device))
    model.eval()

    ds = CMNIST(train=False)
    
    # 1. Biased sample: Digit 0 painted with Red (dominant color of 0)
    img_0, _ = next((ds.data[i][0], 0) for i in range(len(ds)) if ds.data[i][1] == 0)
    visualize_gradcam(model, img_0, colors[0], "Cheater CNN: Biased Sample (Red 0)", "gradcam_biased_0.png")

    # 2. Conflicting sample: Digit 1 painted with Red (dominant color of 0)
    img_1, _ = next((ds.data[i][0], 1) for i in range(len(ds)) if ds.data[i][1] == 1)
    visualize_gradcam(model, img_1, colors[0], "Cheater CNN: Conflicting Sample (Red 1)", "gradcam_conflicting_1.png")

    # 3. Conflicting sample: Digit 0 painted with Green (dominant color of 1)
    visualize_gradcam(model, img_0, colors[1], "Cheater CNN: Conflicting Sample (Green 0)", "gradcam_conflicting_0.png")
