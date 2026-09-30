import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image

p = 0.95
colors = {
    0:(255,0,0),    # red
    1:(0,255,0),    # green
    2:(0,0,255),    # blue
    3:(255,255,0),
    4:(255,0,255),
    5:(0,255,255),
    6:(255,128,0),
    7:(128,0,255),
    8:(128,128,128),
    9:(0,128,128)
}
inv = {0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:0}


all_colors = list(colors.values())

def paint(img, col, noise_scale=60):
    """Colorize foreground strokes and fill background with textured noise."""
    img = np.array(img, dtype=np.float32) / 255.0
    mask = img[..., None]  # foreground mask in [0,1]
    color = np.array(col, dtype=np.float32)

    texture = np.random.randint(0, noise_scale, size=(28, 28, 3)).astype(np.float32)
    out = mask * color + (1.0 - mask) * texture
    return Image.fromarray(out.astype(np.uint8))

class CMNIST(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.data = datasets.MNIST(
            root="data",
            train=train,
            download=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, y = self.data[i]
        dom = colors[y]

        if self.train:
            if np.random.rand() < p:
                c = dom
            else:
                c = all_colors[np.random.randint(10)]
        else:
            c = all_colors[np.random.randint(10)]
            while c == dom:
                c = all_colors[np.random.randint(10)]
        # else:
        #     c = colors[inv[y]]


        img = paint(img, c)
        img = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
        return img, y
