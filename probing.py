import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 4, 3, padding=1)
        self.c2 = nn.Conv2d(4, 6, 3, padding=1)
        self.c3 = nn.Conv2d(6, 12, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(12 * 3 * 3, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.c1(x)))
        x = self.pool(torch.relu(self.c2(x)))
        x = self.pool(torch.relu(self.c3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cheater_model.pth', map_location=device))
model.eval()


def synthesize(module, channel, *, steps=400, lr=0.1, tv_weight=1e-3, l2_weight=1e-4, seed=0):
    activations = None

    def hook_fn(_, __, output):
        nonlocal activations
        activations = output

    hook = module.register_forward_hook(hook_fn)
    torch.manual_seed(seed)
    img = torch.randn(1, 3, 28, 28, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([img], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        model(img)
        act = activations[:, channel]
        loss = -act.mean()

        tv_loss_x = (img[:, :, :, :-1] - img[:, :, :, 1:]).pow(2).mean()
        tv_loss_y = (img[:, :, :-1, :] - img[:, :, 1:, :]).pow(2).mean()
        tv_loss = tv_loss_x + tv_loss_y
        l2_loss = img.pow(2).mean()

        total = loss + tv_weight * tv_loss + l2_weight * l2_loss
        total.backward()
        optimizer.step()
        img.data.clamp_(0, 1)

    hook.remove()
    return img.detach().cpu()


def probe_layers(output_dir='probes', channels_per_layer=4):
    os.makedirs(output_dir, exist_ok=True)
    targets = {
        'c1': model.c1,
        'c2': model.c2,
        'c3': model.c3,
        'fc': model.fc,
    }

    for layer_name, module in targets.items():
        count = getattr(module, 'out_channels', None)
        if count is None:
            count = getattr(module, 'out_features')

        limit = min(channels_per_layer, count)
        print(f'Optimizing layer {layer_name} ({limit} channels)...')

        for ch in range(limit):
            img = synthesize(module, ch, seed=ch)
            filename = os.path.join(output_dir, f'{layer_name}_ch{ch}.png')
            save_image(img, filename, normalize=True)
            print(f'  Saved {filename}')


if __name__ == '__main__':
    probe_layers()