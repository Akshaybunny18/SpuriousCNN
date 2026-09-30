import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mnist_color import CMNIST


ds = CMNIST(train=True)
print(f"Training set size: {len(ds)}")
dl = DataLoader(ds, batch_size=5, shuffle=True)

dst = CMNIST(train=False)
print(f"Test set size: {len(dst)}")
dlt = DataLoader(dst, batch_size=5, shuffle=True)

# Training set samples
x_train, y_train = next(iter(dl))
plt.figure(figsize=(12, 4))
plt.suptitle("Training Set Samples (95% dominant color, 5% random)")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i].permute(1, 2, 0))
    plt.title(f"Label: {int(y_train[i])}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Test set samples
x_test, y_test = next(iter(dlt))
plt.figure(figsize=(12, 4))
plt.suptitle("Test Set Samples (random color, never dominant)")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i].permute(1, 2, 0))
    plt.title(f"Label: {int(y_test[i])}")
    plt.axis("off")
plt.tight_layout()
plt.show()
