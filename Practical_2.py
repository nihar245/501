# -------------------- Cell 1 --------------------
import torch
# arrays
a = torch.tensor([[1,3,4],[4,5,6],[7,3,6],[6,7,8]])
print(a)
print(a.shape)

b = torch.tensor([[1,2,3],[4,5,6]])
print(b)

c = torch.rand(3,4)
print(c)

d = torch.ones(2,3)
print(d)

e = torch.zeros(2,2)
print(e)

# indexing
element = a[1,0]
print(element)

# slicing
slice = a[:2,:]
print(slice)

# reshape
reshaped = a.view(6,2)
print(reshaped)

# transpose
transposed = a.t()
print(transposed)


# -------------------- Cell 2 --------------------
# operation

# addition
a1 = torch.tensor([[1,3,5],[3,5,7]])
a2 = torch.tensor([[4,6,7],[6,7,8]])
a3 = torch.tensor([[2,3],[4,5],[6,7]])

add = a1 + a2
print(add)

# multiplication

mul = torch.matmul(a1, a3)
print(mul)


# -------------------- Cell 3 --------------------
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())


# -------------------- Cell 4 --------------------
torch.manual_seed(5)
torch.rand(2,3)


# -------------------- Cell 5 --------------------
torch.manual_seed(5)
torch.rand(2,3)


# -------------------- Cell 6 --------------------
torch.empty(2,3)


# -------------------- Cell 7 --------------------
torch.eye(3)


# -------------------- Cell 8 --------------------
# making new array with same size as old
a = torch.tensor([[1,2,3.3,5.6],[6.8,7,8.3,9]], dtype=float)
print(a)
print(a.size)
b = torch.zeros_like(a)
print(b)
b.dtype


# -------------------- Cell 9 --------------------
v = torch.tensor([-1,0,4,-7])
print(abs(v))
m = torch.clamp(v, min=1, max=3)
print(m)

print(v)
v.add_(5)   # inplace operator
print(v)


# -------------------- Cell 10 --------------------
# tensor to numpy
import numpy as np
t = torch.ones(5)
n = t.numpy()
print(n)
t.add_(1)
print(n)

# numpy to tensor
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(t)


# -------------------- Cell 11 --------------------
# dataloader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# -------------------- Cell 12 --------------------
# dataset and dataloader
import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset
class NumberDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]
        self.labels = [10, 20, 30, 40, 50]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return one sample (x, y)
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

dataset = NumberDataset()

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print("Length of dataset:", len(dataset))
print("First sample:", dataset[0])
print("Second sample:", dataset[1])
print("-----")

for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}")
    print("Data:", data)
    print("Labels:", labels)
    print("---")


# -------------------- Cell 13 --------------------
# Image Transform
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()  # convert images to tensor [0,1]
)

image, label = dataset[0]
print(type(image), image.shape)  # torch.Tensor, shape: [1, 28, 28]
print("Label:", label)


# -------------------- Cell 14 --------------------
from torchvision.transforms import Lambda

# one-hot encode labels
one_hot = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), 1))

dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=one_hot,
)

# Take one sample
image, label = dataset[0]
print("Image shape:", image.shape)
print("One-hot label:", label)


# -------------------- Cell 15 --------------------
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Device setup (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

# 2. Transform: convert image → tensor → normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # scale to [-1, 1]
])

# 3. Load dataset & dataloader
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 4. Define the Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()   # 28x28 → 784
        self.layers = nn.Sequential(
            nn.Linear(28*28, 128),  # input → hidden1
            nn.ReLU(),
            nn.Linear(128, 64),     # hidden1 → hidden2
            nn.ReLU(),
            nn.Linear(64, 10)       # hidden2 → output (10 classes)
        )

    def forward(self, x):
        x = self.flatten(x)       # flatten image
        logits = self.layers(x)   # forward pass
        return logits

# 5. Create model
model = NeuralNetwork().to(device)
print(model)

# 6. Try one batch
images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)

logits = model(images)                 # raw outputs
probs = nn.Softmax(dim=1)(logits)      # convert to probabilities
preds = probs.argmax(1)                # pick best class

print("Logits shape:", logits.shape)   # [32, 10] → 32 images, 10 classes
print("Predicted classes:", preds[:10])


# -------------------- Cell 16 --------------------
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(3, 2),
    nn.ReLU(),
    nn.Linear(2, 2),
    nn.Softmax(dim=1),
)

print("Model:\n", model)

print("\n--- Model Parameters ---")
for name, param in model.named_parameters():
    print(name, param.shape)

total_params = sum(p.numel() for p in model.parameters())
print("\nTotal trainable parameters:", total_params)

x = torch.tensor([[1.0, 2.0, 3.0]])
y = torch.tensor([1])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("\n--- Before Training ---")
for name, param in model.named_parameters():
    print(name, param.data)

output = model(x)
loss = criterion(output, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("\n--- After Training ---")
for name, param in model.named_parameters():
    print(name, param.data)

print("\nOutput probabilities:", output)
print("Loss value:", loss.item())


# -------------------- Cell 17 --------------------
# (empty cell)