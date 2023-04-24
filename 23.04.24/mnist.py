import functools
import os
import torch
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
from loguru import logger
from tqdm import tqdm


logger.add('file.log')

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    transform=ToTensor(),
)

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs, labels):
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)

    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0,i].imshow(np.array(img))
        axs[0,i].set_title(labels[i])
        axs[0,i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig('image')

# data visualize    
num_list = [training_data[i][0] for i in range(4)]
num_label = [training_data[i][1] for i in range(4)]


show(num_list, num_label)


classes = ('0','1','2','3','4','5','6','7','8','9')
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Check data size
for X, y in test_dataloader:
    print("[INFO] Test dataset \n ------------------------------------- ")
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    print(f"Num_batches : {len(test_dataloader)}")
    print(f"Per batch data size : {batch_size}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
if device=="cuda":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']="0"




# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    # tensorboard
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

@torch.no_grad()            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.eval()
    test_loss, correct = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



# main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-prefix')
    parser.add_argument('--resume-from')

    args = parser.parse_args()

if args.resume_from:
        # 저장했던 중간 모델 정보를 읽습니다.
        model_data = torch.load(args.resume_from)

        model = NeuralNetwork().to(device)
        # 저장했던 모델의 값들로 치환합니다.
        model.load_state_dict(model_data['model_state_dict'])

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # optimizer도 중간에 저장했던 값들로 치환합니다.

        optimizer.load_state_dict(model_data['optimizer_state_dict'])

        # 지금 시작할 epoch은 기존 epoch + 1 즉 다음 epoch입니다.
        start_epoch = model_data['epoch'] + 1
else:
    model = NeuralNetwork()

    optimizer = torch.optim.Adam(model.parameters())

    start_epoch = 1

model.to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()

epochs = 5
for epoch in tqdm(range(start_epoch, epochs+1)):
    logger.info(f"Epoch {epoch}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
        }, f'{args.output_prefix}.{epoch:04d}.pth')
print("Done!")


# predict visualization
test_num_list = [test_data[i][0] for i in range(4)]
num_label = [test_data[i][1] for i in range(4)]

pred_label = []
model.eval()
with torch.no_grad():
    for img in test_num_list:
        img = img.to(device)
        pred = model(img)
        pred_label.append(pred.argmax(1).item())

show(test_num_list, pred_label)
