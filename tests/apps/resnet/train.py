import os
from sys import argv, exit
from time import perf_counter

import torch
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet152
from tqdm import trange


def main():
    if len(argv) != 3:
        print(f"Usage: python3 {argv[0]} num_iter batch_size")
        exit(1)

    num_iter = int(argv[1])
    batch_size = int(argv[2])

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CIFAR10("dataset", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size)
    assert num_iter <= len(loader), len(loader)

    device = torch.device("cuda:0")
    model = resnet152()
    model.to(device)
    model.train()

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    def run(num_iter: int):
        bar = trange(num_iter)
        for _, (inputs, labels) in zip(bar, loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            bar.set_postfix({"loss": loss.item()}, False)

    run(2)

    start = perf_counter()
    run(num_iter)
    elapsed = perf_counter() - start
    print(f"{elapsed * 1000 / num_iter:.2f} ms/batch (B={batch_size})")

    if not os.path.exists(file := "checkpoint/resnet152_cifar10.pt"):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        torch.save(model.state_dict(), file)


if __name__ == "__main__":
    main()
