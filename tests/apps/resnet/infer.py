from sys import argv, exit
from time import perf_counter

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet152
from tqdm import trange

# torch.backends.cudnn.enabled = False

def main():
    if len(argv) != 3:
        print(f"Usage: python3 {argv[0]} num_iter batch_size")
        exit(1)

    num_iter = int(argv[1])
    batch_size = int(argv[2])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CIFAR10("dataset", train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size)
    assert num_iter <= len(loader), len(loader)

    device = torch.device("cuda:0")
    model = resnet152()
    model.load_state_dict(torch.load("checkpoint/resnet152_cifar10.pt", weights_only=True))
    model.to(device)
    model.eval()

    @torch.no_grad()
    def run(num_iter: int):
        bar = trange(num_iter)
        for _, (inputs, labels) in zip(bar, loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct = outputs.argmax(dim=1) == labels
            bar.set_postfix({"correct": f"{correct.sum().item()}/{len(correct)}"})

    run(2)

    start = perf_counter()
    run(num_iter)
    elapsed = perf_counter() - start
    print(f"{elapsed * 1000 / num_iter:.2f} ms/batch (B={batch_size})")


if __name__ == "__main__":
    main()
