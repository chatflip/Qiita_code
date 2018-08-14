from __future__ import print_function
import random
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=0,padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16 * 5 * 5)
        x = self.classifier(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss(size_average=True)(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(size_average=False)(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def opt():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, metavar="M", help="weight decay (default: 0.0001)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--num_workers", type=int, default=0, help="num of pallarel threads(dataloader)")
    parser.add_argument("--log-interval", type=int, default=1, metavar="N", help="how many batches to wait before logging training status")
    args = parser.parse_args()
    return args

#for debug
class RandomPrint(object):
    def __call__(self, samples):
        print (random.random())
        return samples

def worker_init_fn(worker_id):
    random.seed(1+worker_id)

if __name__ == "__main__":
    args = opt()


    random.seed(1)
    torch.manual_seed(1)
    #changed 
    cudnn.deterministic = True

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #train DataLoader
    train_transform = transforms.Compose([transforms.Resize((48, 48)),
                                        transforms.RandomCrop((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        #RandomPrint(),#for debug
                                        ])
    #add worker_init_fn
    train_MNIST = datasets.MNIST("./data", train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_MNIST, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    #test DataLoader
    test_transform = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
    test_MNIST = datasets.MNIST("./data", train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_MNIST, batch_size=args.test_batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)