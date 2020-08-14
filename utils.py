from torchvision import transforms
import torchvision
import torch
from torch.utils.data import Dataset

class Dataloder(Dataset):
    def __init__(self,path):
        self.path = path
    def dataset(self):
        transform = transforms.Compose(
            [transforms.Resize(128),
            transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             ])
        trainset = torchvision.datasets.CIFAR10(root=self.path, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=True)
        testset = torchvision.datasets.CIFAR10(root= self.path, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                 shuffle=False)

        return trainloader,testloader