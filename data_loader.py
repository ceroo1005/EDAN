from torchvision import datasets, transforms
import torch
import os

norm_mean=(0.485, 0.456, 0.406)
norm_std=(0.229, 0.224, 0.225)

def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=norm_mean, std=norm_std)])
    if not os.path.isdir(os.path.join(root_path, dir, 'images')):
        data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    else:
        data = datasets.ImageFolder(root=os.path.join(root_path, dir, 'images'), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=norm_mean, std=norm_std)])
    if not os.path.isdir(os.path.join(root_path, dir, 'images')):
        data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    else:
        data = datasets.ImageFolder(root=os.path.join(root_path, dir, 'images'), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader