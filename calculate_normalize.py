import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from tqdm.notebook import tqdm
from time import time
import data

channels = 3

root_dir = r"F:\Workprojects\TongFu_Bump\data\Manually_select_data2"
transforms = transforms.Compose([transforms.ToTensor()])
allDatasets = data.MyData(root_dir, ["error", "normal"], transforms)

full_loader = torch.utils.data.dataloader.DataLoader(allDatasets, shuffle=False, num_workers=0)

before = time()
mean = torch.zeros(3)
std = torch.zeros(3)
print('==> Computing mean and std..')
for inputs, _labels, _ in tqdm(full_loader):
    for i in range(channels):
        mean[i] += inputs[:, i, :, :].mean()
        std[i] += inputs[:, i, :, :].std()
    # mean += inputs[:, 2, :, :].mean()
    # std += inputs[:, 2, :, :].std()
mean.div_(len(allDatasets))
std.div_(len(allDatasets))
print(mean, std)

print("time elapsed: ", time() - before)
