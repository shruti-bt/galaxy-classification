r"""This file contains function which are not have specific meaning.

The functions like dataloading, trasformation, image show etc.

Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import h5py
import sys
import time
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def read_data(data_path):
    with h5py.File(data_path, "r") as f:
        images = np.array(f["images"])
        # images = np.transpose(images, (0, 3, 2, 1))
        labels = np.array(f["ans"])
    return images, labels


class CustomTensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index] # x.shape == (3, 69, 69)
        if self.transform:
            x = self.transform(x)
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.images)


def load_data(args, is_train=False):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(69, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    images, labels = read_data(args.data_path)
    # print(images.shape, labels.shape)
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]
        
    train_dataset = CustomTensorDataset(train_images, train_labels, transform=transform_train)
    test_dataset = CustomTensorDataset(test_images, test_labels, transform=transform_test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size) 
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size) 

    classes = ['Disk, Face-on, No Spiral', 'Smooth, Completely round', 
                'Smooth, in-between round', 'Smooth, Cigar shaped',
                'Disk, Edge-on, Rounded Bulge','Disk, Edge-on, Boxy Bulge', 
                'Disk, Edge-on, No Bulge', 'Disk, Face-on, Tight Spiral', 
                'Disk, Face-on, Medium Spiral', 'Disk, Face-on, Loose Spiral']
    
    if is_train:
        return train_dataloader, test_dataloader
    else:
        return test_dataloader, classes

def load_model(model_name):
    if model_name.startswith("vgg"):
        from model.vgg import vgg
        return vgg(model_name=model_name)
    elif model_name.startswith("resnet"):
        from model.resnet import resnet
        return resnet(model_name=model_name)
    else:
        raise ValueError("Provide valid model name.")


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 85 # int(term_width)

TOTAL_BAR_LENGTH = 55.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def plot_metrics(series, labels, xlabel,
    ylabel, xticks, yticks, save_path
) -> None:

    plt.figure(figsize=(8, 4), dpi=600)
    for x, x_label in zip(series, labels):
        plt.plot(x, label=x_label)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.savefig(save_path, transperent=True, pad_inches=0.1)






