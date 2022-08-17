import h5py
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


classes = [
    "Disturbed Galaxies",
    "Merging Galaxies",
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge"
]

def read_data(data_path):
    with h5py.File(data_path, "r") as f:
        images = np.array(f["images"])
        # images = T.ToTensor()(images[0]) #np.squeeze(images)
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


def load_data(data_path, batch_size, img_mean, img_std, is_train=False):
    
    transform_train = transforms.Compose([
        # transforms.ToPILImage(), # not needed since the image already np.array
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(img_mean, img_std),
    ])

    transform_test = transforms.Compose([                        
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(img_mean, img_std),
    ])

    images, labels = read_data(data_path)
    train_idx, test_idx = train_test_split(np.arange(images.shape[0]), test_size=0.1, random_state=0)
    train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]
        
    train_dataset = CustomTensorDataset(train_images, train_labels, transform=transform_train)
    test_dataset = CustomTensorDataset(test_images, test_labels, transform=transform_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size) 
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    if is_train:
        return train_dataloader, test_dataloader
    else:
        return test_dataloader, classes

