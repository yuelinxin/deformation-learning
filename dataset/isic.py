from torch.utils.data import Dataset
import glob
import os
from PIL import Image


class ISIC(Dataset):
    def __init__(self, path, transform=None):
        self.root_path = path
        self.transform = transform
        self.img_names = glob.glob(os.path.join(path, "*.JPG"))
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        filename = self.img_names[idx]
        img = Image.open(filename)
        if self.transform:
            img = self.transform(img)
        return img


class DeformISIC(Dataset):
    def __init__(self, path, transform=None):
        self.root_path = path
        self.transform = transform
        self.img_names = glob.glob(os.path.join(path, "*.JPG"))
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        filename = self.img_names[idx]
        img = Image.open(filename)
        if self.transform:
            img, def_img = self.transform(img)
        return img, def_img
