from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os


class HAM10000(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx]['image_id'] + '.jpg')
        image = Image.open(img_name)
        label = self.labels_frame.iloc[idx]['label_idx']

        if self.transform:
            image = self.transform(image)

        return image
