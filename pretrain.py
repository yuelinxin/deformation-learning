import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

# utils
from utils.csv_logger import log_csv_pretrain
from utils.deformation import *
from utils.transforms import *

# datasets
from dataset.HAM10000 import HAM10000
from dataset.isic import ISIC, DeformISIC
from dataset.medmnist import MedMNIST_All

# models
from model.kt.model import DeformationPredictor
from model.mae.mae import mae_vit_base_patch14_dec512d8b


"""training meta setup"""
save_model = True
continue_training = False
model_name = "mae"
# data_dir = '/root/autodl-fs/data/ISIC/'
# label_csv_file = 'data/ham10000/HAM10000_metadata_encoded.csv'
data_dir = '/root/autodl-fs/data/MedMNIST'

# deformations = torch.load('/root/autodl-fs/kernel-transformer/def.pt')
# idx = np.random.choice(len(deformations), 100, replace=False)
# val_deformations = [deformations[i] for i in idx]

def save_checkpoint(state, filename=f'checkpoint/{model_name}_pretrain_medmnist.pth.tar'):
    print('saving model checkpoint ...')
    torch.save(state, filename)


# dataset = HAM10000(csv_file=label_csv_file, root_dir=image_directory)
# dataset = ISIC(path=image_directory)
train_dataset = MedMNIST_All(
    split='train', 
    transform=mae_train_transform, 
    root=data_dir, 
    as_rgb=True
)
val_dataset = MedMNIST_All(
    split='val', 
    transform=mae_test_transform, 
    root=data_dir, 
    as_rgb=True
)

# Determine the lengths for training and testing sets (90:10 split)
# train_size = int(0.9 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# train_dataset.dataset.transform = mae_train_transform
# test_dataset.dataset.transform = mae_test_transform

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=12)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(' - Training device currently set to:', device)


if model_name == "mae":
    model = mae_vit_base_patch14_dec512d8b()
elif model_name == "deform":
    model = DeformationPredictor(in_channels=3, emb_size=96, patch_size=14, 
                                 heads=8, struct=(2, 2, 6, 2)).to(device)
model = nn.DataParallel(model)

criterion = torch.nn.MSELoss()
num_epochs = 300
optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

start_epoch = 0
if continue_training:
    checkpoint = torch.load(f'checkpoint/{model_name}_pretrain_medmnist.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    print('checkpoint loaded, resuming on epoch: ', start_epoch + 1)

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    bar = tqdm(total=len(train_loader))
    for images in train_loader:
        images = images.to(device)
        # def_images = def_images.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            if model_name == "mae":
                loss, _, _ = model(images)
            elif model_name == "deform":
                reconstructed_images = model(def_images)
                loss = criterion(reconstructed_images, images)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        bar.update(1)

    bar.close()

    train_loss = running_loss / len(train_dataset)
    print(f'Training Loss: {train_loss}')

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        bar = tqdm(total=len(val_loader))
        for images in val_loader:
            images = images.to(device)
            # def_images = def_images.to(device)
            
            if model_name == "mae":
                loss, _, _ = model(images)
            elif model_name == "deform":
                reconstructed_images = model(def_images)
                loss = criterion(reconstructed_images, images)
            
            running_val_loss += loss.item() * images.size(0)

            bar.update(1)

        bar.close()

    val_loss = running_val_loss / len(val_dataset)
    print(f'Validation Loss: {val_loss}')

    if save_model:
        save_checkpoint({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        })
    log_csv_pretrain(epoch + 1, train_loss, val_loss)

    scheduler.step()

print('Finished Training')
