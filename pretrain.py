import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from functools import partial
import argparse

# utils
from utils.csv_logger import log_csv_pretrain
from utils.transforms import *

# datasets
from dataset.HAM10000 import HAM10000
from dataset.isic import ISIC, DeformISIC
from dataset.medmnist import MedMNIST_All

# models
from model.kt.model import DeformationPredictor
from model.mae.mae import mae_vit_base_patch14_dec512d8b
from model.moco.builder import MoCo_ViT
from model.moco import vits
from model.defvit.defvit import DeformViT, DeformViTClassifier


parser = argparse.ArgumentParser(description='Pretrain model on MedMNIST')
parser.add_argument('--model', type=str, default='mae', help='model name')
parser.add_argument('--save_model', action='store_true', help='save model checkpoint')
parser.add_argument('--save_csv', action='store_true', help='save csv log')
parser.add_argument('--continue_train', action='store_true', help='continue training from checkpoint')
parser.add_argument('--bs', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=1.5e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
args = parser.parse_args()


"""training files path"""
# data_dir = '/root/autodl-fs/data/ISIC/'
# label_csv_file = 'data/ham10000/HAM10000_metadata_encoded.csv'
data_dir = '/root/autodl-fs/data/MedMNIST'


def save_checkpoint(state, filename=f'checkpoint/{args.model}_pretrain_medmnist.pth.tar'):
    print('saving model checkpoint ...')
    torch.save(state, filename)

def adjust_moco_momentum(epoch, max_epoch, moco_m=0.99):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) * (1. - moco_m)
    return m


"""setup data transforms"""
if args.model == "mae":
    train_transform = mae_train_transform
    val_transform = mae_test_transform
elif args.model == "deform":
    train_transform = deform_train_transform
    val_transform = deform_test_transform
elif args.model == "moco":
    train_transform = MoCoTwoCropsTransform(moco_aug_1, moco_aug_2)
    val_transform = MoCoTwoCropsTransform(moco_aug_1, moco_aug_2)


train_dataset = MedMNIST_All(
    split='train', 
    transform=train_transform, 
    root=data_dir, 
    as_rgb=True,
)
val_dataset = MedMNIST_All(
    split='val', 
    transform=val_transform, 
    root=data_dir, 
    as_rgb=True,
)

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=12)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(' - Training device currently set to:', device)


if args.model == "mae":
    model = mae_vit_base_patch14_dec512d8b()
elif args.model == "deform":
    model = DeformViT()
elif args.model == "moco":
    model = MoCo_ViT(partial(vits.__dict__['vit_base'], stop_grad_conv1=True),
                     256, 4096, 1.0)

model = nn.DataParallel(model)

criterion = torch.nn.MSELoss()
num_epochs = args.epochs
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

start_epoch = 0
if args.continue_train:
    checkpoint = torch.load(f'checkpoint/{args.model}_pretrain_medmnist.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    print('checkpoint loaded, resuming on epoch: ', start_epoch + 1)

for epoch in range(start_epoch, num_epochs):
    """
    train
    """
    model.train()
    running_loss = 0.0
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    bar = tqdm(total=len(train_loader))
    moco_m = 0.99 # for moco
    for i, images in enumerate(train_loader):
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            if args.model == "mae":
                images = images.to(device)
                loss, _, _ = model(images)
            elif args.model == "deform":
                origin = images[0].to(device)
                def_images = images[1].to(device)
                reconstructed_images = model(def_images)
                loss = criterion(reconstructed_images, origin)
            elif args.model == "moco":
                moco_m = adjust_moco_momentum(epoch + i / len(val_loader), num_epochs)
                loss = model(images[0].to(device), images[1].to(device), moco_m)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # if images has two parts, calculate running_loss differently
        if type(images) == list:
            running_loss += loss.item() * images[0].size(0)
        else:
            running_loss += loss.item() * images.size(0)

        bar.update(1)

    bar.close()

    train_loss = running_loss / len(train_dataset)
    print(f'Training Loss: {train_loss}')

    """
    validation
    """
    model.eval()
    running_val_loss = 0.0
    moco_m = 0.99 # for moco
    with torch.no_grad():
        bar = tqdm(total=len(val_loader))
        for i, images in enumerate(val_loader):

            if args.model == "mae":
                images = images.to(device)
                loss, _, _ = model(images)
            elif args.model == "deform":
                origin = images[0].to(device)
                def_images = images[1].to(device)
                reconstructed_images = model(def_images)
                loss = criterion(reconstructed_images, origin)
            elif args.model == "moco":
                moco_m = adjust_moco_momentum(epoch + i / len(val_loader), num_epochs)
                loss = model(images[0].to(device), images[1].to(device), moco_m)
            
            # if images has two parts, calculate running_loss differently
            if type(images) == list:
                running_val_loss += loss.item() * images[0].size(0)
            else:
                running_val_loss += loss.item() * images.size(0)

            bar.update(1)

        bar.close()

    val_loss = running_val_loss / len(val_dataset)
    print(f'Validation Loss: {val_loss}')
    
    """
    save results
    """
    if args.save_model:
        save_checkpoint({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        })
    if args.save_csv:
        log_csv_pretrain(f'{args.model}_pretrain_medmnist.csv', 
                         epoch + 1, train_loss, val_loss)

    scheduler.step()

print('Finished Training')
