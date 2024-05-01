import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score

# utils
from utils.csv_logger import log_csv
from utils.transforms import mae_train_transform, mae_test_transform

# models
import timm
from transformers import SwinForImageClassification, SwinConfig
from model.mae.mae import MAEClassifier
from model.defvit.defvit import DeformViTClassifier

# datasets
from dataset.HAM10000 import HAM10000Classification
from dataset.medmnist import DermaMNIST, BloodMNIST, PathMNIST, OrganSMNIST


parser = argparse.ArgumentParser(description='Pretrain model on MedMNIST')
parser.add_argument('--model', type=str, default='mae', help='model name')
parser.add_argument('--dataset', type=str, default='ham10000', help='dataset name')
parser.add_argument('--save_model', action='store_true', help='save model checkpoint')
parser.add_argument('--save_csv', action='store_true', help='save csv log')
parser.add_argument('--continue_train', action='store_true', help='continue training from checkpoint')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--freeze_encoder', action='store_true', help='freeze the encoder')
args = parser.parse_args()


# training meta setup
image_directory = 'data/ham10000/ham10000-data'
label_csv_file = 'data/ham10000/HAM10000_metadata_encoded.csv'
mednist_root = "/root/autodl-fs/data/MedMNIST"

def save_checkpoint(state, filename=f'checkpoint/{args.model}_transfer_{args.dataset}.pth.tar'):
    print('saving model checkpoint ...')
    torch.save(state, filename)


if args.dataset == "ham10000":
    dataset = HAM10000Classification(csv_file=label_csv_file, root_dir=image_directory)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = mae_train_transform
    val_dataset.dataset.transform = mae_test_transform
    num_classes = 7
elif args.dataset == "derm":
    train_dataset = DermaMNIST(split="train", root=mednist_root, 
                               transform=mae_train_transform, as_rgb=True, size=224)
    val_dataset = DermaMNIST(split="val", root=mednist_root, 
                             transform=mae_test_transform, as_rgb=True, size=224)
    num_classes = 7
elif args.dataset == "blood":
    train_dataset = BloodMNIST(split="train", root=mednist_root, 
                               transform=mae_train_transform, as_rgb=True, size=224)
    val_dataset = BloodMNIST(split="val", root=mednist_root, 
                             transform=mae_test_transform, as_rgb=True, size=224)
    num_classes = 8
elif args.dataset == "path":
    train_dataset = PathMNIST(split="train", root=mednist_root, 
                               transform=mae_train_transform, as_rgb=True, size=224)
    val_dataset = PathMNIST(split="val", root=mednist_root, 
                             transform=mae_test_transform, as_rgb=True, size=224)
    num_classes = 9
elif args.dataset == "organs":
    train_dataset = OrganSMNIST(split="train", root=mednist_root, 
                               transform=mae_train_transform, as_rgb=True, size=224)
    val_dataset = OrganSMNIST(split="val", root=mednist_root, 
                             transform=mae_test_transform, as_rgb=True, size=224)
    num_classes = 11

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=12)
test_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=12)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(' - Training device currently set to:', device)

# config = SwinConfig.from_pretrained("/root/autodl-fs/kernel-transformer/config.json")
# config.num_labels = 7
# model = SwinForImageClassification.from_pretrained(
#     "/root/autodl-fs/kernel-transformer/pytorch_model.bin",
#     config=config,
#     ignore_mismatched_sizes=True
# )
# classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n or 'head' in n]
# encoder_params = [p for n, p in model.named_parameters() if not ('classifier' in n or 'head' in n)]
# assert len(classifier_params) + len(encoder_params) == len(list(model.parameters())), "Parameters not fully separated"

if args.model == "vit":
    model = DeformViTClassifier(num_classes=num_classes, cls_mode="cls")
    model = nn.DataParallel(model)
else:
    weights = torch.load(f'checkpoint/{args.model}_pretrain_medmnist.pth.tar')['state_dict']

    if args.model == "mae":
        model = MAEClassifier(num_classes=num_classes)
        new_state_dict = model.state_dict()
        encoder_weights = {k: v for k, v in weights.items() if not(k.startswith('module.decoder') or k.startswith('module.mask_token'))}
    elif args.model == "deform":
        model = DeformViTClassifier(num_classes=num_classes)
        new_state_dict = model.state_dict()
        encoder_weights = {k: v for k, v in weights.items() if k.startswith('module.encoder')}
    
    model = nn.DataParallel(model)
    
    missing_keys, unexpected_keys = model.load_state_dict(encoder_weights, strict=False)
    print(missing_keys, unexpected_keys)


criterion = torch.nn.CrossEntropyLoss()
num_epochs = args.epochs
scaler = torch.cuda.amp.GradScaler(enabled=True)
if args.freeze_encoder:
    encoder_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
    assert len(classifier_params) + len(encoder_params) == len(list(model.parameters())), "Parameters not fully separated"
    optimizer = torch.optim.Adam(
        [{'params': encoder_params, 'lr': 0},
        {'params': classifier_params, 'lr': args.lr}], 
        weight_decay=args.wd
    )
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)


start_epoch = 0
if args.continue_train:
    checkpoint = torch.load(f'checkpoint/{args.model}_transfer_{args.dataset}.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    print('checkpoint loaded, resuming on epoch: ', start_epoch + 1)

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    top2_correct = 0
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    bar = tqdm(total=len(train_loader))
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.flatten()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            # outputs = model(images, masked=True)
            outputs = model(images)
            # calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            _, top2_predicted = torch.topk(outputs.data, 2)
            # _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top2_correct += ((top2_predicted == labels.unsqueeze(1)).sum(dim=1) > 0).sum().item()
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        bar.update(1)

    bar.close()

    train_accuracy = round((correct / total) * 100, 2)
    print(f'Training Accuracy: {train_accuracy}')

    top2_accuracy = round((top2_correct / total) * 100, 2)
    print(f'Top-2 Accuracy: {top2_accuracy}')

    epoch_loss = running_loss / len(train_dataset)
    print(f'Loss: {epoch_loss}')

    model.eval()
    correct, total = 0, 0
    top2_correct = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        bar = tqdm(total=len(test_loader))
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.flatten()          

            # outputs = model(images, masked=False)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, top2_predicted = torch.topk(outputs.data, 2)
            # _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top2_correct += ((top2_predicted == labels.unsqueeze(1)).sum(dim=1) > 0).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            bar.update(1)

        bar.close()

    accuracy = round((correct / total) * 100, 2)
    print(f'Val. Accuracy: {accuracy}')

    top2_accuracy = round((top2_correct / total) * 100, 2)
    print(f'Val. Top-2 Accuracy: {top2_accuracy}')

    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'F1 Score: {f1}')

    if args.save_model:
        save_checkpoint({
            'epoch': epoch + 1,
            'accuracy': accuracy,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        })
    if args.save_csv:
        log_csv(epoch, train_accuracy, accuracy, epoch_loss)

    scheduler.step()

print('Finished Training')
