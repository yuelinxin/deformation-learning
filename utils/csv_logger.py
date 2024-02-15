import os

def log_csv(epoch, train_acc, val_acc, loss):
    # check if directory exists
    if not os.path.exists('log'):
        os.makedirs('log')
    with open('log/kt_pretrained_cifar10.csv', 'a') as f:
        f.write('{},{},{},{}\n'.format(epoch, train_acc, val_acc, loss))


def log_csv_pretrain(epoch, train_loss, val_loss):
    # check if directory exists
    if not os.path.exists('log'):
        os.makedirs('log')
    with open('log/kt_pretrain_isic.csv', 'a') as f:
        f.write('{},{},{}\n'.format(epoch, train_loss, val_loss))
