import os

def log_csv(filename, epoch, train_acc, val_acc, loss):
    # check if directory exists
    if not os.path.exists('log'):
        os.makedirs('log')
    with open(f'log/{filename}', 'a') as f:
        f.write('{},{},{},{}\n'.format(epoch, train_acc, val_acc, loss))


def log_csv_pretrain(filename, epoch, train_loss, val_loss):
    # check if directory exists
    if not os.path.exists('log'):
        os.makedirs('log')
    with open(f'log/{filename}', 'a') as f:
        f.write('{},{},{}\n'.format(epoch, train_loss, val_loss))
