import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, f1_score

from torch.utils.data.dataloader import DataLoader
from dataset import dataset
import augmentations
from folders_to_csv import CreateFile

# define training logic
def train_epoch(model, train_dataloader, args, optimizer, criterion, scheduler=None, epoch=0, val_results={}):
    model.train()

    running_loss = []
    pbar = tqdm(train_dataloader, desc='epoch {}'.format(epoch), unit='iter')
    for batch, (x, y) in enumerate(pbar):

        x = x.to(args["device"])
        y = y.to(args["device"])

        optimizer.zero_grad()

        outputs = model(x)
        # outputs = torch.softmax(outputs, dim=1)
        # label = torch.argmax(outputs, dim=1).unsqueeze(1)
        # outputs = torch.max(outputs, dim=1).unsqueeze(1).float()
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        running_loss.append(loss.detach().cpu().numpy())

        # log mean loss for the last 10 batches:
        if (batch+1) % 10 == 0:
            wandb.log({'train-step-loss': np.mean(running_loss[-10:])})
            pbar.set_postfix(loss='{:.3f} ({:.3f})'.format(running_loss[-1], np.mean(running_loss)), **val_results)

    # change the position of the scheduler:
    if scheduler is not None:
        scheduler.step()

    train_loss = np.mean(running_loss)

    wandb.log({'train-epoch-loss': train_loss})

    return train_loss


# define validation logic
@torch.no_grad()
def validate_epoch(model, val_dataloader, args, criterion):
    model.eval()

    running_loss, y_true, y_pred = [], [], []

    for x, y in tqdm(val_dataloader):

        x = x.to(args["device"])
        y = y.to(args["device"])
        # print(f"y = {y}")

        outputs = model(x)
        # print(f"outputs = {outputs}")
        loss = criterion(outputs, y)

        # loss calculation over batch
        running_loss.append(loss.cpu().numpy())

        # accuracy calculation over batch
        # outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        y_true.append(y.cpu())
        y_pred.append(outputs.cpu())

    y_true = torch.cat(y_true, 0).numpy()
    # print(f"y_true = {y_true}")
    # print(type(y_true))
    y_pred = torch.cat(y_pred, 0).numpy()
    # print(f"y_pred = {y_pred}")


    val_loss = np.mean(running_loss)
    wandb.log({'validation-loss': val_loss})
    acc = 100. * np.mean(y_true == y_pred)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f"Accuracy is {acc}")
    print(f"sklearn - f1 is {f1_score(y_true=y_true, y_pred=y_pred, average='macro')}")

    wandb.log({'validation-accuracy': acc})
    return {'val_acc': acc, 'val_loss': val_loss}


# MAIN def
def main():
    
    # add parser for obtaining configuration file:
    parser = argparse.ArgumentParser(description='Evaluation Arguments')

    parser.add_argument('-cf', '--conf_file', metavar='conf_file', required=True, type=str,
                         help='Configuration yaml file for the script.')

    parser_args = vars(parser.parse_args())

    # obtain configuration file:
    cf_file = parser_args["conf_file"]

    # initialize args
    with open(cf_file, 'r') as stream:
        args=yaml.safe_load(stream)

    # initialize weights and biases:
    wandb.init(project=args["project_name"], name=args["name"],
               config=args, group=args["group"], save_code=True, mode=args["mode"])

    # initialize model:
    if args["model"] == 'resnet18':
        model = timm.create_model('resnet18', pretrained=args["pretrained"], num_classes=2)
    elif args["model"] == 'resnet50':
        model = timm.create_model('resnet50', pretrained=args["pretrained"], num_classes=2)
    elif args["model"] == 'swin-tiny':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=args["pretrained"], num_classes=3)
    elif args["model"] == 'swin-small':
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=args["pretrained"], num_classes=3)
    elif args["model"] == 'vit-tiny':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=args["pretrained"], num_classes=3)
    elif args["model"] == 'vit-small':
        model = timm.create_model('vit_small_patch16_224', pretrained=args["pretrained"], num_classes=3)
    elif args["model"] == 'xception':
        model = timm.create_model('xception', pretrained=args["pretrained"], num_classes=3)
    else:
        raise Exception('Model architecture not supported.')

    model = model.to(args["device"])

    # DATASET:

    train_transforms = augmentations.get_training_augmentations(args["aug"])
    valid_transforms = augmentations.get_validation_augmentations()

    create_csv = CreateFile(folders=args["folders"], split_list=args["split_list"], path_to_save=args["path_to_save"])
    
    create_csv.images_to_csv(flag=args["flag"])

    if args["flag"]:
        # set the paths for training
        train_dataset = dataset(
            args["train_dir"], train_transforms)
        val_dataset = dataset(
            args["val_dir"], valid_transforms)
    else:
        # set the paths for training
        train_dataset = dataset(
        os.path.join(args["path_to_save"], 'train.csv') , train_transforms)
        val_dataset = dataset(
        os.path.join(args["path_to_save"], 'val.csv') , valid_transforms)

    # defining data loaders:
    train_dataloader = DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args["batch_size"], shuffle=False)

    # setting the optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])

    # setting the scheduler:
    if args["schedule"]:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()


    # directory:
    save_dir = args["save_dir"] + args["name"]
    print(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set value for min-loss:
    min_loss, val_results = float('inf'), {}
    print('Training starts...')
    for epoch in range(args["epochs"]):

        wandb.log({'epoch': epoch})
        train_epoch(model, train_dataloader=train_dataloader, args=args, optimizer=optimizer, criterion=criterion,
                    scheduler=scheduler, epoch=epoch, val_results=val_results)
        val_results = validate_epoch(model, val_dataloader=val_dataloader, args=args, criterion=criterion)

        if val_results['val_loss'] < min_loss:
            min_loss = val_results['val_loss'].copy()
            torch.save(model.state_dict(), os.path.join(
                save_dir,'best-ckpt.pt'))

if __name__ == '__main__':
    main()