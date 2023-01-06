"""
This program implements evaluation on some vision models, for a binary classification with known losses.

It is the simplest implementation of evaluation. It requires a yaml file with the configurations of the program.
Currently it implemetns resnet50, swin-tiny, vit-small and some variations of them. It is also uses BCEWithLogitLoss
but it can be upgraded for better functionality.
Peace.
"""


import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
import wandb

from dataset import dataset
import augmentations
from torch.utils.data.dataloader import DataLoader
from torchmetrics import functional as tmf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

@torch.no_grad()
def testing(model, dataloader, criterion, args):
    model.eval()

    running_loss, y_true, y_pred = [], [], []
    for x, y in tqdm(dataloader):
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
    print(y_true)
    print(y_pred)
    test_loss = np.mean(running_loss)
    # wandb.log({'validation-loss': test_loss})
    # acc = 100. * np.mean(y_true == y_pred)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f"Accuracy is {acc}")

    wandb.log({'test-accuracy': acc})

    return np.mean(running_loss), y_true, y_pred


def log_metrics(y_true, y_pred):

    test_acc = accuracy_score(y_true, y_pred)
    print("acc is ", test_acc)

    test_f1 = f1_score(y_true, y_pred, average="binary")
    print("f1 is ", test_f1)

    test_prec = precision_score(y_true, y_pred, average="binary")
    print("prec is ", test_prec)

    test_rec = recall_score(y_true, y_pred, average="binary")
    print("rec is ", test_rec)

    test_auc = roc_auc_score(y_true, y_pred,)
    print("aucc is ", test_auc)

    wandb.log({
        'Accuracy': test_acc,
        'F1': test_f1,
        'Precision': test_prec,
        'Recall': test_rec,
        # 'ROC-AUC score': test_auc
        })


def log_conf_matrix(y_true, y_pred):
    conf_matrix = tmf.confusion_matrix(y_pred, y_true, num_classes=2)
    conf_matrix = pd.DataFrame(data=conf_matrix, columns=['A', 'B'])
    cf_matrix = wandb.Table(dataframe=conf_matrix)
    wandb.log({'conf_mat': cf_matrix})

def create_csv(y_true, y_pred):

    y_true = np.reshape(y_true, (len(y_true), 1))
    y_pred = np.reshape(y_pred, (len(y_pred), 1))

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    df = pd.concat((y_true, y_pred), axis=1)
    df.columns = ["actual", "pred"]
    df.to_csv("results.csv")


# main def:
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

    # initialize w&b
    print(args["name"])
    wandb.init(project=args["project_name"], name=args["name"],
               config=args, group=args["group"], mode=args["mode"])

    # initialize model:
    if args["model"] == 'resnet50':
        model = timm.create_model('resnet50', num_classes=2)
    elif args["model"] == 'swin-tiny':
        model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=1)
    elif args["model"] == 'swin-small':
        model = timm.create_model('swin_small_patch4_window7_224', num_classes=1)
    elif args["model"] == 'vit-tiny':
        model = timm.create_model('vit_tiny_patch16_224', num_classes=1)
    elif args["model"] == 'vit-small':
        model = timm.create_model('vit_small_patch16_224', num_classes=1)
    elif args["model"] == 'xception':
        model = timm.create_model('xception', num_classes=1)
    else:
        raise Exception('Model architecture not supported.')


    model.load_state_dict(torch.load(args["weights_dir"], map_location='cpu'))

    model = model.eval().to(args["device"])

    # defining transforms:
    transforms = augmentations.get_validation_augmentations()

    # define test dataset:
    test_dataset = dataset(
        args["test_dir"], transforms)

    # define data loaders:
    test_dataloader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)

    # set the criterion:
    criterion = nn.CrossEntropyLoss()

    # testing
    test_loss, y_true, y_pred = testing(
        model=model, dataloader=test_dataloader, criterion=criterion, args=args)

    # calculating and logging results:
    log_metrics(y_true=y_true, y_pred=y_pred)
    # create_csv(y_true=y_true, y_pred=y_pred)
    # log_conf_matrix(y_true=y_true, y_pred=y_pred)

    print(f'Finished Testing with test loss = {test_loss}')


if __name__ == '__main__':
    main()