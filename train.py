import glob
import sys

import os
from os import path as osp
import numpy as np
import math
import torch
import torch.nn as nn
import argparse

import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from thop import profile
from sklearn.metrics import confusion_matrix
from torchvision import transforms, datasets
# from torch.utils import data
from time import perf_counter
from tqdm import tqdm

from construct_dataset import Mydatasetpro
from models.CNN import ConvNet
from models.MLP import MLP4
from models.resnet import resnet50, resnet101, resnet18

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

data_transforms = {
    'train': transforms.Compose({
        # transforms.CenterCrop(48),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    }),
    'valid': transforms.Compose({
        # transforms.CenterCrop(48),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    }),
    'test': transforms.Compose({
        # transforms.CenterCrop(48),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
        # transforms.ToPILImage()
    })
}

_labels = ['0', '1', '2', '3', '4', '5', '6']
num_of_class = len(_labels)

model_type = ['ResNet50', 'MLP4', 'ConvNet', 'ResNet101', 'ResNet18']
model_name = 'ResNet18'  # ResNet MLP4 ConvNet

firstBN = True
if firstBN:
    firstBN_name = '_FirstBN'
else:
    firstBN_name = ''

parser = argparse.ArgumentParser(description='GRU and LSTM of RNN')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
# parser.add_argument('--raw_path', type=str, default='./dataset/Figures_test', help='真实数据集路径')
parser.add_argument('--saved_models_path', type=str, default='./checkpoint/pre_'
                                                             + model_name + '.pkl', help='模型保存')
parser.add_argument('--nepochs', type=str, default=50, help='迭代次数')
parser.add_argument('--batch_size', type=str, default=30, help='Batch大小')
parser.add_argument('--nhid', type=int, default=300, help="size of hidden units per layer")
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='initial momentum')
parser.add_argument('--runtype', type=str, default=model_name, help='type of network model')
parser.add_argument('--bidirect', type=str, default=False, help='bidirectional setting')
parser.add_argument('--dropout', type=str, default=0.1, help='dropout')
parser.add_argument('--train_curve_name', type=str, default='train_curve_' + model_name + '_' + firstBN_name,
                    help='save the train/valid loss curve')

args = parser.parse_args()

# 设置随机种子
torch.manual_seed(1234)

data_directory = './dataset4train/MMAFEDB'
train_set = datasets.ImageFolder(os.path.join(data_directory, 'train'), data_transforms['train'])
test_set = datasets.ImageFolder(os.path.join(data_directory, 'test'), data_transforms['test'])
valid_set = datasets.ImageFolder(os.path.join(data_directory, 'valid'), data_transforms['valid'])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Use gpu or cpu to train
use_gpu = torch.cuda.is_available()

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

if args.runtype in model_type:
    if args.runtype == 'ResNet50':
        model = resnet50(num_classes=num_of_class).to(device)
    elif args.runtype == 'ResNet101':
        model = resnet101(num_classes=num_of_class).to(device)
    elif args.runtype == 'ResNet18':
        model = resnet18(num_classes=num_of_class).to(device)
        # model = torchvision.models.resnet18(pretrained=True).eval().cuda()
    elif args.runtype == 'MLP4':
        model = MLP4(firstBN).to(device)
    elif args.runtype == 'ConvNet':
        model = ConvNet(num_classes=num_of_class).to(device)


def train():
    # 定义损失函数和优化器
    lossfunc = nn.CrossEntropyLoss()
    best_accuracy = 0
    # optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.nepochs)):
        train_loss = 0.0
        epoch_train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            optim.zero_grad()
            outputs = model(inputs)
            loss = lossfunc(outputs, labels)
            loss.backward()
            optim.step()

            # train_loss += loss.item()
            train_loss += loss.item() * inputs.size(0)
            epoch_train_loss += loss.item() * inputs.size(0)
            # train_loss_tmp = 0.0
            if i % 10 == 9:
                print('[Epoch:%d, Batch:%5d] loss:%.6f' % (epoch + 1, i + 1, train_loss / (10 * args.batch_size)))
                train_loss = 0.0

            # if i % 1000 == 999:
            #     valid_loss, acc_valid = valid()
            #     if acc_valid >= best_accuracy:
            #         save_param(model, args.saved_models_path)
            #     print('Epoch: {}, Batch: {}\tTrain Loss: {:.6f}\tValid Loss: {:.6f}\t'
            #           'Valid Accuracy: {:.2%}'.format(epoch + 1, i + 1, epoch_train_loss, valid_loss, acc_valid))
        valid_loss, acc_valid = valid()
        if acc_valid >= best_accuracy:
            save_param(model, args.saved_models_path)
        print('Epoch: {}\tTrain Loss: {:.6f}\tValid Loss: {:.6f}\t'
              'Valid Accuracy: {:.2%}'.format(epoch + 1, epoch_train_loss / len(train_loader.dataset),
                                              valid_loss, acc_valid))

    print('Finished Training')


def valid():
    corr = 0.0
    total = 0.0
    lossfunc = nn.CrossEntropyLoss()
    with torch.no_grad():
        valid_loss = 0.0
        for data, target in valid_loader:
            # Loss
            target = target.long()
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = lossfunc(pred, target)
            valid_loss += loss.item() * data.size(0)
            # valid_loss += loss.item()

            # 准确率
            pred = model(data)
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            corr += (predicted == target).sum().item()
        valid_loss = valid_loss / len(valid_loader.dataset)
        # valid_loss = valid_loss / 100
    return valid_loss, corr / total


def test():
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy on the test set: %d %%' % (100 * correct / total))


def load_param(path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))


def save_param(model, path):
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    load_param(args.saved_models_path)
    start_time = perf_counter()
    train()
    finish_time = perf_counter()
    train_time = finish_time - start_time
    save_param(model, args.saved_models_path)

    print('\033[1;31mThe training time is %ss\033[0m' % str(train_time))
    test()
