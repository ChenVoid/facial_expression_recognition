import glob
import sys

import os
from os import path as osp
import numpy as np
import math
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from thop import profile
from sklearn.metrics import confusion_matrix
from torchvision import transforms, datasets
from torch.utils import data
from time import perf_counter

from construct_dataset import Mydatasetpro
from models.CNN import ConvNet
from models.MLP import MLP4
from models.resnet import resnet50, resnet18

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

# 对数据进行转换处理
transform = transforms.Compose([
    # transforms.Resize((48, 48)),  # 做的第一步转换
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
    transforms.Normalize(mean, std)
])

# data_transforms = {
#     'train': transforms.Compose({
#         # transforms.CenterCrop(48),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean, std)
#     }),
#     'valid': transforms.Compose({
#         # transforms.CenterCrop(48),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean, std)
#     }),
#     'test': transforms.Compose({
#         # transforms.CenterCrop(48),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean, std),
#         # transforms.ToPILImage()
#     })
# }

# data_transforms = {
#     'train': transforms.Compose({
#         transforms.CenterCrop(48),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     }),
#     'test': transforms.Compose({
#         transforms.CenterCrop(48),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     })
# }

labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
num_of_class = len(labels)

tick_marks = np.array(range(len(labels))) + 0.5

model_type = ['ResNet', 'MLP4', 'ConvNet', 'ResNet18']
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
parser.add_argument('--saved_models_path', type=str, default='./checkpoint/'
                                                             + model_name + '.pt', help='模型保存')
parser.add_argument('--nepoches', type=str, default=100, help='迭代次数')
parser.add_argument('--batch_size', type=str, default=32, help='Batch大小')
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

# 使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob('./dataset4train/CKPlus/CKPlus48/*/*.png')
print(len(all_imgs_path))

species = labels
species_to_id = dict((c, i) for i, c in enumerate(species))
# print(species_to_id)
id_to_species = dict((v, k) for k, v in species_to_id.items())
# print(id_to_species)
all_labels = []
# 对所有图片路径进行迭代
for img in all_imgs_path:
    # 区分出每个img，应该属于什么类别
    for i, c in enumerate(species):
        if c in img:
            all_labels.append(i)
print(len(all_labels))  # 得到所有标签

images_dataset = Mydatasetpro(all_imgs_path, all_labels, transform)

images_datalodaer = data.DataLoader(
    images_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

imgs_batch, labels_batch = next(iter(images_datalodaer))
print(imgs_batch.shape)

plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
    img = img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i + 1)
    plt.title(id_to_species.get(label.item()))
    plt.imshow(img)
plt.show()

# 划分测试集和训练集
index = np.random.permutation(len(all_imgs_path))

all_imgs_path = np.array(all_imgs_path)[index]
all_labels = np.array(all_labels)[index]

# 80% as train
train_index = int(len(all_imgs_path) * 0.8)
print(train_index)

test_index = int(len(all_imgs_path) * 0.1)
print(test_index)

train_imgs = all_imgs_path[:train_index]
train_labels = all_labels[:train_index]
test_imgs = all_imgs_path[train_index:train_index + test_index]
test_labels = all_labels[train_index:train_index + test_index]
valid_imgs = all_imgs_path[train_index + test_index:]
valid_labels = all_labels[train_index + test_index:]

train_dataset = Mydatasetpro(train_imgs, train_labels, transform)  # TrainSet TensorData
test_dataset = Mydatasetpro(test_imgs, test_labels, transform)  # TestSet TensorData
valid_dataset = Mydatasetpro(valid_imgs, valid_labels, transform)

# data_directory = './dataset4train/MMAFEDB'
# train_dataset = datasets.ImageFolder(os.path.join(data_directory, 'train'), data_transforms['train'])
# test_dataset = datasets.ImageFolder(os.path.join(data_directory, 'test'), data_transforms['test'])
# valid_dataset = datasets.ImageFolder(os.path.join(data_directory, 'valid'), data_transforms['valid'])
#
train_loader, valid_loader, test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), \
    DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True), \
    DataLoader(test_dataset, batch_size=args.batch_size)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

if args.runtype in model_type:
    if args.runtype == 'ResNet':
        model = resnet50(num_classes=num_of_class).to(device)
    elif args.runtype == 'ResNet18':
        model = resnet18(num_classes=num_of_class).to(device)
    elif args.runtype == 'MLP4':
        model = MLP4(firstBN).to(device)
    elif args.runtype == 'ConvNet':
        model = ConvNet(num_classes=num_of_class).to(device)


def train():
    # 定义损失函数和优化器
    lossfunc = nn.CrossEntropyLoss()
    # lossfunc = nn.NLLLoss()
    # optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    best_accuracy = 0
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(args.nepoches):
        train_loss = 0.0
        for data, target in train_loader:
            target = target.long()
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            pred = model(data)
            loss = lossfunc(pred, target)
            loss.backward()
            optim.step()
            train_loss += loss.item() * data.size(0)
        valid_loss, acc_valid = valid()
        if acc_valid >= best_accuracy:
            torch.save(model, args.saved_models_path)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch: {}\tTraining Loss: {:.6f}\tValid Loss: {:.6f}\tValid Accuracy: {:.2%}'.format(epoch + 1, train_loss,
                                                                                                  valid_loss,
                                                                                                  acc_valid))
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
    return train_loss_list, valid_loss_list


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

            # 准确率
            pred = model(data)
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0)
            corr += (predicted == target).sum().item()
        valid_loss = valid_loss / len(valid_loader.dataset)
    return valid_loss, corr / total


def test():
    model = torch.load(args.saved_models_path).to(device)
    total = 0.0
    corr = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            y_true.extend(target.to('cpu'))
            data, target = data.to(device), target.to(device)
            pred = model(data)
            _, predicted = torch.max(pred.data, 1)
            y_pred.extend(predicted.to('cpu'))
            total += target.size(0)
            corr += (predicted == target).sum().item()
    print("Test Accuracy: {:.2%}".format(corr / total))
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    start_time = perf_counter()
    train_loss_list, valid_loss_list = train()
    finish_time = perf_counter()
    train_time = finish_time - start_time

    print('\033[1;31mThe training time is %ss\033[0m' % str(train_time))
    y_true, y_pred = test()

    # 模型参数规模
    input = torch.randn(args.batch_size, 1, 48, 48).to(device)
    flops, params = profile(model, inputs=(input,))
    print('flops:', flops / 1e6, 'params:', params / 1e6)

    # 学习曲线
    plt.figure()
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.plot(range(len(valid_loss_list)), valid_loss_list)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.savefig(args.train_curve_name)
    # plt.show()

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm_normalized)

    plt.figure(figsize=(12, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    # for x_val, y_val in zip(x.flatten(), y.flatten()):
    #     c = cm_normalized[y_val][x_val]
    #     if c > 0.01:
    #         plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig('confusion_matrix_' + model_name + '_' + firstBN_name + '.png', format='png')
    # plt.show()
