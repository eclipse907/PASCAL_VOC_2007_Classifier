import multiprocessing
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score
from random import randint
from custom_datasets import CustomLabeledDataset


def create_label(input):
    object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']
    labels = input['annotation']['object']
    label_classes = []
    if type(labels) == dict:
        if int(labels['difficult']) == 0:
            label_classes.append(object_categories.index(labels['name']))
    else:
        for i in range(len(labels)):
            if int(labels[i]['difficult']) == 0:
                label_classes.append(object_categories.index(labels[i]['name']))
    correct = np.zeros(len(object_categories))
    correct[label_classes] = 1
    return torch.from_numpy(correct)


def get_average_precision(y_true, y_scores):
    scores = 0.0
    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true=y_true[i], y_score=y_scores[i])
    return scores


def main():
    directory = './data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Selected device: " + str(device))
    if not os.path.exists(directory):
        os.mkdir(directory)
    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
    transformations = transforms.Compose([transforms.Resize((300, 300)),
                                          transforms.RandomChoice([
                                              transforms.ColorJitter(brightness=(0.80, 1.20)),
                                              transforms.RandomGrayscale(p=0.25)
                                          ]),
                                          transforms.RandomHorizontalFlip(p=0.25),
                                          transforms.RandomRotation(25),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std),
                                          ])
    transformations_test = transforms.Compose([transforms.Resize(330),
                                               transforms.FiveCrop(300),
                                               transforms.Lambda(lambda crops: torch.stack(
                                                   [transforms.ToTensor()(crop) for crop in crops])),
                                               transforms.Lambda(lambda crops: torch.stack(
                                                   [transforms.Normalize(mean=mean, std=std)(crop) for crop in crops])),
                                               ])
    train_set = datasets.VOCDetection(directory, year="2007", image_set="train", transform=transformations,
                                      target_transform=create_label, download=True)
    test_set = datasets.VOCDetection(directory, year="2007", image_set="val", transform=transformations_test,
                                     target_transform=create_label, download=True)
    num_of_data = int(input("Enter number of data to train on: "))
    if num_of_data == len(train_set):
        train_dataset = train_set
    else:
        indexes = list()
        while len(indexes) < num_of_data:
            index = randint(0, len(train_set) - 1)
            if index not in indexes:
                indexes.append(index)
        train_dataset = CustomLabeledDataset(train_set, indexes)
    batch_size = int(input("Enter desired batch size: "))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=multiprocessing.cpu_count())
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=10, pin_memory=True)

    model = models.resnet18(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(model.fc.in_features, 20)
    model.to(device)
    loss_func = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.SGD([
        {'params': list(model.parameters())[:-1], 'lr': 1e-5, 'momentum': 0.9},
        {'params': list(model.parameters())[-1], 'lr': 5e-3, 'momentum': 0.9}
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

    print("Started training.")

    for epoch in range(15):
        for i, data in enumerate(train_loader, 1):
            inputs, labels = data[0].to(device), data[1].float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("Epoch: %d, batch: %d, loss: %f" % (epoch + 1, i, loss.item()))
        scheduler.step()

    print("Finished Training.")
    print("Evaluating model on train set.")

    average_precision = 0.0
    average_loss = 0.0
    m = nn.Sigmoid()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].float().to(device)
            bs, ncrops, c, h, w = inputs.size()
            outputs = model(inputs.view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1)
            average_loss += loss_func(outputs, labels).item()
            average_precision += get_average_precision(torch.Tensor.cpu(labels).detach().numpy(),
                                                       torch.Tensor.cpu(m(outputs)).detach().numpy())

    # print("Average precision of neural network is: %.2f%%" % (100 * (average_precision / len(test_set))))
    # print("Average loss of neural network is: %f" % (average_loss / len(test_set)))
    return len(train_dataset), round(average_precision / len(test_set), 2)
