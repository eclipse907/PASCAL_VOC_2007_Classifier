import multiprocessing
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.metrics import average_precision_score
from torch.distributions.categorical import Categorical
from random import randint
from custom_datasets import CustomLabeledDataset
from custom_datasets import CustomUnlabeledDataset


def least_confidence_sampling(num_of_data_to_add, unlabeled_loader, model, device):
    with torch.no_grad():
        new_indexes = list()
        to_delete = list()
        while len(new_indexes) < num_of_data_to_add:
            for data_unlabeled in unlabeled_loader:
                indexes, images = data_unlabeled[0].to(device), data_unlabeled[1].to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                max_probs, _ = torch.max(probs.data, 1)
                diff = torch.ones_like(max_probs) - max_probs
                _, to_return = torch.topk(diff, 1)
                for index in torch.gather(indexes, 0, to_return).tolist():
                    if len(new_indexes) < num_of_data_to_add:
                        new_indexes.append(index)
                        to_delete.append(index)
                    else:
                        break
                if len(new_indexes) >= num_of_data_to_add:
                    break
            for index in to_delete:
                unlabeled_loader.dataset.list_of_indexes.remove(index)
            to_delete.clear()
        return new_indexes


def margin_sampling(num_of_data_to_add, unlabeled_loader, model, device):
    with torch.no_grad():
        new_indexes = list()
        to_delete = list()
        while len(new_indexes) < num_of_data_to_add:
            for data_unlabeled in unlabeled_loader:
                indexes, images = data_unlabeled[0].to(device), data_unlabeled[1].to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                two_biggest_maxs, _ = torch.topk(probs, 2, dim=1)
                diff = torch.abs(two_biggest_maxs[:, 0] - two_biggest_maxs[:, 1])
                _, to_return = torch.topk(diff, 1, largest=False)
                for index in torch.gather(indexes, 0, to_return).tolist():
                    if len(new_indexes) < num_of_data_to_add:
                        new_indexes.append(index)
                        to_delete.append(index)
                    else:
                        break
                if len(new_indexes) >= num_of_data_to_add:
                    break
            for index in to_delete:
                unlabeled_loader.dataset.list_of_indexes.remove(index)
            to_delete.clear()
        return new_indexes


def entropy_sampling(num_of_data_to_add, unlabeled_loader, model, device):
    with torch.no_grad():
        new_indexes = list()
        to_delete = list()
        while len(new_indexes) < num_of_data_to_add:
            for data_unlabeled in unlabeled_loader:
                indexes, images = data_unlabeled[0].to(device), data_unlabeled[1].to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                entropy = Categorical(probs=probs).entropy()
                _, to_return = torch.topk(entropy, 1)
                for index in torch.gather(indexes, 0, to_return).tolist():
                    if len(new_indexes) < num_of_data_to_add:
                        new_indexes.append(index)
                        to_delete.append(index)
                    else:
                        break
                if len(new_indexes) >= num_of_data_to_add:
                    break
            for index in to_delete:
                unlabeled_loader.dataset.list_of_indexes.remove(index)
            to_delete.clear()
        return new_indexes


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


def main(sampling_parameter=None):
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
    seed_size = int(input("Enter desired number of data in initial seed: "))
    train_batch_size = int(input("Enter size of batch for training: "))
    unlabeled_batch_size = int(input("Enter size of batch from which to add unlabeled data: "))
    num_of_data_to_add = int(input("Enter number of new data to add to the train set in each epoch: "))
    while True:
        if not sampling_parameter:
            method = input("Choose sampling method for unlabeled data from the available methods:\n"
                           "1 - Least Confidence Sampling\n"
                           "2 - Margin Sampling\n"
                           "3 - Entropy Sampling\n"
                           "Enter one of the numbers: ")
        else:
            method = sampling_parameter
        try:
            number = int(method)
            if number == 1:
                sampling_method = least_confidence_sampling
                break
            elif number == 2:
                sampling_method = margin_sampling
                break
            elif number == 3:
                sampling_method = entropy_sampling
                break
            else:
                raise ValueError
        except ValueError:
            print("Wrong argument entered.")
    labeled_data_indexes = list()
    while len(labeled_data_indexes) < seed_size:
        index = randint(0, len(train_set) - 1)
        if index not in labeled_data_indexes:
            labeled_data_indexes.append(index)
    labeled_dataset = CustomLabeledDataset(train_set, labeled_data_indexes)
    unlabeled_data_indexes = list()
    for index in range(0, len(train_set)):
        if index not in labeled_data_indexes:
            unlabeled_data_indexes.append(index)
    unlabeled_dataset = CustomUnlabeledDataset(train_set, unlabeled_data_indexes)
    labeled_loader = data.DataLoader(labeled_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                                     num_workers=multiprocessing.cpu_count())
    unlabeled_loader = data.DataLoader(unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=True,
                                       num_workers=multiprocessing.cpu_count())
    test_loader = data.DataLoader(dataset=test_set, batch_size=10, pin_memory=True)

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

    for data_labeled in labeled_loader:
        inputs, labels = data_labeled[0].to(device), data_labeled[1].float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    m = nn.Sigmoid()
    sizes_of_data = list()
    average_precisions = list()
    for epoch in range(15):
        print("Started training in epoch %d." % (epoch + 1))
        if len(unlabeled_dataset) > 0:
            labeled_dataset.list_of_indexes += sampling_method(num_of_data_to_add, unlabeled_loader, model, device)
        for i, data_labeled in enumerate(labeled_loader, 1):
            inputs, labels = data_labeled[0].to(device), data_labeled[1].float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("Epoch: %d, batch: %d, loss: %f" % (epoch + 1, i, loss.item()))
        scheduler.step()
        print("Finished training in epoch %d." % (epoch + 1))
        print("Evaluating model on train set.")
        average_precision = 0.0
        average_loss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                inputs, labels = test_data[0].to(device), test_data[1].float().to(device)
                bs, ncrops, c, h, w = inputs.size()
                outputs = model(inputs.view(-1, c, h, w))
                outputs = outputs.view(bs, ncrops, -1).mean(1)
                average_loss += loss_func(outputs, labels).item()
                average_precision += get_average_precision(torch.Tensor.cpu(labels).detach().numpy(),
                                                           torch.Tensor.cpu(m(outputs)).detach().numpy())
        average_precision = round(average_precision / len(test_set), 2)
        # average_loss = average_loss / len(test_set)
        sizes_of_data.append(len(labeled_dataset))
        average_precisions.append(average_precision)
        print("Evaluation complete.")
        # print("Average precision of neural network on %d samples is: %.2f%%" % (len(labeled_dataset),
        #                                                                         average_precision))
        # print("Average loss of neural network on %d samples is: %f" % (len(labeled_dataset), average_loss))

    return sizes_of_data, average_precisions
