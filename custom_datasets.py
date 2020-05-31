import torch.utils.data as data


class CustomLabeledDataset(data.Dataset):

    def __init__(self, dataset, list_of_indexes):
        self.dataset = dataset
        self.list_of_indexes = list_of_indexes

    def __len__(self):
        return len(self.list_of_indexes)

    def __getitem__(self, item):
        return self.dataset.__getitem__(self.list_of_indexes[item])


class CustomUnlabeledDataset(data.Dataset):

    def __init__(self, dataset, list_of_indexes):
        self.dataset = dataset
        self.list_of_indexes = list_of_indexes

    def __len__(self):
        return len(self.list_of_indexes)

    def __getitem__(self, item):
        data = self.dataset.__getitem__(self.list_of_indexes[item])
        return self.list_of_indexes[item], data[0]
