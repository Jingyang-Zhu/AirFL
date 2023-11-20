import numpy as np
import torch
import torch.backends.cudnn as cudnn
cudnn.banchmark = True
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from option import args_parser



def mnist_iid(dataset, num_users):
    num_items = len(dataset)//num_users
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    # 60,000 training imgs -->  (num_users*2) shards
    per_shard = 2
    num_shards = num_users*per_shard
    num_imgs = len(dataset)//num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/users
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, per_shard, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def cifar10_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar10_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    per_shard = 1
    num_shards = num_users*per_shard
    num_imgs = len(dataset)//num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, per_shard, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_dataset(args):
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST(args.dataset_path, train =True, download=True,
                                       transform = transform)
        test_dataset = datasets.MNIST(args.dataset_path, train = False, download=True,
                                      transform = transform)
        # sample training data amongst users
            # Sample IID user data from Mnist
        if args.iid:
            user_groups_train = mnist_iid(train_dataset, args.num_users)
            # Sample NIID user data from Mnist
        else:
            user_groups_train = mnist_noniid(train_dataset, args.num_users)
        user_groups_test = mnist_iid(test_dataset, args.num_users)
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(args.dataset_path, train=True, download=True,
                                       transform=transform_train)

        test_dataset = datasets.CIFAR10(args.dataset_path, train=False, download=True,
                                      transform=transform_test)
        # sample training data amongst users
            # Sample IID user data from Cifar10
        if args.iid:
            user_groups_train = cifar10_iid(train_dataset, args.num_users)
            # Sample NIID user data from Cifar10
        else:
            user_groups_train = cifar10_noniid(train_dataset, args.num_users)
        user_groups_test = cifar10_iid(test_dataset, args.num_users)
    else:
        raise NotImplementedError()

    # trainloader,testloader = {},{}
    # for user_id in range(args.num_users):
    #     trainloader[user_id] = DataLoader(DatasetSplit(train_dataset, user_groups_train[user_id]),
    #                              batch_size = args.batch_size, shuffle=True)
    #     testloader[user_id] = DataLoader(DatasetSplit(test_dataset, user_groups_test[user_id]),
    #                             batch_size = args.batch_size, shuffle=False)
    #     v_train_loader = DataLoader(train_dataset, batch_size=args.batch_size * args.num_users,
    #                                 shuffle=True)
    #     v_test_loader = DataLoader(test_dataset, batch_size=args.batch_size * args.num_users,
    #                                shuffle=False)
    # return trainloader, testloader, v_train_loader, v_test_loader

    trainloader,testloader = {},{}
    for user_id in range(args.num_users):
        trainloader[user_id] = DataLoader(DatasetSplit(train_dataset, user_groups_train[user_id]),
                                 batch_size = args.batch_size, shuffle=True)
        # testloader[user_id] = DataLoader(DatasetSplit(test_dataset, user_groups_test[user_id]),
        #                         batch_size = args.batch_size, shuffle=False)
        # v_train_loader = DataLoader(train_dataset, batch_size=args.batch_size * args.num_users,
        #                             shuffle=True)
        testloader[user_id] = None
        v_train_loader = None
        v_test_loader = DataLoader(test_dataset, batch_size=256,
                                   shuffle=False)
    return trainloader, testloader, v_train_loader, v_test_loader


if __name__ == '__main__':
    args = args_parser()
    trainloader, testloader, v_train_loader, v_test_loader = get_mnist(args)
    print("Successful")