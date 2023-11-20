import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from os.path import dirname, abspath, join
from torch.autograd import Variable
import numpy as np
import copy
from option import args_parser
from read_mnist import get_dataset
from model import CNN,CNNCifar
from user import User
from cloud import Cloud



def fast_inference(total_testloader, model, device):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(total_testloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Inference
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()*labels.size(0)/len(total_testloader.dataset)
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct/total
        return accuracy, loss

def FedAvg(users, cloud, model, total_testloader, device, args, h):
    users = copy.deepcopy(users)
    cloud = copy.deepcopy(cloud)
    global_model = copy.deepcopy(model)
    test_accuracy, test_loss = [], []
    users_id = np.arange(args.num_users)
    #Federated learning begin to train
    print(args.algorithm)
    initial_lr = args.lr
    for num_com in range(args.num_communication):
        cur_lr = cloud.set_learning_rate(num_com, initial_lr)

        # local update
        message_lst = []
        # scheduling_users_group = np.random.choice(users_id, 30, replace=False)
        scheduling_users_group = users_id
        for user_id in scheduling_users_group:
            if args.case == 0: # gradient
                message = users[user_id].local_update_0(cur_lr)
            elif args.case == 1: # one-step model
                message = users[user_id].local_update_1(cur_lr)
            elif args.case == 2: # model differentiation
                message = users[user_id].local_update_2(cur_lr)
            else: # multi-step model
                message = users[user_id].local_update_3(cur_lr)
            message_lst.append(message)

        # cloud aggregation
        if args.case == 0: # gradient
            cloud.aggregate_0(message_lst, args.noise_var, cur_lr, device, args.P, h[:, args.num_communication-1])
        elif args.case == 1: # one-step model
            cloud.aggregate_1(message_lst, args.noise_var, device, args.P, h[:, args.num_communication-1])
        elif args.case == 2: # model differentiation
            cloud.aggregate_2(message_lst, args.noise_var, device, args.P, h[:, args.num_communication-1])
        else: # multi-step model
            cloud.aggregate_3(message_lst, args.noise_var, device, args.P, h[:, args.num_communication-1])
        global_weight = copy.deepcopy(cloud.global_weight)

        # model disseminat
        for user_id in users_id:
            users[user_id].model_update(global_weight)
        global_model.load_state_dict(global_weight)

        # inference
        acc, loss = fast_inference(total_testloader, global_model, device)
        test_accuracy.append(acc)
        test_loss.append(loss)
        print('(%d|%d) Test Loss: %.3f | Test Acc: %.3f%% | lr = %.3f' % (num_com+1, args.num_communication, loss, acc*100, cur_lr))

    return test_accuracy, test_loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def Initial(args):
    setup_seed(args.seed)
    if args.cuda:
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(device)
    if args.dataset == 'mnist':
        global_model = CNN(input_channels = args.input_channels,output_channels =  args.output_channels)
    else:
        global_model = CNNCifar(input_channels = args.input_channels,output_channels =  args.output_channels)
    #load dataset
    user_trainloader, user_testloader, total_trainloader, total_testloader = get_dataset(args)
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # copy weights
    global_weight = global_model.state_dict()
    # Initialize users, cloud
    users = []
    users_id = np.arange(args.num_users)
    for user_id in users_id:
        users.append(User(id=user_id,
                          train=user_trainloader[user_id],
                          test=user_testloader[user_id],
                          args=args,
                          model=copy.deepcopy(global_model),
                          device=device))
    cloud = Cloud(global_weight=global_weight)
    return users, cloud, global_model, total_testloader, device