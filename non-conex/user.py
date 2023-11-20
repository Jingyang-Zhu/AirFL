from torch.autograd import Variable
import torch
import torch.optim as optim
import copy
import torch.nn as nn
import numpy as np
from option import args_parser
class User():
    def __init__(self, id, train,test,model,args,device):
        self.id = id
        self.train = train
        self.test = test
        self.model = model
        self.optimizer = optim.SGD(params = model.parameters(),
                                  lr = args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = args.batch_size
        self.num_local_update = args.num_local_update
        self.device = device

    def local_update_0(self, lr):
        grad = copy.deepcopy(self.model.state_dict())
        self.optimizer.param_groups[0]['lr'] = lr
        for batch_idx, (inputs, labels) in enumerate(self.train):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            log_probs = self.model(inputs)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            self.optimizer.step()
            if batch_idx >= 0: break
        for key, param in self.model.named_parameters():
            grad[key] = param.grad.data.detach()
        return grad

    def local_update_1(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
        for batch_idx, (inputs, labels) in enumerate(self.train):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            log_probs = self.model(inputs)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            self.optimizer.step()
            if batch_idx >= 0: break
        return copy.deepcopy(self.model.state_dict())

    def local_update_2(self, lr):
        diff_model = copy.deepcopy(self.model.state_dict())
        self.optimizer.param_groups[0]['lr'] = lr
        for batch_idx, (inputs, labels) in enumerate(self.train):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            log_probs = self.model(inputs)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            self.optimizer.step()
            if batch_idx >= self.num_local_update - 1: break
        for param in self.model.state_dict():
            diff_model[param] -= self.model.state_dict()[param]
        return diff_model

    def local_update_3(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
        for batch_idx, (inputs, labels) in enumerate(self.train):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            log_probs = self.model(inputs)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            self.optimizer.step()
            if batch_idx >= self.num_local_update - 1: break
        return copy.deepcopy(self.model.state_dict())

    def model_update(self,weight):
        self.model.load_state_dict(weight)

    # def inference(self,weight):
    #     self.model.load_state_dict(weight)
    #     self.model.eval()
    #     loss, total, correct = 0.0, 0.0, 0.0
    #     for batch_idx, (inputs, labels) in enumerate(self.test):
    #         inputs, labels = inputs.to(self.device), labels.to(self.device)
    #
    #         # Inference
    #         outputs = self.model(inputs)
    #         batch_loss = self.criterion(outputs, labels)
    #         loss += batch_loss.item()
    #
    #         # Prediction
    #         _, pred_labels = torch.max(outputs, 1)
    #         pred_labels = pred_labels.view(-1)
    #         correct += torch.sum(torch.eq(pred_labels, labels)).item()
    #         total += len(labels)
    #
    #     accuracy = correct/total
    #     return accuracy, loss