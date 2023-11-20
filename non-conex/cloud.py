import copy
import torch.nn as nn
import torch
import math
import numpy as np

class Cloud():
    def __init__(self,global_weight):
        self.global_weight = global_weight

    def set_learning_rate(self, num_communication, inital_lr):
        return inital_lr / (0.001*num_communication + 1)

    # 0: gradient; 1: one-step model; 2: model differentiation; 3: multi-step model
    def aggregate_0(self, edge_weights, noise_var, lr, device, P, h):
        w_avg = copy.deepcopy(edge_weights[0])
        info_magnitude = torch.zeros([len(edge_weights), 1]).to(device)
        param_num = sum(param.numel() for param in edge_weights[0].values())
        # print(param_num)
        for key in w_avg.keys():
            for i in range(1, len(edge_weights)):
                w_avg[key] += edge_weights[i][key]
                info_magnitude[i] += torch.sum(edge_weights[i][key]**2/(h[i]**2))
            w_avg[key] = torch.div(w_avg[key], len(edge_weights))
        power_scalar = torch.max(info_magnitude)/param_num/P
        # AWGN noise  full power transmit
        if noise_var > 0:
            for key, val in w_avg.items():
                val += torch.normal(torch.zeros_like(val), math.sqrt(noise_var*power_scalar/len(edge_weights))).to(device);
        for param in self.global_weight:
            self.global_weight[param] -= lr*w_avg[param]
            
    def aggregate_1(self, edge_weights, noise_var, device, P, h):
        self.aggregate_3(edge_weights, noise_var, device, P, h)

    def aggregate_2(self, edge_weights, noise_var, device, P, h):
        w_avg = copy.deepcopy(edge_weights[0])
        info_magnitude = torch.zeros([len(edge_weights), 1]).to(device)
        param_num = sum(param.numel() for param in edge_weights[0].values())
        for key in w_avg.keys():
            for i in range(1, len(edge_weights)):
                w_avg[key] += edge_weights[i][key]
                info_magnitude[i] += torch.sum(edge_weights[i][key]**2)/(h[i]**2)
            w_avg[key] = torch.div(w_avg[key], len(edge_weights))
        power_scalar = torch.max(info_magnitude)/param_num/P
        # AWGN noise  full power transmit
        if noise_var > 0:
            for key, val in w_avg.items():
                val += torch.normal(torch.zeros_like(val), math.sqrt(noise_var*power_scalar/len(edge_weights))).to(device);
        for param in self.global_weight:
            self.global_weight[param] = self.global_weight[param] - w_avg[param]

    def aggregate_3(self, edge_weights, noise_var, device, P, h):
        w_avg = copy.deepcopy(edge_weights[0])
        info_magnitude = torch.zeros([len(edge_weights), 1]).to(device)
        param_num = sum(param.numel() for param in edge_weights[0].values())
        for key in w_avg.keys():
            for i in range(1, len(edge_weights)):
                w_avg[key] += edge_weights[i][key]
                info_magnitude[i] += torch.sum(edge_weights[i][key]**2/(h[i]**2))
            w_avg[key] = torch.div(w_avg[key], len(edge_weights))
        power_scalar = torch.max(info_magnitude)/param_num/P*len(edge_weights)
        # AWGN noise  full power transmit
        if noise_var > 0:
            for key, val in w_avg.items():
                val += torch.normal(torch.zeros_like(val), math.sqrt(noise_var*power_scalar/len(edge_weights))).to(device);
        self.global_weight = w_avg