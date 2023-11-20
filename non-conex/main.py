import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from os.path import dirname, abspath, join
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xlsxwriter
from option import args_parser
from utils import Initial, FedAvg
import scipy.io
matlab_data = scipy.io.loadmat('channel.mat')
h = matlab_data['h']


if __name__ == '__main__':

    savePath = "./logs/"
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # Setting
    args = args_parser()
    filename = "{dataset}_bathSize{bath_size}_lr{lr}_{num_users}users_seed{seed}_{iid}iid_{num_local_update}localStep_{num_communication}communication_noiseVar{noise_var}_case{case}".format(
    	dataset = args.dataset,
    	bath_size = args.batch_size,
    	lr = args.lr,
    	num_users = args.num_users,
    	seed = args.seed,
    	iid = "" if args.iid else "non",
    	num_local_update = args.num_local_update,
    	num_communication = args.num_communication,
    	noise_var = args.noise_var,
    	case = args.case)
    otherInfo = "_InverseLr_0"
    filename = filename + otherInfo
    print("Save in " + savePath + filename + ".xlsx")
    # Initalization and training
    users, cloud, global_model, total_testloader, device = Initial(args)
    acc, loss = FedAvg(users, cloud, global_model, total_testloader, device, args, h)
    #np.savez("./logs/"+filename+".npz", acc = acc, loss = loss)
    workbook = xlsxwriter.Workbook("./logs/"+filename+".xlsx")
    worksheet = workbook.add_worksheet("sheet1")
    worksheet.write(0,0,'loss')
    worksheet.write(0,1,'accuracy')
    for i in range(0,args.num_communication):
        worksheet.write(i+1,0,loss[i])
        worksheet.write(i+1,1,acc[i])

    workbook.close()