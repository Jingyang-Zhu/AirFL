import os
import numpy as np
import matplotlib.pyplot as plt

savePath = "./imgs/"
if not os.path.exists(savePath):
    os.makedirs(savePath)
filename = "mnist_bathSize10_lr0.1_50users_seed1_noniid_10localStep_100communication_noiseVar0.0_case0_InverseLr_0"

data = np.load("./logs/"+filename+".npz")
acc = data['acc']
loss = data['loss']

f1 = plt.figure(1)
plt.plot(acc,linewidth = 1, color = 'b', label = "accuracy")
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Test Accuracy')
f1.savefig(savePath+"[ACC]"+filename+".png")

f2 = plt.figure(2)
plt.plot(loss,linewidth = 1, color = 'b', label = "loss")
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Test Loss')
f2.savefig(savePath+"[LOSS]"+filename+".png")

plt.show()