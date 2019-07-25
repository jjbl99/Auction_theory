import importlib
import regretNet_pytorch_2
importlib.reload(regretNet_pytorch_2)
from regretNet_pytorch_2 import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#importlib.reload(losses)
#importlib.reload(utils)
import numpy as np
import distributions
importlib.reload(distributions)

n_epochs = 3
n_bidders = 1
n_objects = 2
distrib_list = [distributions.UniformDistrib(), distributions.UniformDistrib()]
env = Auction_Environment(n_bidders, n_objects, distrib_list)
net = Add_Network(env)
#print(net.alloc_net)
#print(net.pay_net)
torch.cuda.set_device(0)

trainer = Trainer(env, net, n_epochs)
trainer.net.cuda()
for q in range(n_epochs):
    print('###################################################################################')
    print('\n')
    print('EPOCH NUMBER: ', q+1)
    print('\n')
    print('###################################################################################')
    trainer.train()
    torch.save(trainer.net, './model_one_bidder_two_objects')

net = trainer.net

st = 0
ed = 1
size = 200

X = np.linspace(st, ed, size)
Y = np.linspace(st, ed, size)

values = np.zeros((size, size))
values_bis = np.zeros((size, size))

for i in range (len(X)):
    for j in range (len(Y)):
        values_bis[i][j] = net.alloc_net(torch.Tensor([X[i],Y[j]]).type(dtype))[1]
        values[i][j] = net.alloc_net(torch.Tensor([X[i],Y[j]]).type(dtype))[0]


fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (13,13))
img = ax.imshow(values[::-1, :], extent=[0,1,0,1], vmin = 0.0, vmax=1.0, cmap = 'YlOrRd')
plt.title('proba of alloc of object 1', size = 19)
plt.xlabel('v1', size = 15)
plt.ylabel('v2', size = 15)
plt.pcolor(X, Y, values, cmap = 'YlOrRd')
plt.colorbar(img, fraction=0.046, pad=0.04)
plt.savefig('proba_alloc_object1.pdf')
plt.show()


fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (13,13))
img = ax.imshow(values_bis[::-1, :], extent=[0,1,0,1], vmin = 0.0, vmax=1.0, cmap = 'YlOrRd')
plt.title('proba of alloc of object 2', size = 19)
plt.xlabel('v1', size = 15)
plt.ylabel('v2', size = 15)
plt.pcolor(X, Y, values_bis, cmap = 'YlOrRd')
plt.colorbar(img, fraction=0.046, pad=0.04)
plt.savefig('proba_alloc_object2.pdf')
plt.show()
