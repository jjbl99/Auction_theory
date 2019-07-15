import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
#import utils
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import distributions
import time
from torch.distributions import Exponential

###QUESTION: Should i merge payment and alloc in a single class?

class Bidder():
    """a bidder represented by a list of distributions (valuation) for each object"""
    def __init__(self, distrib_list): 
        #distrib_list is a list of n_objects distributions
        #distributions are described in distributions.py
        self.distrib_list = distrib_list

class Auction_Environment():
    """Auction Environment = our setup"""
    def __init__(self, n_bidders, n_objects, distrib_list): 
        #here the code is for bidders with same value distributions! distrib_list is a list of m distribs
        #self.seller = seller
        self.n_objects = n_objects
        self.n_bidders = n_bidders
        self.bidders_list = self.initialize_set_bidders(distrib_list)
        if len(distrib_list) != self.n_objects: 
            print('ERROR: need len(distrib_list) == self.n_objects')
    
    def initialize_set_bidders(self,distrib_list): 
        bidders_list = []
        for j in range(self.n_bidders):
            bidders_list.append(Bidder(distrib_list))
        return bidders_list

class Add_Network (nn.Module): 
    """Allocation & Payment networks for NeuralNet for bidders with Additive Valuations"""
    def __init__(self, env, n_hidden = 2, hidden_size = 100): #env is an Auction_Environment
        super().__init__()
        self.n_bidders = env.n_bidders
        self.n_objects = env.n_objects

        #allocation network
        input_size = self.n_bidders * self.n_objects
        net = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for k in range(n_hidden - 1):
            net.append(nn.Linear(hidden_size, hidden_size))
            net.append(nn.Tanh())
        net.append(nn.Linear(hidden_size, (self.n_objects)*(self.n_bidders+1)))
        net.append(nn.Softmax())
        
        self.alloc_net = nn.Sequential(*net)

        #payment network
        net2 = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.Tanh()])
        for k in range(n_hidden - 1):
            net2.append(nn.Linear(hidden_size, hidden_size))
            net2.append(nn.Tanh())
        net2.append(nn.Linear(hidden_size, self.n_bidders))
        net2.append(nn.Softmax())
        self.pay_net = nn.Sequential(*net2)

        #self.full_net = nn.ModuleList([self.alloc_net, self.pay_net])

    def merge(self, x): #x is valuation profile of size (n*m)
        """combines our payment and our allocation rules and gives the payment for each bidder"""
        alloc = self.alloc_net(x)
        pay = self.pay_net(x)
        alloc = alloc.view(self.n_bidders + 1, -1)[:-1] #get rid of line for "proba object not allocated"
        x = x.view(self.n_bidders, self.n_objects)
        coefs = alloc * x
        coefs = torch.sum(coefs, dim = 1).view(1, -1) #dim to check
        final_pay = pay * coefs
        return final_pay #tensor of payment for each player of dim (1, n)

class Trainer (nn.Module):
    def __init__(self, env, net):
        super().__init__()
        self.env = env
        self.net = net
        self.lag = torch.ones(env.n_bidders) #tensor of lagrange multipliers
        #CHANGE THE LAG INIT! MUST BE SOME TRICK 
    
    def utility(self, value, x): #x is a bidding profile, value is a valuation profile
        """returns utility for each player"""
        pay = self.net.pay_net(x) #size n
        alloc = self.net.alloc_net(x) #size (n+1)*m
        alloc = alloc.view(self.env.n_bidders + 1, -1)[:-1]
        value = value.view(self.env.n_bidders, -1)
        utility = torch.sum(value * alloc, dim = 1).view(1, -1) - pay #expected payoff - expected payment
        return utility
    
    def regret(self, value, x):
        """returns regret for each player"""
        return self.utility(value, x) - self.utility(value, value)

    
    def train(self, n_profiles = 30000, batch_size = 128, lrs = [0.1,0.001,1.], R = 20, Z = 100):
        """trains self.net, corresponds to 1 epoch"""
        #lrs is a list of 3 learning rates [misreports, network weights, rho], rho is incremented every 2 epochs
        #in paper: n_profiles = 640000, 80 epochs, R = 25, Z = 100
        count = 0

        #generates all the data
        #CAN BE CHANGED TO GENERATE ONLY 1 MINIBATCH AT A TIME!!! 
        data = []
        for b in self.env.bidders_list: 
            for dis in b.distrib_list:
                data.append(dis.sample(n_profiles))
        data = torch.Tensor(data) 
        self.data = data.t_() #Make your data an attribute
        n_steps = data.size()[0] // batch_size #T in the paper

        time_elapsed = 0

        #vector of expected revenue after looking at each batch
        exp_rev_vect = [torch.Tensor([0]) for _ in range(n_steps)]
        
        while count < n_steps: #replaces the "for t= 0 to T" in the paper
            #take 1 minibatch 
            X = self.data[count*batch_size : (count+1)*batch_size] #size (batch_size, n*m)
            tic = time.time()

            misreps = X.clone() #misreports initialized to be equal to the value of the bidders
            i = 0 #counter due to poor coding skills, allows to iterate over the misreps
            #gradient descent on minibatches (bids that maximizes utility)
            for x in X: #for each valuation profile of size n*m
                a = misreps[i].clone().detach()
                #print(a, x)
                a.requires_grad_(True)
                opt_misrep = torch.optim.Adam([a], lr=lrs[0])
                for r in range(R): #R is the nbr of times you opti misreports
                    opt_misrep.zero_grad()
                    u_pred = - self.utility(x, a) #utility obtained for each player going through the model
                    u_pred.backward(torch.ones(u_pred.size()), retain_graph = True) #EST CE QUE J'AI LE DROIT DE METTRE CE TORCH.ONES???
                    opt_misrep.step()
                    #print(u_pred.detach().numpy())
                #print(a, x)
                #a.requires_grad_(False)
                misreps[i] = a.clone()
                i += 1
                #print(i)

            #regret_vect = torch.empty(batch_size, self.env.n_bidders)
            

            ######### DEBUGGING
            with torch.autograd.set_detect_anomaly(True):
                #now we optimize the network parameters
                opt_weights = torch.optim.Adam(self.net.parameters(), lr=lrs[1])
                opt_weights.zero_grad()
                #Compute augmented lagrangian
                f_lagrange = torch.Tensor([0]) #lagrange function C_rho
                #f_lagrange.requires_grad_(True)
                L = X.size()[0]
                for l in range(L): #for each profile in the minibatch 
                    X[l].requires_grad_(True)
                    exp_rev_vect[count] += 1/L * torch.sum(self.net.merge(X[l])) #just to test that revenue increases
                    f_lagrange -= 1/L * torch.sum(self.net.merge(X[l])) #first term of the Lagrangian, sum of the final payments of each player for the given valuation profile
                    f_lagrange += 1/L * torch.sum(self.lag * self.regret(X[l], misreps[l])) #second term
                    f_lagrange += 1/2 * lrs[2] * (torch.sum(self.regret(X[l], misreps[l]))) ** 2 #third

                #print(count, f_lagrange, exp_rev_vect[count])

                #optimize lagrangian
                f_lagrange.backward(retain_graph = True)
                opt_weights.step()
            ######### DEBUGGING

            if count % Z == 0: #once every Z operations
                #upgrade lagrange coefficients
                regret_vect = torch.zeros(self.env.n_bidders).view(1,-1)
                L = X.size()[0]
                for l in range(L):
                    regret_vect += 1/L * self.regret(X[l], misreps[l])
                self.lag = self.lag + lrs[2] * regret_vect
            
            if count % 2 == 0: 
                #increment rho lagrangian TO CHECK
                lrs[2] += 0.1
            
            #if count % 5 == 1:
            #    plt.plot([i for i in range(count+1)], exp_rev_vect[:count+1].detach().numpy())
            #    plt.show()
            count += 1
            toc = time.time()
            time_elapsed += (toc - tic)
            print('code has been running for {} sec, {} %, exp revenue: {}'.format(round(time_elapsed,2),round(count/n_steps*100, 2), exp_rev_vect[count].detach().numpy()))

class Adversarial (nn.Module): 
    ## LATER ! 
    def __init__(self, env, net):
        super().__init__()
        self.env = env
        self.net = net

    def adv_train(self, n_epochs = 2):
        trainer = Trainer(self.env, self.net)
        for _ in range (n_epochs):
            trainer.train()
        #compute value of R after training
        opt_R = torch.optim.Adam(trainer.env.bidders_list[0])
        R = 0
        #change the distrib of my first bidder using gradient descent 
        #trainer.env.bidders_list[0]s


def main (env, nb_steps=500, lr=0.0001, size_batch = 2500):
    n_bidders = 1 
    n_objects = 2
    distrib_list = [distributions.UniformDistrib(), distributions.UniformDistrib()]
    env = Auction_Environment(n_bidders, n_objects, distrib_list)