import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import distributions
import time
from torch.distributions import Exponential

class Bidder():
    """a bidder represented by a list of distributions (valuation) for each object"""
    def __init__(self, distrib_list):
        #distrib_list is a list of n_objects distributions
        #distributions are described in distributions.py
        self.distrib_list = distrib_list

class Auction_Environment():
    """Auction Environment = our setup"""
    def __init__(self, n_bidders, n_objects, distrib_list, batch_size = 128):
        #here the code is for bidders with same value distributions! distrib_list is a list of m distribs
        #self.seller = seller
        self.n_objects = n_objects
        self.n_bidders = n_bidders
        self.bidders_list = self.initialize_set_bidders(distrib_list)
        self.L = batch_size
        if len(distrib_list) != self.n_objects:
            print('ERROR: need len(distrib_list) == self.n_objects')

    def initialize_set_bidders(self, distrib_list):
        bidders_list = []
        for _ in range(self.n_bidders):
            bidders_list.append(Bidder(distrib_list))
        return bidders_list

class Alloc_Net_Additive(nn.Module):
    def __init__(self, env, n_hidden, hidden_size):
        super().__init__()
        self.n_bidders = env.n_bidders
        self.n_objects = env.n_objects
        self.n_hidden = n_hidden

        input_size = self.n_bidders * self.n_objects

        self.net = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        for _ in range(n_hidden - 1):
            self.net.append(nn.Linear(hidden_size, hidden_size))
        self.net.append(nn.Linear(hidden_size, (self.n_objects)*(self.n_bidders+1)))

    def forward(self, x):
        for i in range(self.n_hidden):
            x = self.net[i](x)
            x = torch.tanh(x)
        x = self.net[-1](x)
        x = x.view(-1, self.n_bidders + 1, self.n_objects)
        x = F.softmax(x, dim = 1)
        return x.view(1, -1)

class Alloc_Net_Unit(nn.Module):
    def __init__(self, env, n_hidden, hidden_size):
        super().__init__()
        self.n_bidders = env.n_bidders
        self.n_objects = env.n_objects
        self.n_hidden = n_hidden

        input_size = self.n_bidders * self.n_objects

        self.net = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        for _ in range(n_hidden - 1):
            self.net.append(nn.Linear(hidden_size, hidden_size))
        self.net.append(nn.Linear(hidden_size, 2*(self.n_objects)*(self.n_bidders+1))) # REMOVE 2*

    def forward(self, x):
        for i in range(self.n_hidden):
            x = self.net[i](x)
            x = torch.tanh(x)
        x = self.net[-1](x)
        
        # added
        # x = x.view(-1, self.n_bidders + 1, self.n_objects)
        # x = torch.min(F.softmax(x, dim = 0), F.softmax(x, dim = 1))

        x1 = x[:, : self.n_objects * (self.n_bidders+1)].view(-1, self.n_bidders + 1, self.n_objects)
        x2 = x[:, self.n_objects * (self.n_bidders+1) : ].view(-1, self.n_bidders + 1, self.n_objects)
        x1 = F.softmax(x1, dim = 0)
        x2 = F.softmax(x2, dim = 1)
        x = torch.min(x1, x2)
        return x.view(1, -1)

class Alloc_Net_Combinatorial(nn.Module):
    def __init__(self, env, n_hidden, hidden_size):
        super().__init__()
        self.n_bidders = env.n_bidders
        self.n_objects = env.n_objects
        self.n_hidden = n_hidden

        input_size = self.n_bidders * self.n_objects

        self.net = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        for _ in range(n_hidden - 1):
            self.net.append(nn.Linear(hidden_size, hidden_size))
        self.net.append(nn.Linear(hidden_size, (self.n_objects)*(self.n_bidders+1)))

    def forward(self, x):
        for i in range(self.n_hidden):
            x = self.net[i](x)
            x = torch.tanh(x)
        x = self.net[-1](x)
        x = x.view(-1, self.n_bidders + 1, self.n_objects)
        x = F.softmax(x, dim = 1)
        return x.view(1, -1)

class Pay_Net(nn.Module):
    def __init__(self, env, n_hidden, hidden_size):
        super().__init__()
        self.n_bidders = env.n_bidders
        self.n_objects = env.n_objects
        self.n_hidden = n_hidden
        input_size = self.n_bidders * self.n_objects

        self.net = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        for _ in range(n_hidden - 1):
            self.net.append(nn.Linear(hidden_size, hidden_size))
        self.net.append(nn.Linear(hidden_size,self.n_bidders))

    def forward(self, x):
        for i in range(self.n_hidden):
            x = self.net[i](x)
            x = torch.tanh(x)
        x = self.net[-1](x)
        x = torch.sigmoid(x)
        return x

class Add_Network (nn.Module):
    """Allocation & Payment networks for NeuralNet for bidders with Additive Valuations"""
    def __init__(self, env, mode, n_hidden = 2, hidden_size = 100): #env is an Auction_Environment
        super().__init__()
        self.n_bidders = env.n_bidders
        self.n_objects = env.n_objects
        self.L = env.L

        input_size = self.n_bidders * self.n_objects

        if mode == 'additive':
            self.alloc_net = Alloc_Net_Additive(env, n_hidden, hidden_size)
        elif mode == 'unit-demand':
            self.alloc_net = Alloc_Net_Unit(env, n_hidden, hidden_size)
        elif mode == 'combinatorial':
            self.alloc_net = Alloc_Net_Combinatorial(env, n_hidden, hidden_size)
        self.pay_net = Pay_Net(env, n_hidden, hidden_size)

    def merge(self, x): #x is valuation profile of size (L, n*m)
        """combines our payment and our allocation rules and gives the payment for each bidder"""
        alloc = self.alloc_net.forward(x)
        pay = self.pay_net.forward(x)
        alloc = alloc.view(self.L, self.n_bidders + 1, self.n_objects)[:, :-1, :] #get rid of line for "proba object not allocated"
        x = x.view(self.L, self.n_bidders, self.n_objects)
        coefs = alloc * x
        coefs = torch.sum(coefs, dim = -1).view(self.L, -1)
        final_pay = pay * coefs
        return final_pay #tensor of payment for each player of dim (L, n)

class Trainer (nn.Module):
    def __init__(self, env, net, n_epochs):
        super().__init__()
        self.env = env
        self.net = net
        self.lag = torch.ones(env.n_bidders) * 5 #tensor of lagrange multipliers
        self.first_epoch = True
        self.L = env.L #batch size
        self.misreps_stock = torch.zeros(self.L, self.env.n_bidders * self.env.n_objects)

        self.n_epochs = n_epochs
        self.epoch_revenue = np.zeros(n_epochs*10)
        self.epoch_regret = np.zeros(n_epochs*10)
        self.epoch_count = 0

        params = list(self.net.alloc_net.parameters()) + list(self.net.pay_net.parameters())
        self.opt_weights = torch.optim.Adam(params, lr = 0.001)

    def utility(self, value, x): #x is a bidding profile, value is a valuation profile
        """returns utility for each player"""
        pay = self.net.merge(x) #size (L, n)
        alloc = self.net.alloc_net.forward(x) #size (n+1)*m
        alloc = alloc.view(self.L, self.env.n_bidders + 1, self.env.n_objects)[:, :-1, :]
        value = value.view(self.L, self.env.n_bidders, -1)
        utility = torch.sum(value * alloc, dim = -1).view(self.L, -1) - pay #expected payoff - expected payment
        return utility

    def regret(self, value, x):
        """returns regret for each player"""
        # print('before: ', self.utility(value, x), 'after: ', self.utility(value, value))
        return self.utility(value, x) - self.utility(value, value)

    def mis_utility(self, value, x, i): #value is the valuation profile of all players the reference, x is player's misreports
        """utility when player i that reports x while the others report x_i"""
        inp = value.detach().clone()
        inp[:, i * self.env.n_objects : (i+1) * self.env.n_objects] = x
        pay = self.net.merge(inp) #size L, n
        alloc = self.net.alloc_net.forward(inp) #size (n+1)*m, L
        alloc = alloc.view(self.L, self.env.n_bidders + 1, self.env.n_objects)[:, :-1, :] #get rid of line that accounts for proba object not allocated
        value = value.view(self.L, self.env.n_bidders, -1)
        utility = torch.sum(value * alloc, dim = -1).view(self.L, -1) - pay #expected payoff - expected payment
        #print('blabla', pay, torch.sum(value * alloc, dim = -1).view(self.L, -1))
        return utility[:, i] #check les merdouilles dans les dimensions


    def train(self, n_profiles = 160000, lrs = [0.1, 0.001, 1.], R = 25, Z = 100):
        """trains self.net, corresponds to 1 epoch"""
        #earlier trianing: took lrs[2] = 0.1
        #lrs is a list of 3 learning rates [misreports, network weights, rho], rho is incremented every 2 epochs
        #in paper: n_profiles = 640000, 80 epochs, R = 25, Z = 100
        count = 0
        n_steps = n_profiles // self.L #T in the paper

        time_elapsed = 0
        #vector of expected revenue after looking at each batch
        exp_rev_vect = [torch.Tensor([0]).detach() for _ in range(n_steps)]
        track_regret = [torch.Tensor([0]).detach() for _ in range(n_steps)]

        st_time = time.time()

        while count < n_steps: #replaces the "for t= 0 to T" in the paper
            #take 1 minibatch
            X = []
            for b in self.env.bidders_list: #for each bidder
                for dis in b.distrib_list:
                    X.append(dis.sample(self.L))
            X = torch.Tensor(X)
            X = X.t_()
            X.requires_grad_(False)

            tic = time.time()

            if self.first_epoch == True:
                misreps = X.detach().clone() #misreports initialized to be equal to the value of the bidders
            else:
                misreps = self.misreps_stock.detach().clone() #after first epoch, initialize the whole thing to be equal to the previous misreport
                # misreps.requires_grad_(False)
            L = self.L
            #gradient descent on misreports (bids that maximizes utility)
            for k in range(self.env.n_bidders): #each bidder optimizes, "best response" to the others valuation
                x = misreps[:, k * self.env.n_objects : (k+1) * self.env.n_objects].detach().clone()
                x.requires_grad_(True)
                opt_misrep = torch.optim.Adam([x], lr=lrs[0]) #only modify one player's bids

                #TEST
                # if k == self.env.n_bidders - 1:
                #     a = -self.mis_utility(X, x, k)
                #     print('utility before opti: ', a)

                for r in range(R): #R is the nbr of times you opti misreports
                    opt_misrep.zero_grad()
                    u_pred = - self.mis_utility(X, x, k) #utility obtained for each player going through the model
                    u_pred.backward(torch.ones(self.L))
                    opt_misrep.step()
                    #print(r, x, u_pred) #SEE IF UTILITY INCREASES

                #TEST
                # if k == self.env.n_bidders - 1:
                #     print('utility after opti: ', u_pred, 'bids made: ', x)
                misreps[ :, k * self.env.n_objects : (k+1) * self.env.n_objects] = x.detach().clone() 
            self.misreps_stock = misreps.detach().clone()
            # self.misreps_stock.requires_grad_(False)

            #self.first_epoch = False

            toc1 = time.time()
            intermediate_time = toc1 - tic

            if count == 0 or count == n_steps - 1:
                print('time misreports: ', intermediate_time)

            #now we optimize the network parameters
            self.opt_weights.zero_grad()

            #Compute augmented lagrangian
            f_lagrange = torch.Tensor([0]) #lagrange function C_rho
            var = X.detach().clone()
            var.requires_grad_(False)
            f_lagrange -= 1/L * torch.sum(self.net.merge(var).view(1,-1)) #first term of the Lagrangian, sum of the final payments of each player for the given valuation profile
            f_lagrange += 1/L * torch.sum((self.lag * self.regret(var, misreps)).view(1,-1)) #second term
            f_lagrange += 1/2 * lrs[2] * (torch.sum(self.regret(var, misreps).view(1,-1))) ** 2 #third

            track_regret[count] += 1/L * torch.sum(self.regret(var, misreps).detach().clone().view(1,-1))
            exp_rev_vect[count] += 1/L * torch.sum(self.net.merge(var).detach().clone()) #just to test that revenue increases
            
            #optimize lagrangian
            time_opt = time.time()
            f_lagrange.backward()
            self.opt_weights.step()
            time_post_opt = time.time()

            if count == 0 or count == n_steps//2 or count == n_steps - 1:
                print('time train model: ', time_post_opt - time_opt)

            #print('lag ', f_lagrange, ' reg ', track_regret[count], ' rev ', exp_rev_vect[count])
            count += 1
            toc = time.time()
            time_elapsed += (toc - tic)
            if count == n_steps - 1:
                print('1 epoch took {} sec, step {}'.format(toc - st_time, count))

            if count % Z == 0: #once every Z operations
                #upgrade lagrange coefficients
                regret_vect = torch.sum(self.regret(X, misreps).detach().clone(), dim = 0)/self.L
                self.lag = self.lag + lrs[2] * regret_vect

            if count % 10000 == 0:
                lrs[2] += 20

            if count == n_steps - 1: #allows to make some graphs of the ongoing situation
                self.exp_rev_vect = np.array([x.numpy()[0] for x in exp_rev_vect])
                self.track_regret = np.array([x.numpy()[0] for x in track_regret])
                plt.plot([i for i in range(count)], self.exp_rev_vect[:count])
                plt.title('Expected Revenue as function of nbr of updates')
                plt.show()
                plt.plot([i for i in range(count)], self.track_regret[:count])
                plt.title('Regret as function of nbr of updates')
                plt.show()
                self.epoch_revenue[self.epoch_count] = self.exp_rev_vect[:count].mean()
                print('epoch revenue: ', self.epoch_revenue[self.epoch_count])
                self.epoch_regret[self.epoch_count] = self.track_regret[:count].mean()

            if count == 1 or count == n_steps - 1:
                print('training weights:', toc - toc1)
            #if count % 5 == 1:
            #    plt.plot([i for i in range(count+1)], exp_rev_vect[:count+1].detach().numpy())
            #    plt.show()
            #print('code has been running for {} sec, {} %, exp revenue: {}'.format(round(time_elapsed,2),round(count/n_steps*100, 2), exp_rev_vect[count]))
        if self.epoch_count % self.n_epochs == self.n_epochs - 1:
            plt.plot([i for i in range(self.epoch_count+1)], self.epoch_revenue[:self.epoch_count+1])
            plt.title('Expected Revenue as function of nbr of epochs')
            plt.show()
            plt.plot([i for i in range(self.epoch_count+1)], self.epoch_regret[:self.epoch_count+1])
            plt.title('Regret as function of nbr of epochs')
            plt.show()
        self.epoch_count += 1

