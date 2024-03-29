import torch
from torch.distributions import Exponential
import numpy as np

class UniformDistrib():
    def __init__(self, mu, sigma, a = 0, b = 1, is_strategic = False): #l is lambda for adversarial training: array
        self.a, self.b, self.mu, self.sigma = a, b, mu, sigma
        self.is_strategic = is_strategic
        self.name="uniform"
        # self.cdf=lambda x : (x-a)/(b-a)
        # self.pdf=lambda x : 1/(b-a)
        # self.inverse_virtual_value = lambda x:(x+1)/2 #useless in regretnet
        # self.boost = 2.0 #don't know what that is
        # self.optimal_reserve_price = 0.5 #useless in regretnet
    def sample(self,size):
        if self.is_strategic: 
            l = np.random.normal(self.mu, self.sigma, size)
            return l * np.random.uniform(self.a, self.b, size)
        else:
            return np.random.uniform(self.a, self.b, size)
        #return torch.rand(size)

class ExponentialDistrib():
    def __init__(self,lambdap=1.0):
        self.name="exponential"
        self.cdf = lambda x : 1 - torch.exp(-lambdap*x)
        self.pdf = lambda x : lambdap*torch.exp(-lambdap*x)
        self.inverse_virtual_value = lambda x : x+lambdap
        self.boost = 1
        self.optimal_reserve_price = lambdap
    def sample(self,size):
        return np.random.exponential(1.0,size)
        #m = Exponential(torch.tensor([1.0]))
        #return m.sample(size)

class AdverseDistrib():
    def __init__(self, distrib, mu = 1, sig = 1):
        self.mu = mu 
        self.sig = sig
        self.distrib = distrib #distrib du player
    
    def sample(self, size): #size is number of objects
        values = self.distrib.sample(size)
        lam = np.random.normal(loc = mu, scale = sig, size = size) #generates the lambdas from the same distribution
        return values * self.lambdas