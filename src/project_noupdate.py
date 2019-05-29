from __future__ import absolute_import, division, print_function

import argparse
import logging

import pandas as pd
import torch

from six.moves import xrange
import numpy as np
from numpy.linalg import inv, cholesky
from numpy.random import RandomState
from scipy.stats import wishart
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.mcmc import MCMC, NUTS

from exceptions import NotFittedError
from utils.datasets import build_user_item_matrix
from utils.validation import check_ratings
from utils.evaluation import RMSE
from utils.datasets import load_movielens_1m_ratings
from utils.progress import printProgressBar
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable

import pdb

logging.basicConfig(format='%(message)s', level=logging.INFO)
pyro.enable_validation(True)
pyro.set_rng_seed(0)

truth = None

class BPMF():
    def __init__(self, n_user, n_item, n_feature, beta=2.0, beta_user=2.0,
                 df_user=None, mu0_user=0., beta_item=2.0, df_item=None,
                 mu0_item=0., converge=1e-5, seed=None, max_rating=None,
                 min_rating=None, output_file = None):

        super(BPMF, self).__init__()

        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.rand_state = RandomState(seed)
        self.max_rating = float(max_rating) if max_rating is not None else None
        self.min_rating = float(min_rating) if min_rating is not None else None
        self.converge = converge

        # Hyper Parameter
        self.beta = beta

        # Inv-Whishart (User features)
        self.WI_user = np.eye(n_feature, dtype='float64')
        self.beta_user = beta_user
        self.df_user = int(df_user) if df_user is not None else n_feature
        self.mu0_user = np.repeat(mu0_user, n_feature).reshape(n_feature, 1) 
        # Inv-Whishart (item features)
        self.WI_item = np.eye(n_feature, dtype='float64')
        self.beta_item = beta_item
        self.df_item = int(df_item) if df_item is not None else n_feature
        self.mu0_item = np.repeat(mu0_item, n_feature).reshape(n_feature, 1)

        self.alpha_user = np.eye(n_feature, dtype='float64')
        self.alpha_item = np.eye(n_feature, dtype='float64')

        # initializes the user features randomly.
        # (There is no special reason to use 0.3)
        self.user_features_ = 0.3 * self.rand_state.rand(n_user, n_feature)
        self.item_features_ = 0.3 * self.rand_state.rand(n_item, n_feature)

        # average user/item features
        self.avg_user_features_ = np.zeros((n_user, n_feature))
        self.avg_item_features_ = np.zeros((n_item, n_feature))

        # data state
        self.iter_ = 0
        self.mean_rating_ = None
        self.ratings_csr_ = None
        self.ratings_csc_ = None

        check_ratings(ratings, self.n_user, self.n_item,
                      self.max_rating, self.min_rating)

        self.mean_rating_ = np.mean(ratings[:, 2])

        # only two different ways of building the matrix.
        # csr user-item matrix for fast row access (user update)
        self.ratings_csr_ = build_user_item_matrix(
            self.n_user, self.n_item, ratings)
        # keep a csc matrix for fast col access (item update)
        self.ratings_csc_ = self.ratings_csr_.tocsc()

        self.iter_ = 0
        self.output_file = output_file
        
    def _update_item_features(self):
        # Gibbs sampling for item features
        mean = np.zeros((self.n_item, self.n_feature), dtype = 'float64')
        mean = Variable(torch.from_numpy(mean))
        mean = torch.reshape(mean,( -1,))
        covar = 2 * torch.eye(int(self.n_item * self.n_feature), dtype = torch.float64) 
        temp_feature = pyro.sample('i_temp_feature', dist.MultivariateNormal(mean, covariance_matrix=covar))
        self.item_features_ = torch.reshape(temp_feature, (self.n_item, self.n_feature)).detach().numpy()
            #-----------------------------------------------------------------------------------------------

    def _update_user_features(self):
        # Gibbs sampling for user features
        mean = np.zeros((self.n_user, self.n_feature), dtype = 'float64')
        mean = Variable(torch.from_numpy(mean))
        mean = torch.reshape(mean, (-1,))
        covar = 2 * torch.eye(int(self.n_user * self.n_feature),dtype = torch.float64)
        temp_feature = pyro.sample('u_temp_feature', dist.MultivariateNormal(mean, covariance_matrix=covar))
        self.user_features_ = torch.reshape(temp_feature, (self.n_user, self.n_feature)).detach().numpy()
          
    def _model(self, sigma):
        global truth
        self.iter_ += 1
        self._update_item_features()
        self._update_user_features()

        Y = np.matmul(self.user_features_,self.item_features_.transpose())

        for x in range(self.n_user):
            for y in range(self.n_item):
                if Y[x,y] > self.max_rating:
                    Y[x,y] = self.max_rating
                elif Y[x,y] < self.min_rating:
                    Y[x,y] = self.min_rating
        
        Y = torch.autograd.Variable(torch.from_numpy(Y))
        Y2 = Y
        for x in range(self.n_user):
            for y in range(self.n_item):
                Y2[x,y] = pyro.sample("obs" + str(x*self.n_item + y),dist.Normal(Y[x,y], 0.5))
        print("RMSE : " + str(RMSE(Y2.detach().numpy(),truth)))
        return Y2 

    def _conditioned_model(self,model, sigma, ratings):
        data = dict()
        for x in range(self.n_user):
            for y in range(self.n_item):
                if ratings[x,y] != 0:
                    data["obs" + str(x * self.n_item + y)] = torch.tensor(ratings[x,y], dtype = torch.float64)
        ratings = torch.autograd.Variable(torch.from_numpy(ratings))
        return poutine.condition(model, data=data)(sigma)
 
    #1000, 1000, 1 for defulat
    def _main(self, ratings,sigma, jit=False, num_samples=10
                  , warmup_steps=4, num_chains=1):
        global truth
        ratingstmp = np.zeros((self.n_user, self.n_item), dtype = 'float64')
        truth = ratingstmp
        for rat in ratings:
            ratingstmp[rat[0],rat[1]] = rat[2]
        nuts_kernel = NUTS(self._conditioned_model, jit_compile=jit,)
        posterior = MCMC(nuts_kernel,
                         num_samples=num_samples,
                         warmup_steps=warmup_steps,
                         num_chains=num_chains,
                         disable_progbar=False).run(self._model,sigma, ratingstmp)
        sites = ['u_temp_feature', 'i_temp_feature']
        #marginal = posterior.marginal(sites=sites)
        Y = np.matmul(self.user_features_,self.item_features_.transpose())
        print("user_feature : ")
        print(self.user_features_)
        print("item_features : ")
        print(self.item_features_)
        print("computed data : ")
        print(Y)
        print("truth data : ")
        print(ratingstmp)
        for x in range(self.n_user):
            for y in range(self.n_item):
                if ratingstmp[x,y] == 0.:
                    Y[x,y] = 0.
        print("0 removed computed data : ")
        print(Y)
        print("RMSE: " + str(RMSE(Y, ratingstmp)))

            
if __name__ == "__main__":
    #print("Loading data....")
    ratings = load_movielens_1m_ratings('../ml-1m/ratings-made.dat')
    #print("Loaded")

    output_file = open("mcmc_result.txt", 'w')
    n_user = max(ratings[:,0])
    n_item = max(ratings[:,1])
    ratings[:,(0,1)] -= 1
    bpmf = BPMF(n_user=n_user,n_item=n_item, n_feature=10,
                max_rating=5., min_rating=1., seed= 0, output_file = output_file)
    sigma = 0.1 * torch.eye(int(n_user * n_item), dtype = torch.float64)
    bpmf._main(ratings, sigma)
