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

import sys
import signal


output_file = None
train = None
bpmf = None
validation = None

def signal_handler(signal,frame):
    print("\nwriting current result:\n")
    if output_file is not None:
        if bpmf is not None:
            output_file.write("item feature : \n")
            output_file.write(np.array2string(bpmf.item_features_))
            output_file.write("\nuser feature : \n")
            output_file.write(np.array2string(bpmf.user_features_))
            output_file.write("\nY : \n")
            Y = np.matmul(bpmf.user_features_,bpmf.item_features_.transpose())
            for x in range(bpmf.n_user):
                for y in range(bpmf.n_item):
                    if Y[x,y] > bpmf.max_rating:
                        Y[x,y] = bpmf.max_rating
                    elif Y[x,y] < bpmf.min_rating:
                        Y[x,y] = bpmf.min_rating
            output_file.write(np.array2string(Y))
            numOfrat = 0
            summ = 0.0
            for rat in validation:
                numOfrat += 1
                summ += np.power(Y[rat[0], rat[1]] - rat[2], 2)
            rmse = np.power(summ / numOfrat, 0.5)
            output_file.write("RMSE: " + str(rmse))
            print("RMSE: " + str(rmse))

    exit(0)

logging.basicConfig(format='%(message)s', level=logging.INFO)
pyro.enable_validation(True)
pyro.set_rng_seed(0)

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

        # Latent Variables
        self.mu_user = np.zeros((n_feature, 1), dtype='float64')
        self.mu_item = np.zeros((n_feature, 1), dtype='float64')

        self.user_var = 5 * np.eye(n_feature, dtype = 'float64')
        self.item_var = 5 * np.eye(n_feature, dtype = 'float64')

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
        output_file = output_file
        
    def _update_item_params(self):
        N = self.n_item
        X_bar = np.mean(self.item_features_, 0).reshape((self.n_feature, 1))
        # print 'X_bar', X_bar.shape
        S_bar = np.cov(self.item_features_.T)
        # print 'S_bar', S_bar.shape

        diff_X_bar = self.mu0_item - X_bar

        # W_{0}_star
        WI_post = inv(inv(self.WI_item) +
                      N * S_bar +
                      np.dot(diff_X_bar, diff_X_bar.T) *
                      (N * self.beta_item) / (self.beta_item + N))

        # Note: WI_post and WI_post.T should be the same.
        #       Just make sure it is symmertic here
        WI_post = (WI_post + WI_post.T) / 2.0

        # update alpha_item
        df_post = self.df_item + N
        self.alpha_item = wishart.rvs(df_post, WI_post, 1, self.rand_state)

        mu_var_tmp = torch.autograd.Variable(torch.from_numpy(self.item_var)) / 2
        while True:
            try:
                self.mu_item = pyro.sample('mu_item', dist.MultivariateNormal(torch.zeros((self.n_feature,), dtype = torch.float64), covariance_matrix= mu_var_tmp)).detach().numpy()
                break
            except (RuntimeError, ValueError):
                mu_var_tmp = torch.eye(int(self.n_feature), dtype = torch.float64)
        self.item_var = inv(np.dot(self.beta_item + N, self.alpha_item))
        return

    def _update_user_params(self):
        N = self.n_user
        X_bar = np.mean(self.user_features_, 0).reshape((self.n_feature, 1))
        # print 'X_bar', X_bar.shape
        S_bar = np.cov(self.user_features_.T)
        # print 'S_bar', S_bar.shape

        diff_X_bar = self.mu0_user - X_bar

        # W_{0}_star
        WI_post = inv(inv(self.WI_item) +
                      N * S_bar +
                      np.dot(diff_X_bar, diff_X_bar.T) *
                      (N * self.beta_item) / (self.beta_item + N))

        # Note: WI_post and WI_post.T should be the same.
        #       Just make sure it is symmertic here
        WI_post = (WI_post + WI_post.T) / 2.0

        # update alpha_item
        df_post = self.df_item + N
        self.alpha_item = wishart.rvs(df_post, WI_post, 1, self.rand_state)

        mu_var_tmp = torch.autograd.Variable(torch.from_numpy(self.user_var)) / 2
        while True:
            try:
                self.mu_user = pyro.sample('mu_user', dist.MultivariateNormal(torch.zeros((self.n_feature,), dtype = torch.float64), covariance_matrix= mu_var_tmp)).detach().numpy()
                break
            except(RuntimeError , ValueError):
                mu_var_tmp = torch.eye(int(self.n_feature), dtype = torch.float64)
        self.user_var = inv(np.dot(self.beta_item + N, self.alpha_item))
        return 

    def _update_item_features(self):
        # Gibbs sampling for item features
        for item_id in xrange(self.n_item):
            mean = Variable(torch.from_numpy(self.mu_item))
            covar = Variable(torch.from_numpy(self.item_var))
            mean = torch.reshape(mean, (-1,))
            while True:
                try:
                    temp_feature = pyro.sample('i_temp_feature' + str(item_id), dist.MultivariateNormal(mean, covariance_matrix=covar))  
                    break
                except (RuntimeError, ValueError):
                    covar = 5 * torch.eye(int(self.n_feature), dtype = torch.float64)
            self.item_features_[item_id, :] = temp_feature.detach().numpy().ravel()

    def _update_user_features(self):
        for user_id in xrange(self.n_user):
            mean = Variable(torch.from_numpy(self.mu_user))
            mean = torch.reshape(mean, (-1,))
            covar = Variable(torch.from_numpy(self.user_var))
            while True:
                try:
                    temp_feature = pyro.sample('u_temp_feature' + str(user_id), dist.MultivariateNormal(mean, covariance_matrix=covar))
                    break
                except (RuntimeError, ValueError):
                    covar = 5 * torch.eye(int(self.n_feature), dtype = torch.float64)
            self.user_features_[user_id, :] = temp_feature.detach().numpy().ravel()
          
    def _model(self):
        self.iter_ += 1
        #print("updating parameters")
        self._update_item_params()
        self._update_user_params()

        for i in range(self.user_var.shape[0]):
            for j in range(self.user_var.shape[1]):
                if self.user_var[i][j] <= 1e-6:
                    self.user_var[i][j] = 1e-6
        for i in range(self.item_var.shape[0]):
            for j in range(self.item_var.shape[1]):
                if self.item_var[i][j] <= 1e-6:
                    self.item_var[i][j] = 1e-6

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
        
        #use either one of these code fragment.
        code = 1
        if (code):
            #code version 1.
            for x in range(self.n_user):
                for y in range(self.n_item):
                    Y2[x,y] = pyro.sample("obs" + str(x*self.n_item + y),
                                           dist.Normal(Y[x,y], 0.5))
        else:
            #code version 2
            Y = torch.reshape(Y, (-1,))
            Y2 = Y
            Y2 = pyro.sample('obs',
                  dist.MultivariateNormal(Y,
                      covariance_matrix=0.5*torch.eye(int(self.n_user* self.n_item)
                                                    , dtype = torch.float64)))
            Y2 = torch.reshape(Y2, (self.n_user, self.n_item))
        return Y2 

    def _conditioned_model(self,model, ratings):
        data = dict()
        for x in range(self.n_user):
            for y in range(self.n_item):
                if ratings[x,y] != 0:
                    data["obs" + str(x * self.n_item + y)] = \
                      torch.tensor(ratings[x,y], dtype = torch.float64)
        return poutine.condition(model, data=data)()
 
    #1000, 1000, 1 for default
    def _main(self, ratings, validation, jit=False, num_samples=30
                  , warmup_steps=5, num_chains=1):
        ratingstmp = np.zeros((self.n_user, self.n_item), dtype = 'float64')
        for rat in ratings:
            ratingstmp[rat[0],rat[1]] = rat[2]

        nuts_kernel = NUTS(self._conditioned_model, jit_compile=jit,)
        posterior = MCMC(nuts_kernel,
                         num_samples=num_samples,
                         warmup_steps=warmup_steps,
                         num_chains=num_chains,
                         disable_progbar=False).run(self._model, ratingstmp)
        sites = ['mu_item', 'mu_user'] + ['u_temp_feature' + str(user_id) for user_id in xrange(self.n_user)] + ['i_temp_feature' + str(item_id) for item_id in xrange(self.n_item)]
        marginal = posterior.marginal(sites=sites)
        marginal = torch.cat(list(marginal.support(flatten=True).values()), dim = -1)

        numOfres = (2 + int(self.n_user) + int(self.n_item)) * int(self.n_feature)
        print("shape : " + str(marginal.shape))
        print("numOfres : " + str(numOfres))
        marginal = marginal[num_samples-1]
        
        ifmatrix = torch.zeros((int(self.n_item), int(self.n_feature))).detach().numpy()
        ufmatrix = torch.zeros((int(self.n_user), int(self.n_feature))).detach().numpy()
        
        marginal = torch.reshape(marginal, (2 +int(self.n_user) + int(self.n_item) , int(self.n_feature)))
        for i in range(int(self.n_user)):
            ufmatrix[i,:] = marginal[2 + i]
        for i in range(int(self.n_item)):
            ifmatrix[i,:] = marginal[2 + int(self.n_user) + i]
        
        Y = np.matmul(ufmatrix,ifmatrix.transpose())


        for x in range(self.n_user):
            for y in range(self.n_item):
                if Y[x,y] > self.max_rating:
                    Y[x,y] = self.max_rating
                elif Y[x,y] < self.min_rating:
                    Y[x,y] = self.min_rating

        output_file.write("item feature : \n")
        output_file.write(str(ifmatrix) + "\n")
        output_file.write("user feature : \n")
        output_file.write(str(ufmatrix) + "\n")
        output_file.write("Y : \n")
        output_file.write(str(Y) + "\n")

        numOfrat = 0
        summ = 0.0
        for rat in validation:
            numOfrat += 1
            summ += np.power(Y[rat[0], rat[1]] - rat[2], 2)
        rmse = np.power(summ / numOfrat, 0.5)
        output_file.write("RMSE: " + str(rmse))
        print("RMSE: " + str(rmse))

            
if __name__ == "__main__":
    #print("Loading data....")
    ratings = load_movielens_1m_ratings('../ml-1m/ratings.dat')
    #print("Loaded")
    np.set_printoptions(threshold=np.inf)

    output_file = open("mcmc_result.txt", 'w')
    n_user = max(ratings[:,0])
    n_item = max(ratings[:,1])
    ratings[:,(0,1)] -= 1
    rand_state = RandomState(0)
    rand_state.shuffle(ratings)
    train_pct = 0.9
    train_size = int(train_pct * ratings.shape[0])
    train = ratings[:train_size]
    validation = ratings[train_size:]
    bpmf = BPMF(n_user=n_user,n_item=n_item, n_feature=30,
                max_rating=5., min_rating=1., seed= 0, output_file = output_file)
    signal.signal(signal.SIGINT, signal_handler)

    bpmf._main(train,validation)
