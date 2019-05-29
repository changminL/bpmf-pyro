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
        
    def _update_item_params(self):
        N = self.n_item
        X_bar = np.mean(self.item_features_, 0).reshape((self.n_feature, 1))
        # #print 'X_bar', X_bar.shape
        S_bar = np.cov(self.item_features_.T)
        # #print 'S_bar', S_bar.shape

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

        # update mu_item
        mu_mean = (self.beta_item * self.mu0_item + N * X_bar) / \
            (self.beta_item + N)

        #mu_mean = [i[0] for i in mu_mean]
        #mu_var = cholesky(inv(np.dot(self.beta_item + N
        #                , self.alpha_item)))[self.n_feature-1]
        #----------------------------------------------------------------------
        mu_var = inv(np.dot(self.beta_item + N, self.alpha_item))
        #----------------------------------------------------------------------
        return mu_mean, mu_var

    def _update_user_params(self):
        # same as _update_user_params
        N = self.n_user
        X_bar = np.mean(self.user_features_, 0).reshape((self.n_feature, 1))
        S_bar = np.cov(self.user_features_.T)
        # mu_{0} - U_bar
        diff_X_bar = self.mu0_user - X_bar
        # W_{0}_star 
        WI_post = inv(inv(self.WI_user) + 
                      N * S_bar + np.dot(diff_X_bar, diff_X_bar.T) * 
                      (N * self.beta_user) / (self.beta_user + N))
        # Note: WI_post and WI_post.T should be the same.
        #       Just make sure it is symmertic here
        WI_post = (WI_post + WI_post.T) / 2.0
        # update alpha_user
        df_post = self.df_user + N
        # LAMBDA_{U} ~ W(W{0}_star, df_post)
        self.alpha_user = wishart.rvs(df_post, WI_post, 1, self.rand_state)
        # update mu_user
        # mu_{0}_star = (beta_{0} * mu_{0} + N * U_bar) / (beta_{0} + N) 
        mu_mean = (self.beta_user * self.mu0_user + N * X_bar) / (self.beta_user + N)
        
        #-------------------------------------------------------------------------
        #mu_mean = [i[0] for i in mu_mean]
        # decomposed inv(beta_{0}_star * LAMBDA_{U}) 
        #mu_var = cholesky(inv(np.dot(self.beta_user + N
        #                , self.alpha_user)))[self.n_feature - 1]
        mu_var = inv(np.dot(self.beta_user + N, self.alpha_user))
        #---------------------------------------------------------------------------
        return mu_mean, mu_var

    def _update_item_features(self):
        # Gibbs sampling for item features
        #printProgressBar(0, self.n_item, prefix = 'Updating item features:', suffix = 'Complete', length = 40)
        for item_id in xrange(self.n_item):
            #printProgressBar(item_id, self.n_item, prefix = 'Updating item features:', suffix = 'Complete', length = 40)
            indices = self.ratings_csc_[:, item_id].indices
            features = self.user_features_[indices, :]
            rating = self.ratings_csc_[:, item_id].data - self.mean_rating_
            rating = np.reshape(rating, (rating.shape[0], 1))

            covar = inv(self.alpha_item +
                        self.beta * np.dot(features.T, features))
            lam = cholesky(covar)

            temp = (self.beta * np.dot(features.T, rating) +
                    np.dot(self.alpha_item, self.mu_item))

            mean = np.dot(covar, temp)
            #-----------------------------------------------------------------------------------------------
            mean = Variable(torch.from_numpy(mean))
            covar = Variable(torch.from_numpy(covar))
            mean = torch.reshape(mean, (-1,))
            temp_feature = pyro.sample('i_temp_feature' + str(item_id), dist.MultivariateNormal(mean, covariance_matrix=covar))  
            self.item_features_[item_id, :] = temp_feature.detach().numpy().ravel()
            #-----------------------------------------------------------------------------------------------

    def _update_user_features(self):
        # Gibbs sampling for user features
        #printProgressBar(0, self.n_user, prefix = 'Updating user features:', suffix = 'Complete', length = 40)
        for user_id in xrange(self.n_user):
            #printProgressBar(user_id, self.n_user, prefix = 'Updating user features:' suffix = 'Complete', length = 40)
            indices = self.ratings_csr_[user_id, :].indices
            features = self.item_features_[indices, :]
            rating = self.ratings_csr_[user_id, :].data - self.mean_rating_
            rating = np.reshape(rating, (rating.shape[0], 1))

            covar = inv(self.alpha_user + self.beta * np.dot(features.T, features))
            lam = cholesky(covar)
            # aplha * sum(V_j * R_ij) + LAMBDA_U * mu_u
            temp = (self.beta * np.dot(features.T, rating) +
                    np.dot(self.alpha_user, self.mu_user))
            # mu_i_star
            mean = np.dot(covar, temp)
            #-----------------------------------------------------------------------------------------------
            mean = Variable(torch.from_numpy(mean))
            mean = torch.reshape(mean, (-1,))
            covar = Variable(torch.from_numpy(covar))
            temp_feature = pyro.sample('u_temp_feature' + str(user_id), dist.MultivariateNormal(mean, covariance_matrix=covar))
            self.user_features_[user_id, :] = temp_feature.detach().numpy().ravel()
            #-----------------------------------------------------------------------------------------------
          
    def _model(self, sigma):
        self.iter_ += 1
        #print("iteration : " + str(self.iter_))
        #print("updating parameters")
        i_mu_mean, i_mu_var = self._update_item_params()
        u_mu_mean, u_mu_var = self._update_user_params()
        for i in range(i_mu_var.shape[0]):
            for j in range(i_mu_var.shape[1]):
                if i_mu_var[i][j] <= 1e-6:
                    i_mu_var[i][j] = 1e-6
        for i in range(u_mu_var.shape[0]):
            for j in range(u_mu_var.shape[1]):
                if u_mu_var[i][j] <= 1e-6:
                    u_mu_var[i][j] = 1e-6
        #----------------------------------------------------------------------------------------------
        i_mu_mean = Variable(torch.from_numpy(i_mu_mean))
        i_mu_var = Variable(torch.from_numpy(i_mu_var))
        u_mu_mean = Variable(torch.from_numpy(u_mu_mean))
        u_mu_var = Variable(torch.from_numpy(u_mu_var))

        i_mu_mean = torch.reshape(i_mu_mean, (-1,))
        u_mu_mean = torch.reshape(u_mu_mean, (-1,))
        while True:
            try:
                self.mu_item = pyro.sample('mu_item', dist.MultivariateNormal(i_mu_mean, covariance_matrix=i_mu_var))
                break
            except (RuntimeError, ValueError):
                i_mu_var = 0.01 * torch.eye(self.n_feature, dtype=torch.float64)

        self.mu_item = torch.reshape(self.mu_item, (self.n_feature,1)).detach().numpy()
        while True:
            try:
                self.mu_user = pyro.sample('mu_user', dist.MultivariateNormal(u_mu_mean, covariance_matrix=u_mu_var))
                break
            except (RuntimeError, ValueError):
                u_mu_var = 0.01 * torch.eye(self.n_feature,dtype=torch.float64)
        self.mu_user = torch.reshape(self.mu_user, (self.n_feature,1)).detach().numpy()
        #-----------------------------------------------------------------------------------------------

        #print("updating item features")
        self._update_item_features()
        #print("Done")
        #print("updating user_features")
        self._update_user_features()
        #print("Done")

        Y = np.add(np.matmul(self.user_features_
                      ,self.item_features_.transpose())
           ,self.mean_rating_ * np.ones((self.n_user, self.n_item), dtype='float64'))

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
        ratingstmp = np.zeros((self.n_user, self.n_item), dtype = 'float64')
        for rat in ratings:
            ratingstmp[rat[0],rat[1]] = rat[2]
        nuts_kernel = NUTS(self._conditioned_model, jit_compile=jit,)
        posterior = MCMC(nuts_kernel,
                         num_samples=num_samples,
                         warmup_steps=warmup_steps,
                         num_chains=num_chains,
                         disable_progbar=False).run(self._model,sigma, ratingstmp)
        sites = ['mu_item', 'mu_user'] + ['u_temp_feature' + str(user_id) for user_id in xrange(self.n_user)] + ['i_temp_feature' + str(item_id) for item_id in xrange(self.n_item)]
        marginal = posterior.marginal(sites=sites)

        Y = np.add(np.matmul(self.user_features_
                      ,self.item_features_.transpose())
           ,self.mean_rating_ * np.ones((self.n_user, self.n_item), dtype='float64'))
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
