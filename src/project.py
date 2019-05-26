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

from .exceptions import NotFittedError
from .utils.datasets import build_user_item_matrix
from .utils.validation import check_ratings
from .utils.evaluation import RMSE

logging.basicConfig(format='%(message)s', level=logging.INFO)
pyro.enable_validation(True)
pyro.set_rng_seed(0)

class BPMF():
    def __init__(self, n_user, n_item, n_feature, beta=2.0, beta_user=2.0,
                 df_user=None, mu0_user=0., beta_item=2.0, df_item=None,
                 mu0_item=0., converge=1e-5, seed=None, max_rating=None,
                 min_rating=None):

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
        self.mu0_user = np.repeat(mu0_user, n_feature).reshape(n_feature, 1)  # a vector

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

        # update mu_item
        mu_mean = (self.beta_item * self.mu0_item + N * X_bar) / \
            (self.beta_item + N)
        mu_var = cholesky(inv(np.dot(self.beta_item + N, self.alpha_item)))
        
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

        # decomposed inv(beta_{0}_star * LAMBDA_{U})                                                             
        mu_var = cholesky(inv(np.dot(self.beta_user + N, self.alpha_user)))

        return mu_mean, mu_var

    def _udpate_item_features(self):
        # Gibbs sampling for item features
        for item_id in xrange(self.n_item):
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
 
            temp_feature = pyro.sample('i_temp_feature' + str(item_id), dist.Normal(mean, lam))
            self.item_features_[item_id, :] = temp_feature.ravel()

    def _update_user_features(self):
        # Gibbs sampling for user features
        for user_id in xrange(self.n_user):
            indices = self.ratings_csr_[user_id, :].indices
            features = self.item_features_[indices, :]
            rating = self.ratings_csr_[user_id, :].data - self.mean_rating_
            rating = np.reshape(rating, (rating.shape[0], 1))

            covar = inv(
                self.alpha_user + self.beta * np.dot(features.T, features))
            lam = cholesky(covar)
            # aplha * sum(V_j * R_ij) + LAMBDA_U * mu_u
            temp = (self.beta * np.dot(features.T, rating) +
                    np.dot(self.alpha_user, self.mu_user))
            # mu_i_star
            mean = np.dot(covar, temp)

            temp_feature = pyro.sample('u_temp_feature' + str(user_id), dist.Normal(mean, lam))
            self.user_features_[user_id, :] = temp_feature.ravel()

          
        def _model(self, sigma):
            i_mu_mean, i_mu_var = self._update_item_params()
            u_mu_mean, u_mu_var = self._update_user_params()
            self.mu_item = pyro.sample('mu_item', dist.Normal(i_mu_mean, i_mu_var))
            self.mu_user = pyro.sample('mu_user', dist.Normal(u_mu_mean, u_mu_var))

            self._update_item_features()
            self._update_user_features()            
    
            return pyro.sample("obs", dist.Normal(np.matmul(self.user_features_, self.item_features_.transpose()) 
                               + self.mean_rating_ * np.ones(self.n_user, self.n_item), sigma))

        def _conditioned_model(self, model, sigma, y):
            return poutine.condtion(model, data={"obs": y})(sigma)
 
        def _main(self, args):
            nuts_kernel = NUTS(_conditioned_model, jit_compile=args.jit,)
            posterior = MCMC(nuts_kernel,
                             num_samples=args.num_samples,
                             warmup_steps=args.warmup_steps,
                             num_chains=args.num_chains).run(_model)
            marginal = posterior.marginal(sites=['mu_item', 'mu_user'] + 
                                                ['u_temp_feature' + str(user_id) for user_id in xrange(self.n_user)] +
                                                ['i_temp_feature' + str(item_id) for item_id in xrange(self.n_item)])
            marginal = torch.cat(list(marginal.support(flatten=True).values()), dim=-1).cpu().numpy()
            

bpmf = BPMF()
parser = argparse.ArgumentParser(description='Eight Schools MCMC')
parser.add_argument('--num-samples', type=int, default=1000,
                        help='number of MCMC samples (default: 1000)')
parser.add_argument('--num-chains', type=int, default=1,
                        help='number of parallel MCMC chains (default: 1)')
parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='number of MCMC samples for warmup (default: 1000)')
parser.add_argument('--jit', action='store_true', default=False)
args = parser.parse_args()
bpmf._main(args)
   
           
        
         


