from __future__ import absolute_import, division, print_function

import argparse
import logging

import pandas as pd
import torch
from multiprocessing import Pool, Array, Process, Value, Manager
import ctypes

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

rand_state = RandomState(0)

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
        self.data = None
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

        mu_var = inv(np.dot(self.beta_item + N, self.alpha_item))
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
            mean = Variable(torch.from_numpy(mean))
            covar = Variable(torch.from_numpy(covar))
            mean = torch.reshape(mean, (-1,))
            temp_feature = pyro.sample('i_temp_feature' + str(item_id), dist.MultivariateNormal(mean, covariance_matrix=covar))  
            self.item_features_[item_id, :] = temp_feature.detach().numpy().ravel()

    def _update_user_features(self):
        # Gibbs sampling for user features
        for user_id in xrange(self.n_user):
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
            mean = Variable(torch.from_numpy(mean))
            mean = torch.reshape(mean, (-1,))
            covar = Variable(torch.from_numpy(covar))
            temp_feature = pyro.sample('u_temp_feature' + str(user_id), dist.MultivariateNormal(mean, covariance_matrix=covar))
            self.user_features_[user_id, :] = temp_feature.detach().numpy().ravel()

    def _predict(self, data, is_train=False, avg_u_f=None, avg_i_f=None):
        if not self.mean_rating_:
            raise NotFittedError()

        if not is_train:
            u_features = self.user_features_.take(data.take(0, axis=1), axis=0)
            i_features = self.item_features_.take(data.take(1, axis=1), axis=0)
        else:
            u_features = avg_u_f.take(data.take(0, axis=1), axis=0)
            i_features = avg_i_f.take(data.take(1, axis=1), axis=0)

        preds = np.sum(u_features * i_features, 1) + self.mean_rating_

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    def _init_shared(self, _shared):
        global shared_pred

        shared_pred = np.frombuffer(_shared.get_obj()).reshape(-1)

    def _sample(self, s, e, p, sigma, p2):
        for i in range(s.value, e.value):
            p2[i] = pyro.sample("obs" + str(i), dist.Normal(p[i], sigma.value))

    def _model(self, sigma):
        self.iter_ += 1

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
                i_mu_var = 0.1 * torch.eye(self.n_feature, dtype=torch.float64)

        self.mu_item = torch.reshape(self.mu_item, (self.n_feature,1)).detach().numpy()
        while True:
            try:
                self.mu_user = pyro.sample('mu_user', dist.MultivariateNormal(u_mu_mean, covariance_matrix=u_mu_var))
                break
            except (RuntimeError, ValueError):
                u_mu_var = 0.1 * torch.eye(self.n_feature,dtype=torch.float64)
                
        self.mu_user = torch.reshape(self.mu_user, (self.n_feature,1)).detach().numpy()
        
        self._update_item_features()
        self._update_user_features()
        
        pred = self._predict(self.data)

        pred_len = len(pred)
        pred = Variable(torch.from_numpy(pred))

        pred2 = pred
        pred2_n = pred2.numpy()
        
        threads = 32
        jobs = []
        batch_size = pred_len // threads
    
        pred2 = Array('d', pred2_n)
        sig = Value('d', sigma)
        start = Value('i', 0)
        end = Value('i', batch_size)
        for i in range(threads):
            if(i == (threads - 1)):
                end = Value('i', pred_len)
            p = Process(target=self._sample, args=[start, end, pred, sig, pred2])
            jobs.append(p)
            start = Value('i', start.value + batch_size)
            end = Value('i', end.value + batch_size)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()
        
        pred2 = Variable(torch.from_numpy(np.array(pred2[:])))
        return pred2
    
    def _conditioned_model(self,model, sigma, ratings):
        data = dict()
        
        rating = ratings.take(2, axis=1)
        rating_len = len(rating)
            
        for i in range(rating_len):
            data["obs" + str(i)] = torch.tensor(rating[i], dtype = torch.float64)
        
        return poutine.condition(model, data=data)(sigma)
 
    #1000, 1000, 1 for defulat
    def _main(self, ratings, sigma, args):

        # split data to training & testing
        train_pct = 0.9

        rand_state.shuffle(ratings)
        train_size = int(train_pct * ratings.shape[0])
        train = ratings[:train_size]
        validation = ratings[train_size:]

        self.data = train
       
        nuts_kernel = NUTS(self._conditioned_model, jit_compile=args.jit,)
    
        posterior = MCMC(nuts_kernel,
                         num_samples=args.num_samples,
                         warmup_steps=args.warmup_steps,
                         num_chains=args.num_chains,
                         disable_progbar=False).run(self._model, sigma, train)
        
        sites = ['mu_item', 'mu_user'] + ['u_temp_feature' + str(user_id) for user_id in xrange(self.n_user)] + ['i_temp_feature' + str(item_id) for item_id in xrange(self.n_item)]
        marginal = posterior.marginal(sites=sites)
        marginal = torch.cat(list(marginal.support(flatten=True).values()), dim=-1).cpu().numpy()
        avg_mu_item = np.average(marginal[:, :self.n_feature], axis=0)
        avg_mu_user = np.average(marginal[:, self.n_feature:2*self.n_feature], axis=0)
        avg_u_temp_feature = np.average(marginal[:, 2*self.n_feature:2*self.n_feature + self.n_user*self.n_feature].reshape(-1,self.n_user,self.n_feature), axis=0)
        avg_i_temp_feature = np.average(marginal[:, 2*self.n_feature + self.n_user*self.n_feature:].reshape(-1,self.n_item, self.n_feature), axis=0)
        
        train_preds = self._predict(train[:, :2], True, avg_u_temp_feature, avg_i_temp_feature)
        train_rmse = RMSE(train_preds, train[:, 2])
        val_preds = self._predict(validation[:, :2], True, avg_u_temp_feature, avg_i_temp_feature)
        val_rmse = RMSE(val_preds, validation[:, 2])
        print("After %d iteration, train RMSE: %.6f, validation RMSE: %.6f" % (self.iter_, train_rmse, val_rmse))

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayesian Probabilistic Matrix Factorization')
    parser.add_argument('--num-samples', type=int, default=8,
                        help='number of MCMC samples (default: 8)')
    parser.add_argument('--num-chains', type=int, default=1,
                        help='number of parallel MCMC chains (default: 1)')
    parser.add_argument('--warmup-steps', type=int, default=8,
                        help='number of MCMC samples for warmup (default: 8)')
    parser.add_argument('--jit', action='store_true', default=False)
    parser.add_argument('--n_feature', type=int, default=30,
                        help='number of feature dimension for each users and items (default: 30)')
    parser.add_argument('--n_threads', type=int, default=20,
                        help='number of threads (default: 20)')

    args = parser.parse_args()
    print("Loading data....")
    ratings = load_movielens_1m_ratings('../ml-1m/ratings-made.dat')
    print("Loaded")

    output_file = open("mcmc_result.txt", 'w')
    n_user = max(ratings[:,0])
    n_item = max(ratings[:,1])
    ratings[:,(0,1)] -= 1
    bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=args.n_feature,
                max_rating=5., min_rating=1., seed= 0, output_file = output_file)
    sigma = 0.5
    bpmf._main(ratings, sigma, args)
