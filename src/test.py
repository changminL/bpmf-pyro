import numpy as np
from six.moves import xrange
from bpmf import BPMF
from utils.evaluation import RMSE
from utils.datasets import load_movielens_1m_ratings

# load user ratings
ratings = load_movielens_1m_ratings('../ml-1m/ratings.dat')
n_user = max(ratings[:, 0])
n_item = max(ratings[:, 1])
ratings[:, (0, 1)] -= 1 # shift ids by 1 to let user_id & movie_id start from 0

# fit model
bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=10,
            max_rating=5., min_rating=1., seed=0).fit(ratings, n_iters=20)
print(RMSE(bpmf.predict(ratings[:, :2]), ratings[:,2])) # training RMSE

# predict ratings for user 0 and item 0 to 9:
print(bpmf.predict(np.array([[0, i] for i in xrange(10)])))
