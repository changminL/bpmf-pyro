# Bayesian Probabilistic Matrix Factorization with Pyro

Bayesian Probabilistic Matrix Factorization implementation with Pyro

Current model:
--------------
- Probabilistic Matrix Factorization
- Bayesian Matrix Factorization
- Bayesian Matrix Factorization(pyro)
- Alternating Least Squares with Weighted Lambda Regularization (ALS-WR)

Reference:
----------
- Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo (paper)
- code: https://github.com/chyikwei/recommend

Install:
--------
```
# clone repoisitory
git clone https://github.com/changminL/bpmf-pyro.git
cd bpmf-pyro
cd src
```

Running Test:
-------------
If you genearte 8 samples by using 20 threads on your computer.
Set the feature dimension as 30.
```
python bpmf-pyro.py --num-samples 8 --warmup-steps 8 --n_feature 30 --n_threads 20
```

If you generate 16 samples by using 20 threads on your computer.
Set the feature dimension as 60.
```
python bpmf-pyro.py --num-samples 16 --warmup-steps 16 --n_feature 60 --n_threads 20
```

Notes:
------
- This code is not perfect. We just suggest how can we implement bpmf with pyro. I hope it will give you some hint when you try to implement other models which use MCMC inference methods. 
- This is a project for a CS423(Probabilistic Programming, Spring 2019, KAIST) class.
