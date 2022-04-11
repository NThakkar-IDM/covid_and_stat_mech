""" distributions.py

Classes and functions for matched distributions."""
import sys
import numpy as np

## For densities
from scipy.special import gamma, gammaln

def poisson_density(x,mu):
	ln_poisson = x*np.log(mu)-mu-gammaln(x+1)
	return np.exp(ln_poisson)

def gamma_poisson_density(x,m,k):
	ln_N = gammaln(x+k)-gammaln(x+1)-gammaln(k)
	ln_sf = x*np.log(m/(m+k))+k*np.log(k/(m+k))
	return np.exp(ln_N+ln_sf)

class NegativeBinomial(object):

	def __init__(self,mu,var,N=500):

		## Store the key variables
		self.mu = mu
		self.var = var
		self.N = N
		self.k = np.arange(self.N)

		## Compute the distribution, using either
		## the negative binomial or the poisson where the variance
		## is poorly defined.
		if self.var > self. mu:
			self.r = (mu**2)/(var-mu)
			self.p_k = gamma_poisson_density(self.k,self.mu,self.r)
			self.success = True
		elif self.var == self.mu or np.isnan(self.var):
			self.p_k = poisson_density(self.k,self.mu)
			self.success = False
		else:
			raise ValueError("mean {}, var {} is an ill-defined pair!".format(self.mu,self.var))
			self.success = False
			self.p_k = 1.+0*self.k
		self.p_k = self.p_k/np.sum(self.p_k)

		## Compute the empirical mean and variance
		self.exp_k = np.sum(self.k*self.p_k)
		self.var_k = np.sum(self.p_k*((self.k-self.exp_k)**2))

class PoissonDistribution(object):

	def __init__(self,mu,var,N=300):

		## Store the key variables
		self.mu = mu
		self.N = N
		self.k = np.arange(self.N)

		## Compute p0 and eta for undefined variance
		self.p_k = np.exp(-mu)*(mu**self.k)/gamma(self.k+1)
		self.p_k *= 1./(self.p_k.sum())
	
		## Compute the empirical mean and variance
		self.exp_k = np.sum(self.k*self.p_k)
		self.var_k = np.sum(self.p_k*((self.k-self.exp_k)**2))

		## Convergence check
		self.success = True
