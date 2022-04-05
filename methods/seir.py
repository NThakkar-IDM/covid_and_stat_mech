""" seir.py

SEIR model class and associated tools. This model is the most basic, with no spatial
correlation or connectivity. For more details, see the doc-string of the model class. """
import sys
import numpy as np

## For good hygiene
import warnings

class LogNormalSEIR(object):

	""" Discrete time SEIR model with log-normally distributed transmission. 

	S_t = S_{t-1} - N_{t-1} + V_{t-1}
	E_t = N_{t-1} + (1-(1/D_e))*E_{t-1}
	I_t = (1/D_e)*E{t-1} + (1-(1/D_i))*I_{t-1}
	N_t = beta*S_{t-1}*(I_{t-1}+z_{t-1})*epsilon_t

	D_e, D_i, and the initial condition are assumed known. beta and 
	transmission variance are meant to be inferred from data. """

	def __init__(self,S0,D_e,D_i,z_t):

		## Store the known model parameters
		self.z_t = z_t
		self.S0 = S0
		self.D_e = D_e
		self.D_i = D_i

		## Create a time axis
		self.T = len(z_t)
		self.time = np.arange(self.T)

		## Mark the model as un-fit, which means parameters
		## are missing.
		self._fit = False

def SampleOutcome(new_exposures,pr_samples,delay_samples):

	""" Sample outcomes accross samples of outcome probability and delays. """

	## Compute destined outcomes with a different Pr for each
	## trajectory.
	if len(pr_samples.shape) == 1:
		destined = np.random.binomial(np.round(new_exposures).astype(int),
									  p=pr_samples[:,np.newaxis])
	elif len(pr_samples.shape) == 2:
		destined = np.random.binomial(np.round(new_exposures).astype(int),
									  p=pr_samples)
	else:
		raise TypeError("IFR samples needs to be a 1 or 2 dimensional array!")

	## Finally, use np.roll to shift by appropriate numbers, and then
	## zero rolled entries for daily outcomes (this is the slow part!)
	daily = advindexing_roll(destined,delay_samples)
	for i,d in enumerate(delay_samples):
		daily[i,:d] = 0

	return destined, daily

def advindexing_roll(A, r):
	rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]
	r[r < 0] += A.shape[1]
	column_indices = column_indices - r[:,np.newaxis]
	return A[rows, column_indices]

def sample_traj(model,beta_t,num_samples=10000,sig_eps=None,z_t=None,avg_v_t=None,cov_v_t=None,ic=None):

	""" Sample the model, num_samples is the number of sample trajectories. Output has shape
	(num_samples,4,len(beta_t)). """

	## Set the variance for these samples
	if sig_eps is None:
		sig_eps = model.sig_eps*np.ones((len(beta_t),))

	## Set the importation scenario
	if z_t is None:
		z_t = model.z_t

	## Set the vaccination scenario
	if avg_v_t is None:
		avg_v_t = np.zeros((len(beta_t),))

	## Allocate storage for the output, and set up the
	## initial condition.
	X = np.zeros((num_samples,3,len(beta_t)))
	if ic is None:
		X[:,0,0] = model.S0
	else:
		X[:,:,0] = ic

	## If a vaccine covariance matrix is provided, pre-sample
	## the raw vaccine derived immunity time series. Otherwise,
	## reshape avg_v_t.
	if cov_v_t is None:
		v_t = avg_v_t[np.newaxis,:]
	else:
		v_t = np.random.multivariate_normal(avg_v_t,cov_v_t,size=(num_samples,))

	## Loop over time and collect samples
	total_exposures = np.zeros((num_samples,))
	for t in range(1,len(beta_t)):

		## Sample eps_t
		eps_t = np.exp(np.random.normal(beta_t[t-1],sig_eps[t-1],size=(num_samples,)))

		## Calculate the fraction exposed (for vaccine adjustment)
		frac_exp = total_exposures/model.S0

		## Calculate the number of exposures
		exposures = X[:,0,t-1]*(X[:,2,t-1]+z_t[t-1])*eps_t

		## Update all the 'deterministic' components (S and I)
		X[:,0,t] = X[:,0,t-1]-exposures-((1.-frac_exp)*v_t[:,t-1])
		X[:,2,t] = X[:,1,t-1]/model.D_e + X[:,2,t-1]*(1.-(1./model.D_i))

		## Update the exposed compartment across samples
		X[:,1,t] = exposures+X[:,1,t-1]*(1.-(1./model.D_e))

		## High sig-eps models require by-hand enforcement
		## of positivity (i.e. truncated gaussians).
		X[X[:,2,t]<0,2,t] = 0
		X[X[:,1,t]<0,1,t] = 0
		X[X[:,0,t]<0,0,t] = 0

		## Update total exposures to keep track of the
		## naturally immune fraction.
		total_exposures += exposures

	return X

def mean_traj(model,beta_t,z_t=None,sig_eps=None,avg_v_t=None,ic=None):

	""" Compute the mean trajectory given a time-varying beta_t series. """

	## Set the importation scenario
	if z_t is None:
		z_t = model.z_t

	## Set the variance over time
	if sig_eps is None:
		sig_eps = model.sig_eps*np.ones((len(beta_t),))

	## And the vaccination trace over time
	if avg_v_t is None:
		avg_v_t = np.zeros((len(beta_t,)))

	## Allocate storage for the output, and set up the
	## initial condition.
	X = np.zeros((3,len(beta_t)))
	if ic is None:
		X[0,0] = model.S0
	else:
		X[:,0] = ic

	## Loop over time and collect samples
	eps_t = np.exp(beta_t+0.5*(sig_eps**2))
	total_exposures = 0
	for t in range(1,len(beta_t)):

		## Update all the deterministic components (all of them in this case)
		frac_exp = total_exposures/model.S0
		X[0,t] = X[0,t-1]-X[0,t-1]*(X[2,t-1]+z_t[t-1])*eps_t[t-1]-(1.-frac_exp)*avg_v_t[t-1]
		X[2,t] = X[1,t-1]/model.D_e + X[2,t-1]*(1.-(1./model.D_i))
		X[1,t] = X[0,t-1]*(X[2,t-1]+z_t[t-1])*eps_t[t-1]+\
				   X[1,t-1]*(1.-(1./model.D_e))
		total_exposures += X[0,t-1]*(X[2,t-1]+z_t[t-1])*eps_t[t-1]

	return X