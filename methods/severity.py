""" severity.py 

Methods and data required to dynamically assess the probability of hospitalization and death given
infection. """
import sys
sys.path.append("..\\")
import warnings

## Standard imports
import numpy as np
import pandas as pd

## Get the probability of death, via
## https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
ifr_table = pd.DataFrame([("0 to 9",0.00003,0.00002,0.0001),
						  ("10 to 19",0.00003,0.00002,0.0001),
						  ("20 to 29",0.0002,0.00007,0.0003),
						  ("30 to 39",0.0002,0.00007,0.0003),
						  ("40 to 49",0.0002,0.00007,0.0003),
						  ("50 to 59",0.005,0.0025,0.010),
						  ("60 to 69",0.005,0.0025,0.010),
						  ("70 to 79",0.054,0.028,0.093),
						  ("over 80",0.054,0.028,0.093)],
						  columns=["age","mid","low","high"]).set_index("age")

## Get the probability infections are hospitalized
## via Table 3
ihr_table = pd.DataFrame([("0 to 9",0,0,0),
						  ("10 to 19",0.000408,0.000243,0.000832),
						  ("20 to 29",0.0104,0.00622,0.0213),
						  ("30 to 39",0.0343,0.0204,0.0700),
						  ("40 to 49",0.0425,0.0253,0.0868),
						  ("50 to 59",0.0816,0.0486,0.167),
						  ("60 to 69",0.118,0.0701,0.24),
						  ("70 to 79",0.166,0.0987,0.338),
						  ("over 80",0.184,0.11,0.376)],
						  columns=["age","mid","low","high"]).set_index("age")


#### Time-varying IFR
##############################################################################
def GaussianProcessIFR(age_structured_cases,pyramid,ifr,
					   num_samples=10000,vectorized=True):

	""" Compute age-trend adjusted IFR over time. age_structured_cases is a weekly
	dataframe, with age-bins for columns. IFR and pyramid are dataframes with IFR estimates
	by age bin and population fraction by age bin respectively. 

	This function convolves an age-pyramid based prior with weekly case-based trends to get
	a weekly IFR estimate with uncertainty that responds to definitive transient changes in 
	the ages of people being infected. 

	Output is a dataframe of weekly IFR estimates and variance, to be resamples and interpolated
	for IFR-based initial condition estimation and mortality fitting. """

	## Get the length of time series and number of age bins for
	## reference throughout.
	T = len(age_structured_cases)
	K = len(ifr)

	## Compute a set of IFR samples in each age bin, assuming
	## the IFR is uniformly distributed on the 95% CI.
	ifr_samples = 100*np.random.uniform(low=ifr["low"].values,
										high=ifr["high"].values,
										size=(num_samples,len(ifr)))

	## Compute the time invariate, demographic based estimate
	time_invariant = np.dot(ifr_samples,pyramid.values)
	#prior_mean = 1.5*np.ones((len(df),)) #time_invariant.mean()*np.ones((len(df),))
	#prior_var = 0.49*np.ones((len(df),)) #time_invariant.var()*np.ones((len(df),))
	prior_mean = time_invariant.mean()*np.ones((T,))
	prior_var = time_invariant.var()*np.ones((T,))
	prior = pd.DataFrame(np.array([prior_mean,prior_var]).T,
						 index=age_structured_cases.index,
						 columns=["mean","var"])

	## Sample a dirichlet distribution to compute the case-based
	## time varying estimate. This is done either with a (less interpretable)
	## vectorized option, or via a loop over time.
	alpha = age_structured_cases.copy()+1
	if vectorized:
		dist_samples = np.random.standard_gamma(alpha.values,size=(num_samples,T,K))
		dist_samples = dist_samples/dist_samples.sum(axis=-1,keepdims=True)
		avg_ifr_samples = (dist_samples*(ifr_samples[:,np.newaxis,:])).sum(axis=-1)
		case_based = pd.DataFrame(np.array([avg_ifr_samples.mean(axis=0),avg_ifr_samples.var(axis=0)]).T,
								  index=alpha.index,
								  columns=["mean","var"])
	else:
		case_based = []
		for t, a in alpha.iterrows():
			dist_samples = np.random.dirichlet(a.values,size=(num_samples,))
			avg_ifr_samples = np.sum(ifr_samples*dist_samples,axis=1)
			case_based.append([avg_ifr_samples.mean(),avg_ifr_samples.var()])
		case_based = pd.DataFrame(case_based,
								  index=alpha.index,
								  columns=["mean","var"])

	## Compute a posterior estimate, using the population based
	## estimate as a regularizing prior on the case-based estimate.
	post_mean = (prior["var"]*case_based["mean"]+case_based["var"]*prior["mean"])/(case_based["var"]+prior["var"])
	post_var = (case_based["var"]*prior["var"])/(case_based["var"]+prior["var"])
	post = pd.concat([post_mean.rename("mean"),post_var.rename("var")],axis=1)
	
	return prior, case_based, post

#### Time-varying IHR
##############################################################################
def GaussianProcessIHR(age_structured_cases,pyramid,ihr,
					   num_samples=10000,vectorized=True):

	""" Compute age-trend adjusted IHR over time. age_structured_cases is a weekly
	dataframe, with age-bins for columns. IHR and pyramid are dataframes with IHR estimates
	by age bin and population fraction by age bin respectively. 

	This function convolves an age-pyramid based prior with weekly case-based trends to get
	a weekly IHR estimate with uncertainty that responds to definitive transient changes in 
	the ages of people being infected. 

	Output is a dataframe of weekly IHR estimates and variance, to be resamples and interpolated
	for IHR-based initial condition estimation and mortality fitting. """

	## Get the length of time series and number of age bins for
	## reference throughout.
	T = len(age_structured_cases)
	K = len(ihr)

	## Compute a set of IFR samples in each age bin, assuming
	## the IFR is uniformly distributed on the 95% CI.
	ihr_samples = 100*np.random.uniform(low=ihr["low"].values,
										high=ihr["high"].values,
										size=(num_samples,len(ihr)))

	## Compute the time invariate, demographic based estimate
	time_invariant = np.dot(ihr_samples,pyramid.values)
	prior_mean = time_invariant.mean()*np.ones((T,))
	prior_var = time_invariant.var()*np.ones((T,))
	prior = pd.DataFrame(np.array([prior_mean,prior_var]).T,
						 index=age_structured_cases.index,
						 columns=["mean","var"])

	## Sample a dirichlet distribution to compute the case-based
	## time varying estimate. This is done either with a (less interpretable)
	## vectorized option, or via a loop over time.
	alpha = age_structured_cases.copy()+1
	if vectorized:
		dist_samples = np.random.standard_gamma(alpha.values,size=(num_samples,T,K))
		dist_samples = dist_samples/dist_samples.sum(axis=-1,keepdims=True)
		avg_ihr_samples = (dist_samples*(ihr_samples[:,np.newaxis,:])).sum(axis=-1)
		case_based = pd.DataFrame(np.array([avg_ihr_samples.mean(axis=0),avg_ihr_samples.var(axis=0)]).T,
								  index=alpha.index,
								  columns=["mean","var"])
	else:
		case_based = []
		for t, a in alpha.iterrows():
			dist_samples = np.random.dirichlet(a.values,size=(num_samples,))
			avg_ihr_samples = np.sum(ihr_samples*dist_samples,axis=1)
			case_based.append([avg_ihr_samples.mean(),avg_ihr_samples.var()])
		case_based = pd.DataFrame(case_based,
								  index=alpha.index,
								  columns=["mean","var"])

	## Compute a posterior estimate, using the population based
	## estimate as a regularizing prior on the case-based estimate.
	post_mean = (prior["var"]*case_based["mean"]+case_based["var"]*prior["mean"])/(case_based["var"]+prior["var"])
	post_var = (case_based["var"]*prior["var"])/(case_based["var"]+prior["var"])
	post = pd.concat([post_mean.rename("mean"),post_var.rename("var")],axis=1)
	
	return prior, case_based, post