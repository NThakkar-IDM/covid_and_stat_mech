""" vaccine.py

Statistical models for vaccine derived immunity given data on doses. These functions
facilitate computing 'raw' vaccine derived immunity, i.e. the number of people who get
vaccine immunity every day (with uncertainty) based only on the probability of vaccine 
failure. In other words, these functions do not accout for vaccine doses given to people
who have already had covid, which is accounted for via adjustment after raw immunity is 
computed. """
import sys
import numpy as np

## For time series manipulation
import pandas as pd

## Basic immunity models
def BinomialVaccineImmunity(dose1,dose2,
							eps1=0.63,eps2=0.63,tau1=10,tau2=10):

	""" Use a gaussian approximation to a binomial distribution to compute daily
	numbers of people who get vaccine immunity. 

	dose1: pd.Series, number of first doses administered every day.
	dose2: pd.Series, number of second doses administered every day.
	eps_i: immunity probability for the ith dose.
	tau_i: int, days to immunity after dose. """

	## Shift both series to account for the time
	## to vaccine derived immunity
	t1 = pd.date_range(dose1.index[0],dose1.index[-1]+pd.to_timedelta(tau1,unit="d"),freq="d")
	t2 = pd.date_range(dose2.index[0],dose2.index[-1]+pd.to_timedelta(tau2,unit="d"),freq="d")
	dose1 = dose1.reindex(t1).shift(tau1).fillna(0)
	dose2 = dose2.reindex(t2).shift(tau2).fillna(0)

	## Compute the daily expected new immune
	expected_immune = (eps1*dose1 + (1-eps1)*eps2*dose2).rename("avg")
	var_immune = (eps1*(1-eps1)*dose1+(1-eps1)*eps2*(1-eps2)*dose2).rename("var")

	## Put it together
	df = pd.concat([expected_immune,var_immune],axis=1)
	return df

def BetaVaccineImmunity(dose1,dose2,
						a1=158.71841496675802,b1=114.73243246094387,
						a2=158.71841496675802,b2=114.73243246094387,
						tau1=21,tau2=21):
	
	""" Use expectation and variance operations to compute expected immunity every day based on
	beta distributed transmission blocking efficacy. 

	dose1: pd.Series, number of first doses administered every day.
	dose2: pd.Series, number of second doses administered every day.
	eps_i ~ Beta(ai,bi)
	tau_i: int, days to immunity after dose. """

	## Shift both series to account for the time
	## to vaccine derived immunity
	t1 = pd.date_range(dose1.index[0],dose1.index[-1]+pd.to_timedelta(tau1,unit="d"),freq="d")
	t2 = pd.date_range(dose2.index[0],dose2.index[-1]+pd.to_timedelta(tau2,unit="d"),freq="d")
	dose1 = dose1.reindex(t1).shift(tau1).fillna(0)
	dose2 = dose2.reindex(t2).shift(tau2).fillna(0)

	## Compute the daily contributions to immunity, using the expecation and variance
	## of 1 beta distributed variable and the product of 2 beta distributed variables.
	N1 = (a1/(a1+b1))*dose1
	V1 = (dose1**2)*(a1*b1)/(((a1+b1)**2)*(a1+b1+1))
	exp_prod = (a2/(a2+b2))*((b1/(a1+b1)))
	N2 = exp_prod*dose2
	V2 = (dose2**2)*exp_prod*((a2+1)*(b1+1)/((a2+b2+1)*(a1+b1+1))-exp_prod)

	## Combine and reshape
	expected_immune = (N1+N2).rename("avg")
	var_immune = (V1+V2).rename("var")
	df = pd.concat([expected_immune,var_immune],axis=1)
	return df

def BetaVaccineFailures(dose1,dose2,
						a1=158.71841496675802,b1=114.73243246094387,
						a2=158.71841496675802,b2=114.73243246094387,
						tau1=21,tau2=21):

	""" Use expectation and variance operations to estimate the running total of 
	vaccine failures given dose data. When modified by symptom suppression probability, this
	can be used to estimate vaccine-related changes in the IFR and IHR. 
	
	dose1: pd.Series, number of first doses administered every day.
	dose2: pd.Series, number of second doses administered every day.
	eps_i ~ Beta(ai,bi)
	tau_i: int, days to immunity after dose. """

	## Shift both series to account for the time
	## to vaccine derived immunity
	t1 = pd.date_range(dose1.index[0],dose1.index[-1]+pd.to_timedelta(tau1,unit="d"),freq="d")
	t2 = pd.date_range(dose2.index[0],dose2.index[-1]+pd.to_timedelta(tau2,unit="d"),freq="d")
	dose1 = dose1.reindex(t1).shift(tau1).fillna(0)
	dose2 = dose2.reindex(t2).shift(tau2).fillna(0)

	## Compute the daily contributions to failure, using the expecation and variance
	## of 1 beta distributed variable and the product of 2 beta distributed variables.
	N1 = (b1/(a1+b1))*dose1
	V1 = (dose1**2)*(a1*b1)/(((a1+b1)**2)*(a1+b1+1))
	exp_prod = (a2/(a2+b2))*((b1/(a1+b1)))
	N2 = exp_prod*dose2
	V2 = (dose2**2)*exp_prod*((a2+1)*(b1+1)/((a2+b2+1)*(a1+b1+1))-exp_prod)

	## Combine according to a source/sink model. That this
	## F_t = N1 - N2 + F_{t-t}
	expected_failures = np.cumsum(N1-N2).rename("avg")
	var_failures = np.cumsum(V1+V2).rename("var")

	## Reshape and return
	df = pd.concat([expected_failures,var_failures],axis=1)
	return df