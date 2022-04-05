""" pathogenesis.py

Methods and classes for pathogensis based signal processing. """
import sys

## Standard imports
import numpy as np
import pandas as pd

## For debug/step-by-step plots
import matplotlib.pyplot as plt

def PathogenesisGPR(model,phi,fmu,debug=False):

	""" Pathogenesis based signal processing to compute I_t, E_t, and N_t up to an unknown constant
	with associated covariance matricies. The method relies on the pathogenesis distribution from
	https://advances.sciencemag.org/content/6/33/eabc1202 to create a linear operator mapping the time
	between exposure and symptom onset (i.e. midpoint infectiousness, by assumption). 

	model: the model class in model.py
	phi: an epi-curve, weekend effects are expected, as a daily pd.timeseries.
	fmu: the random walk output used to construct the epi-curve, aligned in time. """

	## First set up the pathogensis distribution and 
	## operator, normalized to estimate the expected infectious
	## population at time t given exposure at time t'
	alpha = 1.97
	lam = 0.11
	y = np.arange(0,len(phi))
	inc_dist = alpha*lam*((y*lam)**(alpha-1.))*np.exp(-(y*lam)**(alpha))
	inc_dist *= model.D_i/(inc_dist.sum())
	Ps = np.tril(np.array([np.roll(inc_dist,i) for i in np.arange(len(inc_dist))]).T)
	P = Ps[:,:-2]

	## Set the durations
	D_e = model.D_e
	D_i = model.D_i

	## Construct the mapping from I_t to E_t
	D1 = D_e*np.diag((-1.+(1./D_i))*np.ones((len(phi)-1,)))\
		   + D_e*np.diag(np.ones((len(phi)-2,)),k=1)
	D1 = np.hstack([D1,np.zeros((len(phi)-1,1))])
	D1[-1,-1] = D_e

	## And then the mapping from E_t to N_t
	D2 = np.diag((-1.+(1./D_e))*np.ones((len(phi)-2,)))\
		   + np.diag(np.ones((len(phi)-3,)),k=1)
	D2 = np.hstack([D2,np.zeros((len(phi)-2,1))])
	D2[-1,-1] = 1.

	## Construct the pathogensis operator, and compute it's
	## SVD to estimate the effective rank
	A = np.dot(P,np.dot(D2,D1))
	L = A - np.eye(len(A))
	u, s, vt = np.linalg.svd(L)
	p_k = s/np.sum(np.abs(s))
	H = np.sum(-p_k*np.log(p_k))
	erank = np.exp(H)

	## Construct the approximation by taking the effective
	## null space and projecting accordingly
	null_space = vt[int(np.ceil(erank)):,:]
	phi_hat = pd.Series(np.dot(np.dot(null_space,phi.values),null_space),
						index=phi.index,name="phi_hat")

	## Now estimate the variance, via the Gaussian process
	## regression. Compute the residual.
	r_t = phi-phi_hat

	## Do we want there to be missing data days?
	## and using the rolling modification of C_t to fit H_t as a measure
	## of relative observation volatility.
	## First construct the weekend residual.
	outlier = phi.index.weekday.isin({5,6})
	holidays = set([pd.to_datetime(d) for d in ["2020-01-01",
												"2020-01-20",
												"2020-02-17",
												"2020-05-25",
												"2020-07-03",
												"2020-09-07",
												"2020-11-11",
												"2020-11-26",
												"2020-12-24",
												"2020-12-25",
												"2021-01-01",
												"2021-01-18",
												"2021-02-15",
												]])
	holidays = phi.index.isin(holidays)
	outlier = outlier | holidays
	t = phi.loc[~outlier].index
	rw_t = r_t.loc[~outlier].values

	## Then get the observation variance matrix ready
	F = np.diag((fmu.values[~outlier]**2))
	Finv = np.diag(1./np.diag(F))
	w = rw_t.var()
	
	## Modify the pathogensis operator to account for outlier
	## days.
	Pw = P[~outlier,:]

	## Construct the GPR kernel modeling correlation at the
	## infectious duration timescale.
	s = (phi.index-phi.index[0]).days.astype(float).values[:-2]
	s = np.dot(s[:,np.newaxis],np.ones((len(s),1)).T)
	K = (w/((len(rw_t)+1)))*np.exp(-((s - s.T)**2)/(2.*(D_i**2)))
	#print((w/((len(rw_t)+1))))
	#K = 0.002*w*np.exp(-((s - s.T)**2)/(2.*(D_i**2)))
	
	## Construct the residual correlation matrix
	## and use it to solve for the conditional mean and
	## variance estimate of eps_t.
	M = np.dot(Pw,np.dot(K,Pw.T))+w*F
	Minv = np.linalg.inv(M)
	m_eps = np.dot(np.dot(K,np.dot(Pw.T,Minv)),rw_t)
	Veps = K - np.dot(K,np.dot(Pw.T,np.dot(Minv,Pw),K))

	## Use eps_t to construct an estimate of the variance
	#h_t = Veps + np.outer(m_eps,m_eps)
	h_t = np.diag(Veps) + m_eps**2 ## marginal variance estimate.

	## And project the solution to the I_t space with the
	## full projection matrix to get I_t's covariance matrix
	## and associated marginals.
	phi_cov = np.dot(np.dot(P,np.diag(h_t)),P.T)

	## From which we can construct the full suite of estimates
	eta_hat = pd.Series(np.dot(D1,phi_hat.values),
						index=phi.index[:-1])
	eta_cov = np.dot(D1,np.dot(phi_cov,D1.T))
	xi_hat = pd.Series(np.dot(D2,eta_hat.values),
					   index=eta_hat.index[:-1])
	xi_cov = np.dot(D2,np.dot(eta_cov,D2.T))

	if debug:

		## Set up colors
		colors = ["#bd33ff","#ff33db","#ff3375","#ff5733","#ffbd33","#dbff33"]

		## Plot the singular values and the null-space threshold
		fig, axes = plt.subplots(figsize=(8,6))
		axes.spines["left"].set_position(("axes",-0.025))
		axes.spines["top"].set_visible(False)
		axes.spines["right"].set_visible(False)
		axes.grid(color="grey",alpha=0.2)
		axes.plot(p_k,color=colors[3],lw=6)
		axes.axvline(erank,color=colors[4],ls="dashed",lw=3)
		axes.text(erank+10,p_k[0],"Effective\nnull space",
				  color=colors[4],fontsize=22,
				  horizontalalignment="left",verticalalignment="center")
		axes.set_yscale("log")
		axes.set_ylim((1e-5,None))
		axes.set_ylabel(r"$p_k = \sigma_k/|\sigma_k|$")
		axes.set_xlabel(r"Singular value index, $k$")
		fig.tight_layout()

		## Make a signal and noise plot
		phi_std = np.sqrt(np.diag(phi_cov))
		fig, axes = plt.subplots(2,1,sharex=True,figsize=(12,6))
		for ax in axes:
			ax.spines["left"].set_position(("axes",-0.025))
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
		axes[0].plot(phi,color="k",alpha=0.3)
		axes[0].fill_between(phi_hat.index,
							 phi_hat-2.*phi_std,
							 phi_hat+2.*phi_std,
							 facecolor=colors[0],edgecolor="None",alpha=0.6)
		axes[0].plot(phi_hat,color=colors[0],lw=2)
		axes[1].fill_between(r_t.index,-2.*phi_std,2.*phi_std,
							 facecolor=colors[0],edgecolor="None",alpha=0.6)
		axes[1].plot(r_t,color="k",alpha=0.3,lw=1)
		#axes[1].set_ylim((-8,8))
		fig.tight_layout()

		## Plot the estimates
		eta_std = np.sqrt(np.diag(eta_cov))
		xi_std = np.sqrt(np.diag(xi_cov))
		fig, axes = plt.subplots(3,1,sharex=True,figsize=(12,8))
		for ax in axes:
			ax.spines["left"].set_position(("axes",-0.025))
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
		axes[0].plot(phi,color="k",lw=1,alpha=0.3)
		axes[0].fill_between(phi_hat.index,
							 phi_hat-2.*phi_std,
							 phi_hat+2.*phi_std,
							 facecolor=colors[0],edgecolor="None",alpha=0.5)
		axes[0].plot(phi_hat,color=colors[0],lw=3)
		axes[1].fill_between(eta_hat.index,
							 eta_hat-2.*eta_std,
							 eta_hat+2.*eta_std,
							 facecolor=colors[1],edgecolor="None",alpha=0.5)	
		axes[1].plot(eta_hat,color=colors[1],lw=3)
		axes[2].fill_between(xi_hat.index,
							 xi_hat-2.*xi_std,
							 xi_hat+2.*xi_std,
							 facecolor=colors[2],edgecolor="None",alpha=0.5)	
		axes[2].plot(xi_hat,color=colors[2],lw=3)
		axes[0].set_ylabel(r"$\varphi_t$")
		axes[1].set_ylabel(r"$\eta_t$")
		axes[2].set_ylabel(r"$\xi_t$")
		fig.tight_layout()
		plt.show()
		sys.exit()

	return phi_hat, phi_cov, eta_hat, eta_cov, xi_hat, xi_cov