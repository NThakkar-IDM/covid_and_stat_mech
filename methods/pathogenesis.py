""" pathogenesis.py

Methods and classes for pathogensis based signal processing. """
import sys

## Standard imports
import numpy as np
import pandas as pd

## For debug/step-by-step plots
import matplotlib.pyplot as plt

## For spline smoothing
from methods.splines import SmoothingSpline, SampleSpline

## For epi-curve construction
from scipy.optimize import minimize

def AdjustedRWEpiCurve(timeseries,ihr,prior_ihr,
					   correlation_time=28,end_index=None,weekend_smooth=True,hosp_only=False,debug=False):

	""" Construct an epi-curve based on a combination of testing and hospitalization data,
	with an eye towards pooling information for better estimates. This version adjusts the 
	hospitalization data using estimates of the IHR.

	timeseries is a data frame with cols cases, negatives, and hospitalization. 
	correlation_time: in days, for the random walk. corresponds to the time-scale on which
					  hospitalization data should be followed. 

	end_index: None means that random walk weights are used for the whole time series. 
			   Supplying an integer cuts the last n weights and forward fills, so make the
			   connection to cases less flexible at the end of the timeseries in response to
			   hosps being potentially a lagging indicator. """

	## Create a hospitalization adjustment using the ihr values. This takes
	## our IHR estimates and compares them to the population average
	## to account for transients in the age-distribution of the infected population.
	ihr_adj = prior_ihr["mean"].reindex(ihr.index).interpolate().fillna(method="bfill").fillna(method="ffill")
	ihr_adj = (ihr_adj/ihr["mean"]).rename("adjustment")

	## Construct the spline epi curve using the function above.
	## This essentially interprets the cases as directly connected to
	## epi when corrected for weekend effects.
	if weekend_smooth:
		spline = SplineTestingEpiCurve(timeseries,debug=False).rename("spline")
	else:
		spline = timeseries["cases"].rename("spline")

	## Concatenate into a single, aligned dataframe to make
	## referencing easier throughout.
	df = pd.concat([spline,timeseries["hosp"],ihr_adj],axis=1)
	df = df.loc[spline.loc[spline>0].index[0]:spline.index[-1]]

	## Apply the adjustment factor to the hospitalization
	## data used for the random walk.
	df["hosp"] *= df["adjustment"].fillna(1.)

	## Do you want just adjusted hosps?
	if hosp_only:
		rw = df["adjustment"].copy().reindex(spline.index).fillna(1.)
		epi_curve = df["hosp"].copy().reindex(spline.index).fillna(0)
		result = "No regression required."
	
	else:
		## Then set up the regularization matrix for the parameters 
		## (fixed effect for the scale factor, random walk for the weights).
		T = len(df)
		D2 = np.diag(T*[-2])+np.diag((T-1)*[1],k=1)+np.diag((T-1)*[1],k=-1)
		D2[0,2] = 1
		D2[-1,-3] = 1
		lam = np.dot(D2.T,D2)*((correlation_time**4)/8.)
		
		## Set up cost function to be passed to scipy.minimize, and
		## it's gradient to boost efficiency and stability. Then
		## solve the regression problem.
		def cost(theta):
			beta = 1./(1. + np.exp(-theta))
			f = beta*df["spline"].values
			ll = np.sum((df["hosp"].values-f)**2)
			lp = np.dot(theta.T,np.dot(lam,theta))
			return ll+lp
		def grad_cost(theta):
			beta = 1./(1. + np.exp(-theta))
			f = beta*df["spline"].values
			grad = -2.*(df["spline"].values)*(df["hosp"].values-f)*beta*(1.-beta)
			grad += 2.*np.dot(lam,theta)
			return grad
		alpha = np.sum(spline.values*timeseries["hosp"].values)/np.sum(spline.values**2)
		x0 = np.log(alpha/(1-alpha))*np.ones((T,))
		result = minimize(cost,x0=x0,
						  jac=grad_cost,
						  options={"gtol":5e-4})
		beta = 1./(1. + np.exp(-result["x"]))

		## If specified, replace the final values with those at
		## end index to make the connection to cases more rigid, and
		## to use that as a closer proxy for overall infections.
		if end_index is not None:
			beta[-end_index:] = beta[-end_index]
		
		## Finally, construct the fitted epi-curve
		epi_curve = (beta*df["spline"].copy()).reindex(spline.index).fillna(0)
		rw = pd.Series(beta,index=df.index,name="rw").reindex(spline.index).fillna(method="bfill")

	## Plot if debug
	if debug:
		print("\nRegression result")
		print(result)

		## Set up a figure
		fig, axes = plt.subplots(3,1,sharex=True,figsize=(12,11))
		for i, ax in enumerate(axes):
			ax.spines["left"].set_position(("axes",-0.015))
			if i != 2:
				ax.spines["left"].set_visible(False)
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)

		## Plot the case data
		axes[0].plot(timeseries["cases"],ls="None",marker="o",markersize=9,alpha=0.5,
					 markeredgecolor="k",markerfacecolor="None",markeredgewidth=1,
					 lw=1,color="k",label="Daily COVID-19 cases")
		axes[0].plot(spline,lw=3,color="#8F2D56",label=r"$\tilde{C}_t$, "+"epi-curve based\non testing data only")
		axes[0].set_ylim((0,None))
		axes[0].set_yticks([])
		axes[0].legend(loc="upper left",
					   bbox_to_anchor=(-0.08,0.95),
					   fontsize=20,frameon=False)
		
		## Plot the epi curve
		axes[1].plot(timeseries["hosp"],ls="None",marker="o",markersize=9,alpha=0.5,
					 markeredgecolor="k",markerfacecolor="None",markeredgewidth=1,
					 lw=1,color="k",label="Daily hospital admissions")
		axes[1].plot(epi_curve,lw=3,color="#EDAE01",label=r"$\tilde{H}_t$, "+"epi-curve regularized\nby hospitalizations")
		axes[1].set_ylim((0,None))
		axes[1].set_yticks([])
		axes[1].legend(loc="upper left",
					   bbox_to_anchor=(-0.08,0.95),
					   fontsize=20,frameon=False)
		
		## Set up the middle panel
		axes[2].fill_between(rw.index,0,rw.values,color="#662E1C",alpha=0.2)
		axes[2].plot(rw,c="#662E1C",lw=4)
		axes[2].set_ylim((0,None))
		axes[2].set_ylabel(r"Random walk, $f(\mu_t^*)$")

		## Finish up details
		ticks = pd.date_range(timeseries.index[0],timeseries.index[-1],freq="MS")
		tick_labels = [t.strftime("%b") for t in ticks]
		axes[1].set_xticks(ticks)
		axes[1].set_xticklabels(tick_labels)
		fig.tight_layout()
		#fig.savefig("_plots\\debug_blended.png")

		## Testing figure for reference
		timeseries["tests"] = timeseries["cases"]+timeseries["negatives"]
		fig, axes = plt.subplots(figsize=(12,5))
		axes.spines["left"].set_position(("axes",-0.025))
		axes.spines["top"].set_visible(False)
		axes.spines["right"].set_visible(False)
		axes.plot(timeseries["tests"],color="k",lw=1,ls="dashed")
		axes.plot(timeseries["tests"].rolling(7).mean(),color="#505160",lw=3)
		axes.set_ylabel("Daily COVID-19 tests")
		fig.tight_layout()
		#fig.savefig("_plots\\debug_tests.png")

		## IHR adjustment
		fig, axes = plt.subplots(2,1,sharex=True,figsize=(12,8))
		for ax in axes:
			ax.spines["left"].set_position(("axes",-0.025))
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
		axes[0].fill_between(df.index,
							 timeseries.loc[df.index[0]:,"hosp"].values,df["hosp"].values,
							 facecolor="xkcd:red wine",edgecolor="None",alpha=0.6)
		axes[0].plot(timeseries["hosp"],alpha=1,
					 lw=2,color="grey",label="Actual data")
		axes[0].plot(df["hosp"],alpha=1,
					 lw=2,color="k",label="Adjusted data")
		axes[0].legend(loc="upper left",
					   fontsize=20,frameon=False)
		axes[0].set_ylabel("Hospital admissions")
		axes[1].plot(df["adjustment"],color="k",lw=3)
		axes[1].set_ylabel("IHR adjustment")
		fig.tight_layout()
		#fig.savefig("_plots\\debug_adjustment.png")

		plt.show()
		sys.exit()

	return epi_curve, rw, spline

def SplineTestingEpiCurve(dataset,debug=False):

	""" Create an epi curve based on fraction positive and smoothed total tests. dataset is a 
	dataframe with a daily time index with cases and negatives as columns. Smoothing here is done
	using a smoothing spline with a 3 day prior correlation. """

	## Compute fraction positive
	total_tests = dataset["cases"]+dataset["negatives"]
	fraction_positive = (dataset["cases"]/total_tests).fillna(0)

	## Compute spline smoothed total tests
	spline = SmoothingSpline(np.arange(len(total_tests)),total_tests.values,lam=((3**4)/8))
	smooth_tt = pd.Series(spline(np.arange(len(dataset))),
						  index=dataset.index) 
	smooth_tt.loc[smooth_tt<0] = 0

	## Compute the epicurve estimate
	epi_curve = fraction_positive*smooth_tt

	## Make a diagnostic plot if needed
	if debug:
		fig, axes = plt.subplots(3,1,sharex=True,figsize=(12,10))
		axes[0].plot(dataset["cases"],c="k",ls="dashed",lw=1,
					 label="WDRS COVID-19 positives")
		axes[0].plot(fraction_positive*smooth_tt,c="xkcd:red wine",lw=2,
					 label="Epidemiological curve, based on smoothed\ntests, used to estimate " +r"R$_e$")
		axes[1].plot(total_tests,c="k",ls="dashed",lw=1,
					 label="Raw total daily tests")
		axes[1].plot(smooth_tt,c="xkcd:red wine",lw=2,
					 label="Smoothed tests with a 3 day correlation\ntime, correcting for fluctuations")
		axes[1].set_xlim(("2020-02-01",None))
		axes[2].plot(fraction_positive.loc[fraction_positive.loc[fraction_positive!=0].index[0]:],
					 c="grey",lw=3,
					 label="Raw fraction positive, computed with WDRS positive\nand negative tests, declines with increased testing volume")
		for ax in axes:
			ax.legend(frameon=False,fontsize=18)
		axes[0].set_ylabel("Epi-curve")
		axes[1].set_ylabel("Total COVID-19 tests")
		axes[2].set_ylabel("Fraction positive")
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug.png")
		plt.show()
		sys.exit()

	return epi_curve

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