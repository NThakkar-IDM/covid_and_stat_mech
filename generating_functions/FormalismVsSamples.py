""" FormalismVsSamples.py

Script to create figure 3 from the paper, which compares 3 different
recursive computations to the empirical estimates via samples generated in
SampleTransmissionTrees.py """
import sys
sys.path.append("..\\")
import methods

## standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For ticks
import matplotlib.ticker as plt_ticker

## For probability calculations
from scipy.special import gamma, gammaln, binom

## For reproducability
np.random.seed(2)

## Some useful colors
plt.rcParams["font.size"] = 22.
c8c = {"red":"#ED0A3F","orange":"#FF8833",
	   "yellow":"#FBE870","green":"#01A638",
	   "blue":"#0066FF","violet":"#803790",
	   "brown":"#AF593E","black":"#000000"}
size_dist_color = "#F7AE30"
km_est_color = "#F74A30"
svl_est_color = "#F73079"

def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

def gamma_poisson_density(x,m,k):
	ln_N = gammaln(x+k)-gammaln(x+1)-gammaln(k)
	ln_sf = x*np.log(m/(m+k))+k*np.log(k/(m+k))
	return np.exp(ln_N+ln_sf)

def KaplanMeier(data,final_days=0,lifetime="duration"):

	## Get the observed events deaths, and align to a
	## specific index.
	d_i = data.loc[data["observed"] == 1,lifetime].value_counts().sort_index()
	d_i = d_i.reindex(np.arange(d_i.index[-1]+1+final_days)).fillna(0).rename("d_i")
	
	## Compute the individuals at risk at each time point. This would be based on
	## total individuals minus those with observed deaths, except that people's durations
	## can end for unobserved reasons. So we compute "exits" which is both observed and 
	## unobserved duration endings. n_i is shifted since it's the number of remaining folks
	## from the previous timestep.
	exits = data[lifetime].value_counts().sort_index().reindex(d_i.index).fillna(0)
	n_i = pd.Series(len(data)*np.ones((len(d_i),))-np.cumsum(exits),
					index=d_i.index,name="n_i").shift(1).fillna(method="bfill")
	df = pd.concat([d_i,n_i],axis=1)

	## Compute the MLE survival curve and associated
	## uncertainty.
	df["h_i"] = df["d_i"]/df["n_i"]
	df["var_h_i"] = (df["h_i"]*(1-df["h_i"]))/df["n_i"]
	df["S_t"] = np.cumprod(1.-df["h_i"])
	df["var_S_t"] = np.cumprod(df["var_h_i"]+(1-df["h_i"])**2)-df["S_t"]**2
	df["std_S_t"] = np.sqrt(df["var_S_t"])

	return df

if __name__ == "__main__":

	## Basic timescale business
	d_E = 5
	d_I = 4

	## Get the relevant inferences from the WA data
	t_df = pd.read_pickle("..\\pickle_jar\\T_dist_df.pkl")
	t_df["k"] = (t_df["avg"]**2)/(t_df["var"]-t_df["avg"])
	t_df["p"] = t_df["avg"]/(t_df["avg"]+t_df["k"])

	## Set up the relevant arrays defining the generating
	## function for stars.
	time = np.arange(len(t_df))
	k_t = t_df["k"].values
	p_t = t_df["p"].values

	## And the infectious periods
	rho = np.zeros((len(time),))
	rho[d_E:d_E+d_I] = 1
	rho = np.triu(np.vstack([np.roll(rho,i) 
							 for i in range(len(time))]))

	## Get the sampled trees for comparison
	samples = pd.read_pickle("..\\pickle_jar\\sampled_trees.pkl")

	## And compute the fraction of trees rooted at different
	## times.
	seed_dist = samples["start_date"].value_counts().sort_index()/(len(samples))
	
	## Set up an aggreated overview plot
	fig = plt.figure(figsize=(12,10))
	size_ax = fig.add_subplot(2,2,(1,1))
	km_ax = fig.add_subplot(2,2,(2,2))
	svsl_ax = fig.add_subplot(2,2,(3,4))

	## Should you compute from scratch or retrieve via
	## serialized output from a previous run?
	_compute = True

	## SIZE DIST COMPUTATION
	## Set up the discretization of the unit
	## circle in the complex plane
	if _compute:
		print("\nComputing the size dist via fft...")
		M = 2000
		m = np.arange(M)
		theta = 2*np.pi*m/M
		z = np.exp(1j*theta)

		## Initialize the recursion
		T = np.zeros((M,len(time)),dtype=np.cdouble)
		T[:,-1] = 1*z
		S_t = ((1. - p_t)/(1-p_t*T))**k_t

		## And step backwards through time.
		for t in np.arange(len(time)-1)[::-1]:
			
			## Get the infectious distribution
			rho_t = rho[t]

			## Add reporting?
			root = 1*z

			## Evaluate the tree function
			S_t[:,t+1:] = ((1. - p_t[t+1:])/(1-p_t[t+1:]*T[:,t+1:]))**(k_t[t+1:])
			T_t = root*np.prod(1 - rho_t + rho_t*S_t,
							axis=1)
			
			## Store the result
			T[:,t] = T_t

		## Evaluate the distribution via FFT
		fft_l = np.abs(np.fft.fft(T,axis=0)/M)
		fft_l = pd.DataFrame(fft_l.T,
							 columns=m,
							 index=t_df.index)
		fft_l.to_pickle("..\\pickle_jar\\fft_size_dist.pkl")

	else:
		fft_l = pd.read_pickle("..\\pickle_jar\\fft_size_dist.pkl")

	## Get the output from sampling
	sample_dist = samples["chain_size"].value_counts().sort_index()/(len(samples))
	
	## Model dist
	model_dist = fft_l.loc["2020-03-05":seed_dist.index[-1],1:]
	model_dist = model_dist.mean(axis=0)
	model_dist = model_dist/(model_dist.sum())

	## Make the size panel
	axes_setup(size_ax)
	size_ax.grid(color="grey",alpha=0.2)
	size_ax.set_yscale("log")
	size_ax.plot(sample_dist,color="k",lw=4,label="Sampled\ndistribution")
	size_ax.plot(model_dist,color=size_dist_color,lw=4,label="Recursive estimate")
	size_ax.set_xlim((-10,1000))
	size_ax.set_ylim((sample_dist.min(),None))
	size_ax.set_ylabel("Probability")
	size_ax.set_xlabel("Tree size")
	y_minor = plt_ticker.LogLocator(base=10.,
									subs=np.arange(1, 10,)*0.1,
									numticks=10)
	size_ax.yaxis.set_minor_locator(y_minor)
	size_ax.yaxis.set_minor_formatter(plt_ticker.NullFormatter())
	size_ax.legend(loc=1,frameon=False,fontsize=18)

	#### THE SURVIVAL FUNCTION
	## Set up all the Zs
	if _compute:
		print("\nComputing the survival function...")
		Z = np.tril(np.ones((len(time),len(time))))

		## And storage for the generating
		## function evaluations
		T = np.copy(Z)
		S_t = np.zeros((len(time),))

		## And step backwards through time over heaviside 
		## functions for z.
		for t in np.arange(len(time)-1)[::-1]:
			
			## fix z for the function evaluation
			z = Z[t]

			## Loop over seed times and evaluate the
			## recursion.
			for t0 in np.arange(t+1)[::-1]:
				S_t[t0+1:] = ((1. - p_t[t0+1:])/(1-p_t[t0+1:]*T[t,t0+1:]))**(k_t[t0+1:])
				T[t,t0] = np.prod(1 - rho[t0] + rho[t0]*S_t)

		## Reshape and format.
		T = pd.DataFrame(T,
						 index=t_df.index,
						 columns=t_df.index)
		survival = (1.-T).loc[:t_df.index[-2]]
		survival.to_pickle("..\\pickle_jar\\survival_functions.pkl")

	else:
		survival = pd.read_pickle("..\\pickle_jar\\survival_functions.pkl")

	## And the KM estimate for the samples
	samples["observed"] = (samples["end_date"] <= "2021-03-10").astype(int)
	samples["duration"] = 1 + samples["chain_life"]
	kmf = KaplanMeier(samples)
	
	## Set up the model dist
	model_dist = survival.loc["2020-03-04":,"2020-03-05"].copy().reset_index(drop=True)
	model_hazard = (1. - model_dist/(model_dist.shift(1))).loc[1:]

	## Compare with the KM estimate
	axes_setup(km_ax)
	km_ax.plot(kmf.index,
			  kmf["S_t"].values,
			  color="k",lw=4,label="KM estimate\nbased on samples")
	km_ax.plot(model_dist,
			  color=km_est_color,lw=4,label="Recursive estimate")
	km_ax.set_yscale("log")
	km_ax.set_xlim((kmf.index[0]-10,kmf.index[-1]))
	ylim = km_ax.get_ylim()
	km_ax.fill_between(model_hazard.index,0,model_hazard.values,
					  facecolor="grey",edgecolor="None",alpha=0.3,zorder=-1,
					  label="Daily hazard")
	km_ax.set_ylim(ylim)
	km_ax.set_ylabel("Survival function")
	km_ax.set_xlabel("Lifetime (days)")
	km_ax.legend(frameon=False,fontsize=18)

	## And then compute the size vs life relationship
	## Set up all the Zs
	if _compute:
		print("\nComputing the size vs life relationship...")
		Z = np.tril(np.ones((len(time),len(time))))

		## And storage for the generating
		## function evaluations
		T = np.copy(Z)
		S_t = np.zeros((len(time),))

		## And step backwards through time over heaviside 
		## functions for z (by stepping along the diagonal on
		## T, backwards).
		print("...starting with the cumulative probability...")
		for tL in np.arange(len(time)-1)[::-1]:
			for t0 in np.arange(tL+1)[::-1]:
				S_t[t0+1:] = ((1. - p_t[t0+1:])/(1-p_t[t0+1:]*T[tL,t0+1:]))**(k_t[t0+1:])
				T[tL,t0] = np.prod(1 - rho[t0] + rho[t0]*S_t)

		## Reshape and format.
		T = pd.DataFrame(T,
						 index=t_df.index.rename("lifetime"),
						 columns=t_df.index.rename("seedtime"))

		## Compute the derivative wrt to u, in a similar loop
		## along the diagonal.
		print("...then the first moment...")
		dT = np.copy(Z)
		dS_t = np.zeros(S_t.shape)
		for tL in np.arange(len(time)-1)[::-1]:
			for t0 in np.arange(tL+1)[::-1]:
				S_t[t0+1:] = ((1. - p_t[t0+1:])/(1-p_t[t0+1:]*T.values[tL,t0+1:]))**(k_t[t0+1:])
				dS_t[t0+1:] = (p_t[t0+1:]*k_t[t0+1:]*S_t[t0+1:])/(1.-p_t[t0+1:]*T.values[tL,t0+1:])
				comps = (rho[t0]*dS_t*dT[tL])/(1 - rho[t0] + rho[t0]*S_t)
				dT[tL,t0] = T.values[tL,t0]*(1. + np.sum(comps))
		
		## Reshape and format.
		dT = pd.DataFrame(dT,
						 index=t_df.index.rename("lifetime"),
						 columns=t_df.index.rename("seedtime"))

		## And compute the second derivative, to get to the variance.
		print("...and finally the second factorial moment...")
		d2T = np.zeros(Z.shape)
		d2S_t = np.zeros(S_t.shape)
		for tL in np.arange(len(time)-1)[::-1]:
			for t0 in np.arange(tL+1)[::-1]:
				S_t[t0+1:] = ((1. - p_t[t0+1:])/(1-p_t[t0+1:]*T.values[tL,t0+1:]))**(k_t[t0+1:])
				dS_t[t0+1:] = (p_t[t0+1:]*k_t[t0+1:]*S_t[t0+1:])/(1.-p_t[t0+1:]*T.values[tL,t0+1:])
				d2S_t[t0+1:] = (p_t[t0+1:]*(k_t[t0+1:]+1)*dS_t[t0+1:])/(1.-p_t[t0+1:]*T.values[tL,t0+1:])
				comps = (rho[t0]*dS_t*dT.values[tL])/(1 - rho[t0] + rho[t0]*S_t)
				term1 = 2*np.sum(comps)
				term2 = np.triu(np.outer(comps[t0:t0+d_E+d_I],comps[t0:t0+d_E+d_I]),k=1).sum().sum()
				term3 = (rho[t0]*d2S_t*dT.values[tL]*dT.values[tL] + rho[t0]*dS_t*d2T[tL])/(1 - rho[t0] + rho[t0]*S_t)
				d2T[tL,t0] = T.values[tL,t0]*(term1+term2+np.sum(term3))

		## Reshape and format.
		d2T = pd.DataFrame(d2T,
						 index=t_df.index.rename("lifetime"),
						 columns=t_df.index.rename("seedtime"))

		## Compute the expectation
		ExpN = (dT.diff(1))/(T.diff(1))
		ExpN = ExpN.loc["2020-03-01":t_df.index[-2]]

		## And the second factorial moment
		ExpN2 = (d2T.diff(1))/(T.diff(1))
		ExpN2 = ExpN2.loc["2020-03-01":t_df.index[-2]]

		## Compute the variance
		VarN = ExpN2 + ExpN - (ExpN**2)
		StdN = np.sqrt(VarN)

		## Save the results
		ExpN.to_pickle("..\\pickle_jar\\expN.pkl")
		StdN.to_pickle("..\\pickle_jar\\StdN.pkl")

	else:
		ExpN = pd.read_pickle("..\\pickle_jar\\expN.pkl")
		StdN = pd.read_pickle("..\\pickle_jar\\StdN.pkl")

	## Prepare plot samples
	plt_samples = samples.loc[samples["chain_size"] > 1]

	## Get the model estimate
	model_mean = ExpN.loc["2020-03-05":,"2020-03-05"].reset_index(drop=True)
	model_std = StdN.loc["2020-03-05":,"2020-03-05"].reset_index(drop=True)
	
	## And finish the figure
	axes_setup(svsl_ax)
	svsl_ax.set_yscale("log")
	svsl_ax.set_xscale("log")
	svsl_ax.grid(color="grey",alpha=0.2)
	svsl_ax.plot(plt_samples["chain_life"].values,plt_samples["chain_size"].values,
			  marker=".",markersize=5,
			  markeredgecolor="None",
			  ls="None",
			  color="k",
			  alpha=0.2)
	svsl_ax.plot([],[],marker="o",color="k",ls="None",label="Sampled trees\nwith non-trivial size")
	svsl_ax.fill_between(model_mean.index,
					  model_mean-1.*model_std,
					  model_mean+1.*model_std,
					  facecolor=svl_est_color,edgecolor="None",alpha=0.3)
	svsl_ax.plot(model_mean,
			  color=svl_est_color,lw=4,label="Recursive estimate")#,label="KM estimate based on samples")
	svsl_ax.set_ylabel("Tree size")
	svsl_ax.set_xlabel("Lifetime (days)")
	svsl_ax.set_ylim((1,None))
	y_minor = plt_ticker.LogLocator(base=10.,
									subs=np.arange(1, 10,)*0.1,
									numticks=10)
	svsl_ax.yaxis.set_minor_locator(y_minor)
	svsl_ax.yaxis.set_minor_formatter(plt_ticker.NullFormatter())
	svsl_ax.legend(frameon=False)
	
	## Finish up
	fig.tight_layout()

	## Add panel labels
	size_ax.text(-0.25,1.,"a.",fontsize=20,color="k",transform=size_ax.transAxes)
	km_ax.text(-0.25,1.,"b.",fontsize=20,color="k",transform=km_ax.transAxes)
	svsl_ax.text(-0.11,1.,"c.",fontsize=20,color="k",transform=svsl_ax.transAxes)

	## Done.
	fig.savefig("..\\_plots\\formalism_vs_samples.png")
	plt.show()
